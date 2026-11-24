import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fsdp_resolution as workflow
import physical_fsdp as physical
import sam_refine_data as views
from sam_refine32 import SamRefine32, interactions
from sam_refine_resolution import SamRefineResolution
from test_sam_refine32 import FakeSAM, FakePrompt, FakeDecoder


class DynamicPrompt(FakePrompt):
    image_embedding_size = (8,) * 3
    input_image_size = (128,) * 3

    def forward(self, points, boxes, masks):
        return self.point(points[0] / self.input_image_size[0]) + self.label(points[1]), self.mask(masks)

    def get_dense_pe(self):
        return torch.zeros((1, 8) + self.image_embedding_size, device=self.point.weight.device)


class DynamicDecoder(FakeDecoder):
    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output):
        assert image_embeddings.shape == image_pe.shape == dense_prompt_embeddings.shape
        logits = self.project(image_embeddings + dense_prompt_embeddings)
        logits = logits + self.point(sparse_prompt_embeddings.mean(1))[..., None, None, None]
        shape = tuple(4 * n for n in image_embeddings.shape[-3:])
        return F.interpolate(logits, size=shape, mode='trilinear', align_corners=False), torch.zeros(1)


def inputs(context_size=128):
    image = torch.randn(1, 1, 64, 80, 96)
    context = torch.randn((1, 1) + (context_size,) * 3)
    target = torch.zeros_like(image, dtype=torch.uint8)
    target[..., 28:35, 38:45, 45:52] = 1
    return image, context, torch.tensor([.5, .75, 1.]), torch.tensor([.25] * 3), target, torch.ones_like(target, dtype=torch.bool)


class ResolutionTests(unittest.TestCase):
    def test_baseline_is_numerically_identical_for_all_five_clicks_with_trained_refiner(self):
        torch.manual_seed(2026)
        original = SamRefine32(FakeSAM()).eval()
        nn.init.normal_(original.residual.weight, std=.01)
        candidate = SamRefineResolution(copy.deepcopy(original.sam)).eval()
        candidate.load_state_dict(original.state_dict(), strict=True)
        candidate.adapt('baseline')
        image, context, ratio, spacing, target, valid = inputs()
        with torch.inference_mode():
            _, before = interactions(original, image, context, ratio, spacing, target, valid, (1, 5), 2026)
            _, after = candidate.interaction_function(candidate, image, context, ratio, spacing, target, valid, (1, 5), 2026)
        for click in (1, 5):
            torch.testing.assert_close(before[click]['refined'], after[click]['refined'], rtol=0, atol=0)
        self.assertEqual(set(original.state_dict()), set(candidate.state_dict()))

    def test_native_half_uses_actual_rectangular_features_and_reaches_trained_cnn(self):
        model = SamRefineResolution(FakeSAM())
        source = copy.deepcopy(model.state_dict())
        model.adapt('detail')
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, source[key], rtol=0, atol=0)
        nn.init.normal_(model.residual.weight, std=.01)
        image, context, ratio, spacing, target, valid = inputs()
        cache = model.encode(image, context, ratio, spacing)
        self.assertEqual(cache['detail32'].shape[-3:], (32, 40, 48))
        self.assertEqual(cache['semantic32'].shape[-3:], (32, 40, 48))
        from monai.losses import DiceCELoss
        context_target = torch.zeros_like(context, dtype=torch.uint8)
        context_target[..., 60:68, 60:68, 60:68] = 1
        loss, _ = model.interaction_function(model, image, context, ratio, spacing, target, valid, (1,), 2026,
            DiceCELoss(sigmoid=True, squared_pred=True), context_target, torch.ones_like(context_target, dtype=torch.bool))
        loss.backward()
        self.assertGreater(model.detail32[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.sam.mask_decoder.project.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.sam.image_encoder.project.weight.grad.abs().sum().item(), 0)

    def test_token16_keeps_prompt_dense_encoding_feedback_and_coordinates_aligned(self):
        sam = FakeSAM(); sam.prompt_encoder = DynamicPrompt(); sam.mask_decoder = DynamicDecoder()
        model = SamRefineResolution(sam).eval()
        source = copy.deepcopy(model.state_dict())
        with patch('sam_refine_resolution.configure_encoder_grid') as configure:
            model.adapt('tokens')
        configure.assert_called_once_with(sam.image_encoder, 16)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, source[key], rtol=0, atol=0)
        image, context, ratio, spacing, target, valid = inputs(256)
        with torch.inference_mode():
            cache = model.encode(image, context, ratio, spacing)
            center = torch.tensor([[[31.5, 39.5, 47.5]]])
            torch.testing.assert_close(model.map_points(cache, center), torch.full((1, 1, 3), 127.5))
            _, outputs = model.interaction_function(model, image, context, ratio, spacing, target, valid, (1, 5), 2026)
        self.assertEqual(sam.prompt_encoder.input_image_size, (256,) * 3)
        self.assertEqual(outputs[5]['refined'].shape, image.shape)
        self.assertEqual(cache['embedding'].shape[-3:], (16,) * 3)
        self.assertEqual(model.coarse_size, 64)

    def test_direct_context_reconstruction_preserves_detail_tensors_center_and_field(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, cohort = root / 'source', root / 'cohort'
            cohort.mkdir(); source.mkdir()
            (root / 'config.json').write_text('{"cohort_run_dir": "' + str(cohort) + '"}')
            physical.save(cohort / 'config.json', {'data_dir': str(root), 'source_root': str(source)})
            physical.save(root / 'manifest.json', {})
            array = np.random.default_rng(2026).normal(size=(64,) * 3).astype(np.float32) * 100
            meta = {'size': [64] * 3, 'spacing': [.35] * 3, 'origin': [0.] * 3, 'direction': np.eye(3).ravel().tolist()}
            image = physical.image_xyz(array, meta)
            labels = np.zeros((64,) * 3, np.uint8); labels[28:34, 28:34, 28:34] = 1
            norm = {'clip_lower': -1000., 'clip_upper': 1000., 'mean': 0., 'std': 100.}
            plan = {'shape': [64, 80, 96], 'spacing': [.25] * 3}
            sample = physical.make_sample(image, array, labels == 1, 'train_01', 'RFSDP', plan, norm)
            prepared = {'samples': {'train': [sample], 'val': []}, 'policy': {'selected': plan, 'normalization': norm},
                        'source_provenance': {'train': [{'patient_id': '1'}], 'val': []}}
            def records(unused, split):
                return [{'patient_id': '1', 'source_image': str(source / 'imagesTr/1.nii.gz'),
                         'source_label': str(source / 'labelsTr/1-label.nii.gz')}] if split == 'train' else []
            def load(*unused):
                return image, None, array, labels, {}
            with patch.object(physical, 'records_for', records), patch.object(physical, 'load_case', load):
                views.add_contexts(prepared, {}, source, prepared['source_provenance'])
                old_grid = copy.deepcopy(sample['sam_grid'])
                pointers = [sample[k].data_ptr() for k in ('image', 'target', 'valid')]
                workflow.replace_contexts(prepared, SimpleNamespace(foundation_run_dir=root))
                actual_root = (root / 'actual data').resolve()
                def relocated(record, bound_root, expected, alias):
                    self.assertEqual(bound_root, actual_root)
                    self.assertEqual(record['source_image'], str(actual_root / 'imagesTr/1.nii.gz'))
                    self.assertEqual(record['source_label'], str(actual_root / 'labelsTr/1-label.nii.gz'))
                    return load()
                with patch.object(workflow, 'load_bound_case', relocated):
                    workflow.replace_contexts(prepared, SimpleNamespace(foundation_run_dir=root, source_root=actual_root))
            self.assertEqual(sample['sam_image'].shape[-3:], (256,) * 3)
            self.assertEqual(pointers, [sample[k].data_ptr() for k in ('image', 'target', 'valid')])
            center = lambda g: np.asarray(g['origin']) + (np.asarray(g['size']) - 1) * g['spacing'] / 2
            np.testing.assert_allclose(center(old_grid), center(sample['sam_grid']), atol=1e-12)
            np.testing.assert_allclose(np.asarray(old_grid['size']) * old_grid['spacing'],
                                       np.asarray(sample['sam_grid']['size']) * sample['sam_grid']['spacing'])
            expected = physical.xyz(physical.resample(image, sample['sam_grid'], physical.sitk.sitkLinear, -1000.)) / 100
            expected[~sample['sam_valid'][0, 0].numpy()] = 0
            np.testing.assert_allclose(sample['sam_image'][0, 0].numpy(), expected)


if __name__ == '__main__':
    unittest.main()
