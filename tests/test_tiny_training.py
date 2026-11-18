import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_private_tiny as tiny
from train_private_long import interactive_loss
from evaluate_private_validation import infer_clicks


class ToyPrompt(torch.nn.Module):
    def forward(self, points, boxes, masks):
        return points[0], masks

    def get_dense_pe(self):
        return torch.zeros(1)


class ToySAM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(.1))
        self.prompt_encoder = ToyPrompt()

    def image_encoder(self, image):
        if self.training:
            raise AssertionError('Diagnostic must use the same eval mode for optimization and scoring')
        return F.avg_pool3d(image, 4)

    def mask_decoder(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output):
        point = sparse_prompt_embeddings[0, 0] / 4
        grid = torch.stack(torch.meshgrid(*[torch.arange(32, device=point.device) for _ in range(3)], indexing='ij'))
        distance = ((grid - point[:, None, None, None]) ** 2).sum(0)
        return self.bias + .01 * image_embeddings + .05 * torch.exp(-distance / 8), torch.zeros(1)


def fixture(root, image_split='train'):
    source = root / 'normal case_86'
    data = root / 'prepared'
    groups = {'train': [str(1000 + i) for i in range(54)], 'val': [str(2000 + i) for i in range(10)],
              'test': [str(3000 + i) for i in range(20)]}
    manifest = {'splits': groups, 'target': {'1': 'RFSDP', '9': 'LFSDP'}, 'cases': []}
    selected = tiny.select_patients(manifest, 2026) if image_split == 'train' else groups['val']
    for folder in ('imagesTr', 'labelsTr', 'ColorTable'):
        (source / folder).mkdir(parents=True)
    for folder in ('imagesTr', 'labelsTr'):
        (data / image_split / 'ct_private' / folder).mkdir(parents=True)
    for split, ids in groups.items():
        for key in ids:
            ct_path = source / 'imagesTr' / f'{key}.nii.gz'
            label_path = source / 'labelsTr' / f'{key}-label.nii.gz'
            manifest['cases'].append({'patient_id': key, 'split': split, 'source_image': str(ct_path),
                                      'source_label': str(label_path), 'right_voxels': 27, 'left_voxels': 27})
            if key not in selected:
                continue
            array = (np.arange(18 * 20 * 22).reshape(18, 20, 22) % 1100 - 200).astype(np.int16)
            ct = sitk.GetImageFromArray(array)
            ct.SetSpacing((.5, .7, 1.2))
            ct.SetOrigin((-10., -20., -30.))
            ct.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
            labels = np.zeros(array.shape, np.uint8)
            labels[3:6, 4:7, 5:8] = 1
            labels[10:13, 12:15, 14:17] = 9
            label = sitk.GetImageFromArray(labels)
            label.CopyInformation(ct)
            sitk.WriteImage(ct, str(ct_path))
            sitk.WriteImage(label, str(label_path))
            (data / image_split / 'ct_private/imagesTr' / f'{key}.nii.gz').hardlink_to(ct_path)
            binary = sitk.GetImageFromArray(np.isin(labels, [1, 9]).astype(np.uint8))
            binary.CopyInformation(ct)
            sitk.WriteImage(binary, str(data / image_split / 'ct_private/labelsTr' / f'{key}.nii.gz'))
            (source / 'ColorTable' / f'{key}_ColorTable.csv').write_text('LabelValue,Name\n1,RFSDP\n9,LFSDP\n')
    (data / 'manifest.json').write_text(json.dumps(manifest))
    return source, data, manifest


class TinyTrainingTests(unittest.TestCase):
    def test_frozen_selection_does_not_depend_on_input_order_and_rejects_leakage(self):
        manifest = {'splits': {'train': [str(i) for i in range(54)], 'val': [str(100 + i) for i in range(10)],
                               'test': [str(200 + i) for i in range(20)]}, 'target': {'1': 'RFSDP', '9': 'LFSDP'}}
        selected = tiny.select_patients(manifest, 2026)
        manifest['splits']['train'].reverse()
        self.assertEqual(tiny.select_patients(manifest, 2026), selected)
        self.assertTrue(set(selected) <= set(manifest['splits']['train']))
        manifest['splits']['test'][0] = selected[0]
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            tiny.select_patients(manifest, 2026)

    def test_side_crops_restore_original_rotated_grid_and_union_without_double_counting(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, data, manifest = fixture(root)
            run = root / 'run'
            run.mkdir()
            samples, patients = tiny.prepare_samples(source, data, manifest, run, 2026)
            self.assertEqual(len(samples), 4)
            self.assertEqual(len(patients), 2)
            restored = {}
            for sample in samples:
                self.assertEqual(sample['metadata']['target_retention'], 1)
                prediction = sample['target'][0, 0].numpy().astype(bool)
                indices = tiny.native_prediction_indices(prediction, sample['valid'], sample['metadata'], patients[sample['case']]['shape'])
                self.assertEqual(len(indices), 27)
                restored.setdefault(sample['case'], []).append(indices)
                # Padding predictions must not create native-grid false positives.
                padded = prediction | ~sample['valid']
                np.testing.assert_array_equal(indices, tiny.native_prediction_indices(padded, sample['valid'], sample['metadata'], patients[sample['case']]['shape']))
            for alias, parts in restored.items():
                gt = patients[alias]['target_indices']
                result = tiny.sparse_scores(np.concatenate(parts + [parts[0]]), gt)
                self.assertEqual(result['dice_full_target'], 1)
                self.assertEqual(result['predicted_voxels_valid'], 54)
                self.assertAlmostEqual(tiny.sparse_scores(parts[0], gt)['dice_full_target'], 2 / 3)
            selected = tiny.select_patients(manifest, 2026)[0]
            first = next(row for row in manifest['cases'] if row['patient_id'] == selected)
            image = sitk.ReadImage(first['source_label'])
            image.SetOrigin((99., 99., 99.))
            sitk.WriteImage(image, first['source_label'])
            bad_run = root / 'bad'
            bad_run.mkdir()
            with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
                tiny.prepare_samples(source, data, manifest, bad_run, 2026)

    def test_same_seed_training_interactions_reproduce_frozen_inference_and_gradients(self):
        from monai.losses import DiceCELoss
        model = ToySAM().eval()
        image = torch.linspace(-1, 1, 128 ** 3).reshape(1, 1, 128, 128, 128)
        target = torch.zeros_like(image).long()
        target[:, :, 60:64, 60:64, 60:64] = 1
        criterion = DiceCELoss(sigmoid=True, squared_pred=True)
        loss, logits = interactive_loss(model, image, target, criterion, 5, 42)
        frozen = infer_clicks(model, image, target, [1, 5], 42)
        np.testing.assert_array_equal(logits[0, 0].numpy(), frozen[5])
        loss.backward()
        self.assertTrue(torch.isfinite(model.bias.grad))
        self.assertNotEqual(model.bias.grad.item(), 0)

    def test_cpu_diagnostic_real_loss_exports_and_does_not_claim_success_for_low_dice(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, data, _ = fixture(root)
            run = root / 'run'
            run.mkdir()
            checkpoint = root / 'turbo.pth'
            torch.save({'model_state_dict': ToySAM().state_dict()}, checkpoint)
            registry = types.ModuleType('segment_anything.build_sam3D')
            registry.sam_model_registry3D = {'vit_b_ori': lambda checkpoint: ToySAM()}
            modules = {'segment_anything': types.ModuleType('segment_anything'), 'segment_anything.build_sam3D': registry}
            args = argparse.Namespace(sam_dir=root / 'sam', source_root=source, data_dir=data, checkpoint=checkpoint,
                                      run_dir=run, epochs=1, eval_every=1, lr=.001, target_dice=.9, seed=2026, device='cpu')
            def stub_overlay(path, *args):
                path.write_text('overlay invocation checked; renderer tested separately')
            previous = {name: sys.modules.get(name) for name in modules}
            sys.modules.update(modules)
            try:
                with patch.object(tiny, 'overlay', stub_overlay):
                    tiny.execute(args)
            finally:
                for name, value in previous.items():
                    if value is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = value
            summary = json.loads((run / 'summary.json').read_text())
            self.assertEqual(summary['completed_epochs'], 1)
            self.assertEqual(summary['optimizer_updates'], 4)
            self.assertFalse(summary['memorization_check_passed'])
            self.assertFalse(summary['validation_images_read'])
            self.assertFalse(summary['test_images_read'])
            self.assertEqual(summary['bilateral_total_clicks']['5'], 10)
            last = torch.load(run / 'sam_model_tiny_last.pth', weights_only=False)
            self.assertLess(last['model_state_dict']['bias'].item(), torch.tensor(.1).item())
            self.assertEqual(len(list((run / 'best_training').glob('*prediction_crop.nii.gz'))), 8)
            self.assertEqual(len(json.loads((run / 'selected_training_patients.json').read_text())), 2)


if __name__ == '__main__':
    unittest.main()
