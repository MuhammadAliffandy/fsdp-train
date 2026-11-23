import contextlib
import io
import json
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
import sam_refine32 as architecture
import sam_refine_data as views
import physical_fsdp as physical
from test_tiny_training import fixture


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Conv3d(1, 8, 1)

    def forward(self, image):
        return self.project(F.avg_pool3d(image, 16))


class FakePrompt(nn.Module):
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.point = nn.Linear(3, 8)
        self.label = nn.Embedding(2, 8)
        self.mask = nn.Conv3d(1, 8, 4, stride=4)

    def forward(self, points, boxes, masks):
        return self.point(points[0] / 128) + self.label(points[1]), self.mask(masks)

    def get_dense_pe(self):
        return torch.zeros((1, 8, 8, 8, 8), device=self.point.weight.device)


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Conv3d(8, 1, 1)
        self.point = nn.Linear(8, 1)

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output):
        logits = self.project(image_embeddings + dense_prompt_embeddings)
        logits = logits + self.point(sparse_prompt_embeddings.mean(1))[..., None, None, None]
        return F.interpolate(logits, size=(32,) * 3, mode='trilinear', align_corners=False), torch.zeros(1)


class FakeSAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = FakeEncoder()
        self.prompt_encoder = FakePrompt()
        self.mask_decoder = FakeDecoder()


def tensors():
    image = torch.randn((1, 1, 64, 64, 64))
    context = torch.randn((1, 1, 128, 128, 128))
    target = torch.zeros_like(image, dtype=torch.uint8)
    target[..., 28:36, 28:36, 28:36] = 1
    context_target = torch.zeros_like(context, dtype=torch.uint8)
    context_target[..., 60:68, 60:68, 60:68] = 1
    return image, context, target, context_target


class RefineArchitectureTests(unittest.TestCase):
    def test_physical_warp_and_click_coordinate_centers(self):
        coords = 2 * (torch.arange(128).float() + .5) / 128 - 1
        feature = coords[None, None, :, None, None].expand(1, 1, 128, 128, 128)
        ratio = torch.tensor([.5, .75, 1.])
        warped = architecture.context_to_detail(feature, ratio, (32, 32, 32))
        expected = (2 * (torch.arange(32).float() + .5) / 32 - 1) * .5
        torch.testing.assert_close(warped[0, 0, :, 16, 16], expected)
        cache = {'shape': (64, 96, 128), 'ratio': ratio}
        center = torch.tensor([[[31.5, 47.5, 63.5]]])
        torch.testing.assert_close(architecture.SamRefine32.map_points(cache, center), torch.full((1, 1, 3), 63.5))

    def test_zero_head_preserves_sam_and_real_cnn32_features(self):
        torch.manual_seed(2026)
        sam = FakeSAM()
        before = {k: v.detach().clone() for k, v in sam.state_dict().items()}
        model = architecture.SamRefine32(sam).eval()
        image, context, _, _ = tensors()
        with torch.inference_mode():
            cache = model.encode(image, context, torch.tensor([.5] * 3), torch.tensor([.35] * 3))
            self.assertEqual(tuple(cache['detail32'].shape), (1, 32, 32, 32, 32))
            self.assertEqual(tuple(cache['embedding'].shape[-3:]), (8, 8, 8))
            point, label = torch.tensor([[[32., 32., 32.]]]), torch.ones((1, 1), dtype=torch.long)
            refined, coarse, baseline = model.decode(cache, point, label, torch.zeros((1, 1, 32, 32, 32)), [(point, label)])
        torch.testing.assert_close(refined, baseline, rtol=0, atol=0)
        self.assertEqual(tuple(coarse.shape), (1, 1, 32, 32, 32))
        for key, value in sam.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_two_updates_reach_encoder_decoder_and_new_detail_branch(self):
        from monai.losses import DiceCELoss
        torch.manual_seed(2026)
        model = architecture.SamRefine32(FakeSAM()).train()
        image, context, target, context_target = tensors()
        before = model.sam.image_encoder.project.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, outputs = architecture.interactions(model, image, context, torch.tensor([.5] * 3), torch.tensor([.35] * 3),
                target, torch.ones_like(target, dtype=torch.bool), (1,), 2026,
                DiceCELoss(sigmoid=True, squared_pred=True), context_target, torch.ones_like(context_target, dtype=torch.bool))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(model.sam.image_encoder.project.weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.sam.mask_decoder.project.weight.grad.abs().sum().item(), 0)
            if step == 1:
                self.assertGreater(model.full[0].weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.detail32[0].weight.grad.abs().sum().item(), 0)
            optimizer.step()
        self.assertFalse(torch.equal(before, model.sam.image_encoder.project.weight))

    def test_click_sampling_excludes_padding_and_handles_no_error(self):
        target = torch.zeros((1, 1, 8, 8, 8), dtype=torch.uint8)
        target[..., 4, 4, 4] = 1
        valid = torch.zeros_like(target, dtype=torch.bool)
        valid[..., 4, 4, 4] = True
        logits = torch.ones_like(target).float() * 100
        point, label = architecture.next_click(logits, target, valid, np.random.default_rng(2026))
        torch.testing.assert_close(point, torch.tensor([[[4., 4., 4.]]]))
        self.assertEqual(label.item(), 1)

    def test_context_preprocessing_and_shared_lr_augmentation(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory).resolve()
            source, dataset, manifest = fixture(root)
            chosen = next(r for r in manifest['cases'] if Path(r['source_image']).exists())
            image = physical.sitk.ReadImage(chosen['source_image'])
            ct = physical.xyz(image)
            label = physical.xyz(physical.sitk.ReadImage(chosen['source_label']))
            norm = {'clip_lower': -1000., 'clip_upper': 1000., 'mean': 0., 'std': 1000.}
            plan = {'shape': [64, 112, 144], 'spacing': [.35] * 3}
            sample = physical.make_sample(image, ct, label == 1, 'train_01', 'RFSDP', plan, norm)
            prepared = {'policy': {'selected': plan, 'normalization': norm}, 'samples': {'train': [sample], 'val': []}}
            from evaluate_private_validation import file_hash
            previous = {'patient_id': chosen['patient_id'], 'source_image': chosen['source_image'], 'source_label': chosen['source_label'],
                'source_image_sha256': file_hash(Path(chosen['source_image'])), 'source_label_sha256': file_hash(Path(chosen['source_label']))}
            def records(unused, split):
                return [chosen] if split == 'train' else []
            with patch.object(physical, 'records_for', records):
                views.add_contexts(prepared, manifest, source, {'train': [previous], 'val': []})
            self.assertEqual(tuple(sample['sam_image'].shape), (1, 1, 128, 128, 128))
            a = np.asarray(sample['grid']['origin']) + (np.asarray(plan['shape']) - 1) * .35 / 2
            b = np.asarray(sample['sam_grid']['origin']) + 127 * np.asarray(sample['sam_grid']['spacing']) / 2
            np.testing.assert_allclose(a, b)
            first = views.paired_inputs(sample, torch.device('cpu'), 2026)
            second = views.paired_inputs(sample, torch.device('cpu'), 2026)
            for key in ('image', 'target', 'sam_image', 'sam_target', 'valid', 'sam_valid'):
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
            self.assertFalse(first['target'][~first['valid']].any())
            self.assertFalse(first['sam_target'][~first['sam_valid']].any())

class WorkflowTests(unittest.TestCase):
    def test_threshold_selected_on_train_before_validation(self):
        import train_sam_refine32 as trainer
        calls = []
        def evaluate(model, prepared, split, device, thresholds, seed, **kwargs):
            calls.append(split)
            def bundle(score):
                return {'metrics': {'mean_bilateral_patient_dice': score}}
            result = {.3: bundle(.8), .5: bundle(.2)} if split == 'train' else {.3: bundle(.1), .5: bundle(.9)}
            return {'refined': {5: result}}
        with patch.object(trainer, 'evaluate', evaluate):
            threshold, _, _ = trainer.calibrate_validate(None, {}, torch.device('cpu'), 2026)
        self.assertEqual(calls, ['train', 'val'])
        self.assertEqual(threshold, .3)

    def test_warmup_then_partial_turbo_trainability(self):
        import train_sam_refine32 as trainer
        sam = FakeSAM()
        sam.image_encoder.blocks = nn.ModuleList([nn.Conv3d(8, 8, 1) for _ in range(12)])
        model = architecture.SamRefine32(sam)
        self.assertEqual(trainer.set_stage(model, 0, 5), 'refiner_warmup')
        self.assertTrue(all(not p.requires_grad for p in sam.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.full.parameters()))
        self.assertEqual(trainer.set_stage(model, 5, 5), 'partial_turbo_and_refiner')
        for name, value in sam.named_parameters():
            expected = name.startswith(('image_encoder.blocks.10.', 'image_encoder.blocks.11.', 'mask_decoder.'))
            self.assertEqual(value.requires_grad, expected, name)
        self.assertFalse(sam.prompt_encoder.training)
        self.assertTrue(sam.mask_decoder.training)
        self.assertTrue(sam.image_encoder.blocks[10].training)
        self.assertFalse(sam.image_encoder.blocks[9].training)

    def test_export_actual_native_overlay_and_curve(self):
        import train_sam_refine32 as trainer
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            mask = np.zeros((16, 20, 24), bool)
            mask[7:10, 8:11, 10:14] = True
            sample = {'native_target': mask, 'raw_native': np.zeros_like(mask, dtype=np.float32),
                      'native': {'spacing': [.3, .4, 1.]}, 'name': 'val_01_RFSDP'}
            trainer.overlay(sample, mask, 1., .3, output)
            bundle = {'refined': {5: {.3: {'patients': [{'dice_full_target': .8}]}}}}
            trainer.plot_results([{'epoch': 1, 'train_dice_5': .8, 'val_dice_5': .4}], bundle, bundle, .3, output)
            self.assertEqual(len(list(output.glob('*.png'))), 2)

    def test_training_checkpoint_reload_export_and_resume(self):
        import argparse
        import tarfile
        import train_sam_refine32 as trainer
        from train_private_unet import state_hash
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            image = physical.sitk.GetImageFromArray(np.zeros((16, 16, 16), dtype=np.int16))
            image.SetSpacing((.35,) * 3)
            mask = np.zeros((16,) * 3, bool)
            mask[6:10, 6:10, 6:10] = True
            norm = {'clip_lower': -1000., 'clip_upper': 1000., 'mean': 0., 'std': 1000.}
            plan = {'shape': [64] * 3, 'spacing': [.35] * 3}
            prepared = {'samples': {}, 'patients': {}, 'policy': {}, 'coverage': {}}
            for split in ('train', 'val'):
                alias = split + '_01'
                prepared['samples'][split] = [physical.make_sample(image, np.zeros((16,) * 3), mask,
                    alias, side, plan, norm) for side in ('RFSDP', 'LFSDP')]
                prepared['patients'][split] = {alias: {'shape': (16,) * 3, 'target_indices': np.flatnonzero(mask)}}
            def build(args):
                sam = FakeSAM()
                sam.image_encoder.blocks = nn.ModuleList([nn.Conv3d(8, 8, 1) for _ in range(12)])
                return architecture.SamRefine32(sam), state_hash(sam.state_dict())
            def run_sample(model, sample, device, seed, criterion=None, augment=False, auxiliary_weight=.25):
                image = sample['image'].to(device)
                features = model.full(image)
                logits = model.residual(torch.cat((features, features), 1))
                logits = logits + model.sam.mask_decoder.project(model.sam.image_encoder.project(image))
                outputs = {click: {'refined': logits, 'coarse_on_detail': logits} for click in (1, 5)}
                loss = physical.masked_loss(criterion, logits, sample['target'], sample['valid']) if criterion is not None else None
                return loss, outputs
            args = argparse.Namespace(run_dir=root / 'run', resume=False, seed=2026, device='cpu', precision='bf16',
                epochs=2, val_every=1, accumulation=8, sam_lr=1e-5, head_lr=1e-4, head_warmup_epochs=1,
                auxiliary_weight=.25, augmentation='none', patience_epochs=50, min_delta=.001)
            args.run_dir.mkdir()
            config = {'experiment': 'synthetic_test_only', 'protocol': 'synthetic_test_only'}
            with patch.object(trainer, 'prepare', return_value=(prepared, config)), patch.object(trainer, 'build', build), \
                 patch.object(trainer, 'run_sample', run_sample):
                # The archive includes config.json as in a real prepared run.
                for name in ('config.json',):
                    physical.save(args.run_dir / name, config)
                original = trainer.train_epoch
                def interrupted(*values):
                    if values[6] == 1:
                        raise RuntimeError('simulated interruption')
                    return original(*values)
                with patch.object(trainer, 'train_epoch', interrupted), self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    trainer.execute(args)
                checkpoint = torch.load(args.run_dir / 'hybrid_latest.pth', weights_only=False)
                self.assertEqual(checkpoint['epoch'], 1)
                args.resume = True
                trainer.execute(args)
            summary = physical.read(args.run_dir / 'summary.json')
            self.assertTrue(summary['complete'])
            self.assertFalse(summary['test_images_read'])
            latest = torch.load(args.run_dir / 'hybrid_latest.pth', weights_only=False)
            self.assertEqual([r['stage'] for r in latest['history']], ['refiner_warmup', 'partial_turbo_and_refiner'])
            self.assertTrue(all(r['optimizer_updates'] == 1 for r in latest['history']))
            with tarfile.open(args.run_dir / 'results.tar.gz') as archive:
                names = archive.getnames()
            self.assertIn('results/summary.json', names)
            self.assertEqual(len([n for n in names if n.endswith('.png')]), 3)
            self.assertFalse(any(n.endswith('.pth') for n in names))


if __name__ == '__main__':
    unittest.main()
