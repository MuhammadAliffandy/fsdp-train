import contextlib
import io
import numpy as np
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import SimpleITK as sitk
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import physical_fsdp as physical
from train_private_tiny import native_prediction_indices


def sample_fixture(direction=None, spacing=(.5, .7, 1.25), origin=(-10., -20., -30.)):
    shape = (32, 36, 28)
    mask = np.zeros(shape, bool)
    mask[10:17, 12:21, 8:16] = True
    raw = np.where(mask, -700., 100.).astype(np.float32)
    meta = {'size': list(shape), 'spacing': list(spacing), 'origin': list(origin),
            'direction': list(direction or np.eye(3).ravel())}
    image = physical.image_xyz(raw, meta)
    norm = {'clip_lower': -1000., 'clip_upper': 200., 'mean': -700., 'std': 300.}
    return image, raw, mask, norm


class PhysicalDataTests(unittest.TestCase):
    def test_lip_and_oblique_physical_roundtrip_original_indices_and_padding(self):
        directions = [(1., 0., 0., 0., 0., 1., 0., -1., 0.),
                      tuple(np.array([[np.cos(.17), -np.sin(.17), 0], [np.sin(.17), np.cos(.17), 0], [0, 0, 1]]).ravel())]
        for direction in directions:
            image, raw, mask, norm = sample_fixture(direction)
            sample = physical.make_sample(image, raw, mask, 'train_01', 'RFSDP',
                {'shape': [64, 112, 144], 'spacing': [.35] * 3}, norm)
            self.assertEqual(sample['grid']['direction'], np.eye(3).ravel().tolist())
            self.assertEqual(sample['metadata']['target_retention'], 1.)
            self.assertGreaterEqual(sample['roundtrip_dice'], .95)
            self.assertTrue(torch.isfinite(sample['image']).all())
            self.assertTrue((sample['image'][~sample['valid']] == 0).all())
            pred = physical.restore_logits(sample['target'][0, 0].numpy() * 2. - 1., sample) > 0
            indices = native_prediction_indices(pred, np.ones(pred.shape, bool), sample['metadata'], mask.shape)
            self.assertTrue(np.isin(np.flatnonzero(mask), indices).mean() > .95)
            self.assertEqual(int(sample['native_target'].sum()), int(mask.sum()))

    def test_world_coordinates_invert_with_rotated_nonisotropic_source(self):
        image, _, mask, _ = sample_fixture((0., -1., 0., 1., 0., 0., 0., 0., 1.))
        meta = physical.geometry(image)
        points = np.array([[1., 2., 3.], [5., 6., 7.]])
        np.testing.assert_allclose(physical.source_points(physical.world_points(points, meta), meta), points, atol=1e-10)
        lo, hi = physical.world_bounds(mask, meta)
        np.testing.assert_allclose(hi - lo, [9 * .7, 7 * .5, 8 * 1.25])

    def test_padding_has_no_effect_on_loss_or_gradient(self):
        from monai.losses import DiceCELoss
        target = torch.zeros((1, 1, 3, 4, 5))
        target[..., 1, 1, 1] = 1
        valid = torch.ones_like(target, dtype=torch.bool)
        valid[..., 0, :, :] = False
        logits = torch.randn_like(target).requires_grad_()
        criterion = DiceCELoss(sigmoid=True, squared_pred=True)
        first = physical.masked_loss(criterion, logits, target, valid)
        first.backward()
        self.assertTrue((logits.grad[~valid] == 0).all())
        changed = logits.detach().clone()
        changed[~valid] = 9999
        second = physical.masked_loss(criterion, changed, target, valid)
        torch.testing.assert_close(first, second)

    def test_seeded_augmentation_keeps_pairing_validity_and_target(self):
        image, raw, mask, norm = sample_fixture()
        sample = physical.make_sample(image, raw, mask, 'train_01', 'RFSDP',
            {'shape': [48, 48, 64], 'spacing': [.35] * 3}, norm)
        a = physical.augment(sample, 2026, True)
        b = physical.augment(sample, 2026, True)
        for x, y in zip(a[:3], b[:3]):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertTrue(a[1].any())
        self.assertTrue((a[0][~a[2]] == 0).all())
        self.assertFalse(a[1][~a[2]].any())
        original = physical.augment(sample, 0, False)
        torch.testing.assert_close(original[0], sample['image'])

    def test_train_only_policy_fallback_happens_before_validation_preparation(self):
        def samples(quality):
            return [{'case': 'x', 'side': side, 'metadata': {'target_retention': 1.},
                     'physical_margin_mm': [5.] * 3,
                     'roundtrip_dice': quality, 'valid': torch.ones((1, 1, 2, 2, 2), dtype=torch.bool),
                     'target': torch.ones((1, 1, 2, 2, 2), dtype=torch.uint8)} for side in ('RFSDP', 'LFSDP')]
        policy = {'normalization': {}, 'candidates': [{'spacing': [.35] * 3, 'shape': [16] * 3},
                  {'spacing': [.3] * 3, 'shape': [32] * 3}]}
        calls = []
        def prepare(manifest, source, prior, split, plan, norm):
            calls.append(split)
            quality = .9 if len(calls) == 1 else .99
            patients = {str(i): {} for i in range(54 if split == 'train' else 10)}
            if split == 'val':
                self.assertEqual(calls, ['train', 'train', 'val'])
                self.assertEqual(physical.read(output / 'policy.json')['selected']['shape'], [32] * 3)
            return samples(quality), patients, []
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(physical, 'inspect_training', return_value=(policy, [])), \
             patch.object(physical, 'prepare_split', prepare):
            output = Path(directory) / 'prepared'
            result = physical.prepare_dataset({}, Path(directory), {'train': [], 'val': []}, output)
            self.assertEqual(result['policy']['selected']['shape'], [32] * 3)
            self.assertFalse(result['policy']['attempts'][0]['eligible'])
            self.assertTrue(result['policy']['attempts'][1]['eligible'])

    def test_failed_training_preflight_never_prepares_validation(self):
        samples = [{'case': 'train_01', 'side': 'RFSDP', 'metadata': {'target_retention': 1.}, 'roundtrip_dice': .8,
                    'physical_margin_mm': [5.] * 3, 'valid': torch.ones((1, 1, 2, 2, 2), dtype=torch.bool),
                    'target': torch.ones((1, 1, 2, 2, 2), dtype=torch.uint8)}]
        policy = {'normalization': {}, 'candidates': [{'spacing': [.35] * 3, 'shape': [16] * 3}]}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(physical, 'inspect_training', return_value=(policy, [])), \
             patch.object(physical, 'prepare_split', return_value=(samples, {}, [])) as prepare:
            output = Path(directory) / 'prepared'
            with self.assertRaisesRegex(ValueError, 'No training-only physical grid'):
                physical.prepare_dataset({}, Path(directory), {'train': [], 'val': []}, output)
            self.assertEqual(prepare.call_count, 1)
            self.assertEqual(prepare.call_args.args[3], 'train')
            self.assertIsNone(physical.read(output / 'policy.json')['selected'])

    def test_half_voxel_phase_avoids_thin_label_reconstruction_loss(self):
        generator = np.random.default_rng(1)
        mask = np.zeros((24, 24, 32), bool)
        mask[8:15, 8:15, 10:21] = generator.random((7, 7, 11)) > .5
        raw = np.zeros(mask.shape, np.float32)
        meta = {'spacing': [.293, .293, .25], 'direction': np.eye(3).ravel().tolist(),
                'size': list(mask.shape), 'origin': [0.] * 3}
        image = physical.image_xyz(raw, meta)
        norm = {'clip_lower': -1., 'clip_upper': 1., 'mean': 0., 'std': 1.}
        plan = {'shape': [64, 64, 64], 'spacing': [.25] * 3}
        original = physical.make_sample(image, raw, mask, 'train_26', 'RFSDP', plan, norm)
        shifted = physical.make_sample(image, raw, mask, 'train_26', 'RFSDP',
            {**plan, 'origin_shift_fraction_xyz': [.5] * 3}, norm)
        self.assertLess(original['roundtrip_dice'], .95)
        self.assertGreater(shifted['roundtrip_dice'], .99)
        self.assertEqual(shifted['metadata']['target_retention'], 1.)
        self.assertGreaterEqual(min(shifted['physical_margin_mm']), 5.)
        np.testing.assert_allclose(np.asarray(shifted['grid']['origin']) - original['grid']['origin'], [.125] * 3)
        np.testing.assert_array_equal(physical.xyz(image), raw)

    def test_phase_is_shared_and_frozen_before_validation(self):
        policy = {'normalization': {}, 'candidates': [{'spacing': [.25] * 3, 'shape': [96, 160, 208]}]}
        calls = []
        def prepare(manifest, source, prior, split, plan, norm):
            phase = plan['origin_shift_fraction_xyz']
            calls.append((split, phase))
            quality = .91 if phase == [0.] * 3 else .99
            samples = [{'case': 'train_26', 'side': side, 'metadata': {'target_retention': 1.},
                        'physical_margin_mm': [5.1] * 3, 'roundtrip_dice': quality,
                        'valid': torch.ones((1, 1, 2, 2, 2), dtype=torch.bool),
                        'target': torch.ones((1, 1, 2, 2, 2), dtype=torch.uint8)} for side in ('RFSDP', 'LFSDP')]
            return samples, {str(i): {} for i in range(54 if split == 'train' else 10)}, []
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(physical, 'inspect_training', return_value=(policy, [])), \
             patch.object(physical, 'prepare_split', prepare):
            result = physical.prepare_dataset({}, Path(directory), {'train': [], 'val': []}, Path(directory) / 'new',
                origin_phases=((0., 0., 0.), (.5, .5, .5)))
        self.assertEqual(calls, [('train', [0.] * 3), ('train', [.5] * 3), ('val', [.5] * 3)])
        self.assertEqual(result['policy']['selected']['origin_shift_fraction_xyz'], [.5] * 3)
        self.assertEqual(len(result['policy']['attempts'][0]['failed_sides']), 2)

    def test_threshold_tie_policy_never_uses_validation(self):
        curves = {t: {'metrics': {'mean_bilateral_patient_dice': score}} for t, score in ((.3, .8), (.5, .8), (.7, .7))}
        self.assertEqual(physical.choose_threshold(curves), .5)

    def test_native_evaluation_counts_targets_outside_physical_roi_and_deduplicates(self):
        class Identity(torch.nn.Module):
            def forward(self, image):
                return image
        image, raw, mask, norm = sample_fixture()
        sample = physical.make_sample(image, raw, mask, 'val_01', 'RFSDP',
            {'shape': [16, 16, 16], 'spacing': [.35] * 3}, norm)
        self.assertLess(sample['metadata']['target_retention'], 1.)
        sample['image'] = sample['target'].float() * 2 - 1
        other = {**sample, 'side': 'LFSDP', 'name': 'val_01_LFSDP'}
        patients = {'val_01': {'shape': list(mask.shape), 'target_indices': np.flatnonzero(mask)}}
        with contextlib.redirect_stdout(io.StringIO()):
            result = physical.evaluate(Identity(), [sample, other], patients, torch.device('cpu'), [.5])[.5]
        self.assertLess(result['metrics']['mean_bilateral_patient_dice'], 1.)
        self.assertLess(result['metrics']['mean_recall'], 1.)
        self.assertEqual(result['patients'][0]['full_target_voxels'], int(mask.sum()))
        self.assertEqual(result['patients'][0]['predicted_voxels_valid'], result['sides'][0]['predicted_voxels_valid'])


if __name__ == '__main__':
    unittest.main()
