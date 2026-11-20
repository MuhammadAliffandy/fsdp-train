import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import roi_zoom
from train_private_tiny import native_prediction_indices


def policy_samples(edge=64):
    samples = []
    for patient in range(54):
        for side in ('RFSDP', 'LFSDP'):
            target = torch.zeros((1, 1, 128, 128, 128), dtype=torch.uint8)
            point = edge if patient == 0 and side == 'RFSDP' else 64
            target[0, 0, point, 64, 64] = 1
            samples.append({'case': f'train_{patient:02d}', 'side': side, 'target': target})
    return samples


class ROIZoomTests(unittest.TestCase):
    def test_policy_uses_per_axis_bounds_and_selects_smallest_train_only_crop(self):
        result = roi_zoom.choose_policy(policy_samples())
        self.assertEqual(result['selected_native_size_xyz'], [10, 10, 10])
        self.assertTrue(all(candidate['fully_retained_sides'] == 108 for candidate in result['cubic_diagnostics']))
        moved = policy_samples(edge=20)
        result = roi_zoom.choose_policy(moved)
        self.assertEqual(result['selected_native_size_xyz'], [96, 10, 10])

    def test_policy_rejects_validation_and_invalid_side_coverage(self):
        samples = policy_samples()
        samples[-1]['case'] = 'val_01'
        with self.assertRaisesRegex(ValueError, 'training patients only'):
            roi_zoom.choose_policy(samples)
        samples = policy_samples()
        samples[-1]['side'] = 'RFSDP'
        with self.assertRaisesRegex(ValueError, 'Both training sides'):
            roi_zoom.choose_policy(samples)

    def test_zoom_resizes_image_and_labels_and_restores_native_geometry(self):
        target = torch.zeros((1, 1, 128, 128, 128), dtype=torch.uint8)
        target[0, 0, 50:52, 60:62, 70:72] = 1
        raw = np.zeros((128, 128, 128), dtype=np.float32)
        raw[50:52, 60:62, 70:72] = 500
        image = torch.from_numpy(raw.copy())[None, None]
        sample = {'case': 'val_01', 'side': 'RFSDP', 'name': 'val_01_RFSDP', 'image': image,
                  'target': target, 'raw': raw, 'valid': np.ones(raw.shape, dtype=bool),
                  'prompt_seed': 7, 'metadata': {'full_target_voxels': 8, 'crop_target_voxels': 8,
                    'target_retention': 1., 'crop_origin_mm': [10., 20., 30.], 'spacing_mm': [1., 2., 3.],
                    'direction': [0., -1., 0., 1., 0., 0., 0., 0., 1.], 'crop_offset_xyz': [10, 20, 30]}}
        zoomed = roi_zoom.zoom_samples([sample], (64, 80, 96))[0]
        self.assertEqual(tuple(zoomed['image'].shape), (1, 1, 128, 128, 128))
        self.assertEqual(tuple(zoomed['target'].shape), (1, 1, 128, 128, 128))
        self.assertEqual(zoomed['metadata']['target_retention'], 1.)
        self.assertEqual(zoomed['metadata']['crop_offset_xyz'], [42, 44, 46])
        self.assertEqual(zoomed['metadata']['crop_origin_mm'], [-38., 52., 78.])
        self.assertEqual(zoomed['native_target'].sum(), 8)
        self.assertEqual(zoomed['valid'].shape, (64, 80, 96))
        prediction = np.zeros((64, 80, 96), dtype=bool)
        prediction[18, 36, 54] = True
        indices = native_prediction_indices(prediction, zoomed['valid'], zoomed['metadata'], (200, 200, 200))
        expected = np.ravel_multi_index(([60], [80], [100]), (200, 200, 200))
        np.testing.assert_array_equal(indices, expected)

    def test_label_grid_roundtrip_is_explicitly_measured(self):
        labels = torch.zeros((1, 1, 64, 64, 64), dtype=torch.uint8)
        labels[..., 20:24, 20:24, 20:24] = 1
        up = roi_zoom.resize_labels(labels)
        self.assertEqual(tuple(up.shape), (1, 1, 128, 128, 128))
        restored = roi_zoom.restore_logits(up[0, 0].float().numpy() * 2 - 1, (64, 64, 64)) > 0
        self.assertAlmostEqual(float(restored.sum()), float(labels.sum()))
        self.assertAlmostEqual(roi_zoom.scores(restored, labels[0, 0].numpy().astype(bool), int(labels.sum()))['dice_full_target'], 1.)


if __name__ == '__main__':
    unittest.main()
