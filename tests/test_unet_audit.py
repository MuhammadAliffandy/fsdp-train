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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_unet_tiny as audit
import train_private_unet as baseline
from evaluate_private_validation import file_hash
from test_unet_training import ProxyUNet, setup


class UNetAuditTests(unittest.TestCase):
    def test_subthreshold_foreground_can_recover_without_changing_05_control(self):
        records, patients = [], {}
        for alias in ('train_01', 'train_02'):
            targets = []
            for side, point in [('RFSDP', (1, 1, 1)), ('LFSDP', (2, 2, 2))]:
                target = np.zeros((4, 4, 4), bool)
                target[point] = True
                probability = np.full(target.shape, .001)
                probability[target] = .45
                logits = np.log(probability / (1 - probability))
                sample = {'case': alias, 'side': side, 'native_target': target,
                          'valid': np.ones(target.shape, bool),
                          'metadata': {'full_target_voxels': 1, 'crop_offset_xyz': [0, 0, 0]}}
                records.append({'sample': sample, 'fp32': logits, 'amp': logits.copy()})
                targets.append(np.ravel_multi_index(point, target.shape))
            patients[alias] = {'shape': target.shape, 'target_indices': np.array(targets)}
        curves = audit.threshold_sweep(records, patients, thresholds=(.1, .3, .4, .5))
        for result in curves.values():
            self.assertEqual(result['best_train_only_candidate']['threshold'], .4)
            self.assertEqual(result['best_train_only_candidate']['minimum_diagnostic_dice'], 1.)
            half = next(row for row in result['curve'] if row['threshold'] == .5)
            self.assertEqual(half['mean_bilateral_patient_dice'], 0.)
        with self.assertRaisesRegex(ValueError, 'both sides'):
            audit.threshold_sweep(records[:-1], patients)

    def test_independent_dice_keeps_missing_targets_and_discards_padding(self):
        logits, target = np.ones((3, 3, 3)), np.zeros((3, 3, 3), bool)
        target[1, 1, 1] = True
        valid = np.zeros_like(target)
        valid[1, 1, 1] = True
        metrics = audit.hard_dice(logits, target, full_count=2, valid=valid)
        self.assertEqual(metrics['predicted_voxels'], 1)
        self.assertAlmostEqual(metrics['dice'], 2 / 3)

    def test_full_tiny_audit_reproduces_score_without_changing_training_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(baseline, 'build_model', ProxyUNet), \
             patch.object(baseline, 'overlay_native', lambda path, *args: path.touch()):
            root = Path(directory).resolve()
            args = setup(root)
            args.epochs = 1
            baseline.execute(args)
            protected = [args.run_dir / name for name in ('config.json', 'summary.json', 'unet_best.pth', 'unet_latest.pth', 'history.csv')]
            before = {str(path): file_hash(path) for path in protected}
            output = root / 'audit'
            output.mkdir()
            audit.execute(SimpleNamespace(run_dir=args.run_dir, output_dir=output, device='cpu'))
            report = json.loads((output / 'summary.json').read_text())
            overview = json.loads((output / 'overview.json').read_text())
            threshold_report = json.loads((output / 'threshold_overview.json').read_text())
            self.assertFalse(threshold_report['gate_modified'])
            self.assertFalse(threshold_report['validation_images_read'])
            self.assertFalse(threshold_report['test_images_read'])
            self.assertTrue(report['complete'])
            self.assertFalse(report['validation_images_read'])
            self.assertFalse(report['test_images_read'])
            self.assertFalse(report['training_artifacts_modified'])
            self.assertEqual(report['optimizer_updates'], 0)
            self.assertEqual(len(report['per_side']), 8)
            self.assertEqual(report['training_history']['successful_optimizer_updates'], 4)
            self.assertEqual(report['training_history']['amp_skipped_updates'], 0)
            self.assertEqual(set(report['checkpoints']), {'unet_best', 'unet_latest'})
            self.assertEqual(overview['optimizer_updates_in_audit'], 0)
            for value in overview['checkpoints'].values():
                self.assertAlmostEqual(value['mean_side_dice_native_amp'], value['mean_side_dice_native_fp32'])
                self.assertEqual(value['first_roi_gradient_probe']['nonfinite_gradient_elements'], 0)
            self.assertEqual({str(path): file_hash(path) for path in protected}, before)
            self.assertFalse((args.data_dir / 'val').exists())
            self.assertFalse((args.data_dir / 'test').exists())


if __name__ == '__main__':
    unittest.main()
