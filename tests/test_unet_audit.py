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
