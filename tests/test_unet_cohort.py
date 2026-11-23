import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_unet_tiny as audit
import train_private_tiny as loader
import train_private_unet as base
import train_unet_cohort as cohort
from evaluate_private_validation import file_hash
from test_side_training import small_preprocess
from test_unet_training import ProxyUNet, setup


def native_proxy_zoom(samples, shape):
    return [{**sample, 'native_target': sample['target'][0, 0].numpy().astype(bool),
             'native_size_xyz': list(sample['target'].shape[-3:]), 'gt_label_roundtrip_dice': 1.} for sample in samples]


class CohortTests(unittest.TestCase):
    def test_fixed_threshold_after_native_resize_and_half_control(self):
        class Known(torch.nn.Module):
            def forward(self, value):
                return value
        gt = np.zeros((4, 4, 4), bool)
        gt[1, 1, 1] = True
        probability = np.full(gt.shape, .001, np.float32)
        probability[gt] = .45
        probability[0, 0, 0] = .9
        valid = np.ones(gt.shape, bool)
        valid[0, 0, 0] = False
        image = torch.from_numpy(np.log(probability / (1 - probability)))[None, None]
        samples = [{'case': 'train_01', 'side': side, 'name': side, 'image': image,
                    'native_target': gt, 'valid': valid,
                    'metadata': {'target_retention': .5, 'full_target_voxels': 2, 'crop_offset_xyz': [0, 0, 0]}}
                   for side in ('RFSDP', 'LFSDP')]
        patients = {'train_01': {'shape': gt.shape, 'target_indices': np.array([21, 22])}}
        with contextlib.redirect_stdout(io.StringIO()):
            means, sides, rows, control = cohort.evaluate(Known(), samples, patients, torch.device('cpu'), .3)
        self.assertAlmostEqual(means['mean_bilateral_patient_dice'], 2 / 3)
        self.assertEqual(rows[0]['predicted_voxels_valid'], 1)
        self.assertEqual(control['mean_bilateral_patient_dice'], 0.)
        self.assertTrue(all(row['threshold'] == .3 and row['dice_full_target'] == 2 / 3 for row in sides))

    def test_real_calibration_full_cohort_resume_and_provenance_without_test_images(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(base, 'build_model', ProxyUNet), \
             patch.object(loader, 'preprocess', small_preprocess), \
             patch.object(base, 'zoom_samples', native_proxy_zoom), \
             patch.object(audit, 'zoom_samples', native_proxy_zoom), \
             patch.object(cohort, 'zoom_samples', native_proxy_zoom), \
             patch.object(base, 'overlay_native', lambda path, *args: path.touch()), \
             patch.object(cohort, 'overlay_native', lambda path, *args: path.touch()):
            root = Path(directory).resolve()
            tiny_args = setup(root, cohort=True)
            tiny_args.epochs = 1
            base.execute(tiny_args)
            audit_dir = root / 'audit'
            audit_dir.mkdir()
            audit.execute(SimpleNamespace(run_dir=tiny_args.run_dir, output_dir=audit_dir, device='cpu'))
            threshold = cohort.read(audit_dir / 'thresholds.json')['checkpoints']['unet_latest']['amp']['best_train_only_candidate']['threshold']
            args = argparse.Namespace(source_root=tiny_args.source_root, data_dir=tiny_args.data_dir,
                v4_run_dir=tiny_args.v4_run_dir, tiny_run_dir=tiny_args.run_dir, audit_dir=audit_dir,
                run_dir=root / 'cohort', exploratory_cohort=True, threshold=threshold, epochs=2,
                val_every=1, accumulation=8, lr=.001, seed=2026, patience_epochs=50, min_delta=.001,
                device='cpu', resume=False)
            protected = {str(p): file_hash(p) for folder in (tiny_args.run_dir, audit_dir) for p in folder.rglob('*') if p.is_file()}
            args.run_dir.mkdir()
            cohort.execute(args)
            summary = cohort.read(args.run_dir / 'summary.json')
            self.assertEqual((summary['train_patients'], summary['validation_patients'], summary['reserved_test_patients']), (54, 10, 20))
            self.assertEqual(summary['threshold'], threshold)
            self.assertFalse(summary['test_images_read'])
            self.assertFalse(summary['calibration']['original_tiny_diagnostic_passed'])
            self.assertTrue(summary['calibration']['exploratory_cohort_authorized'])
            self.assertEqual(summary['coverage']['train']['side_rois'], 108)
            self.assertEqual(summary['coverage']['val']['side_rois'], 20)
            self.assertEqual(summary['initial_state_sha256'], cohort.read(tiny_args.run_dir / 'summary.json')['initial_state_sha256'])
            reference = torch.load(args.run_dir / 'unet_latest.pth', weights_only=False)
            self.assertTrue(all(row['optimizer_updates'] == 14 for row in reference['history']))
            self.assertTrue(all('moving_model_grid_train_dice_at_0_5' in row for row in reference['history']))
            self.assertEqual(reference['config']['initialization'], 'fresh random weights with the tiny seed; no tiny/SAM learned weights')
            self.assertFalse((args.data_dir / 'test').exists())
            args.run_dir = root / 'resumed'
            args.run_dir.mkdir()
            original = base.train_epoch
            def interrupted(*values, **kwargs):
                if values[6] == 1:
                    raise RuntimeError('simulated interruption')
                return original(*values, **kwargs)
            with patch.object(base, 'train_epoch', interrupted), self.assertRaisesRegex(RuntimeError, 'simulated'):
                cohort.execute(args)
            args.resume = True
            cohort.execute(args)
            resumed = torch.load(args.run_dir / 'unet_latest.pth', weights_only=False)
            self.assertEqual(reference['history'], resumed['history'])
            for name in reference['model_state_dict']:
                torch.testing.assert_close(reference['model_state_dict'][name], resumed['model_state_dict'][name], rtol=0, atol=0)
            self.assertEqual({path: file_hash(Path(path)) for path in protected}, protected)
            versions = cohort.read(tiny_args.run_dir / 'config.json')['versions']
            policy = cohort.read(args.v4_run_dir / 'roi_policy.json')
            args.threshold = .99
            with self.assertRaisesRegex(ValueError, 'Fixed threshold'):
                cohort.verify_calibration(args, versions, policy)
            args.threshold = threshold
            altered = cohort.read(audit_dir / 'thresholds.json')
            altered['validation_images_read'] = True
            base.save_json(audit_dir / 'thresholds.json', altered)
            with self.assertRaisesRegex(ValueError, 'training-only'):
                cohort.verify_calibration(args, versions, policy)

    def test_launcher_uses_nohup_fixed_threshold_and_explicit_exploratory_protocol(self):
        with tempfile.TemporaryDirectory(prefix='fsdp cohort ') as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            launcher = root / 'scripts/launch_unet_cohort_nohup.sh'
            shutil.copyfile(Path(cohort.__file__).with_name(launcher.name), launcher)
            for name in ('data/manifest.json', 'v4/summary.json', 'v4/roi_policy.json',
                         'tiny/summary.json', 'tiny/unet_latest.pth', 'audit/summary.json', 'audit/thresholds.json'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (root / 'bin').mkdir()
            python = root / 'bin/python'
            python.write_text('#!/bin/bash\nprintf "GPU=%s\\n" "$CUDA_VISIBLE_DEVICES"\nprintf "%s\\n" "$@"\n')
            python.chmod(0o755)
            env = {**os.environ, 'PATH': str(root / 'bin') + os.pathsep + os.environ['PATH'],
                   'CONDA_DEFAULT_ENV': 'fsdp-med3d', 'GPU_ID': '2', 'PRIVATE_V4_RUN_DIR': str(root / 'v4'),
                   'PRIVATE_DATA_DIR': str(root / 'data'), 'UNET_TINY_RUN_DIR': str(root / 'tiny'),
                   'UNET_AUDIT_DIR': str(root / 'audit'), 'UNET_RUN_NAME': 'fsdp_private_unet_cohort_check', 'RESUME': '0'}
            result = subprocess.run(['bash', str(launcher)], env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            log = (root / 'logs/unet_cohort/latest.log').read_text()
            self.assertIn('--exploratory-cohort', log)
            self.assertIn('--threshold\n0.3', log)
            self.assertIn('--epochs\n200', log)
            self.assertIn('GPU=2', log)
            self.assertNotIn('--resume', log)
            env['RESUME'] = '1'
            result = subprocess.run(['bash', str(launcher)], env=env, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('unfinished run', result.stderr)
            env['RESUME'] = '0'
            (root / 'audit/thresholds.json').unlink()
            result = subprocess.run(['bash', str(launcher)], env=env, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('artifact missing', result.stderr)


if __name__ == '__main__':
    unittest.main()
