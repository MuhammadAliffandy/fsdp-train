import argparse
import contextlib
import csv
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_private_unet as baseline
from evaluate_private_validation import file_hash, write_csv
from test_tiny_training import fixture
from test_side_training import small_preprocess
import train_private_tiny as tiny_loader


class ProxyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 1, 1)

    def forward(self, image):
        return self.conv(image)


def setup(root, cohort=False):
    source, data, manifest = fixture(root, all_training_patients=cohort, include_validation=cohort)
    v4 = root / 'v4'
    v4.mkdir()
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'torchio', 'monai', 'numpy', 'SimpleITK')}
    (v4 / 'config.json').write_text(json.dumps({'experiment': 'v4_anisotropic_roi_zoom', 'seed': 2026,
        'manifest_sha256': file_hash(data / 'manifest.json'), 'versions': versions,
        'shared_code_sha256': {name: file_hash(Path(baseline.__file__).with_name(name)) for name in
                              ('train_private_tiny.py', 'evaluate_private_validation.py')},
        'zoom_module_sha256': file_hash(Path(baseline.__file__).with_name('roi_zoom.py'))}))
    (v4 / 'summary.json').write_text(json.dumps({'complete': True, 'test_images_read': False,
        'train_patients': 54, 'validation_patients': 10, 'native_roi_size_xyz': [64, 80, 96],
        'best_validation_dice': {'5': {'mean_bilateral_patient_dice': .22}}}))
    (v4 / 'roi_policy.json').write_text(json.dumps({'selected_native_size_xyz': [64, 80, 96],
        'training_patients': 54, 'training_sides': 108, 'margin_voxels': 4}))
    provenance = {split: [] for split in ('train', 'val')}
    for row in manifest['cases']:
        if row['split'] == 'test':
            continue
        paths = [Path(row[key]) for key in ('source_image', 'source_label')]
        provenance[row['split']].append({'patient_id': row['patient_id'],
            'source_image': row['source_image'], 'source_label': row['source_label'],
            'source_image_sha256': file_hash(paths[0]) if paths[0].exists() else 'unread',
            'source_label_sha256': file_hash(paths[1]) if paths[1].exists() else 'unread'})
    (v4 / 'source_provenance.json').write_text(json.dumps(provenance))
    run = root / 'tiny'
    run.mkdir()
    return argparse.Namespace(source_root=source, data_dir=data, v4_run_dir=v4, run_dir=run, tiny_run_dir=None,
        stage='tiny', epochs=2, val_every=1, accumulation=8, lr=.001, seed=2026,
        patience_epochs=50, min_delta=.001, device='cpu', resume=False)


class UNetTests(unittest.TestCase):
    def test_launcher_starts_tiny_by_default_and_refuses_cohort_without_gate(self):
        with tempfile.TemporaryDirectory(prefix='fsdp unet ') as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            launcher = root / 'scripts/launch_private_unet_nohup.sh'
            shutil.copyfile(Path(baseline.__file__).with_name('launch_private_unet_nohup.sh'), launcher)
            for name in ('data/manifest.json', 'v4/summary.json', 'v4/roi_policy.json'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (root / 'bin').mkdir()
            executable = root / 'bin/python'
            executable.write_text('#!/bin/bash\nprintf "GPU=%s\\n" "$CUDA_VISIBLE_DEVICES"\nprintf "%s\\n" "$@"\n')
            executable.chmod(0o755)
            env = {**os.environ, 'PATH': str(root / 'bin') + os.pathsep + os.environ['PATH'],
                   'CONDA_DEFAULT_ENV': 'fsdp-med3d', 'GPU_ID': '2', 'PRIVATE_V4_RUN_DIR': str(root / 'v4'),
                   'PRIVATE_DATA_DIR': str(root / 'data'), 'PRIVATE_SOURCE_ROOT': str(root / 'normal case_86'),
                   'UNET_STAGE': 'tiny', 'UNET_RUN_NAME': 'fsdp_private_unet_tiny_check', 'RESUME': '0'}
            result = subprocess.run(['bash', str(launcher)], env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            log = (root / 'logs/unet_tiny/latest.log').read_text()
            self.assertIn('GPU=2', log)
            self.assertIn('--stage\ntiny', log)
            self.assertNotIn('--tiny-run-dir', log)
            self.assertIn('--source-root\n' + str(root / 'normal case_86'), log)
            env.update(UNET_STAGE='cohort', UNET_RUN_NAME='fsdp_private_unet_cohort_check')
            refused = subprocess.run(['bash', str(launcher)], env=env, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn('Complete the tiny diagnostic first', refused.stderr)

    def test_actual_monai_unet_has_full_size_logits_and_trainable_gradients(self):
        from monai.losses import DiceCELoss
        model = baseline.build_model()
        image = torch.randn((1, 1, 32, 32, 32))
        logits = baseline.checked_logits(model, image)
        self.assertEqual(logits.shape, image.shape)
        loss = DiceCELoss(sigmoid=True, squared_pred=True)(logits, (image > 0).float())
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
        model.eval()
        with torch.inference_mode():
            full = baseline.checked_logits(model, torch.zeros((1, 1, 128, 128, 128)))
        self.assertEqual(tuple(full.shape), (1, 1, 128, 128, 128))

    def test_tiny_reads_only_two_patients_exports_native_and_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(baseline, 'build_model', ProxyUNet), \
             patch.object(baseline, 'overlay_native', lambda path, *args: path.touch()):
            root = Path(directory).resolve()
            args = setup(root)
            baseline.execute(args)
            summary = json.loads((args.run_dir / 'summary.json').read_text())
            self.assertEqual(summary['train_patients'], 2)
            self.assertEqual(summary['validation_patients'], 0)
            self.assertFalse(summary['validation_images_read'])
            self.assertFalse(summary['test_images_read'])
            self.assertFalse(summary['diagnostic_passed'])
            self.assertFalse((args.data_dir / 'val').exists())
            self.assertFalse((args.data_dir / 'test').exists())
            self.assertEqual(len(json.loads((args.run_dir / 'source_provenance.json').read_text())['train']), 2)
            self.assertEqual(summary['coverage']['train']['side_rois'], 4)
            reference = torch.load(args.run_dir / 'unet_latest.pth', weights_only=False)
            self.assertTrue(all(row['optimizer_updates'] == 4 for row in reference['history']))
            with self.assertRaisesRegex(ValueError, 'has not passed'):
                baseline.check_gate(args.run_dir, json.loads((args.run_dir / 'config.json').read_text()))
            args.run_dir = root / 'resumed'
            args.run_dir.mkdir()
            original = baseline.train_epoch
            def interrupt(*values, **kwargs):
                if values[6] == 1:
                    raise RuntimeError('simulated interruption')
                return original(*values, **kwargs)
            with patch.object(baseline, 'train_epoch', interrupt), self.assertRaisesRegex(RuntimeError, 'simulated'):
                baseline.execute(args)
            args.resume = True
            baseline.execute(args)
            resumed = torch.load(args.run_dir / 'unet_latest.pth', weights_only=False)
            self.assertEqual(reference['history'], resumed['history'])
            for name in reference['model_state_dict']:
                torch.testing.assert_close(reference['model_state_dict'][name], resumed['model_state_dict'][name], rtol=0, atol=0)

    def test_native_evaluation_counts_truncated_target_and_deduplicates_bilateral_union(self):
        class Known(torch.nn.Module):
            def forward(self, image):
                return image
        gt = np.zeros((4, 4, 4), bool)
        gt[1, 1, 1] = True
        logits = torch.full((1, 1, 4, 4, 4), -1.)
        logits[0, 0, 1, 1, 1] = 1
        samples = [{'case': 'train_01', 'side': side, 'name': side, 'image': logits,
                    'native_target': gt, 'valid': np.ones(gt.shape, bool),
                    'metadata': {'target_retention': .5, 'full_target_voxels': 2, 'crop_offset_xyz': [0, 0, 0]}}
                   for side in ('RFSDP', 'LFSDP')]
        patients = {'train_01': {'shape': (4, 4, 4), 'target_indices': np.array([21, 22])}}
        means, sides, rows = baseline.evaluate(Known(), samples, patients, torch.device('cpu'))
        self.assertEqual(rows[0]['predicted_voxels_valid'], 1)
        self.assertAlmostEqual(means['mean_bilateral_patient_dice'], 2 / 3)
        self.assertTrue(all(row['oracle_clicks'] == 0 and row['dice_full_target'] == 2 / 3 for row in sides))

    def test_cohort_gate_rejects_fabricated_pass_without_supporting_csvs(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(baseline, 'build_model', ProxyUNet), \
             patch.object(baseline, 'overlay_native', lambda path, *args: path.touch()):
            args = setup(Path(directory).resolve())
            args.epochs = 1
            baseline.execute(args)
            config = json.loads((args.run_dir / 'config.json').read_text())
            summary_path = args.run_dir / 'summary.json'
            summary = json.loads(summary_path.read_text())
            summary['diagnostic_passed'] = True
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'minimum Dice0.90'):
                baseline.check_gate(args.run_dir, config)

    def test_cohort_is_refused_before_any_image_preparation_when_gate_is_absent(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args = setup(Path(directory).resolve())
            args.stage = 'cohort'
            with self.assertRaisesRegex(ValueError, 'requires --tiny-run-dir'):
                baseline.execute(args)
            self.assertFalse((args.run_dir / 'preparation').exists())
            self.assertFalse((args.data_dir / 'val').exists())

    def test_mocked_gate_cohort_uses_all_54_10_patients_and_14_updates_without_test_files(self):
        def native_proxy_zoom(samples, shape):
            return [{**sample, 'native_target': sample['target'][0, 0].numpy().astype(bool),
                     'native_size_xyz': list(sample['target'].shape[-3:]), 'gt_label_roundtrip_dice': 1.} for sample in samples]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(baseline, 'build_model', ProxyUNet), \
             patch.object(baseline, 'overlay_native', lambda path, *args: path.touch()), \
             patch.object(tiny_loader, 'preprocess', small_preprocess), \
             patch.object(baseline, 'zoom_samples', native_proxy_zoom), \
             patch.object(baseline, 'check_gate', return_value={'test_proxy_gate': True}):
            root = Path(directory).resolve()
            args = setup(root, cohort=True)
            args.stage, args.epochs = 'cohort', 1
            args.tiny_run_dir = root / 'test_proxy_gate'
            args.tiny_run_dir.mkdir()
            v4_sources = json.loads((args.v4_run_dir / 'source_provenance.json').read_text())
            selected = tiny_loader.select_patients(json.loads((args.data_dir / 'manifest.json').read_text()), args.seed)
            save_sources = {'train': [row for row in v4_sources['train'] if row['patient_id'] in selected]}
            (args.tiny_run_dir / 'source_provenance.json').write_text(json.dumps(save_sources))
            torch.manual_seed(args.seed)
            (args.tiny_run_dir / 'summary.json').write_text(json.dumps({'initial_state_sha256': baseline.state_hash(ProxyUNet().state_dict())}))
            baseline.execute(args)
            summary = json.loads((args.run_dir / 'summary.json').read_text())
            self.assertEqual(summary['train_patients'], 54)
            self.assertEqual(summary['validation_patients'], 10)
            self.assertEqual(summary['coverage']['train']['side_rois'], 108)
            self.assertEqual(summary['coverage']['val']['side_rois'], 20)
            self.assertEqual(summary['oracle_clicks'], 0)
            self.assertFalse(summary['test_images_read'])
            self.assertFalse((args.data_dir / 'test').exists())
            latest = torch.load(args.run_dir / 'unet_latest.pth', weights_only=False)
            self.assertEqual(latest['history'][0]['optimizer_updates'], 14)
            self.assertEqual(latest['history'][0]['amp_skipped_updates'], 0)
            self.assertAlmostEqual(summary['validation_dice_change_vs_v4'], summary['best_selected_metrics']['mean_bilateral_patient_dice'] - .22)


if __name__ == '__main__':
    unittest.main()
