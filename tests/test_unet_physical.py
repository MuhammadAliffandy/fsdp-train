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
import time
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import physical_fsdp as physical
import train_private_unet as base
import train_unet_physical as trainer
from evaluate_private_validation import file_hash
from test_unet_training import ProxyUNet, setup


class PhysicalTrainingTests(unittest.TestCase):
    def test_actual_model_rectangular_grid_and_masked_gradients(self):
        from monai.losses import DiceCELoss
        model = base.build_model().eval()
        with torch.inference_mode():
            result = base.checked_logits(model, torch.zeros((1, 1, 64, 112, 144)))
        self.assertEqual(tuple(result.shape), (1, 1, 64, 112, 144))
        image = torch.randn((1, 1, 32, 32, 32))
        logits = base.checked_logits(model, image)
        valid = torch.ones_like(image, dtype=torch.bool)
        valid[..., 0:4, :, :] = False
        target = (image > 0).float()
        target[~valid] = 0
        loss = physical.masked_loss(DiceCELoss(sigmoid=True, squared_pred=True), logits, target, valid)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))

    def test_threshold_is_chosen_on_training_before_validation(self):
        calls = []
        prepared = {'samples': {'train': 'train_samples', 'val': 'val_samples'},
                    'patients': {'train': {}, 'val': {}}}
        def evaluate(model, samples, patients, device, thresholds, label):
            calls.append(samples)
            if samples == 'train_samples':
                return {.3: {'metrics': {'mean_bilateral_patient_dice': .8}}, .5: {'metrics': {'mean_bilateral_patient_dice': .7}}}
            self.assertEqual(calls, ['train_samples', 'val_samples'])
            return {.3: {'metrics': {'mean_bilateral_patient_dice': .1}}, .5: {'metrics': {'mean_bilateral_patient_dice': .9}}}
        with patch.object(physical, 'evaluate', evaluate):
            threshold, _, _ = trainer.calibration_validation(None, prepared, torch.device('cpu'), 'test')
        self.assertEqual(threshold, .3)

    def test_full_physical_preprocess_training_resume_and_originals_without_test_files(self):
        real_export = trainer.export_representatives
        def no_render(model, prepared, device, threshold, expected, output):
            output.mkdir()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(base, 'build_model', ProxyUNet), \
             patch.object(physical, 'MINIMUM_FOV_MM', (6., 6., 6.)), \
             patch.object(physical, 'MARGIN_MM', .5), \
             patch.object(trainer, 'export_representatives', no_render):
            root = Path(directory).resolve()
            fixture_args = setup(root, cohort=True)
            control = root / 'cohort'
            control.mkdir()
            v4config = physical.read(fixture_args.v4_run_dir / 'config.json')
            torch.manual_seed(2026)
            control_summary = {'complete': True, 'experiment': 'unet_cohort_train_calibrated',
                'train_patients': 54, 'validation_patients': 10, 'test_images_read': False,
                'initial_state_sha256': base.state_hash(ProxyUNet().state_dict()),
                'best_validation_metrics': {'mean_bilateral_patient_dice': .1864}}
            control_config = {'data_dir': str(fixture_args.data_dir), 'source_root': str(fixture_args.source_root),
                'seed': 2026, 'manifest_sha256': file_hash(fixture_args.data_dir / 'manifest.json'), 'versions': v4config['versions']}
            for name, value in (('summary.json', control_summary), ('config.json', control_config),
                                 ('source_provenance.json', physical.read(fixture_args.v4_run_dir / 'source_provenance.json'))):
                (control / name).write_text(json.dumps(value))
            protected = [p for folder in (control, fixture_args.source_root, fixture_args.data_dir) for p in folder.rglob('*') if p.is_file()]
            before = {str(p): file_hash(p) for p in protected}
            args = argparse.Namespace(cohort_run_dir=control, run_dir=root / 'physical', epochs=2, val_every=1,
                accumulation=8, lr=.001, seed=2026, patience_epochs=50, min_delta=.001,
                augmentation='light', device='cpu', resume=False)
            args.run_dir.mkdir()
            trainer.execute(args)
            summary = physical.read(args.run_dir / 'summary.json')
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['train_patients'], 54)
            self.assertEqual(summary['validation_patients'], 10)
            self.assertEqual(summary['reserved_test_patients'], 20)
            self.assertFalse(summary['test_images_read'])
            self.assertEqual(summary['coverage']['train']['minimum_target_retention'], 1.)
            self.assertGreaterEqual(summary['coverage']['train']['minimum_roundtrip_dice'], .95)
            self.assertEqual(summary['initial_state_sha256'], control_summary['initial_state_sha256'])
            self.assertFalse((fixture_args.data_dir / 'test').exists())
            reference = torch.load(args.run_dir / 'unet_latest.pth', weights_only=False)
            self.assertTrue(all(r['optimizer_updates'] == 14 for r in reference['history']))
            self.assertTrue(all(r['amp_skipped_updates'] == 0 for r in reference['history']))
            self.assertIn(reference['best_threshold'], physical.THRESHOLDS)
            selected = torch.load(args.run_dir / 'unet_best.pth', weights_only=False)
            prepared = torch.load(args.run_dir / 'physical_data.pt', weights_only=False)
            selected_model = ProxyUNet()
            selected_model.load_state_dict(selected['model_state_dict'])
            threshold = selected['best_threshold']
            expected = physical.evaluate(selected_model, prepared['samples']['val'], prepared['patients']['val'],
                                         torch.device('cpu'), [threshold])[threshold]
            visual_output = root / 'representative_export'
            real_export(selected_model, prepared, torch.device('cpu'), threshold, expected, visual_output)
            self.assertEqual(len(list(visual_output.glob('*.png'))), 6)
            self.assertEqual(len(list(visual_output.glob('*_prediction.nii.gz'))), 6)
            exported_image = physical.sitk.ReadImage(str(next(visual_output.glob('*_prediction.nii.gz'))))
            self.assertEqual(exported_image.GetDirection(), tuple(prepared['samples']['val'][0]['native']['direction']))
            args.run_dir = root / 'resumed'
            args.run_dir.mkdir()
            original = trainer.train_epoch
            def interrupted(*values, **kwargs):
                if values[6] == 1:
                    raise RuntimeError('simulated interruption')
                return original(*values, **kwargs)
            with patch.object(trainer, 'train_epoch', interrupted), self.assertRaisesRegex(RuntimeError, 'simulated'):
                trainer.execute(args)
            args.resume = True
            trainer.execute(args)
            resumed = torch.load(args.run_dir / 'unet_latest.pth', weights_only=False)
            self.assertEqual(reference['history'], resumed['history'])
            self.assertEqual(reference['best_threshold'], resumed['best_threshold'])
            for name in reference['model_state_dict']:
                torch.testing.assert_close(reference['model_state_dict'][name], resumed['model_state_dict'][name], rtol=0, atol=0)
            self.assertEqual({str(p): file_hash(p) for p in protected}, before)
            cache = args.run_dir / 'physical_data.pt'
            cache.write_bytes(cache.read_bytes() + b'tamper')
            with self.assertRaisesRegex(ValueError, 'cache changed'):
                trainer.prepare(args)

    def test_launcher_chains_preprocessing_training_with_nohup_and_separate_outputs(self):
        with tempfile.TemporaryDirectory(prefix='fsdp physical ') as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            launcher = root / 'scripts/launch_unet_physical_nohup.sh'
            shutil.copyfile(Path(trainer.__file__).with_name(launcher.name), launcher)
            control = root / 'cohort'
            control.mkdir()
            for name in ('summary.json', 'config.json', 'source_provenance.json'):
                (control / name).touch()
            (root / 'bin').mkdir()
            executable = root / 'bin/python'
            executable.write_text('#!/bin/bash\nprintf "GPU=%s\\n" "$CUDA_VISIBLE_DEVICES"\nprintf "%s\\n" "$@"\n')
            executable.chmod(0o755)
            env = {**os.environ, 'PATH': str(root / 'bin') + os.pathsep + os.environ['PATH'], 'CONDA_DEFAULT_ENV': 'fsdp-med3d',
                   'GPU_ID': '2', 'UNET_COHORT_RUN_DIR': str(control), 'UNET_RUN_NAME': 'fsdp_private_unet_physical_check', 'RESUME': '0'}
            result = subprocess.run(['bash', str(launcher)], env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            path = root / 'logs/unet_physical/latest.log'
            for _ in range(100):
                log = path.read_text()
                if '--cohort-run-dir' in log:
                    break
                time.sleep(.05)
            self.assertIn('GPU=2', log)
            self.assertIn('--augmentation\nlight', log)
            self.assertIn('--epochs\n200', log)
            self.assertIn('--cohort-run-dir\n' + str(control), log)
            self.assertTrue((root / 'outputs/private/unet_physical/latest').is_symlink())
            env['RESUME'] = '1'
            refused = subprocess.run(['bash', str(launcher)], env=env, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn('unfinished physical run', refused.stderr)


if __name__ == '__main__':
    unittest.main()
