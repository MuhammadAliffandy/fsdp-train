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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import profile_fsdp_training_data as profiler
import train_private_tiny as loader
from evaluate_private_validation import file_hash
from test_side_training import small_preprocess
from test_unet_cohort import native_proxy_zoom
from test_unet_training import setup
from types import SimpleNamespace


class TrainingDataProfileTests(unittest.TestCase):
    def test_physical_volume_components_and_pixel_identity(self):
        target = np.zeros((5, 5, 5), bool)
        target[0, 0, 0] = target[4, 4, 4] = True
        result = profiler.target_stats(target, np.array([.5, .5, 2.]))
        self.assertEqual(result['components_26_connected'], 2)
        self.assertEqual(result['volume_mm3'], 1.)
        self.assertEqual(result['bbox_extent_mm_xyz'], [2.5, 2.5, 10.])
        self.assertEqual(result['largest_component_fraction'], .5)
        self.assertEqual(profiler.pixel_hash(target), profiler.pixel_hash(target.copy()))
        self.assertNotEqual(profiler.pixel_hash(target), profiler.pixel_hash(target.astype(np.uint8)))
        with self.assertRaisesRegex(ValueError, 'finite'):
            profiler.intensity_stats(np.array([float('nan')]))

    def test_full_training_profile_needs_no_validation_or_test_files_and_preserves_sources(self):
        def preprocess(*args):
            image, target, raw, metadata, valid = small_preprocess(*args)
            metadata['normalization_positive_voxels'] = int((raw > 0).sum())
            return image, target, raw, metadata, valid
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(loader, 'preprocess', preprocess), patch.object(profiler, 'zoom_samples', native_proxy_zoom):
            root = Path(directory).resolve()
            args = setup(root, cohort=True)
            source_rows = profiler.read(args.v4_run_dir / 'source_provenance.json')
            for row in source_rows['val']:
                Path(row['source_image']).unlink()
                Path(row['source_label']).unlink()
            shutil.rmtree(args.data_dir / 'val')
            run = root / 'cohort'
            run.mkdir()
            versions = profiler.read(args.v4_run_dir / 'config.json')['versions']
            config = {'source_root': str(args.source_root), 'data_dir': str(args.data_dir),
                      'v4_run_dir': str(args.v4_run_dir), 'seed': 2026, 'versions': versions,
                      'manifest_sha256': file_hash(args.data_dir / 'manifest.json'),
                      'native_roi_size_xyz': [64, 80, 96], 'code_sha256': {
                          name: file_hash(Path(profiler.__file__).with_name(name)) for name in
                          ('train_unet_cohort.py', 'train_private_unet.py', 'roi_zoom.py')}}
            summary = {'complete': True, 'experiment': 'unet_cohort_train_calibrated',
                       'train_patients': 54, 'validation_patients': 10, 'test_images_read': False,
                       'threshold': .3, 'best_epoch': 65,
                       'best_validation_metrics': {'mean_bilateral_patient_dice': .1864},
                       'best_frozen_training_metrics': {'mean_bilateral_patient_dice': .3250}}
            references = [{**row, 'case': f'train_{i:02d}'} for i, row in enumerate(source_rows['train'], 1)]
            for name, value in (('config.json', config), ('summary.json', summary), ('source_provenance.json', {'train': references})):
                (run / name).write_text(json.dumps(value))
            protected = [p for folder in (args.source_root, args.data_dir, run) for p in folder.rglob('*') if p.is_file()]
            before = {str(p): file_hash(p) for p in protected}
            output = root / 'profile'
            output.mkdir()
            profiler.execute(SimpleNamespace(run_dir=run, output_dir=output))
            result = profiler.read(output / 'summary.json')
            compact = profiler.read(output / 'overview.json')
            self.assertTrue(result['complete'])
            self.assertEqual(result['training_patients'], 54)
            self.assertEqual(result['training_sides'], 108)
            self.assertEqual(sum(result['orientation_counts'].values()), 54)
            self.assertAlmostEqual(result['patient_distributions']['spacing_z_mm']['median'], 1.2)
            self.assertAlmostEqual(result['side_distributions']['volume_mm3']['median'], 11.34, places=5)
            self.assertEqual(result['side_distributions']['components_26_connected']['maximum'], 1.)
            self.assertEqual(result['side_distributions']['target_retention']['minimum'], 1.)
            pooled = result['pooled_training_target_source_intensities']
            self.assertLessEqual(pooled['minimum'], pooled['p0_5'])
            self.assertLessEqual(pooled['p0_5'], pooled['p99_5'])
            self.assertLessEqual(pooled['p99_5'], pooled['maximum'])
            self.assertGreater(pooled['std'], 0.)
            self.assertEqual(len(result['identical_source_pixel_groups_training_only'][0]), 54)
            self.assertFalse(result['validation_images_read'])
            self.assertFalse(result['test_images_read'])
            self.assertEqual(result['optimizer_updates'], 0)
            self.assertEqual(compact['side_distributions']['voxels']['median'], 27.)
            self.assertFalse((args.data_dir / 'val').exists())
            self.assertFalse((args.data_dir / 'test').exists())
            self.assertEqual({str(p): file_hash(p) for p in protected}, before)
            self.assertNotIn('patient_id', (output / 'patients.csv').read_text())
            self.assertNotIn('patient_id', (output / 'sides.csv').read_text())

    def test_launcher_uses_cpu_nohup_and_isolated_report(self):
        with tempfile.TemporaryDirectory(prefix='fsdp data profile ') as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            launcher = root / 'scripts/launch_dataset_profile_nohup.sh'
            shutil.copyfile(Path(profiler.__file__).with_name(launcher.name), launcher)
            run = root / 'cohort'
            run.mkdir()
            for name in ('summary.json', 'config.json', 'source_provenance.json'):
                (run / name).touch()
            (root / 'bin').mkdir()
            executable = root / 'bin/python'
            executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
            executable.chmod(0o755)
            env = {**os.environ, 'PATH': str(root / 'bin') + os.pathsep + os.environ['PATH'],
                   'CONDA_DEFAULT_ENV': 'fsdp-med3d', 'UNET_COHORT_RUN_DIR': str(run)}
            result = subprocess.run(['bash', str(launcher)], env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            log_path = root / 'logs/dataset_profile/latest.log'
            for _ in range(100):
                log = log_path.read_text()
                if '--run-dir' in log:
                    break
                time.sleep(.05)
            self.assertIn('--run-dir\n' + str(run), log)
            self.assertNotIn('--device', log)
            self.assertIn('outputs/private/dataset_profile/', log)
            (run / 'summary.json').unlink()
            refused = subprocess.run(['bash', str(launcher)], env=env, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn('Completed cohort', refused.stderr)


if __name__ == '__main__':
    unittest.main()
