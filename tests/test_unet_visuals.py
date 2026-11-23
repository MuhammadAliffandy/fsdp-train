import contextlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_unet_tiny as audit
import create_unet_visuals as visual
import train_private_unet as baseline
from evaluate_private_validation import file_hash
from test_unet_training import ProxyUNet, setup


class VisualTests(unittest.TestCase):
    def test_confusion_matrix_counts_duplicate_predictions_once(self):
        row = visual.confusion_row('train_01', 'unet_latest', 200, .3,
                                   np.array([1, 1, 4]), np.array([1, 2]), np.arange(10))
        self.assertEqual((row['tp'], row['fp'], row['fn'], row['tn']), (1, 1, 1, 7))
        self.assertEqual(row['domain_voxels'], 10)
        self.assertEqual(row['dice_full_target'], .5)
        with self.assertRaisesRegex(ValueError, 'Confusion domain'):
            visual.confusion_row('x', 'x', 0, .5, np.array([20]), np.array([1]), np.arange(10))

    def test_all_four_sides_and_both_checkpoints_export_verified_pngs_without_source_changes(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(baseline, 'build_model', ProxyUNet), \
             patch.object(baseline, 'overlay_native', lambda path, *args: path.touch()):
            root = Path(directory).resolve()
            args = setup(root)
            args.epochs = 1
            baseline.execute(args)
            audit_dir = root / 'audit'
            audit_dir.mkdir()
            audit.execute(SimpleNamespace(run_dir=args.run_dir, output_dir=audit_dir, device='cpu'))
            files = [path for folder in (args.run_dir, audit_dir) for path in folder.rglob('*') if path.is_file()]
            before = {str(path): file_hash(path) for path in files}
            output = root / 'visuals'
            output.mkdir()
            visual.execute(SimpleNamespace(run_dir=args.run_dir, audit_dir=audit_dir, output_dir=output, device='cpu'))
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertTrue(manifest['complete'])
            self.assertFalse(manifest['validation_images_read'])
            self.assertFalse(manifest['test_images_read'])
            self.assertEqual(manifest['optimizer_updates'], 0)
            self.assertEqual(len(list((output / 'figures').glob('*three_planes.png'))), 8)
            self.assertGreaterEqual(len(manifest['figures']), 18)
            for name in manifest['figures']:
                with Image.open(output / 'figures' / name) as image:
                    self.assertGreater(image.width, 500)
                    self.assertGreater(image.height, 300)
            archive = output.with_suffix('.tar.gz')
            self.assertTrue(archive.is_file())
            with tarfile.open(archive) as bundle:
                names = bundle.getnames()
            self.assertTrue(any(name.endswith('patient_confusion_metrics.csv') for name in names))
            self.assertFalse(any('/preparation' in name or name.endswith('.nii.gz') for name in names))
            self.assertEqual({str(path): file_hash(path) for path in files}, before)
            self.assertFalse((args.data_dir / 'val').exists())
            self.assertFalse((args.data_dir / 'test').exists())


if __name__ == '__main__':
    unittest.main()
