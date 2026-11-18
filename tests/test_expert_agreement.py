import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_expert_agreement import agreement, audit, index_files


class AgreementTests(unittest.TestCase):
    def test_dice_empty_policy_and_physical_centroid_distance(self):
        a = np.zeros((3, 4, 6), bool)
        b = a.copy()
        empty = agreement(a, b, np.array([.5, 1., 2.]), np.eye(3).flatten())
        self.assertIsNone(empty['dice'])
        self.assertTrue(empty['both_empty'])
        a[1, 1, 1:3] = True
        b[1, 1, 2:4] = True
        result = agreement(a, b, np.array([.5, 1., 2.]), np.eye(3).flatten())
        self.assertEqual(result['dice'], .5)
        self.assertEqual(result['centroid_distance_mm'], .5)
        missing = agreement(a, np.zeros_like(a), np.ones(3), np.eye(3).flatten())
        self.assertEqual(missing['dice'], 0.)
        self.assertTrue(missing['one_empty'])

    def test_train_only_audit_known_overlap_and_geometry_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'source with spaces'
            for folder in ('imagesTr', 'labelsTr', 'labels_exp1', 'labels_exp2', 'labels_exp3', 'ColorTable'):
                (root / folder).mkdir(parents=True)
            ids = [str(1000 + index) for index in range(54)]
            records = []
            for index, key in enumerate(ids):
                image = sitk.GetImageFromArray(np.zeros((4, 5, 6), np.int16))
                image.SetSpacing((.5, .7, 1.5))
                image.SetOrigin((-5., -6., -7.))
                image.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
                image_path = root / 'imagesTr' / f'{key}.nii.gz'
                sitk.WriteImage(image, str(image_path))
                ref = np.zeros((4, 5, 6), np.uint8)
                if index != 53:
                    ref[1, 1, 1:3] = 1
                ref[2, 3, 3:5] = 9
                second = ref.copy()
                if index != 53:
                    second[second == 1] = 0
                    second[1, 1, 2:4] = 1
                third = ref.copy()
                third[third == 1] = 0
                for folder, values, filename in [
                    ('labelsTr', ref, f'{key}-label.nii.gz'), ('labels_exp1', ref, f'{key}.nii.gz'),
                    ('labels_exp2', second, f'{key}_label_exp2.nii.gz'), ('labels_exp3', third, f'{key}-label.nii.gz')]:
                    label = sitk.GetImageFromArray(values)
                    label.CopyInformation(image)
                    sitk.WriteImage(label, str(root / folder / filename))
                (root / 'ColorTable' / f'{key}_ColorTable.csv').write_text('LabelValue,Name\n1,RFSDP\n9,LFSDP\n')
                records.append({'patient_id': key, 'source_image': str(image_path),
                                'source_label': str(root / 'labelsTr' / f'{key}-label.nii.gz')})
            manifest = Path(directory) / 'manifest.json'
            # Validation/test files deliberately do not exist.
            manifest.write_text(json.dumps({'splits': {'train': ids, 'val': [str(2000+n) for n in range(10)],
                                                       'test': [str(3000+n) for n in range(20)]}, 'cases': records}))
            with contextlib.redirect_stdout(io.StringIO()):
                result = audit(root, manifest, Path(directory) / 'audit', confirmed=True)
            self.assertTrue(result['complete'])
            self.assertEqual(result['fully_comparable_patients'], 54)
            self.assertFalse(result['validation_images_read'])
            self.assertFalse(result['test_images_read'])
            row = next(row for row in result['pairwise_agreement'] if row['source_a'] == 'labelsTr' and row['source_b'] == 'labels_exp2' and row['target'] == 'RFSDP')
            self.assertEqual(row['mean_dice_nonempty'], .5)
            self.assertEqual(row['both_empty_pairs'], 1)
            self.assertEqual(row['nonempty_pairs'], 53)
            self.assertAlmostEqual(row['mean_centroid_distance_mm'], .5)
            self.assertTrue((Path(directory) / 'audit/agreement_heatmap.png').is_file())
            mapping = Path(directory) / 'mapping.json'
            mapping.write_text(json.dumps({name: {'RFSDP': 1, 'LFSDP': 9} for name in ('labels_exp1', 'labels_exp2', 'labels_exp3')}))
            with contextlib.redirect_stdout(io.StringIO()):
                mapped = audit(root, manifest, Path(directory) / 'mapped', mapping_path=mapping)
            self.assertTrue(mapped['complete'])
            self.assertEqual(mapped['pairwise_agreement'], result['pairwise_agreement'])
            with contextlib.redirect_stdout(io.StringIO()):
                inventory = audit(root, manifest, Path(directory) / 'inventory')
            self.assertEqual(inventory['mode'], 'inventory_only_mapping_pending')
            self.assertEqual(inventory['pairwise_agreement'], [])
            altered = root / 'labels_exp2' / f'{ids[0]}_label_exp2.nii.gz'
            label = sitk.ReadImage(str(altered))
            label.SetOrigin((10., 20., 30.))
            sitk.WriteImage(label, str(altered))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                audit(root, manifest, Path(directory) / 'mismatch', confirmed=True)
            failed = json.loads((Path(directory) / 'mismatch/summary.json').read_text())
            self.assertFalse(failed['complete'])
            self.assertEqual(failed['fully_comparable_patients'], 53)
            self.assertEqual(failed['issues'], 1)

    def test_filename_collisions_remain_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('123-label.nii.gz', '123_other.nii.gz', 'unknown.nii.gz'):
                (root / name).touch()
            indexed, unrecognized = index_files(root)
            self.assertEqual(len(indexed['123']), 2)
            self.assertEqual(unrecognized, 1)


if __name__ == '__main__':
    unittest.main()
