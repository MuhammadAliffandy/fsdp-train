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
from audit_label_correspondence import audit as correspondence_audit, counts, summarize


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
                if index != 0:
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
            with contextlib.redirect_stdout(io.StringIO()):
                correspondence = correspondence_audit(root, manifest, Path(directory) / 'correspondence')
            self.assertTrue(correspondence['complete'])
            self.assertFalse(correspondence['semantic_mapping_verified'])
            self.assertFalse(correspondence['test_images_read'])
            candidate = next(row for row in correspondence['candidates'] if row['source'] == 'labels_exp2' and row['expert_raw_value'] == 1)
            self.assertEqual(candidate['top_reference_overlap_candidates'][0]['reference_value'], 1)
            self.assertEqual(candidate['top_reference_overlap_candidates'][0]['mean_patient_dice_nonempty'], .5)
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
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                correspondence_audit(root, manifest, Path(directory) / 'correspondence_mismatch')
            partial = json.loads((Path(directory) / 'correspondence_mismatch/summary.json').read_text())
            self.assertFalse(partial['complete'])
            self.assertEqual(partial['compared_patients']['labels_exp1'], 53)
            label.CopyInformation(sitk.ReadImage(str(root / 'imagesTr' / f'{ids[0]}.nii.gz')))
            sitk.WriteImage(label, str(altered))
            for folder in ('labels_exp1', 'labels_exp2', 'labels_exp3'):
                for path in (root / folder).glob('*.nii.gz'):
                    image = sitk.ReadImage(str(path))
                    array = sitk.GetArrayFromImage(image)
                    array[array == 9] = 2
                    changed = sitk.GetImageFromArray(array)
                    changed.CopyInformation(image)
                    sitk.WriteImage(changed, str(path))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'target IDs absent'):
                audit(root, manifest, Path(directory) / 'absent_ids', confirmed=True)
            unsupported = json.loads((Path(directory) / 'absent_ids/summary.json').read_text())
            self.assertFalse(unsupported['complete'])
            self.assertEqual(unsupported['mode'], 'mapping_unsupported')
            self.assertEqual(unsupported['pairwise_agreement'], [])
            self.assertEqual(sum(row['asserted_id'] == 9 for row in unsupported['mapping_issues']), 3)
            self.assertFalse((Path(directory) / 'absent_ids/agreement_heatmap.png').exists())

    def test_raw_correspondence_counts_absent_labels_and_bilateral_union(self):
        reference = np.array([0, 1, 1, 9, 9, 3], dtype=np.uint8)
        expert = np.array([0, 2, 2, 2, 2, 1], dtype=np.uint8)
        ref, raw, overlap = counts(reference, expert)
        cases = [{'ref': ref, 'raw': raw, 'overlap': overlap},
                 {'ref': {0: 6}, 'raw': {0: 6}, 'overlap': {0: {0: 6}}}]
        rows = summarize(cases, [0, 1, 2], {1: 'RFSDP', 9: 'LFSDP', 3: 'RSAC'})
        union = next(row for row in rows if row['expert_raw_value'] == 2 and row['reference_value'] == '1+9')
        self.assertEqual(union['mean_patient_dice_nonempty'], 1)
        self.assertEqual(union['pooled_voxel_dice'], 1)
        self.assertEqual(union['both_empty_pairs'], 1)
        self.assertEqual(union['overlap_voxels'], 4)
        self.assertEqual(overlap[0][0], 1)

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
