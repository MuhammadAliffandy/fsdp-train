import json
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_foundation_sources as audit


def fixture(root):
    row = {'case': 'train_01', 'patient_id': '1000001'}
    for role, folder in audit.FOLDERS.items():
        directory = root / folder; directory.mkdir(parents=True)
        path = directory / ('1000001_ColorTable.csv' if role == 'color_table' else '1000001-label.nii.gz' if role == 'source_label' else '1000001.nii.gz')
        path.write_bytes(role.encode())
        row[role], row[role + '_sha256'] = str(path), audit.digest(path)
    return {'train': [row], 'val': []}


class SourceAuditTests(unittest.TestCase):
    def test_relocated_reader_preserves_voxels_geometry_and_checksums_with_symlinked_folders(self):
        import numpy as np
        import SimpleITK as sitk
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); original = fixture(root / 'storage')
            row = original['train'][0]
            labels = np.zeros((5, 6, 7), dtype=np.uint8)
            labels[1, 2, 3], labels[3, 4, 5] = 1, 9
            image = sitk.GetImageFromArray(np.arange(labels.size, dtype=np.float32).reshape(labels.shape))
            image.SetSpacing((.3, .4, .7)); image.SetOrigin((-20., 10., -40.))
            label = sitk.GetImageFromArray(labels); label.CopyInformation(image)
            for key, value in (('source_image', image), ('source_label', label)):
                sitk.WriteImage(value, row[key])
            Path(row['color_table']).write_text('LabelValue,Name\n1,RFSDP\n9,LFSDP\n')
            row.update({key + '_sha256': audit.digest(Path(row[key])) for key in audit.FOLDERS})
            actual = root / 'actual data'; actual.mkdir()
            for name in audit.FOLDERS.values():
                (actual / name).symlink_to(root / 'storage' / name, target_is_directory=True)
            record = audit.bind_record({**row, 'right_voxels': 1, 'left_voxels': 1}, actual)
            expected = audit.bind_sources(original, actual)['train'][0]
            ct_image, _, ct, target, _ = audit.load_bound_case(record, actual, expected, 'train_01')
            np.testing.assert_array_equal(target, labels.transpose(2, 1, 0))
            np.testing.assert_array_equal(ct, sitk.GetArrayFromImage(image).transpose(2, 1, 0))
            np.testing.assert_allclose(ct_image.GetSpacing(), image.GetSpacing())
            self.assertEqual(ct_image.GetOrigin(), image.GetOrigin())
            with self.assertRaisesRegex(ValueError, 'counts differ'):
                audit.load_bound_case({**record, 'right_voxels': 2}, actual, expected, 'train_01')
            Path(expected['color_table']).write_text('LabelValue,Name\n1,LFSDP\n9,RFSDP\n')
            with self.assertRaisesRegex(ValueError, 'content differs'):
                audit.load_bound_case(record, actual, expected, 'train_01')

    def test_explicit_root_rebinds_paths_without_changing_original_names_or_checksums(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); provenance = fixture(root / 'old')
            before = copy.deepcopy(provenance)
            (root / 'old').rename(root / 'actual data')
            bound = audit.bind_sources(provenance, root / 'actual data')
            self.assertTrue(audit.inventory(bound, verify=True)['ready'])
            self.assertEqual(provenance, before)
            for key in audit.FOLDERS:
                self.assertEqual(bound['train'][0][key + '_sha256'], before['train'][0][key + '_sha256'])
                self.assertEqual(Path(bound['train'][0][key]).name, Path(before['train'][0][key]).name)
            record = {**before['train'][0], 'right_voxels': 7, 'left_voxels': 9}
            moved = audit.bind_record(record, root / 'actual data')
            self.assertEqual(moved['right_voxels'], 7)
            self.assertEqual(moved['source_label'], bound['train'][0]['source_label'])

    def test_explicit_root_rejects_different_labels_even_if_original_location_is_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); original = fixture(root / 'old')
            fixture(root / 'new')
            bound = audit.bind_sources(original, root / 'new')
            Path(bound['train'][0]['source_label']).write_bytes(b'other_final_annotation')
            report = audit.inventory(bound, verify=True)
            self.assertFalse(report['ready'])
            self.assertEqual(report['status_counts']['changed'], 1)
            self.assertTrue(audit.inventory(original, verify=True)['ready'])

    def test_missing_renamed_label_is_found_by_hash_without_modifying_source_or_using_experts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            provenance = fixture(root)
            original = Path(provenance['train'][0]['source_label'])
            renamed = original.with_name('1000001_seg.nii.gz'); original.rename(renamed)
            (root / 'labels_exp1').mkdir()
            (root / 'labels_exp1/1000001.nii.gz').write_bytes(renamed.read_bytes())
            (root / 'labelsTr/10000010.nii.gz').write_bytes(renamed.read_bytes())
            report = audit.inventory(provenance, verify=True)
            self.assertFalse(report['ready'])
            problem = report['problems'][0]
            self.assertEqual(problem['status'], 'missing')
            self.assertEqual(problem['alternatives'], [{'path': str(renamed), 'sha256_matches': True,
                'actual_sha256': provenance['train'][0]['source_label_sha256']}])
            self.assertFalse(original.exists())
            self.assertFalse(report['data_modified'])

    def test_changed_label_is_rejected_even_when_identical_backup_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); provenance = fixture(root)
            original = Path(provenance['train'][0]['source_label'])
            original.with_name('1000001_backup.nii.gz').write_bytes(original.read_bytes())
            original.write_bytes(b'different_annotation')
            report = audit.inventory(provenance, verify=True)
            self.assertFalse(report['ready'])
            self.assertEqual(report['status_counts']['changed'], 1)
            self.assertTrue(report['problems'][0]['alternatives'][0]['sha256_matches'])

    def test_moved_dataset_requires_explicit_search_root_and_is_not_applied(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); provenance = fixture(root / 'old')
            (root / 'old').rename(root / 'new')
            report = audit.inventory(provenance, search_roots=[root / 'new'], verify=True)
            self.assertEqual(report['status_counts'], {'missing': 3})
            self.assertTrue(all(row['alternatives'][0]['sha256_matches'] for row in report['problems']))
            self.assertFalse((root / 'old').exists())

    def test_presence_precheck_is_distinguished_from_content_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); provenance = fixture(root / 'data')
            base = root / 'base'; base.mkdir()
            (base / 'source_provenance.json').write_text(json.dumps(provenance))
            report = audit.check_folder(base)
            self.assertTrue(report['ready'])
            self.assertFalse(report['content_verified'])
            self.assertEqual(audit.inventory(provenance, verify=True)['status_counts'], {'ok': 3})
            Path(provenance['train'][0]['source_label']).unlink()
            with self.assertRaisesRegex(ValueError, 'No model training started'):
                audit.check_folder(base)

    def test_test_split_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'Only train/val'):
            audit.inventory({'test': [{'patient_id': 'must_not_read'}]})


if __name__ == '__main__':
    unittest.main()
