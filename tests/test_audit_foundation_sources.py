import json
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
