import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import contextlib
import io

spec = importlib.util.spec_from_file_location('audit', Path(__file__).resolve().parents[1] / 'scripts/audit_fsdp_sam_results.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)


class AuditTests(unittest.TestCase):
    def test_counts_describe_volume_error_and_reject_inconsistent_dice(self):
        row = dict(predicted_voxels_valid=200, full_target_voxels=100, true_positive_voxels=75, dice_full_target=.5)
        result = audit.counts(row)
        self.assertEqual((result['fp'], result['fn'], result['volume_ratio']), (125, 25, 2))
        row['dice_full_target'] = .9
        with self.assertRaises(ValueError):
            audit.counts(row)

    def test_completed_pair_is_audited_without_mutating_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm, tp in (('control', 40), ('affine', 50)):
                exports = root / 'run' / arm / 'best_validation'
                exports.mkdir(parents=True)
                summary = dict(complete=True, test_images_read=False, augmentation_arm=arm,
                    validation_patients=10, oracle_clicks_per_side=[1,5], threshold=.5,
                    research_source={'checkpoint_sha256':'same'}, best_epoch=1, completed_epochs=35,
                    best_validation_metrics={'5':{'mean_bilateral_patient_dice':tp/100}})
                (exports.parent / 'summary.json').write_text(json.dumps(summary))
                patients = [dict(case=f'val_{i:02d}', predicted_voxels_valid=100, full_target_voxels=100,
                    true_positive_voxels=tp, dice_full_target=tp/100) for i in range(1,11)]
                sides = [dict(r, side=side) for r in patients for side in ('RFSDP','LFSDP')]
                boundaries = [dict(case=r['case'], surface_dice_0_5mm=.7, surface_dice_1mm=.8, hd95_mm=2) for r in patients]
                for name, rows in (('patients',patients),('sides',sides),('boundary_metrics',boundaries)):
                    write_csv(exports / f'val_5_{name}.csv', rows)
            originals = {p:p.read_bytes() for p in (root / 'run').rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):
                audit.audit(root / 'run', root / 'audit')
            result=json.loads((root / 'audit/summary.json').read_text())
            self.assertAlmostEqual(result['paired_patients'][0]['delta'], .1)
            self.assertTrue((root / 'audit/review_bundle.tar.gz').is_file())
            self.assertTrue(all(p.read_bytes()==value for p,value in originals.items()))


if __name__ == '__main__':
    unittest.main()
