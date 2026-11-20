import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from inspect_side_training import summarize


class InspectionTests(unittest.TestCase):
    def test_macro_scores_volume_ratios_and_tamper_rejection_without_images(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory).resolve()
            export = run / 'best_validation'
            export.mkdir()
            for filename, count in [('frozen_training_patients.csv', 54), ('patients.csv', 10)]:
                rows = [{'case': f'case_{i}', 'clicks': 5, 'dice_full_target': .5 if i % 2 else .16, 'precision': .5 if i % 2 else .1,
                         'recall_full_target': .5 if i % 2 else .4, 'predicted_voxels_valid': 10 if i % 2 else 20,
                         'full_target_voxels': 10 if i % 2 else 5} for i in range(count)]
                with (export / filename).open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            reference = {'5': {'mean_bilateral_patient_dice': .33}}
            summary = {'complete': True, 'train_patients': 54, 'validation_patients': 10, 'best_epoch': 5,
                       'export_directory': str(export), 'best_checkpoint_frozen_training_dice': reference, 'best_validation_dice': reference}
            (run / 'summary.json').write_text(json.dumps(summary))
            (run / 'history.csv').write_text('epoch,train_dice_crop_augmented,val_bilateral_dice_5_per_side,lr\n1,0.1,0.1,0.001\n2,0.2,,0.001\n5,0.3,0.2,0.0005\n')
            result = summarize(run)
            self.assertEqual(result['selected_checkpoint_metrics']['val']['mean_precision'], .3)
            self.assertEqual(result['selected_checkpoint_metrics']['val']['mean_recall'], .45)
            self.assertEqual(result['selected_checkpoint_metrics']['val']['mean_predicted_to_target_voxel_ratio'], 2.5)
            self.assertEqual(result['selected_checkpoint_metrics']['val']['pooled_predicted_to_target_voxel_ratio'], 2)
            self.assertEqual([row['epoch'] for row in result['validation_curve']], [1, 5])
            self.assertFalse(result['images_or_weights_loaded'])
            summary['best_validation_dice'] = {'5': {'mean_bilateral_patient_dice': .8}}
            (run / 'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'does not reproduce'):
                summarize(run)


if __name__ == '__main__':
    unittest.main()
