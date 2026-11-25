import csv
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import inspect_fsdp_bottleneck as audit


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    parent = root / 'study'; run = parent / 'history'; run.mkdir(parents=True)
    summary = {'complete': True, 'variant': 'history', 'test_images_read': False,
        'train_patients': 54, 'validation_patients': 10, 'oracle_clicks_per_side': [1, 5],
        'completed_epochs': 2, 'planned_epochs': 100, 'stopped_early': True, 'best_epoch': 0, 'threshold': .45,
        'initial_candidate_metrics': {'train_threshold': .45, 'train': {'5': {'mean_bilateral_patient_dice': .6}},
                                      'val': {'5': {'mean_bilateral_patient_dice': .5}}},
        'research_source': {'source_plus_additional_epoch': 35},
        'best_frozen_training_metrics': {'1': {'mean_bilateral_patient_dice': .55}, '5': {'mean_bilateral_patient_dice': .6}},
        'best_validation_metrics': {'1': {'mean_bilateral_patient_dice': .4}, '5': {'mean_bilateral_patient_dice': .5}},
        'validation_control_fixed_0_5': {'mean_bilateral_patient_dice': .49}, 'detail_grid': [32] * 3, 'token_grid': [8] * 3,
        'native_boundary_metrics': {'mean_surface_dice_1mm': .9}}
    save(run / 'summary.json', summary)
    save(parent / 'summary.json', {'complete': True, 'entries': [{'variant': 'history', 'directory': str(run.resolve())}]})
    save(run / 'config.json', {'physical_policy': {'selected': {'shape': [96, 160, 208], 'spacing': [.25] * 3},
                                                  'sam_context': {'physical_extent_mm': [52] * 3}},
        'foundation': {'source': 'same'}, 'cache_sha256': 'cache', 'manifest_sha256': 'manifest', 'versions': {},
        'parameters': {'epochs': 100, 'lr_multiplier': .1, 'patience_epochs': 25, 'seed': 2026, 'augmentation': 'light',
                       'accumulation': 8, 'auxiliary_weight': .25, 'precision': 'bf16', 'val_every': 1, 'min_delta': .001}})
    fields = ('epoch', 'source_plus_additional_epoch', 'train_dice_5', 'val_dice_5', 'val_control_0_5',
              'train_threshold', 'loss', 'sam_lr', 'head_lr')
    with (run / 'history.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(fields)
        writer.writerow([1, 36, .65, .48, .47, .45, .6, 1e-6, 1e-5])
        writer.writerow([2, 37, .7, .45, .44, .45, .5, 1e-6, 1e-5])
    export = run / 'best_validation'; export.mkdir()
    for split, count, tp in (('train', 54, 6), ('val', 10, 5)):
        with (export / f'{split}_5_patients.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['case', 'clicks_per_side', 'threshold', 'predicted_voxels_valid', 'full_target_voxels',
                             'true_positive_voxels', 'dice_full_target'])
            for index in range(1, count + 1):
                writer.writerow([f'{split}_{index:02d}', 5, .45, 10, 10, tp, tp / 10])
    return parent, run


class BottleneckAuditTests(unittest.TestCase):
    def test_distinguishes_selected_checkpoint_from_last_epoch_and_does_not_modify_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, run = fixture(Path(folder))
            before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.rglob('*') if p.is_file()}
            report = audit.inspect(parent)
            self.assertAlmostEqual(report['selected_checkpoint']['train']['mean_dice'], .6)
            self.assertAlmostEqual(report['learning_curve'][-1]['train_dice_5'], .7)
            self.assertAlmostEqual(report['learning_curve'][-1]['train_val_gap_pp'], 25.)
            self.assertIn('consistent with overfitting', report['pattern_hypothesis'])
            self.assertAlmostEqual(report['selected_checkpoint']['threshold_vs_fixed_0_5_gain_pp'], 1.)
            self.assertAlmostEqual(report['selected_checkpoint']['one_to_five_click_gain_pp'], 10.)
            self.assertEqual(report['feature_geometry']['fusion_cell_mm_xyz'], [.75, 1.25, 1.625])
            self.assertEqual(report['feature_geometry']['sam_token_cell_mm_xyz'], [6.5] * 3)
            self.assertFalse(report['gpu_used']); self.assertFalse(report['images_or_weights_loaded'])
            after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.rglob('*') if p.is_file()}
            self.assertEqual(before, after)

    def test_invalid_patient_confusion_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, run = fixture(Path(folder))
            path = run / 'best_validation/val_5_patients.csv'
            path.write_text(path.read_text().replace(',10,10,5,0.5', ',10,10,11,0.5', 1))
            with self.assertRaisesRegex(ValueError, 'confusion counts'):
                audit.inspect(parent)

    def test_incomplete_history_does_not_create_a_learning_diagnosis(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, run = fixture(Path(folder))
            path = run / 'history.csv'
            path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
            with self.assertRaisesRegex(ValueError, 'complete contiguous'):
                audit.inspect(parent)

    def test_wrong_click_budget_or_test_read_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, run = fixture(Path(folder))
            summary = audit.read(run / 'summary.json'); summary['test_images_read'] = True
            save(run / 'summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'test unread'):
                audit.inspect(parent)

    def test_parent_cannot_redirect_inspection_to_an_unrelated_run(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, _ = fixture(Path(folder))
            save(parent / 'summary.json', {'entries': [{'variant': 'history', 'directory': str(Path(folder) / 'other')}]})
            with self.assertRaisesRegex(ValueError, 'belong to this parent'):
                audit.inspect(parent)

    def test_original_screening_pairs_use_same_patients_and_recipe(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, run = fixture(Path(folder)); control = parent / 'control'
            summary = copy.deepcopy(audit.read(run / 'summary.json')); summary['variant'] = 'control'
            summary['best_validation_metrics']['5']['mean_bilateral_patient_dice'] = .4
            save(control / 'summary.json', summary)
            save(control / 'config.json', audit.read(run / 'config.json'))
            export = control / 'best_validation'; export.mkdir()
            source = run / 'best_validation/val_5_patients.csv'
            (export / source.name).write_text(source.read_text().replace(',10,10,5,0.5', ',10,10,4,0.4'))
            result = audit.inspect(parent)['matched_original_screening']
            self.assertEqual(result['patients_improved'], 10)
            self.assertAlmostEqual(result['mean_paired_delta_pp'], 10.)
            config = audit.read(control / 'config.json'); config['parameters']['seed'] = 5
            save(control / 'config.json', config)
            with self.assertRaisesRegex(ValueError, 'not matched'):
                audit.inspect(parent)


if __name__ == '__main__':
    unittest.main()
