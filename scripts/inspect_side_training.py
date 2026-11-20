#!/usr/bin/env python3
"""Summarize existing v3 training artifacts without loading images or model weights."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


def summarize(run):
    run = run.expanduser().resolve()
    summary = json.loads((run / 'summary.json').read_text())
    if not summary['complete'] or summary['train_patients'] != 54 or summary['validation_patients'] != 10:
        raise ValueError('Expected a complete 54/10-patient side-training run')
    export = Path(summary['export_directory']).resolve()
    if not export.is_relative_to(run):
        raise ValueError('Best-checkpoint exports must belong to this run')
    result = {}
    for split, filename, count, reference in [
        ('train', 'frozen_training_patients.csv', 54, summary['best_checkpoint_frozen_training_dice']),
        ('val', 'patients.csv', 10, summary['best_validation_dice'])]:
        with (export / filename).open(newline='') as stream:
            rows = [row for row in csv.DictReader(stream) if int(row['clicks']) == 5]
        if len(rows) != count or len({row['case'] for row in rows}) != count:
            raise ValueError(f'{split}: expected all {count} distinct patients at five clicks per side')
        values = {field: [float(row[field]) for row in rows] for field in
                  ('dice_full_target', 'precision', 'recall_full_target', 'predicted_voxels_valid', 'full_target_voxels')}
        if any(not math.isfinite(value) for column in values.values() for value in column) or min(values['full_target_voxels']) <= 0:
            raise ValueError('Invalid metrics or empty final targets')
        dice = mean(values['dice_full_target'])
        if not math.isclose(dice, reference['5']['mean_bilateral_patient_dice'], abs_tol=1e-6, rel_tol=0):
            raise ValueError(f'{split}: CSV Dice does not reproduce the selected-checkpoint summary')
        result[split] = {'patients': count, 'mean_dice': dice,
            'mean_precision': mean(values['precision']), 'mean_recall': mean(values['recall_full_target']),
            'mean_predicted_to_target_voxel_ratio': mean(a / b for a, b in zip(values['predicted_voxels_valid'], values['full_target_voxels'])),
            'pooled_predicted_to_target_voxel_ratio': sum(values['predicted_voxels_valid']) / sum(values['full_target_voxels']),
            'target_voxels_min_median_max': [min(values['full_target_voxels']), median(values['full_target_voxels']), max(values['full_target_voxels'])]}
    with (run / 'history.csv').open(newline='') as stream:
        curve = [{'epoch': int(row['epoch']), 'moving_augmented_train_dice': float(row['train_dice_crop_augmented']),
                  'validation_bilateral_dice_5_per_side': float(row['val_bilateral_dice_5_per_side']), 'lr': float(row['lr'])}
                 for row in csv.DictReader(stream) if row['val_bilateral_dice_5_per_side']]
    return {'run': run.name, 'best_epoch': summary['best_epoch'], 'selected_checkpoint_metrics': result,
            'validation_curve': curve,
            'note': 'Patient-macro scores; moving training curve is not frozen training inference. Recall/precision are averaged separately.',
            'images_or_weights_loaded': False, 'artifacts_modified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'outputs/private/side_training/latest')
    args = parser.parse_args()
    try:
        print(json.dumps(summarize(args.run_dir), indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f'Run inspection stopped: {error}\n')


if __name__ == '__main__':
    main()
