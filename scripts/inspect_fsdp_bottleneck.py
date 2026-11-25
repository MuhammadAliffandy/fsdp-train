#!/usr/bin/env python3
"""Audit saved FSDP research metrics without loading images, model weights or a GPU."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


def read(path):
    return json.loads(path.read_text())


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite recorded metric')
    return result


def score(summary, split, clicks=5):
    key = 'best_frozen_training_metrics' if split == 'train' else 'best_validation_metrics'
    return number(summary[key][str(clicks)]['mean_bilateral_patient_dice'])


def learning_curve(run, summary):
    initial = summary['initial_candidate_metrics']
    rows = [{'epoch': 0, 'source_plus_additional_epoch': summary.get('research_source', {}).get('source_plus_additional_epoch'),
             'train_dice_5': number(initial['train']['5']['mean_bilateral_patient_dice']),
             'val_dice_5': number(initial['val']['5']['mean_bilateral_patient_dice']),
             'threshold': number(initial['train_threshold'])}]
    with (run / 'history.csv').open(newline='') as stream:
        history = list(csv.DictReader(stream))
    if len(history) != summary['completed_epochs'] or [int(row['epoch']) for row in history] != list(range(1, len(history) + 1)):
        raise ValueError('History is not the complete contiguous recorded epoch sequence')
    for row in history:
        if row.get('val_dice_5') in (None, ''):
            continue
        rows.append({'epoch': int(row['epoch']), 'source_plus_additional_epoch': int(row['source_plus_additional_epoch']),
            'train_dice_5': number(row['train_dice_5']), 'val_dice_5': number(row['val_dice_5']),
            'val_control_0_5': number(row['val_control_0_5']), 'threshold': number(row['train_threshold']),
            'loss': number(row['loss']), 'sam_lr': number(row['sam_lr']), 'head_lr': number(row['head_lr'])})
    for row in rows:
        row['train_val_gap_pp'] = 100 * (row['train_dice_5'] - row['val_dice_5'])
    best = max(row['val_dice_5'] for row in rows)
    if abs(best - score(summary, 'val')) > 1e-6:
        raise ValueError('History best validation does not reproduce summary')
    return rows


def patients(run, summary, split):
    path = run / 'best_validation' / f'{split}_5_patients.csv'
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = summary['train_patients' if split == 'train' else 'validation_patients']
    if len(rows) != expected or len({row['case'] for row in rows}) != expected:
        raise ValueError(f'{split}: incomplete or duplicate patient rows')
    result = []
    for row in rows:
        if int(row['clicks_per_side']) != 5 or number(row['threshold']) != number(summary['threshold']):
            raise ValueError('Patient row has a different click budget/threshold')
        predicted, target, tp = [int(row[key]) for key in ('predicted_voxels_valid', 'full_target_voxels', 'true_positive_voxels')]
        if target <= 0 or predicted < 0 or not 0 <= tp <= min(predicted, target):
            raise ValueError('Invalid native sparse confusion counts')
        dice = 2 * tp / (predicted + target)
        if abs(dice - number(row['dice_full_target'])) > 1e-8:
            raise ValueError('Patient Dice does not reproduce confusion counts')
        result.append({'case': row['case'], 'dice': dice, 'precision': tp / predicted if predicted else 0.,
            'recall': tp / target, 'tp': tp, 'fp': predicted - tp, 'fn': target - tp,
            'target_voxels': target, 'predicted_voxels': predicted, 'predicted_to_target_ratio': predicted / target})
    if abs(mean(row['dice'] for row in result) - score(summary, split)) > 1e-6:
        raise ValueError(f'{split}: patient scores do not reproduce selected checkpoint')
    return sorted(result, key=lambda row: (row['dice'], row['case']))


def summary_counts(rows):
    target = [row['target_voxels'] for row in rows]
    return {'patients': len(rows), 'mean_dice': mean(row['dice'] for row in rows),
            'median_dice': median(row['dice'] for row in rows), 'minimum_dice': min(row['dice'] for row in rows),
            'empty_predictions': sum(row['predicted_voxels'] == 0 for row in rows),
            'patients_below_dice_0_3': sum(row['dice'] < .3 for row in rows),
            'patients_below_dice_0_5': sum(row['dice'] < .5 for row in rows),
            'pooled_fp': sum(row['fp'] for row in rows), 'pooled_fn': sum(row['fn'] for row in rows),
            'pooled_predicted_to_target_ratio': sum(row['predicted_voxels'] for row in rows) / sum(target),
            'target_voxels_min_median_max': [min(target), median(target), max(target)]}


def geometry(config, summary):
    policy = config['physical_policy']
    detail = policy['selected']
    shape, spacing = detail['shape'], detail['spacing']
    context = policy['sam_context']['physical_extent_mm']
    grids = (shape, spacing, context, summary['detail_grid'], summary['token_grid'])
    if any(len(values) != 3 or any(number(value) <= 0 for value in values) for values in grids):
        raise ValueError('Feature geometry must have three positive finite axes')
    extent = [number(n) * number(s) for n, s in zip(shape, spacing)]
    return {'detail_spacing_mm_xyz': spacing, 'detail_shape_xyz': shape, 'detail_extent_mm_xyz': extent,
            'fusion_grid_xyz': summary['detail_grid'],
            'fusion_cell_mm_xyz': [extent[i] / summary['detail_grid'][i] for i in range(3)],
            'sam_token_grid_xyz': summary['token_grid'],
            'sam_token_cell_mm_xyz': [number(context[i]) / summary['token_grid'][i] for i in range(3)],
            'note': 'Full/half-resolution CNN skips are already present. Token spacing alone is not a Dice ceiling.'}


def matched_screening(run, summary):
    source = Path(summary.get('research_source', {}).get('directory', str(run))).resolve()
    control = source.parent / 'control'
    if not (control / 'summary.json').exists():
        return None
    source_summary, control_summary = [read(path / 'summary.json') for path in (source, control)]
    source_config, control_config = [read(path / 'config.json') for path in (source, control)]
    recipe = ('epochs', 'lr_multiplier', 'patience_epochs', 'seed', 'augmentation', 'accumulation',
              'auxiliary_weight', 'precision', 'val_every', 'min_delta')
    if (not source_summary['complete'] or not control_summary['complete']
            or source_summary['variant'] != 'history' or control_summary['variant'] != 'control'
            or source_summary['test_images_read'] or control_summary['test_images_read']
            or source_summary['oracle_clicks_per_side'] != control_summary['oracle_clicks_per_side']
            or any(source_config[key] != control_config[key] for key in
                   ('foundation', 'cache_sha256', 'manifest_sha256', 'physical_policy', 'versions'))
            or any(source_config['parameters'][key] != control_config['parameters'][key] for key in recipe)):
        raise ValueError('Original history/control screening configuration is not matched')
    left = {row['case']: row for row in patients(source, source_summary, 'val')}
    right = {row['case']: row for row in patients(control, control_summary, 'val')}
    if set(left) != set(right) or any(left[key]['target_voxels'] != right[key]['target_voxels'] for key in left):
        raise ValueError('Screening patient IDs or native target sizes differ')
    deltas = [{'case': key, 'history_dice': left[key]['dice'], 'control_dice': right[key]['dice'],
               'delta_pp': 100 * (left[key]['dice'] - right[key]['dice'])} for key in sorted(left)]
    return {'source_screening_run': source.parent.name, 'budget_epochs_per_arm': source_config['parameters']['epochs'],
            'mean_paired_delta_pp': mean(row['delta_pp'] for row in deltas),
            'patients_improved': sum(row['delta_pp'] > 1e-8 for row in deltas),
            'patients_worse': sum(row['delta_pp'] < -1e-8 for row in deltas),
            'patients_tied': sum(abs(row['delta_pp']) <= 1e-8 for row in deltas),
            'patients': sorted(deltas, key=lambda row: (row['delta_pp'], row['case'])),
            'scope': 'Selected checkpoints from matched original screening budgets, not a matched control for the later continuation.'}


def inspect(run):
    run = run.expanduser().resolve()
    if not (run / 'summary.json').exists() and (run / 'history/summary.json').exists():
        run = run / 'history'
    summary = read(run / 'summary.json')
    if 'entries' in summary:
        entry = next((entry for entry in summary['entries'] if entry['variant'] == 'history'), None)
        if entry is None:
            raise ValueError('Parent report does not contain a history arm')
        child = Path(entry['directory']).resolve()
        if child.parent != run:
            raise ValueError('History entry must belong to this parent workflow')
        run = child
        summary = read(run / 'summary.json')
    if (not summary['complete'] or summary['variant'] != 'history' or summary['test_images_read']
            or summary['train_patients'] != 54 or summary['validation_patients'] != 10
            or summary['oracle_clicks_per_side'] != [1, 5]):
        raise ValueError('Expected complete frozen 54/10 history results with test unread and original click budgets')
    config = read(run / 'config.json')
    curve = learning_curve(run, summary)
    train, validation = [patients(run, summary, split) for split in ('train', 'val')]
    first, last = curve[0], curve[-1]
    train_change = last['train_dice_5'] - first['train_dice_5']
    val_change = last['val_dice_5'] - first['val_dice_5']
    if train_change > .02 and val_change < -.01:
        pattern = 'Frozen train improves while validation declines: consistent with overfitting under this recipe, not proof of its cause.'
    elif train_change <= .02 and val_change <= .01:
        pattern = 'Recorded improvement is limited on both splits; inspect optimization, supervision and feature resolution before assuming generalization alone.'
    else:
        pattern = 'Mixed learning pattern; inspect the complete measured curve and patient-level errors.'
    selected = score(summary, 'val')
    result = {'run': run.parent.name, 'variant': run.name, 'artifacts': str(run), 'complete': True, 'completed_epochs': summary['completed_epochs'],
        'planned_epochs': summary['planned_epochs'], 'stopped_early': summary['stopped_early'], 'best_epoch': summary['best_epoch'],
        'learning_curve': curve, 'first_to_last_measured_change_pp': {'train': train_change * 100, 'val': val_change * 100},
        'pattern_hypothesis': pattern, 'selected_checkpoint': {'train': summary_counts(train), 'val': summary_counts(validation),
            'train_val_gap_pp': 100 * (score(summary, 'train') - selected), 'threshold': summary['threshold'],
            'threshold_vs_fixed_0_5_gain_pp': 100 * (selected - number(summary['validation_control_fixed_0_5']['mean_bilateral_patient_dice'])),
            'one_to_five_click_gain_pp': 100 * (selected - score(summary, 'val', 1))},
        'worst_train_patients': train[:8], 'validation_patients_sorted': validation,
        'native_boundary_metrics': summary.get('native_boundary_metrics'), 'feature_geometry': geometry(config, summary),
        'target_gaps_pp': {'to_dice_0_8': 100 * (.8 - selected), 'to_dice_0_9': 100 * (.9 - selected)},
        'interpretation': 'Volume Dice and tolerance-based surface Dice are different metrics. Best-checkpoint train metrics are not the last-epoch train metrics.',
        'limitations': 'Patterns are hypotheses from a single run on a repeatedly inspected 10-patient validation split; no promised accuracy gain or anatomical label validation.',
        'pattern_rule': 'Descriptive heuristic only: train gain >2pp and val decline >1pp flags divergence; not a statistical test or an automatic training decision.',
        'images_or_weights_loaded': False, 'gpu_used': False, 'test_images_read': False, 'artifacts_modified': False}
    paired = matched_screening(run, summary)
    if paired is not None:
        result['matched_original_screening'] = paired
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'outputs/private/fsdp_research/latest')
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.run_dir), indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError, StopIteration) as error:
        parser.exit(1, f'Bottleneck audit stopped: {error}\n')


if __name__ == '__main__':
    main()
