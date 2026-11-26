#!/usr/bin/env python3
"""Audit existing SAM control/affine exports on CPU without model inference."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import tarfile


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def counts(row):
    pred, target, tp = [int(row[k]) for k in
                        ('predicted_voxels_valid', 'full_target_voxels', 'true_positive_voxels')]
    if target <= 0 or pred < 0 or not 0 <= tp <= min(pred, target):
        raise ValueError('Invalid prediction/reference voxel counts')
    dice = 2 * tp / (pred + target)
    if not math.isclose(dice, float(row['dice_full_target']), abs_tol=1e-6):
        raise ValueError('CSV Dice differs from voxel counts')
    fp, fn = pred - tp, target - tp
    return dict(dice=dice, predicted_voxels=pred, target_voxels=target, tp=tp, fp=fp, fn=fn,
                volume_ratio=pred / target, precision=tp / pred if pred else 0., recall=tp / target,
                empty_prediction=pred == 0,
                voxel_error_balance='FP>FN' if fp > fn else 'FN>FP' if fn > fp else 'FP=FN')


def audit(run, output):
    run, output = run.resolve(), output.resolve()
    if output == run or run.is_relative_to(output):
        raise ValueError('Audit output must not contain or replace the source run')
    if output.exists():
        raise ValueError('Choose a new audit output directory')
    summaries, rows, side_rows, sources, package_files = {}, {}, {}, {}, []
    for arm in ('control', 'affine'):
        folder = run / arm
        summary_file = folder / 'summary.json'
        summary = json.loads(summary_file.read_text())
        if (not summary['complete'] or summary['test_images_read']
                or summary.get('augmentation_arm') != arm or summary['validation_patients'] != 10
                or summary['oracle_clicks_per_side'] != [1, 5]):
            raise ValueError(f'Unexpected completed SAM arm: {arm}')
        summaries[arm] = summary
        exports = folder / 'best_validation'
        patients_file, sides_file = [exports / f'val_5_{suffix}.csv' for suffix in ('patients', 'sides')]
        boundary_file = exports / 'val_5_boundary_metrics.csv'
        patients, sides, boundaries = [read_rows(p) for p in (patients_file, sides_file, boundary_file)]
        if (len(patients) != 10 or len(sides) != 20 or len(boundaries) != 10
                or len({r['case'] for r in patients}) != 10
                or len({(r['case'], r['side']) for r in sides}) != 20):
            raise ValueError('Expected ten unique patients and twenty unique sides')
        boundary = {r['case']: r for r in boundaries}
        if set(boundary) != {r['case'] for r in patients}:
            raise ValueError('Boundary and overlap patient identities differ')
        rows[arm] = {}
        for row in patients:
            case = row['case']
            b = boundary[case]
            rows[arm][case] = dict(case=case, **counts(row),
                surface_dice_0_5mm=float(b['surface_dice_0_5mm']), surface_dice_1mm=float(b['surface_dice_1mm']),
                hd95_mm=float(b['hd95_mm']) if b['hd95_mm'] else None)
        expected = summary['best_validation_metrics']['5']['mean_bilateral_patient_dice']
        if not math.isclose(mean(r['dice'] for r in rows[arm].values()), expected, abs_tol=1e-6):
            raise ValueError('Summary and per-patient CSV mean differ')
        side_rows[arm] = [dict(case=r['case'], side=r['side'], **counts(r)) for r in sides]
        for path in (summary_file, patients_file, sides_file, boundary_file):
            sources[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
            package_files.append(path)
        package_files.extend(exports.glob('val_*.png'))
        if (folder / 'history.csv').exists():
            package_files.append(folder / 'history.csv')
    if set(rows['control']) != set(rows['affine']):
        raise ValueError('Arms use different validation patients')
    if summaries['control']['research_source'] != summaries['affine']['research_source']:
        raise ValueError('Arms have different initialization provenance')
    for case in rows['control']:
        if rows['control'][case]['target_voxels'] != rows['affine'][case]['target_voxels']:
            raise ValueError('Arms have different target voxel counts')
    comparisons = [dict(case=case, control_dice=rows['control'][case]['dice'], affine_dice=rows['affine'][case]['dice'],
                        delta=rows['affine'][case]['dice'] - rows['control'][case]['dice']) for case in rows['control']]
    comparisons.sort(key=lambda r: r['delta'])
    report = dict(complete=True, gpu_used=False, optimizer_updates=0, test_images_read=False,
        source_run=str(run), source_hashes=sources,
        same_selected_threshold=summaries['control']['threshold'] == summaries['affine']['threshold'],
        protocol='Ten repeatedly inspected validation patients; GT ROI and five oracle clicks per side.',
        interpretation='FP/FN and volume balance describe errors; root cause requires overlays. Surface Dice at1mm is not80%volume Dice.',
        paired_patients=comparisons, arms={})
    output.mkdir(parents=True)
    def save_csv(path, records):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    for arm in ('control', 'affine'):
        patients = sorted(rows[arm].values(), key=lambda r: r['dice'])
        report['arms'][arm] = dict(mean_dice=mean(r['dice'] for r in patients),
            best_epoch=summaries[arm]['best_epoch'], completed_epochs=summaries[arm]['completed_epochs'],
            threshold=summaries[arm]['threshold'],
            patients_with_more_fp_than_fn=sum(r['fp'] > r['fn'] for r in patients),
            patients_with_more_fn_than_fp=sum(r['fn'] > r['fp'] for r in patients),
            empty_predictions=sum(r['empty_prediction'] for r in patients), worst_patients=patients[:3])
        save_csv(output / f'{arm}_patients.csv', patients)
        save_csv(output / f'{arm}_sides.csv', sorted(side_rows[arm], key=lambda r: r['dice']))
    save_csv(output / 'paired_patients.csv', comparisons)
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    with tarfile.open(output / 'review_bundle.tar.gz', 'w:gz') as archive:
        for path in package_files:
            archive.add(path, arcname='exports/' + str(path.relative_to(run)))
        for path in output.glob('*.csv'):
            archive.add(path, arcname='audit/' + path.name)
        archive.add(output / 'summary.json', arcname='audit/summary.json')
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f'Review bundle: {output / "review_bundle.tar.gz"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    audit(args.run_dir, args.output_dir)
