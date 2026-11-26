#!/usr/bin/env python3
"""Export verified whole CT volumes and score automatic nnU-Net predictions."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import SimpleITK as sitk
import torch

import physical_fsdp as data
from evaluate_private_validation import file_hash
from fsdp_verified_cache import image_inventory, require_images, validate_targets, read_image, source_label


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def prepare(args):
    foundation = args.foundation_run_dir.resolve()
    config = data.read(foundation / 'config.json')
    if file_hash(foundation / 'hybrid_data.pt') != config['cache_sha256']:
        raise ValueError('Foundation cache checksum differs')
    cohort = data.read(Path(config['cohort_run_dir']) / 'config.json')
    manifest_path = Path(cohort['data_dir']) / 'manifest.json'
    if file_hash(manifest_path) != config['manifest_sha256']:
        raise ValueError('Frozen manifest checksum differs')
    prepared = torch.load(foundation / 'hybrid_data.pt', map_location='cpu', weights_only=False)
    integrity = validate_targets(prepared, data.read(manifest_path))
    inventory = image_inventory(prepared['source_provenance'], args.source_root, verify=True)
    require_images(inventory)
    if not inventory['ct_content_verified']:
        raise ValueError('CT hash verification was not performed')
    source_bytes = sum(Path(row['path']).stat().st_size for row in inventory['rows'] if row['case'] != 'train_28')
    ancestor = args.output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    minimum_free = 3 * source_bytes + 20 * 2**30
    if shutil.disk_usage(ancestor).free < minimum_free:
        raise ValueError(f'Insufficient free disk for export/preprocessing; require at least {minimum_free / 2**30:.1f} GiB')
    dataset = f'Dataset{args.dataset_id:03d}_FSDPAuto'
    raw = args.output / 'nnUNet_raw' / dataset
    preprocessed = args.output / 'nnUNet_preprocessed'
    results = args.output / 'nnUNet_results'
    if args.output.exists():
        raise ValueError('Choose a new output directory; existing artifacts are protected')
    for folder in (raw / 'imagesTr', raw / 'labelsTr', args.output / 'validation_images',
                   args.output / 'validation_labels', preprocessed, results):
        folder.mkdir(parents=True, exist_ok=True)
    records = []
    for split in ('train', 'val'):
        sources = {row['case']: row for row in prepared['source_provenance'][split]}
        for alias in sorted(prepared['patients'][split]):
            if alias == 'train_28':
                continue
            samples = [s for s in prepared['samples'][split] if s['case'] == alias]
            if len(samples) != 2 or {s['side'] for s in samples} != {'RFSDP', 'LFSDP'}:
                raise ValueError(f'Invalid bilateral samples: {alias}')
            expected = sources[alias]
            image = read_image(expected, args.source_root, expected, samples)
            labels = [source_label(s, image) for s in samples]
            target = sitk.Or(labels[0], labels[1])
            native = data.xyz(target).astype(bool)
            indices = np.flatnonzero(native.ravel())
            wanted = prepared['patients'][split][alias]['target_indices']
            if not np.array_equal(indices, wanted):
                raise ValueError(f'Full-volume target reconstruction differs: {alias}')
            key = f'FSDP_{alias}'
            image_dir = raw / 'imagesTr' if split == 'train' else args.output / 'validation_images'
            label_dir = raw / 'labelsTr' if split == 'train' else args.output / 'validation_labels'
            image_path = image_dir / f'{key}_0000.nii.gz'
            label_path = label_dir / f'{key}.nii.gz'
            sitk.WriteImage(image, str(image_path), True)
            sitk.WriteImage(target, str(label_path), True)
            records.append({'split': split, 'case': alias, 'key': key,
                            'source_sha256': expected['source_image_sha256'],
                            'image_sha256': file_hash(image_path), 'label_sha256': file_hash(label_path),
                            'target_voxels': int(native.sum()), 'shape': list(image.GetSize())})
            print(f'{split}: {alias} full CT and target exported', flush=True)
    if sum(r['split'] == 'train' for r in records) != 53 or sum(r['split'] == 'val' for r in records) != 10:
        raise ValueError('Expected 53 train and 10 validation patients')
    write_json(raw / 'dataset.json', {'channel_names': {'0': 'zscore'},
        'labels': {'background': 0, 'FSDP': 1}, 'numTraining': 53, 'file_ending': '.nii.gz'})
    write_json(args.output / 'export_report.json', {'complete': True, 'dataset': dataset,
        'foundation_cache_sha256': config['cache_sha256'], 'manifest_sha256': config['manifest_sha256'],
        'target_integrity': integrity, 'source_inventory': inventory, 'records': records,
        'normalization': 'nnU-Net per-case zscore; avoids foreground-only CT percentile clipping',
        'validation_used_for_training_or_planning': False, 'test_images_read': False,
        'protocol': 'Whole native CT and full bilateral FSDP target; train_28 excluded; no GT ROI or clicks at inference.'})


def score(args):
    report = data.read(args.output / 'export_report.json')
    if not report['complete'] or report['test_images_read']:
        raise ValueError('Export report incomplete or test data accessed')
    rows = []
    for record in report['records']:
        if record['split'] != 'val':
            continue
        truth_path = args.output / 'validation_labels' / f'{record["key"]}.nii.gz'
        prediction_path = args.predictions / f'{record["key"]}.nii.gz'
        if file_hash(truth_path) != record['label_sha256'] or not prediction_path.is_file():
            raise ValueError(f'Missing prediction or changed target: {record["case"]}')
        truth_image, pred_image = sitk.ReadImage(str(truth_path)), sitk.ReadImage(str(prediction_path))
        if (truth_image.GetSize() != pred_image.GetSize() or
                not np.allclose(truth_image.GetSpacing(), pred_image.GetSpacing(), atol=1e-5) or
                not np.allclose(truth_image.GetOrigin(), pred_image.GetOrigin(), atol=1e-4) or
                not np.allclose(truth_image.GetDirection(), pred_image.GetDirection(), atol=1e-5)):
            raise ValueError(f'Prediction geometry differs: {record["case"]}')
        gt, pred = data.xyz(truth_image).astype(bool), data.xyz(pred_image).astype(bool)
        tp = int(np.logical_and(gt, pred).sum()); predicted = int(pred.sum()); target = int(gt.sum())
        rows.append({'case': record['case'], 'dice': 2 * tp / (predicted + target),
                     'precision': tp / predicted if predicted else 0., 'recall': tp / target,
                     'tp': tp, 'fp': predicted - tp, 'fn': target - tp,
                     'target_voxels': target, 'predicted_voxels': predicted})
    if len(rows) != 10:
        raise ValueError('Expected all ten validation patients')
    with (args.predictions / 'patients.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {'complete': True, 'configuration': args.configuration, 'epoch_budget': 100,
        'validation_patients': 10, 'mean_bilateral_patient_dice': float(np.mean([r['dice'] for r in rows])),
        'minimum_bilateral_patient_dice': min(r['dice'] for r in rows),
        'mean_precision': float(np.mean([r['precision'] for r in rows])),
        'mean_recall': float(np.mean([r['recall'] for r in rows])),
        'oracle_clicks': 0, 'gt_assisted_roi_at_inference': False, 'test_images_read': False,
        'predictions': str(args.predictions), 'patients': rows}
    write_json(args.predictions / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('configuration', 'epoch_budget', 'mean_bilateral_patient_dice')}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'score'))
    parser.add_argument('--foundation-run-dir', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-id', type=int, default=901)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--configuration', choices=('2d', '3d_fullres'))
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    if args.action == 'prepare':
        if not args.foundation_run_dir or not args.source_root or not 1 <= args.dataset_id <= 999:
            parser.error('prepare requires foundation, source root and valid dataset ID')
        args.foundation_run_dir = args.foundation_run_dir.expanduser().resolve()
        args.source_root = args.source_root.expanduser().resolve()
        prepare(args)
    else:
        if not args.predictions or not args.configuration:
            parser.error('score requires predictions and configuration')
        args.predictions = args.predictions.expanduser().resolve()
        score(args)


if __name__ == '__main__':
    main()
