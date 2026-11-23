#!/usr/bin/env python3
"""Compare native label reconstruction and fixed subvoxel grid shifts on train only."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import numpy as np
import torch

import physical_fsdp as data


def inspect(image, labels, value, sample):
    binary = data.image_xyz((labels == value).astype(np.uint8), data.geometry(image))
    support = data.sitk.Image(image.GetSize(), data.sitk.sitkUInt8) + 1
    support.CopyInformation(image)
    rows = []
    phases = list(itertools.product((0., .25), repeat=3)) + [(.5, .5, .5)]
    for phase in phases:
        grid = {**sample['grid'], 'origin': (np.asarray(sample['grid']['origin']) +
                np.asarray(phase) * sample['grid']['spacing']).tolist()}
        target_image = data.resample(binary, grid, data.sitk.sitkNearestNeighbor, pixel=data.sitk.sitkUInt8)
        target = data.xyz(target_image).astype(bool)
        valid = data.xyz(data.resample(support, grid, data.sitk.sitkNearestNeighbor, pixel=data.sitk.sitkUInt8)).astype(bool)
        shifted = {**sample, 'grid': grid, 'valid': torch.from_numpy(valid)[None, None]}
        logits = data.restore_logits(target.astype(np.float32) * 2 - 1, shifted)
        native = sample['native_target']
        nearest = data.xyz(data.resample(target_image, sample['native'], data.sitk.sitkNearestNeighbor)) > .5
        def score(pred):
            tp = int((pred & native).sum())
            return {'dice': 2 * tp / (int(pred.sum()) + int(native.sum())),
                    'false_positive_voxels': int((pred & ~native).sum()),
                    'false_negative_voxels': int((~pred & native).sum())}
        rows.append({'grid_shift_fraction_xyz': phase,
                     'linear_signed_label_roundtrip': score(logits > 0),
                     'nearest_label_roundtrip': score(nearest),
                     'native_fg_near_zero_logit_voxels': int((native & (np.abs(logits) < 1e-5)).sum()),
                     'model_foreground_voxels': int(target.sum())})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('outputs/private/sam_refine32/latest'))
    parser.add_argument('--train-alias', default='train_26')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    requested = data.read(run / 'requested_config.json')
    control = Path(requested['cohort_run_dir'])
    config = data.read(control / 'config.json')
    manifest = data.read(Path(config['data_dir']) / 'manifest.json')
    records = data.records_for(manifest, 'train')
    if not args.train_alias.startswith('train_'):
        parser.error('Only train aliases are accepted')
    index = int(args.train_alias.removeprefix('train_')) - 1
    if index not in range(len(records)):
        parser.error('Train alias index outside frozen train split')
    record = records[index]
    sources = {r['patient_id']: r for r in data.read(control / 'source_provenance.json')['train']}
    image, _, ct, labels, _ = data.load_case(record, Path(config['source_root']).resolve(), sources[record['patient_id']], args.train_alias)
    policy = data.read(run / 'preprocessing/policy.json')
    plan = policy['attempts'][-1]['plan']
    report = {'source_run': run.name, 'case': args.train_alias, 'source_geometry': data.geometry(image),
              'plan': plan, 'sides': {}, 'validation_images_read': False, 'test_images_read': False,
              'interpretation': 'Training-only reconstruction diagnostic. Phase sweeps and nearest-return scores do not change training, select validation preprocessing, or establish anatomical accuracy.'}
    for side, value in (('RFSDP', 1), ('LFSDP', 9)):
        sample = data.make_sample(image, ct, labels == value, args.train_alias, side, plan, policy['normalization'])
        report['sides'][side] = inspect(image, labels, value, sample)
    output = Path('outputs/private/roundtrip_audit') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    data.save(output / 'summary.json', report)
    compact = {key: value for key, value in report.items() if key != 'sides'}
    compact['sides'] = {side: {'unchanged_grid': rows[0], 'highest_linear_roundtrip_phase':
        max(rows, key=lambda r: r['linear_signed_label_roundtrip']['dice'])} for side, rows in report['sides'].items()}
    print(json.dumps(compact, indent=2))
    print(f'Report: {output / "summary.json"}')


if __name__ == '__main__':
    main()
