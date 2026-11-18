#!/usr/bin/env python3
"""Compare train and validation with the same frozen model and inference protocol."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from evaluate_private_validation import file_hash, infer_clicks, overlay, preprocess, save_crop, scores, write_csv
from train_private_long import Crops, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sam-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    data = args.data_dir.resolve()
    run = args.run_dir.resolve() if args.run_dir else None
    if run is None:
        candidates = list((Path(__file__).resolve().parents[1] / 'runs').glob('fsdp_private_v2_*/summary.json'))
        if not candidates:
            parser.error('No completed v2 run found; supply --run-dir')
        run = max(candidates, key=lambda path: path.stat().st_mtime).parent
    if args.output_dir.exists():
        parser.error('Output already exists')
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    manifest_path = data / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    groups = manifest['splits']
    all_ids = sum([groups[name] for name in ('train', 'val', 'test')], [])
    if len(set(all_ids)) != len(all_ids) or len(groups['val']) != 10 or len(groups['test']) != 20:
        parser.error('Frozen patient splits are inconsistent')
    checkpoint_path = run / 'sam_model_val_best.pth'
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if state['config']['manifest_sha256'] != file_hash(manifest_path):
        parser.error('Checkpoint belongs to a different frozen manifest')
    if [Path(path).resolve() for path in state['used_datas']] != [data / 'train/ct_private']:
        parser.error('Checkpoint training path does not match the private dataset')
    best_epoch, seed = state['epoch'], state['config']['seed']
    logged = next((row for row in state['history'] if row['epoch'] == best_epoch), None)
    previous = json.loads((run / 'summary.json').read_text())
    if previous['best_epoch'] != best_epoch:
        parser.error('Selected checkpoint epoch and completed run summary differ')
    sys.path.insert(0, str(args.sam_dir.resolve()))
    from segment_anything.build_sam3D import sam_model_registry3D
    model = sam_model_registry3D['vit_b_ori'](checkpoint=None).to(device).eval()
    model.load_state_dict(state['model_state_dict'], strict=True)
    del state
    rows, comparison = [], []
    for split in ('train', 'val'):
        ids = sorted(groups[split])
        for folder in ('imagesTr', 'labelsTr'):
            found = {path.name for path in (data / split / 'ct_private' / folder).glob('*.nii.gz')}
            if found != {key + '.nii.gz' for key in ids}:
                parser.error(f'{split} files differ from the frozen split')
        dataset = Crops(data / split / 'ct_private', ids, augment=False)
        means, split_rows = validate(model, dataset, device, seed)
        for row in split_rows:
            row['case'] = split + '_' + row['case'].split('_')[-1]
            row['split'] = split
        rows.extend(split_rows)
        for clicks in (1, 5):
            selected = [row for row in split_rows if row['clicks'] == clicks]
            comparison.append({'split': split, 'clicks': clicks, 'cases': len(ids),
                               'mean_dice_full_target': means[str(clicks)],
                               'mean_dice_crop': float(np.mean([row['dice_crop'] for row in selected])),
                               'mean_precision': float(np.mean([row['precision'] for row in selected])),
                               'mean_recall': float(np.mean([row['recall_full_target'] for row in selected])),
                               'mean_target_retention': float(np.mean([row['target_retention'] for row in selected]))})
        ranked = sorted([row for row in split_rows if row['clicks'] == 5], key=lambda row: row['dice_full_target'])
        write_csv(output / f'{split}_ranked_cases.csv', ranked)
        for rank, row in [('worst', ranked[0]), ('middle', ranked[len(ranked) // 2]), ('best', ranked[-1])]:
            index = int(row['case'].split('_')[-1]) - 1
            key = ids[index]
            image, target, raw, metadata, _ = preprocess(dataset.root / 'imagesTr' / f'{key}.nii.gz', dataset.root / 'labelsTr' / f'{key}.nii.gz')
            prediction = infer_clicks(model, image.to(device), target.to(device), [5],
                                      seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))[5] > 0
            stem = f'{split}_{rank}_{row["case"]}'
            overlay(output / f'{stem}.png', raw, target[0, 0].numpy().astype(bool), prediction, metadata['target_retention'])
            save_crop(output / f'{stem}_prediction.nii.gz', prediction.astype(np.uint8), metadata)
        print(f'{split} completed: {means}', flush=True)
        del dataset
    write_csv(output / 'cases.csv', rows)
    write_csv(output / 'train_val_comparison.csv', comparison)
    matched = all(abs(row['mean_dice_full_target'] - previous['best_validation_dice'][str(row['clicks'])]) < 1e-4
                  for row in comparison if row['split'] == 'val')
    result = {'run': run.name, 'checkpoint': str(checkpoint_path), 'checkpoint_sha256': file_hash(checkpoint_path),
              'best_epoch': best_epoch, 'comparison': comparison, 'training_logged_at_best_epoch': logged,
              'validation_reproduces_saved_summary': matched, 'test_images_read': False,
              'protocol': 'same eval mode, no augmentation, GT-guided crop and oracle prompts for both splits',
              'five_click_train_val_gap': next(row['mean_dice_full_target'] for row in comparison if row['split'] == 'train' and row['clicks'] == 5)
                - next(row['mean_dice_full_target'] for row in comparison if row['split'] == 'val' and row['clicks'] == 5)}
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2), flush=True)
    if not matched:
        raise SystemExit('Validation differs from the saved result; investigate before interpreting the generalization gap')


if __name__ == '__main__':
    main()
