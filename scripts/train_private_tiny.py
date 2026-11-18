#!/usr/bin/env python3
"""Test SAM-Med3D learnability on two frozen training patients, one ROI per FSDP side."""

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import random
import sys

import numpy as np
import SimpleITK as sitk
import torch

from audit_expert_agreement import header, same_geometry
from evaluate_private_validation import file_hash, infer_clicks, overlay, preprocess, save_crop, scores, write_csv
from train_private_long import atomic_save, interactive_loss


SIDES = {'RFSDP': 1, 'LFSDP': 9}


def select_patients(manifest, seed):
    groups = manifest['splits']
    flat = [key for split in ('train', 'val', 'test') for key in groups[split]]
    if [len(groups[split]) for split in ('train', 'val', 'test')] != [54, 10, 20] or len(set(flat)) != len(flat):
        raise ValueError('Expected frozen disjoint 54/10/20 patient split')
    if not all(isinstance(key, str) and key.isdigit() for key in flat):
        raise ValueError('Expected numeric patient IDs')
    if manifest['target'] != {'1': 'RFSDP', '9': 'LFSDP'}:
        raise ValueError('Unexpected final FSDP target mapping')
    return sorted(random.Random(seed).sample(sorted(groups['train']), 2))


def native_prediction_indices(prediction, valid, metadata, shape):
    """Place XYZ crop voxels back in the original XYZ grid, discarding padding."""
    points = np.argwhere(prediction & valid) + np.asarray(metadata['crop_offset_xyz'], dtype=np.int64)
    if not len(points):
        return np.array([], dtype=np.int64)
    if ((points < 0) | (points >= np.asarray(shape))).any():
        raise ValueError('Valid crop voxel maps outside the original CT')
    return np.ravel_multi_index(points.T, shape)


def sparse_scores(predicted, target):
    predicted = np.unique(predicted)
    target = np.unique(target)
    intersection = len(np.intersect1d(predicted, target, assume_unique=True))
    return {'dice_full_target': 2 * intersection / (len(predicted) + len(target)) if len(predicted) + len(target) else 1.,
            'precision': intersection / len(predicted) if len(predicted) else 0.,
            'recall_full_target': intersection / len(target) if len(target) else 1.,
            'predicted_voxels_valid': len(predicted), 'full_target_voxels': len(target),
            'true_positive_voxels': intersection}


def prepare_samples(source_root, data, manifest, run, seed, split='train'):
    source_root, data = source_root.expanduser().resolve(), data.resolve()
    training_ids = select_patients(manifest, seed)
    if split not in ('train', 'val'):
        raise ValueError('Only training diagnostic or frozen validation data are supported')
    ids = training_ids if split == 'train' else sorted(manifest['splits']['val'])
    records = {row['patient_id']: row for row in manifest['cases']}
    inputs = run / 'inputs'
    inputs.mkdir()
    samples, patients, provenance = [], {}, []
    for index, key in enumerate(ids, 1):
        alias = f'{split}_{index:02d}'
        record = records[key]
        image_path, label_path = [Path(record[field]).resolve() for field in ('source_image', 'source_label')]
        if record['split'] != split or not image_path.is_relative_to(source_root / 'imagesTr') or not label_path.is_relative_to(source_root / 'labelsTr'):
            raise ValueError(f'Selected source is not an original {split} image/final label')
        with (source_root / 'ColorTable' / f'{key}_ColorTable.csv').open(encoding='utf-8-sig', newline='') as stream:
            colors = {int(row['LabelValue']): row['Name'].strip().upper() for row in csv.DictReader(stream)}
        if colors.get(1) != 'RFSDP' or colors.get(9) != 'LFSDP':
            raise ValueError('Final ColorTable does not confirm FSDP IDs')
        geometry = header(image_path)
        if not same_geometry(geometry, header(label_path)):
            raise ValueError('Original final label/CT geometry mismatch')
        label_image = sitk.ReadImage(str(label_path))
        label = sitk.GetArrayViewFromImage(label_image)
        if not np.isfinite(label).all() or not np.equal(label, np.floor(label)).all() or not np.isin(label, range(18)).all():
            raise ValueError('Invalid original final-label values')
        bilateral = np.isin(label, [1, 9])
        prepared = data / split / 'ct_private'
        if not same_geometry(geometry, header(prepared / 'imagesTr' / f'{key}.nii.gz')):
            raise ValueError('Prepared CT geometry differs from original')
        prepared_image = sitk.ReadImage(str(prepared / 'labelsTr' / f'{key}.nii.gz'))
        if not same_geometry(geometry, prepared_image) or not np.array_equal(sitk.GetArrayViewFromImage(prepared_image), bilateral):
            raise ValueError(f'Prepared {split} target differs from original labelsTr 1/9')
        del prepared_image
        patients[alias] = {'shape': tuple(geometry.GetSize()),
                           'target_indices': np.flatnonzero(bilateral.transpose(2, 1, 0).reshape(-1))}
        provenance.append({'case': alias, 'patient_id': key, 'source_image': str(image_path), 'source_label': str(label_path),
                           'source_image_sha256': file_hash(image_path), 'source_label_sha256': file_hash(label_path)})
        for side_index, (side, value) in enumerate(SIDES.items()):
            mask = (label == value).astype(np.uint8)
            expected = record['right_voxels' if value == 1 else 'left_voxels']
            if not mask.any() or int(mask.sum()) != expected:
                raise ValueError('Side is empty or source target count changed since preparation')
            path = inputs / f'{alias}_{side}_native_target.nii.gz'
            binary = sitk.GetImageFromArray(mask)
            binary.CopyInformation(label_image)
            sitk.WriteImage(binary, str(path), useCompression=True)
            image, target, raw, metadata, valid = preprocess(image_path, path)
            if metadata['target_retention'] != 1.:
                raise ValueError('Side ROI does not retain the entire target; no resizing/resampling performed')
            offset = label_image.TransformPhysicalPointToContinuousIndex(tuple(metadata['crop_origin_mm']))
            if not np.allclose(offset, np.round(offset), atol=1e-4, rtol=0):
                raise ValueError('Crop does not map to integer native voxels')
            metadata['crop_offset_xyz'] = np.round(offset).astype(int).tolist()
            name = f'{alias}_{side}'
            save_crop(inputs / f'{name}_ct_crop.nii.gz', raw, metadata)
            save_crop(inputs / f'{name}_target_crop.nii.gz', target[0, 0].numpy().astype(np.uint8), metadata)
            samples.append({'case': alias, 'side': side, 'name': name, 'image': image, 'target': target,
                            'raw': raw, 'metadata': metadata, 'valid': valid, 'prompt_seed': seed + index * 10 + side_index})
        print(f'{alias}: both side ROIs prepared with 100% target retention', flush=True)
    name = 'selected_training_patients.json' if split == 'train' else 'selected_validation_patients.json'
    (run / name).write_text(json.dumps(provenance, indent=2))
    return samples, patients


@torch.inference_mode()
def evaluate(model, samples, patients, device, export=None, progress_label=None):
    model.eval()
    side_rows, patient_rows, restored = [], [], {}
    if export is not None:
        export.mkdir(parents=True)
    for index, sample in enumerate(samples, 1):
        predictions = infer_clicks(model, sample['image'].to(device), sample['target'].to(device), [1, 5], sample['prompt_seed'])
        for clicks, logits in predictions.items():
            prediction = logits > 0
            metadata = sample['metadata']
            side_rows.append({'case': sample['case'], 'side': sample['side'], 'clicks': clicks,
                              'target_retention': metadata['target_retention'],
                              **scores(prediction, sample['target'][0, 0].numpy().astype(bool), metadata['full_target_voxels'], sample['valid'])})
            shape = patients[sample['case']]['shape']
            restored.setdefault((sample['case'], clicks), []).append(native_prediction_indices(prediction, sample['valid'], metadata, shape))
            if export is not None:
                name = f"{sample['name']}_{clicks}_clicks"
                save_crop(export / f'{name}_prediction_crop.nii.gz', prediction.astype(np.uint8), metadata)
                overlay(export / f'{name}_overlay.png', sample['raw'], sample['target'][0, 0].numpy().astype(bool), prediction, metadata['target_retention'])
        if progress_label:
            print(f'{progress_label}: {sample["name"]} complete ({index}/{len(samples)})', flush=True)
    for (alias, clicks), parts in sorted(restored.items()):
        patient_rows.append({'case': alias, 'clicks': clicks,
                             **sparse_scores(np.concatenate(parts), patients[alias]['target_indices'])})
    means = {str(clicks): {'mean_side_dice': float(np.mean([r['dice_full_target'] for r in side_rows if r['clicks'] == clicks])),
                          'minimum_side_dice': min(r['dice_full_target'] for r in side_rows if r['clicks'] == clicks),
                          'mean_bilateral_patient_dice': float(np.mean([r['dice_full_target'] for r in patient_rows if r['clicks'] == clicks])),
                          'minimum_bilateral_patient_dice': min(r['dice_full_target'] for r in patient_rows if r['clicks'] == clicks)} for clicks in (1, 5)}
    if export is not None:
        write_csv(export / 'sides.csv', side_rows)
        write_csv(export / 'patients.csv', patient_rows)
    return means, side_rows, patient_rows


def diagnostic_score(means):
    return min(means['5']['minimum_side_dice'], means['5']['minimum_bilateral_patient_dice'])


def execute(args):
    device = torch.device(args.device)
    run, data = args.run_dir, args.data_dir
    manifest_path = data / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    selected = select_patients(manifest, args.seed)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(manifest_sha256=file_hash(manifest_path), initial_checkpoint_sha256=file_hash(args.checkpoint),
                  script_sha256=file_hash(Path(__file__)),
                  shared_code_sha256={name: file_hash(Path(__file__).with_name(name)) for name in ('train_private_long.py', 'evaluate_private_validation.py', 'audit_expert_agreement.py')},
                  versions={name: importlib.metadata.version(name) for name in ('torch', 'torchio', 'monai', 'numpy', 'SimpleITK')},
                  train_patients=2, side_rois=4, augmentation='none', optimizer='AdamW all parameters; constant LR; weight_decay=0',
                  model_mode='eval mode with gradients during optimization; identical mode and AMP policy during scoring',
                  prompt_protocol='fixed per-side seed; adaptive oracle single-click interactions with detached feedback; native coordinates',
                  roi_protocol='native 128 cube centered on side GT; no resampling; full target retention required',
                  interpretation='Training-only memorization diagnostic, not validation/test performance',
                  validation_images_read=False, test_images_read=False)
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    samples, patients = prepare_samples(args.source_root, data, manifest, run, args.seed)
    if len(samples) != 4 or len(patients) != len(selected):
        raise ValueError('Expected two training patients and four nonempty side ROIs')
    sys.path.insert(0, str(args.sam_dir))
    from segment_anything.build_sam3D import sam_model_registry3D
    from monai.losses import DiceCELoss
    model = sam_model_registry3D['vit_b_ori'](checkpoint=None).to(device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)
    del checkpoint
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    criterion = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    history, updates, skips = [], 0, 0
    means, side_rows, patient_rows = evaluate(model, samples, patients, device, run / 'initial')
    initial = means
    best_score, best_epoch, best_means = diagnostic_score(means), 0, means
    def save(path, epoch):
        atomic_save(path, {'epoch': epoch, 'model_state_dict': model.state_dict(), 'config': config,
                           'optimizer_updates': updates, 'amp_skipped_updates': skips, 'training_only_dice': means})
    save(run / 'sam_model_tiny_best.pth', 0)
    completed = 0
    print(json.dumps({'initial_training_only_dice': means}), flush=True)
    for epoch in range(1, args.epochs + 1) if best_score < args.target_dice else []:
        # Keep dropout and normalization behavior identical to frozen-model inference.
        model.eval()
        losses, epoch_updates = [], 0
        for sample in samples:
            optimizer.zero_grad(set_to_none=True)
            image, target = sample['image'].to(device), sample['target'].to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                loss, _ = interactive_loss(model, image, target, criterion, 5, sample['prompt_seed'])
            if not torch.isfinite(loss):
                raise ValueError('Non-finite diagnostic loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm) and not scaler.is_enabled():
                raise ValueError('Non-finite diagnostic gradient')
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            success = scaler.get_scale() >= scale
            updates += int(success)
            skips += int(not success)
            epoch_updates += int(success)
            losses.append(loss.item())
        completed = epoch
        row = {'epoch': epoch, 'mean_interaction_loss': float(np.mean(losses)),
               'optimizer_updates': epoch_updates, 'total_optimizer_updates': updates, 'total_amp_skipped_updates': skips,
               'minimum_training_dice_5_clicks': None, 'mean_training_bilateral_dice_5_clicks': None}
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            means, side_rows, patient_rows = evaluate(model, samples, patients, device)
            score = diagnostic_score(means)
            row.update(minimum_training_dice_5_clicks=score, mean_training_bilateral_dice_5_clicks=means['5']['mean_bilateral_patient_dice'])
            write_csv(run / 'sides_latest.csv', side_rows)
            write_csv(run / 'patients_latest.csv', patient_rows)
            if score > best_score:
                best_score, best_epoch, best_means = score, epoch, means
                save(run / 'sam_model_tiny_best.pth', epoch)
        history.append(row)
        write_csv(run / 'history.csv', history)
        (run / 'progress.json').write_text(json.dumps({'completed_epochs': epoch, 'planned_epochs': args.epochs,
            'best_epoch': best_epoch, 'best_minimum_training_dice_5_clicks': best_score,
            'training_only': True, 'target_dice': args.target_dice}, indent=2))
        print(json.dumps({'tiny_epoch': row, 'best_minimum_training_dice_5_clicks': best_score}), flush=True)
        if best_score >= args.target_dice:
            break
    save(run / 'sam_model_tiny_last.pth', completed)
    best = torch.load(run / 'sam_model_tiny_best.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(best['model_state_dict'], strict=True)
    del best
    means, _, _ = evaluate(model, samples, patients, device, run / 'best_training')
    if means != best_means:
        for clicks in ('1', '5'):
            if not all(np.isclose(means[clicks][key], best_means[clicks][key], atol=1e-6, rtol=0) for key in means[clicks]):
                raise ValueError('Reloaded best checkpoint fails to reproduce diagnostic scores')
    summary = {'completed_epochs': completed, 'planned_epochs': args.epochs, 'best_epoch': best_epoch,
               'train_patients': 2, 'side_rois': 4, 'initial_training_only_dice': initial, 'best_training_only_dice': means,
               'diagnostic_target_dice': args.target_dice, 'memorization_check_passed': diagnostic_score(means) >= args.target_dice,
               'optimizer_updates': updates, 'amp_skipped_updates': skips,
               'checkpoint_sha256': file_hash(run / 'sam_model_tiny_best.pth'),
               'protocol': config['roi_protocol'] + '; fixed-seed oracle prompts; 5 clicks PER SIDE (10 total for bilateral @5)',
               'click_budget_per_side': [1, 5], 'bilateral_total_clicks': {'1': 2, '5': 10},
               'validation_images_read': False, 'test_images_read': False,
               'interpretation': 'Held-in training diagnostic only; no validation improvement or 80% generalization established',
               'artifacts': str(run)}
    (run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sam-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--target-dice', type=float, default=.9)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if min(args.epochs, args.eval_every) < 1 or not np.isfinite(args.lr) or args.lr <= 0 or not 0 < args.target_dice <= 1:
        parser.error('Invalid diagnostic counts, LR or target Dice')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    if args.run_dir.exists():
        parser.error('Run directory already exists; choose a fresh diagnostic run')
    args.run_dir.mkdir(parents=True)
    try:
        execute(args)
    except Exception as error:
        (args.run_dir / 'failure.json').write_text(json.dumps({'error_type': type(error).__name__, 'error': str(error)}, indent=2))
        raise


if __name__ == '__main__':
    main()
