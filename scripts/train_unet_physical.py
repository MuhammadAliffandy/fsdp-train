#!/usr/bin/env python3
"""Physical-grid FSDP preprocessing followed by full-cohort 3D U-Net training."""

import argparse
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

import physical_fsdp as data
import train_private_unet as base
from evaluate_private_validation import file_hash, save_crop, write_csv
from train_private_long import accumulation_window, atomic_save, early_stop_update
from train_private_tiny import native_prediction_indices, select_patients, sparse_scores


CODE_FILES = ('train_unet_physical.py', 'physical_fsdp.py', 'train_private_unet.py',
              'train_private_tiny.py', 'train_private_long.py', 'evaluate_private_validation.py', 'audit_expert_agreement.py')


def verify_sources(provenance):
    for rows in provenance.values():
        for row in rows:
            for name in ('source_image', 'source_label', 'color_table'):
                if file_hash(Path(row[name])) != row[name + '_sha256']:
                    raise ValueError('Source CT/label/ColorTable changed since preprocessing')


def prepare(args):
    from monai.networks.nets import UNet
    from monai.transforms import RandAffined
    prior = args.cohort_run_dir
    old, summary = data.read(prior / 'config.json'), data.read(prior / 'summary.json')
    if (not summary['complete'] or summary['experiment'] != 'unet_cohort_train_calibrated'
            or summary['train_patients'] != 54 or summary['validation_patients'] != 10 or summary['test_images_read']):
        raise ValueError('Expected the completed previous54/10 U-Net cohort control')
    manifest_path = Path(old['data_dir']) / 'manifest.json'
    manifest = data.read(manifest_path)
    select_patients(manifest, args.seed)
    if args.seed != old['seed'] or file_hash(manifest_path) != old['manifest_sha256']:
        raise ValueError('Frozen manifest/seed differs from prior cohort')
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'torchio', 'monai', 'numpy', 'SimpleITK')}
    if versions != old['versions']:
        raise ValueError('Use the same package versions as the previous cohort')
    requested = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != 'resume'}
    requested.update(experiment='unet_physical_v2', architecture=base.ARCHITECTURE, versions=versions,
        code_sha256={name: file_hash(Path(__file__).with_name(name)) for name in CODE_FILES},
        monai_source_sha256={obj.__name__: file_hash(Path(inspect.getfile(obj))) for obj in (UNet, RandAffined)},
        prior_control_sha256={name: file_hash(prior / name) for name in ('config.json', 'summary.json', 'source_provenance.json')},
        manifest_sha256=file_hash(manifest_path), initialization='fresh seeded random U-Net weights; no learned checkpoint initialization',
        normalization='train-only pooled raw FSDP foreground; percentile0.5/99.5 clipping; shared mean/std',
        loss='FP32 DiceCELoss sigmoid=True squared_pred=True over valid CT voxels only; padding excluded',
        augmentation='none; seeded shuffle only' if args.augmentation == 'none' else
            'paired affine p0.5: rotations+-5deg, scaling+-5%, translation+-1mm; LR flip0.5; gain/shift p0.3; noise sigma0.02 p0.15; border-touching transforms rejected',
        threshold_protocol='native restored logits; choose threshold using all54 frozen train patients only before each validation; mean bilateral train Dice; ties closest0.5 then lower threshold',
        threshold_candidates=list(data.THRESHOLDS), selection='mean bilateral native full-target validation Dice at the train-chosen threshold',
        optimizer='AdamW lr configured; weight_decay1e-4; clip_grad_norm1', schedule='five-epoch warmup then cosine to0.05 of LR',
        protocol='GT-side physical bbox-centered LPS ROI; train-derived common physical grid; original-native full-target bilateral union; no oracle clicks',
        test_images_read=False)
    cache = args.run_dir / 'physical_data.pt'
    if args.resume:
        config = data.read(args.run_dir / 'config.json')
        if any(config[key] != value for key, value in requested.items()):
            raise ValueError('Resume code/settings/control differs from frozen run')
        if file_hash(cache) != config['physical_cache_sha256']:
            raise ValueError('Physical preprocessing cache changed')
        prepared = torch.load(cache, map_location='cpu', weights_only=False)
        if prepared['policy'] != config['physical_policy']:
            raise ValueError('Cached preprocessing policy differs')
        verify_sources(prepared['source_provenance'])
    else:
        data.save(args.run_dir / 'requested_config.json', requested)
        prior_sources = data.read(prior / 'source_provenance.json')
        prepared = data.prepare_dataset(manifest, Path(old['source_root']).resolve(), prior_sources, args.run_dir / 'preprocessing')
        atomic_save(cache, prepared)
        config = {**requested, 'physical_cache_sha256': file_hash(cache), 'physical_policy': prepared['policy']}
        data.save(args.run_dir / 'config.json', config)
        data.save(args.run_dir / 'source_provenance.json', prepared['source_provenance'])
    return prepared, config, summary


def train_epoch(model, samples, optimizer, scaler, criterion, device, epoch, args):
    model.train()
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
    optimizer.zero_grad(set_to_none=True)
    losses, dice, updates, skipped, rejected = [], [], 0, 0, 0
    for step, index in enumerate(order):
        image, target, valid, fallback = data.augment(samples[index], args.seed + epoch * 1000 + step, args.augmentation == 'light')
        rejected += int(fallback)
        image, target, valid = image.to(device), target.to(device).float(), valid.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits = base.checked_logits(model, image)
        loss = data.masked_loss(criterion, logits, target, valid)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite valid-voxel training loss')
        size, should_step = accumulation_window(step, len(samples), args.accumulation)
        scaler.scale(loss / size).backward()
        if should_step:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm) and not scaler.is_enabled():
                raise ValueError('Nonfinite training gradients')
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            success = scaler.get_scale() >= old_scale
            updates += int(success)
            skipped += int(not success)
            optimizer.zero_grad(set_to_none=True)
            print(f'epoch={epoch + 1} roi={step + 1}/{len(samples)} loss={loss.item():.6f} updates={updates} skips={skipped}', flush=True)
        pred, gt = ((logits.detach() > 0) & valid), target.bool()
        denominator = pred.sum() + gt.sum()
        dice.append(float((2 * (pred & gt).sum() / denominator).item()) if denominator else 1.)
        losses.append(loss.item())
    expected = math.ceil(len(samples) / args.accumulation)
    if updates + skipped != expected:
        raise ValueError('Incomplete optimizer accumulation')
    return {'loss': float(np.mean(losses)), 'moving_augmented_model_grid_dice_at_0_5': float(np.mean(dice)),
            'optimizer_updates': updates, 'amp_skipped_updates': skipped, 'expected_updates': expected,
            'rejected_spatial_augmentations': rejected}


def calibration_validation(model, prepared, device, label):
    train = data.evaluate(model, prepared['samples']['train'], prepared['patients']['train'], device, data.THRESHOLDS, label + '_train_calibration')
    threshold = data.choose_threshold(train)
    validation = data.evaluate(model, prepared['samples']['val'], prepared['patients']['val'], device,
                               sorted(set((threshold, .3, .5))), label + '_validation')
    return threshold, train, validation


@torch.inference_mode()
def export_representatives(model, prepared, device, threshold, expected, output):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    model.eval()
    ranked = sorted(expected['patients'], key=lambda r: (r['dice_full_target'], r['case']))
    aliases = {ranked[0]['case'], ranked[len(ranked) // 2]['case'], ranked[-1]['case']}
    lookup = {(r['case'], r['side']): r['dice_full_target'] for r in expected['sides']}
    output.mkdir()
    for sample in prepared['samples']['val']:
        if sample['case'] not in aliases:
            continue
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits = base.checked_logits(model, sample['image'].to(device))
        prediction = data.restore_logits(logits[0, 0].float().cpu().numpy(), sample) > math.log(threshold / (1 - threshold))
        patients = prepared['patients']['val'][sample['case']]
        indices = native_prediction_indices(prediction, np.ones(prediction.shape, bool), sample['metadata'], patients['shape'])
        target_indices = native_prediction_indices(sample['native_target'], np.ones(prediction.shape, bool), sample['metadata'], patients['shape'])
        score = sparse_scores(indices, target_indices)['dice_full_target']
        if abs(score - lookup[(sample['case'], sample['side'])]) > 1e-4:
            raise ValueError('Representative inference does not reproduce selected scores')
        for suffix, mask in (('prediction', prediction), ('target', sample['native_target'])):
            save_crop(output / f'{sample["name"]}_{suffix}.nii.gz', mask.astype(np.uint8), sample['metadata'])
        fig, axes = plt.subplots(3, 3, figsize=(10, 9))
        for row, axis in enumerate((2, 1, 0)):
            index = int(np.argmax(sample['native_target'].sum(axis=tuple(i for i in range(3) if i != axis))))
            image, gt, pred = [np.take(a, index, axis=axis).T for a in (sample['raw_native'], sample['native_target'], prediction)]
            remaining = [i for i in range(3) if i != axis]
            aspect = sample['native']['spacing'][remaining[1]] / sample['native']['spacing'][remaining[0]]
            for col, ax in enumerate(axes[row]):
                ax.imshow(image, cmap='gray', vmin=-1000, vmax=1000, origin='lower', aspect=aspect)
                for mask, color, show in ((gt, 'cyan', col != 1), (pred, 'magenta', col != 0)):
                    if show and mask.any():
                        ax.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([color]), alpha=.3, origin='lower', aspect=aspect)
                ax.set_title(f'Native axis{axis} slice{index}: ' + ('GT', 'prediction', 'GT + prediction')[col], fontsize=8)
                ax.axis('off')
        fig.suptitle(f'{sample["name"]}; GT-assisted physical ROI; threshold{threshold:g}; native side Dice{score:.4f}')
        fig.tight_layout()
        fig.savefig(output / f'{sample["name"]}.png', dpi=140)
        plt.close(fig)


def execute(args):
    from monai.losses import DiceCELoss
    prepared, config, previous = prepare(args)
    run, device = args.run_dir, torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    model = base.build_model().to(device)
    initial_hash = base.state_hash(model.state_dict())
    if initial_hash != previous['initial_state_sha256']:
        raise ValueError('Random initialization differs from the previous U-Net architecture/seed')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    def schedule(epoch):
        warmup = min(5, args.epochs)
        if epoch < warmup:
            return (epoch + 1) / warmup
        fraction = min(1., (epoch - warmup) / max(1, args.epochs - warmup))
        return .05 + .95 * .5 * (1 + math.cos(math.pi * fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    criterion = DiceCELoss(sigmoid=True, squared_pred=True)
    start, best, best_epoch, best_threshold, history, initial = 0, -1., 0, .5, [], None
    early = {'anchor': -1., 'last_improvement_epoch': 0}
    if args.resume:
        saved = torch.load(run / 'unet_latest.pth', map_location='cpu', weights_only=False)
        if saved['config'] != config or saved['initial_state_sha256'] != initial_hash:
            raise ValueError('Resume checkpoint/config/initialization differs')
        for obj, key in ((model, 'model'), (optimizer, 'optimizer'), (scheduler, 'scheduler'), (scaler, 'scaler')):
            obj.load_state_dict(saved[key + '_state_dict'])
        start, best, best_epoch, best_threshold, history, initial, early = [saved[key] for key in
            ('epoch', 'best_score', 'best_epoch', 'best_threshold', 'history', 'initial_metrics', 'early_state')]
        torch.set_rng_state(saved['torch_rng_state'])
        np.random.set_state(saved['numpy_rng_state'])
        random.setstate(saved['python_rng_state'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
        del saved
    def state(epoch):
        return {'epoch': epoch, 'best_score': best, 'best_epoch': best_epoch, 'best_threshold': best_threshold,
                'history': history, 'initial_metrics': initial, 'early_state': early, 'config': config,
                'initial_state_sha256': initial_hash, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(), 'torch_rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(), 'python_rng_state': random.getstate(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
    if not args.resume:
        threshold, train, val = calibration_validation(model, prepared, device, 'initial')
        best_threshold = threshold
        best = val[threshold]['metrics']['mean_bilateral_patient_dice']
        early = {'anchor': best, 'last_improvement_epoch': 0}
        initial = {'threshold_from_train': threshold, 'train': train[threshold]['metrics'], 'val': val[threshold]['metrics']}
        atomic_save(run / 'unet_best.pth', state(0))
        atomic_save(run / 'unet_latest.pth', state(0))
        data.save(run / 'preflight.json', {'passed': True, 'physical_policy': prepared['policy'], 'coverage': prepared['coverage'],
            'initial_metrics': initial, 'initial_state_sha256': initial_hash, 'train_patients': 54, 'validation_patients': 10,
            'test_images_read': False, 'exploratory_cohort': True, 'original_tiny_90_percent_gate_not_claimed_passed': True})
    stopped = args.patience_epochs > 0 and start - early['last_improvement_epoch'] >= args.patience_epochs
    for epoch in range(start, args.epochs) if not stopped else []:
        lr = optimizer.param_groups[0]['lr']
        metrics = train_epoch(model, prepared['samples']['train'], optimizer, scaler, criterion, device, epoch, args)
        scheduler.step()
        row = {'epoch': epoch + 1, 'lr': lr, **metrics, 'train_chosen_threshold': None,
               'frozen_train_dice': None, 'validation_dice': None, 'validation_dice_at_0_3': None, 'validation_dice_at_0_5': None}
        improved = False
        if epoch == 0 or (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            threshold, train, val = calibration_validation(model, prepared, device, f'epoch_{epoch + 1}')
            score = val[threshold]['metrics']['mean_bilateral_patient_dice']
            row.update(train_chosen_threshold=threshold, frozen_train_dice=train[threshold]['metrics']['mean_bilateral_patient_dice'],
                       validation_dice=score, validation_dice_at_0_3=val[.3]['metrics']['mean_bilateral_patient_dice'],
                       validation_dice_at_0_5=val[.5]['metrics']['mean_bilateral_patient_dice'])
            data.save(run / 'latest_train_threshold_curve.json', {str(t): result['metrics'] for t, result in train.items()})
            write_csv(run / 'validation_latest_sides.csv', val[threshold]['sides'])
            write_csv(run / 'validation_latest_patients.csv', val[threshold]['patients'])
            early, stopped = early_stop_update(early, score, epoch + 1, args.patience_epochs, args.min_delta)
            improved = score > best
            if improved:
                best, best_epoch, best_threshold = score, epoch + 1, threshold
        history.append(row)
        write_csv(run / 'history.csv', history)
        data.save(run / 'progress.json', {'complete': False, 'completed_epochs': epoch + 1, 'planned_epochs': args.epochs,
            'best_epoch': best_epoch, 'best_validation_dice': best, 'best_train_chosen_threshold': best_threshold,
            'stopped_early': stopped, 'latest': row})
        if improved:
            atomic_save(run / 'unet_best.pth', state(epoch + 1))
        atomic_save(run / 'unet_latest.pth', state(epoch + 1))
        print(json.dumps({'epoch_result': row, 'best_validation_dice': best}), flush=True)
        if stopped:
            print(f'Early stopping after epoch{epoch + 1}; no significant validation improvement for{args.patience_epochs}epochs.', flush=True)
            break
    saved = torch.load(run / 'unet_best.pth', map_location='cpu', weights_only=False)
    if saved['epoch'] != best_epoch or saved['best_score'] != best or saved['best_threshold'] != best_threshold or saved['config'] != config:
        raise ValueError('Best checkpoint metadata differs')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    del saved
    threshold, train, val = calibration_validation(model, prepared, device, 'selected')
    if threshold != best_threshold or abs(val[threshold]['metrics']['mean_bilateral_patient_dice'] - best) > 1e-4:
        raise ValueError('Selected checkpoint/threshold fails score reproduction')
    output, number = run / 'best_exports', 0
    while output.exists():
        number += 1
        output = run / f'best_exports_resume_{number}'
    output.mkdir()
    for split, result in (('validation', val[threshold]), ('frozen_training', train[threshold])):
        write_csv(output / f'{split}_sides.csv', result['sides'])
        write_csv(output / f'{split}_patients.csv', result['patients'])
    data.save(output / 'train_threshold_curve.json', {str(t): result['metrics'] for t, result in train.items()})
    export_representatives(model, prepared, device, threshold, val[threshold], output / 'representatives')
    means, training = val[threshold]['metrics'], train[threshold]['metrics']
    summary = {'complete': True, 'experiment': config['experiment'], 'completed_epochs': history[-1]['epoch'] if history else start,
        'planned_epochs': args.epochs, 'stopped_early': stopped, 'best_epoch': best_epoch,
        'train_patients': 54, 'validation_patients': 10, 'reserved_test_patients': 20,
        'threshold': threshold, 'threshold_selected_on': 'all54training patients only; before validation; checkpoint metadata frozen',
        'best_validation_metrics': means, 'best_frozen_training_metrics': training,
        'fixed_threshold_validation_controls': {str(t): val[t]['metrics'] for t in (.3, .5)},
        'train_validation_gap': training['mean_bilateral_patient_dice'] - means['mean_bilateral_patient_dice'],
        'previous_cohort_validation_dice': previous['best_validation_metrics']['mean_bilateral_patient_dice'],
        'validation_delta_vs_previous_cohort': means['mean_bilateral_patient_dice'] - previous['best_validation_metrics']['mean_bilateral_patient_dice'],
        'physical_policy': prepared['policy'], 'coverage': prepared['coverage'],
        'initial_state_sha256': initial_hash, 'best_checkpoint_sha256': file_hash(run / 'unet_best.pth'),
        'oracle_clicks': 0, 'validation_images_read': True, 'test_images_read': False,
        'protocol': config['protocol'], 'augmentation': args.augmentation,
        'interpretation': 'GT-assisted physical ROI validation. Grid/normalization/padding loss/augmentation/calibration changed together; pipeline comparison, not a causal single-factor ablation. Not automatic full-scan/test performance.',
        'artifacts': str(run), 'export_directory': str(output)}
    data.save(run / 'summary.json', summary)
    data.save(run / 'progress.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort-run-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--patience-epochs', type=int, default=50)
    parser.add_argument('--min-delta', type=float, default=.001)
    parser.add_argument('--augmentation', choices=('none', 'light'), default='light')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.run_dir, args.cohort_run_dir = args.run_dir.expanduser().resolve(), args.cohort_run_dir.expanduser().resolve()
    prior_config = data.read(args.cohort_run_dir / 'config.json')
    if any(args.run_dir.is_relative_to(folder) for folder in
           (args.cohort_run_dir, Path(prior_config['source_root']).resolve(), Path(prior_config['data_dir']).resolve())):
        parser.error('Run outputs must be outside the original control/source/prepared dataset')
    if (min(args.epochs, args.val_every, args.accumulation) < 1 or not np.isfinite(args.lr) or args.lr <= 0
            or args.patience_epochs < 0 or not np.isfinite(args.min_delta) or args.min_delta < 0):
        parser.error('Invalid optimizer/epoch/early-stop settings')
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    if args.resume:
        if not (args.run_dir / 'unet_latest.pth').exists() or (args.run_dir / 'summary.json').exists():
            parser.error('Resume requires an unfinished original physical run/latest checkpoint')
    elif args.run_dir.exists() or args.run_dir.is_relative_to(args.cohort_run_dir):
        parser.error('Use a new run directory outside the previous control')
    else:
        args.run_dir.mkdir(parents=True)
    try:
        execute(args)
    except Exception as error:
        data.save(args.run_dir / 'failure.json', {'error_type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
