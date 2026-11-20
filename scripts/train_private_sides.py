#!/usr/bin/env python3
"""Train side-specific SAM-Med3D on 54 patients, selecting by frozen ten-patient validation."""

import argparse
import csv
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch

from evaluate_private_validation import file_hash, scores, write_csv
from train_private_long import accumulation_window, atomic_save, early_stop_update, interactive_loss
from train_private_tiny import evaluate, prepare_samples, select_patients
from compare_side_validation import export_representatives


def baseline_info(directory, manifest_hash, checkpoint_hash, seed, sam_builder_hash):
    summary = json.loads((directory / 'summary.json').read_text())
    protocol = json.loads((directory / 'protocol.json').read_text())
    if not summary['complete'] or summary['validation_patients'] != 10 or summary['side_rois'] != 20 or summary['test_images_read']:
        raise ValueError('A complete ten-patient side-validation comparison is required')
    if protocol['manifest_sha256'] != manifest_hash or protocol['seed'] != seed:
        raise ValueError('Baseline comparison uses a different manifest or prompt seed')
    if protocol['shared_code_sha256']['evaluate_private_validation.py'] != file_hash(Path(__file__).with_name('evaluate_private_validation.py')) or protocol['sam_builder_sha256'] != sam_builder_hash:
        raise ValueError('Core preprocessing/inference or SAM builder differs from the completed side-validation comparison')
    turbo = next(row for row in summary['checkpoints'] if row['checkpoint'] == 'turbo_original')
    if turbo['sha256'] != checkpoint_hash:
        raise ValueError('Initialization must match the original Turbo used in the side-validation comparison')
    return {name: file_hash(directory / name) for name in
            ('summary.json', 'protocol.json', 'turbo_original/sides.csv', 'turbo_original/patients.csv')}


def reproduce_baseline(directory, sides, patients):
    maximum_error = 0.
    for name, actual, fields, expected_count in [('sides', sides, ('case', 'side', 'clicks'), 40),
                                                ('patients', patients, ('case', 'clicks'), 20)]:
        with (directory / 'turbo_original' / f'{name}.csv').open(newline='') as stream:
            saved = list(csv.DictReader(stream))
        key = lambda row: tuple(str(row[field]) for field in fields)
        lookup = {key(row): float(row['dice_full_target']) for row in saved}
        observed = {key(row): row['dice_full_target'] for row in actual}
        if len(saved) != expected_count or len(lookup) != expected_count or len(actual) != expected_count or set(lookup) != set(observed):
            raise ValueError('Baseline cases/click budgets differ from current validation')
        for alias, value in observed.items():
            error = abs(value - lookup[alias])
            maximum_error = max(maximum_error, error)
            if error > 1e-4:
                raise ValueError('Original Turbo fails to reproduce saved side-validation Dice; training has not started')
    return maximum_error


def train_epoch(model, samples, optimizer, scaler, criterion, device, epoch, seed, accumulation):
    model.train()
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    optimizer.zero_grad(set_to_none=True)
    losses, dices, updates, skips = [], [], 0, 0
    for step, index in enumerate(order):
        sample = samples[index]
        image, target = sample['image'], sample['target']
        axes = [axis for axis in (2, 3, 4) if torch.rand(()).item() < .5]
        if axes:
            image, target = torch.flip(image, axes), torch.flip(target, axes)
        image, target = image.to(device), target.to(device=device, dtype=torch.long)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            loss, logits = interactive_loss(model, image, target, criterion, 5, seed + epoch * len(samples) + step)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite side-training loss')
        size, should_step = accumulation_window(step, len(samples), accumulation)
        scaler.scale(loss / size).backward()
        if should_step:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm) and not scaler.is_enabled():
                raise ValueError('Non-finite side-training gradients')
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            success = scaler.get_scale() >= scale
            updates += int(success)
            skips += int(not success)
            optimizer.zero_grad(set_to_none=True)
        gt = target[0, 0].bool().cpu().numpy()
        losses.append(loss.item())
        dices.append(scores((logits[0, 0] > 0).cpu().numpy(), gt, int(gt.sum()))['dice_crop'])
        if should_step:
            print(f'epoch={epoch + 1} side_roi={step + 1}/{len(samples)} loss={loss.item():.6f} optimizer_updates={updates} amp_skips={skips}', flush=True)
    return {'mean_interaction_loss': float(np.mean(losses)), 'train_dice_crop_augmented': float(np.mean(dices)),
            'optimizer_updates': updates, 'amp_skipped_updates': skips, 'expected_updates': math.ceil(len(samples) / accumulation)}


def execute(args):
    run, data = args.run_dir, args.data_dir
    manifest_file = data / 'manifest.json'
    manifest = json.loads(manifest_file.read_text())
    select_patients(manifest, args.seed)
    baseline_hashes = baseline_info(args.baseline_dir, file_hash(manifest_file), file_hash(args.checkpoint), args.seed,
                                    file_hash(args.sam_dir / 'segment_anything/build_sam3D.py'))
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != 'resume'}
    config.update(manifest_sha256=file_hash(manifest_file), initial_checkpoint_sha256=file_hash(args.checkpoint),
                  script_sha256=file_hash(Path(__file__)),
                  shared_code_sha256={name: file_hash(Path(__file__).with_name(name)) for name in
                    ('train_private_tiny.py', 'evaluate_private_validation.py', 'train_private_long.py', 'compare_side_validation.py', 'audit_expert_agreement.py')},
                  sam_builder_sha256=file_hash(args.sam_dir / 'segment_anything/build_sam3D.py'),
                  sam_source_sha256={str(path.relative_to(args.sam_dir)): file_hash(path) for path in sorted((args.sam_dir / 'segment_anything').rglob('*.py'))},
                  baseline_sha256=baseline_hashes,
                  versions={name: importlib.metadata.version(name) for name in ('torch', 'torchio', 'monai', 'numpy', 'SimpleITK')},
                  train_patients=54, train_side_rois=108, validation_patients=10, validation_side_rois=20,
                  model='vit_b_ori; 128 input cube', optimizer='AdamW; all parameters; weight_decay=0.1',
                  augmentation='independent XYZ flips; shuffled side ROIs; epoch-varying adaptive oracle prompt seeds',
                  loss='MONAI DiceCELoss sigmoid/squared prediction, averaged across five detached-feedback interactions',
                  selection='maximum mean validation full-target bilateral patient Dice at five clicks per side',
                  protocol='native 128 side-GT-centered ROIs, no resampling, full target retention, native prompt order; 1/5 clicks PER SIDE equals 2/10 per patient',
                  test_images_read=False)
    if args.resume:
        if json.loads((run / 'config.json').read_text()) != config:
            raise ValueError('Resume code/data/config differs from the frozen run; use a new run')
    else:
        (run / 'config.json').write_text(json.dumps(config, indent=2))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    preparation = run / 'preparation'
    if args.resume:
        attempt = 1
        while (run / f'preparation_resume_{attempt}').exists():
            attempt += 1
        preparation = run / f'preparation_resume_{attempt}'
    preparation.mkdir()
    samples_by_split, patients_by_split, provenance = {}, {}, {}
    for split in ('train', 'val'):
        directory = preparation / split
        directory.mkdir()
        samples, patients = prepare_samples(args.source_root, data, manifest, directory, args.seed,
                                            split=split, all_training_patients=split == 'train')
        expected = 54 if split == 'train' else 10
        if len(samples) != 2 * expected or len(patients) != expected:
            raise ValueError(f'{split}: missing patients or side targets')
        for sample in samples:
            sample['target'] = sample['target'].to(torch.uint8)
        samples_by_split[split], patients_by_split[split] = samples, patients
        name = 'selected_training_patients.json' if split == 'train' else 'selected_validation_patients.json'
        provenance[split] = json.loads((directory / name).read_text())
    if args.resume:
        if json.loads((run / 'source_provenance.json').read_text()) != provenance:
            raise ValueError('Source scans/final masks changed since the frozen run')
    else:
        (run / 'source_provenance.json').write_text(json.dumps(provenance, indent=2))
    sys.path.insert(0, str(args.sam_dir))
    from segment_anything.build_sam3D import sam_model_registry3D
    from monai.losses import DiceCELoss
    model = sam_model_registry3D['vit_b_ori'](checkpoint=None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.1)
    warmup = min(5, args.epochs)
    def schedule(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        fraction = min(1., (epoch - warmup) / max(1, args.epochs - warmup))
        return .05 + .95 * .5 * (1 + math.cos(math.pi * fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    criterion = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    source = torch.load(run / 'sam_model_side_latest.pth' if args.resume else args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(source.get('model_state_dict', source), strict=True)
    history, start, best_epoch = [], 0, 0
    best = -1.
    early = {'anchor': -1., 'last_improvement_epoch': 0}
    if args.resume:
        optimizer.load_state_dict(source['optimizer_state_dict'])
        scheduler.load_state_dict(source['lr_scheduler_state_dict'])
        scaler.load_state_dict(source['scaler_state_dict'])
        history, start, best, best_epoch = source['history'], source['epoch'], source['best_validation_dice'], source['best_epoch']
        early = source['early_stop_state']
        torch.set_rng_state(source['torch_rng_state'])
        np.random.set_state(source['numpy_rng_state'])
        random.setstate(source['python_rng_state'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(source['cuda_rng_state'])
    del source
    def state(epoch):
        return {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(),
                'history': history, 'best_validation_dice': best, 'best_epoch': best_epoch, 'early_stop_state': early, 'config': config,
                'torch_rng_state': torch.get_rng_state(), 'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate(), 'cuda_rng_state': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
    validation, val_patients = samples_by_split['val'], patients_by_split['val']
    if not args.resume:
        means, side_rows, patient_rows = evaluate(model, validation, val_patients, device, progress_label='initial_validation')
        error = reproduce_baseline(args.baseline_dir, side_rows, patient_rows)
        best = means['5']['mean_bilateral_patient_dice']
        early = {'anchor': best, 'last_improvement_epoch': 0}
        (run / 'preflight.json').write_text(json.dumps({'passed': True, 'source_patients_verified': 64,
            'train_side_rois': 108, 'validation_side_rois': 20, 'maximum_baseline_dice_error': error,
            'initial_validation_dice': means, 'test_images_read': False}, indent=2))
        write_csv(run / 'validation_initial_sides.csv', side_rows)
        write_csv(run / 'validation_initial_patients.csv', patient_rows)
        atomic_save(run / 'sam_model_side_val_best.pth', state(0))
        atomic_save(run / 'sam_model_side_latest.pth', state(0))
        print(json.dumps({'preflight_passed': True, 'initial_validation': means}), flush=True)
    stopped = args.resume and args.patience_epochs > 0 and start - early['last_improvement_epoch'] >= args.patience_epochs
    for epoch in range(start, args.epochs) if not stopped else []:
        lr = optimizer.param_groups[0]['lr']
        metrics = train_epoch(model, samples_by_split['train'], optimizer, scaler, criterion, device, epoch, args.seed, args.accumulation)
        scheduler.step()
        row = {'epoch': epoch + 1, 'lr': lr, **metrics, 'val_bilateral_dice_1_per_side': None,
               'val_bilateral_dice_5_per_side': None, 'val_mean_side_dice_5': None, 'val_minimum_patient_dice_5': None}
        improved = False
        if epoch == 0 or (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            means, side_rows, patient_rows = evaluate(model, validation, val_patients, device, progress_label=f'validation_epoch_{epoch + 1}')
            score = means['5']['mean_bilateral_patient_dice']
            row.update(val_bilateral_dice_1_per_side=means['1']['mean_bilateral_patient_dice'], val_bilateral_dice_5_per_side=score,
                       val_mean_side_dice_5=means['5']['mean_side_dice'], val_minimum_patient_dice_5=means['5']['minimum_bilateral_patient_dice'])
            write_csv(run / 'validation_latest_sides.csv', side_rows)
            write_csv(run / 'validation_latest_patients.csv', patient_rows)
            improved = score > best
            early, stopped = early_stop_update(early, score, epoch + 1, args.patience_epochs, args.min_delta)
            if improved:
                best, best_epoch = score, epoch + 1
        history.append(row)
        write_csv(run / 'history.csv', history)
        (run / 'progress.json').write_text(json.dumps({'completed_epochs': epoch + 1, 'planned_epochs': args.epochs,
            'best_epoch': best_epoch, 'best_validation_bilateral_dice_5_per_side': best,
            'stopped_early': stopped, 'latest': row, 'early_stop_state': early}, indent=2))
        if improved:
            atomic_save(run / 'sam_model_side_val_best.pth', state(epoch + 1))
        if improved or stopped or (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            atomic_save(run / 'sam_model_side_latest.pth', state(epoch + 1))
        print(json.dumps({'epoch_result': row, 'best_validation_bilateral_dice_5_per_side': best}), flush=True)
        if stopped:
            print(f'Early stopping after epoch {epoch + 1}: no validation gain >{args.min_delta} for {args.patience_epochs} epochs', flush=True)
            break
    chosen = torch.load(run / 'sam_model_side_val_best.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(chosen['model_state_dict'], strict=True)
    selected_epoch, expected_best = chosen['epoch'], chosen['best_validation_dice']
    if selected_epoch != best_epoch or not np.isclose(expected_best, best, atol=1e-8, rtol=0):
        raise ValueError('Best/latest checkpoint metadata are inconsistent; inspect an interrupted checkpoint save')
    del chosen
    means, side_rows, patient_rows = evaluate(model, validation, val_patients, device, progress_label='selected_validation')
    if not np.isclose(means['5']['mean_bilateral_patient_dice'], expected_best, atol=1e-4, rtol=0):
        raise ValueError('Reloaded validation-best checkpoint does not reproduce its selection score')
    export = run / 'best_validation'
    if export.exists():
        attempt = 1
        while (run / f'best_validation_resume_{attempt}').exists():
            attempt += 1
        export = run / f'best_validation_resume_{attempt}'
    export.mkdir()
    write_csv(export / 'sides.csv', side_rows)
    write_csv(export / 'patients.csv', patient_rows)
    export_representatives(model, validation, val_patients, device, export, side_rows, patient_rows)
    training_means, training_sides, training_patients = evaluate(model, samples_by_split['train'], patients_by_split['train'], device,
                                                              progress_label='selected_frozen_training')
    write_csv(export / 'frozen_training_sides.csv', training_sides)
    write_csv(export / 'frozen_training_patients.csv', training_patients)
    completed = history[-1]['epoch'] if history else start
    baseline_summary = json.loads((args.baseline_dir / 'summary.json').read_text())
    previous_v2 = next((row['mean_bilateral_patient_dice'] for row in baseline_summary.get('comparison', [])
                        if row['checkpoint'] == 'v2_best' and row['clicks_per_side'] == 5), None)
    change_vs_v2 = means['5']['mean_bilateral_patient_dice'] - previous_v2 if previous_v2 is not None else None
    summary = {'complete': True, 'completed_epochs': completed, 'planned_epochs': args.epochs, 'stopped_early': stopped,
               'best_epoch': selected_epoch, 'train_patients': 54, 'train_side_rois': 108, 'validation_patients': 10, 'validation_side_rois': 20,
               'best_validation_dice': means, 'best_checkpoint_frozen_training_dice': training_means,
               'train_validation_gap_5_per_side': training_means['5']['mean_bilateral_patient_dice'] - means['5']['mean_bilateral_patient_dice'],
               'previous_v2_validation_bilateral_dice_5_per_side': previous_v2,
               'validation_dice_change_vs_previous_v2': change_vs_v2,
               'best_checkpoint_sha256': file_hash(run / 'sam_model_side_val_best.pth'),
               'protocol': config['protocol'], 'test_images_read': False, 'artifacts': str(run),
               'interpretation': 'GT-assisted interactive validation; not automatic test/deployment performance', 'export_directory': str(export)}
    (run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sam-dir', 'source-root', 'data-dir', 'checkpoint', 'baseline-dir', 'run-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--patience-epochs', type=int, default=50)
    parser.add_argument('--min-delta', type=float, default=.001)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if min(args.epochs, args.val_every, args.accumulation) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('Training counts and LR must be positive')
    if args.patience_epochs < 0 or not np.isfinite(args.min_delta) or args.min_delta < 0:
        parser.error('Invalid early-stopping settings')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    if args.resume:
        if not (args.run_dir / 'sam_model_side_latest.pth').is_file():
            parser.error('Resume requires an existing run and its latest side checkpoint')
    elif args.run_dir.exists():
        parser.error('Run directory exists; use a new run or --resume')
    else:
        args.run_dir.mkdir(parents=True)
    try:
        execute(args)
    except Exception as error:
        (args.run_dir / 'failure.json').write_text(json.dumps({'error_type': type(error).__name__, 'error': str(error)}, indent=2))
        raise


if __name__ == '__main__':
    main()
