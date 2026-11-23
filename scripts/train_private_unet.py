#!/usr/bin/env python3
"""Train-only learnability check or gated cohort baseline with MONAI 3D U-Net."""

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from evaluate_private_validation import file_hash, save_crop, scores, write_csv
from roi_zoom import overlay_native, restore_logits, zoom_samples
from train_private_long import accumulation_window, atomic_save, early_stop_update
from train_private_tiny import native_prediction_indices, prepare_samples, select_patients, sparse_scores


ARCHITECTURE = {'spatial_dims': 3, 'in_channels': 1, 'out_channels': 1,
                'channels': [16, 32, 64, 128, 256], 'strides': [2, 2, 2, 2],
                'num_res_units': 2, 'norm': 'INSTANCE', 'dropout': 0.0}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_model():
    from monai.networks.nets import UNet
    return UNet(**ARCHITECTURE)


def v4_control(args, manifest_path):
    summary = json.loads((args.v4_run_dir / 'summary.json').read_text())
    config = json.loads((args.v4_run_dir / 'config.json').read_text())
    policy = json.loads((args.v4_run_dir / 'roi_policy.json').read_text())
    if (not summary['complete'] or summary['test_images_read'] or summary['train_patients'] != 54
            or summary['validation_patients'] != 10 or config['experiment'] != 'v4_anisotropic_roi_zoom'):
        raise ValueError('A completed 54/10-patient v4 anisotropic run is required')
    if config['manifest_sha256'] != file_hash(manifest_path) or config['seed'] != args.seed:
        raise ValueError('Manifest/seed differs from frozen v4')
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'torchio', 'monai', 'numpy', 'SimpleITK')}
    if config['versions'] != versions:
        raise ValueError('Use the same package versions as v4')
    for name in ('train_private_tiny.py', 'evaluate_private_validation.py'):
        if config['shared_code_sha256'][name] != file_hash(Path(__file__).with_name(name)):
            raise ValueError('Source extraction or native inference/scoring differs from v4')
    if config['zoom_module_sha256'] != file_hash(Path(__file__).with_name('roi_zoom.py')):
        raise ValueError('ROI zoom implementation differs from v4')
    shape = policy['selected_native_size_xyz']
    if (policy['training_patients'] != 54 or policy['training_sides'] != 108 or policy['margin_voxels'] != 4
            or shape != summary['native_roi_size_xyz']):
        raise ValueError('Unsupported or inconsistent frozen v4 ROI policy')
    return summary, policy, versions


def checked_logits(model, image):
    logits = model(image)
    if logits.shape != image.shape or not torch.isfinite(logits).all():
        raise ValueError('U-Net must return finite one-channel logits matching the model input')
    return logits


@torch.inference_mode()
def evaluate(model, samples, patients, device, export=None, progress_label=None):
    model.eval()
    sides, bilateral, restored = [], [], {}
    if export is not None:
        export.mkdir(parents=True)
    for index, sample in enumerate(samples, 1):
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits = checked_logits(model, sample['image'].to(device))
        native_logits = restore_logits(logits[0, 0].float().cpu().numpy(), sample['native_target'].shape)
        prediction = native_logits > 0
        metadata = sample['metadata']
        sides.append({'case': sample['case'], 'side': sample['side'], 'oracle_clicks': 0,
                      'target_retention': metadata['target_retention'],
                      **scores(prediction, sample['native_target'], metadata['full_target_voxels'], sample['valid'])})
        restored.setdefault(sample['case'], []).append(native_prediction_indices(
            prediction, sample['valid'], metadata, patients[sample['case']]['shape']))
        if export is not None:
            name = sample['name']
            save_crop(export / f'{name}_prediction_crop.nii.gz', prediction.astype(np.uint8), metadata)
            save_crop(export / f'{name}_target_crop.nii.gz', sample['native_target'].astype(np.uint8), metadata)
            overlay_native(export / f'{name}_overlay.png', sample, prediction)
        if progress_label:
            print(f'{progress_label}: {sample["name"]} complete ({index}/{len(samples)})', flush=True)
    for alias, parts in sorted(restored.items()):
        bilateral.append({'case': alias, 'oracle_clicks': 0,
                          **sparse_scores(np.concatenate(parts), patients[alias]['target_indices'])})
    if (len(restored) != len(patients) or len(sides) != 2 * len(patients)
            or any(sorted(sample['side'] for sample in samples if sample['case'] == alias) != ['LFSDP', 'RFSDP']
                   for alias in patients)):
        raise ValueError('Evaluation requires both sides for every patient')
    means = {'mean_side_dice': float(np.mean([row['dice_full_target'] for row in sides])),
             'minimum_side_dice': min(row['dice_full_target'] for row in sides),
             'mean_bilateral_patient_dice': float(np.mean([row['dice_full_target'] for row in bilateral])),
             'minimum_bilateral_patient_dice': min(row['dice_full_target'] for row in bilateral),
             'mean_precision': float(np.mean([row['precision'] for row in bilateral])),
             'mean_recall': float(np.mean([row['recall_full_target'] for row in bilateral]))}
    if export is not None:
        write_csv(export / 'sides.csv', sides)
        write_csv(export / 'patients.csv', bilateral)
    return means, sides, bilateral


def selection_score(means, stage):
    return min(means['minimum_side_dice'], means['minimum_bilateral_patient_dice']) if stage == 'tiny' else means['mean_bilateral_patient_dice']


def train_epoch(model, samples, optimizer, scaler, criterion, device, epoch, seed, accumulation, augment):
    model.train(augment)
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    optimizer.zero_grad(set_to_none=True)
    losses, moving_dice, updates, skips = [], [], 0, 0
    for step, index in enumerate(order):
        image, target = samples[index]['image'], samples[index]['target']
        axes = [axis for axis in (2, 3, 4) if torch.rand(()).item() < .5] if augment else []
        if axes:
            image, target = torch.flip(image, axes), torch.flip(target, axes)
        image, target = image.to(device), target.to(device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits = checked_logits(model, image)
            loss = criterion(logits.float(), target)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite U-Net training loss')
        window, should_step = accumulation_window(step, len(samples), accumulation)
        scaler.scale(loss / window).backward()
        if should_step:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm) and not scaler.is_enabled():
                raise ValueError('Non-finite U-Net gradients')
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            success = scaler.get_scale() >= old_scale
            updates += int(success)
            skips += int(not success)
            optimizer.zero_grad(set_to_none=True)
            print(f'epoch={epoch + 1} roi={step + 1}/{len(samples)} loss={loss.item():.6f} updates={updates} skips={skips}', flush=True)
        losses.append(loss.item())
        gt = target[0, 0].bool().cpu().numpy()
        moving_dice.append(scores((logits.detach()[0, 0] > 0).cpu().numpy(), gt, int(gt.sum()))['dice_crop'])
    return {'loss': float(np.mean(losses)), 'moving_model_grid_train_dice': float(np.mean(moving_dice)),
            'optimizer_updates': updates, 'amp_skipped_updates': skips,
            'expected_updates': math.ceil(len(samples) / accumulation)}


def check_gate(directory, config):
    if directory is None:
        raise ValueError('Cohort training requires --tiny-run-dir with a passed U-Net diagnostic')
    summary = json.loads((directory / 'summary.json').read_text())
    old = json.loads((directory / 'config.json').read_text())
    if (not summary['complete'] or summary['stage'] != 'tiny' or not summary['diagnostic_passed']
            or summary['validation_images_read'] or summary['test_images_read'] or summary['train_patients'] != 2):
        raise ValueError('Tiny U-Net diagnostic has not passed or used the wrong cohort')
    for name in ('manifest_sha256', 'v4_control_sha256', 'architecture', 'versions', 'monai_source_sha256',
                 'script_sha256', 'shared_code_sha256', 'seed', 'native_roi_size_xyz'):
        if old[name] != config[name]:
            raise ValueError(f'Tiny diagnostic differs from cohort source/config: {name}')
    path = directory / 'unet_best.pth'
    if file_hash(path) != summary['best_checkpoint_sha256']:
        raise ValueError('Tiny checkpoint checksum differs from completed summary')
    export = Path(summary['export_directory']).resolve()
    if not export.is_relative_to(directory.resolve()):
        raise ValueError('Tiny exports must belong to their run')
    minimum = 1.
    for filename, count, fields in [('sides.csv', 4, ('case', 'side')), ('patients.csv', 2, ('case',))]:
        with (export / filename).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != count or len({tuple(row[key] for key in fields) for row in rows}) != count:
            raise ValueError('Tiny diagnostic exports are incomplete')
        values = [float(row['dice_full_target']) for row in rows]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError('Invalid tiny diagnostic Dice')
        minimum = min(minimum, *values)
    if minimum < .90 or not np.isclose(minimum, summary['best_selection_score'], atol=1e-6, rtol=0):
        raise ValueError('Tiny diagnostic CSVs do not establish minimum Dice0.90')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if (checkpoint['epoch'] != summary['best_epoch'] or checkpoint['config'] != old
            or not np.isclose(checkpoint['best_score'], minimum, atol=1e-6, rtol=0)):
        raise ValueError('Tiny checkpoint metadata differs from its completed report')
    return {**{name: file_hash(directory / name) for name in ('summary.json', 'config.json', 'source_provenance.json', 'unet_best.pth')},
            'sides_csv': file_hash(export / 'sides.csv'), 'patients_csv': file_hash(export / 'patients.csv')}


def execute(args):
    from monai.losses import DiceCELoss
    from monai.networks.nets import UNet
    run = args.run_dir
    manifest_path = args.data_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    select_patients(manifest, args.seed)
    v4, policy, versions = v4_control(args, manifest_path)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != 'resume'}
    config.update(manifest_sha256=file_hash(manifest_path), architecture=ARCHITECTURE, versions=versions,
                  script_sha256=file_hash(Path(__file__)),
                  shared_code_sha256={name: file_hash(Path(__file__).with_name(name)) for name in
                    ('train_private_tiny.py', 'evaluate_private_validation.py', 'roi_zoom.py', 'train_private_long.py')},
                  monai_source_sha256=file_hash(Path(inspect.getfile(UNet))),
                  v4_control_sha256={name: file_hash(args.v4_run_dir / name) for name in
                    ('summary.json', 'config.json', 'roi_policy.json', 'source_provenance.json')},
                  native_roi_size_xyz=policy['selected_native_size_xyz'],
                  initialization='random; from scratch; fixed seed; never tiny or SAM checkpoint weights',
                  optimizer='AdamW; all U-Net parameters; weight_decay=0' if args.stage == 'tiny' else 'AdamW; all U-Net parameters; weight_decay=1e-4',
                  effective_accumulation=1 if args.stage == 'tiny' else args.accumulation,
                  augmentation='none' if args.stage == 'tiny' else 'independent XYZ flips and seeded ROI shuffle',
                  schedule='constant LR' if args.stage == 'tiny' else 'five-epoch warmup then cosine to0.05 of LR',
                  model_mode='eval with gradients during optimization' if args.stage == 'tiny' else 'train during optimization; eval for scoring',
                  loss='DiceCELoss sigmoid=True squared_pred=True; one forward pass; no oracle clicks',
                  protocol='Same v4 GT-side-centered fixed ROI, normalization and128³ input; inverse-resize logits then threshold0.5; original-grid full-target bilateral Dice',
                  selection='minimum of all side/patient native Dice' if args.stage == 'tiny' else 'mean native-grid bilateral validation Dice',
                  test_images_read=False)
    config['tiny_gate_sha256'] = check_gate(args.tiny_run_dir, config) if args.stage == 'cohort' else None
    if args.resume:
        if json.loads((run / 'config.json').read_text()) != config:
            raise ValueError('Resume code/data/config differs; use an identical run or start a new one')
    else:
        save_json(run / 'config.json', config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    attempt = 0
    preparation = run / 'preparation'
    while preparation.exists():
        attempt += 1
        preparation = run / f'preparation_resume_{attempt}'
    preparation.mkdir()
    splits = ('train',) if args.stage == 'tiny' else ('train', 'val')
    samples_by_split, patients_by_split, provenance, coverage = {}, {}, {}, {}
    v4_sources = json.loads((args.v4_run_dir / 'source_provenance.json').read_text())
    for split in splits:
        folder = preparation / split
        folder.mkdir()
        samples, patients = prepare_samples(args.source_root, args.data_dir, manifest, folder, args.seed,
                                            split=split, all_training_patients=args.stage == 'cohort' and split == 'train')
        expected = (2 if args.stage == 'tiny' else 54) if split == 'train' else 10
        if len(patients) != expected or len(samples) != 2 * expected:
            raise ValueError('Unexpected U-Net patient/side coverage')
        name = 'selected_training_patients.json' if split == 'train' else 'selected_validation_patients.json'
        provenance[split] = json.loads((folder / name).read_text())
        reference = {row['patient_id']: row for row in v4_sources[split]}
        for row in provenance[split]:
            if any(row[key] != reference[row['patient_id']][key] for key in
                   ('source_image', 'source_label', 'source_image_sha256', 'source_label_sha256')):
                raise ValueError('U-Net source CT/final label differs from completed v4')
        for sample in samples:
            sample['target'] = sample['target'].to(torch.uint8)
        samples = zoom_samples(samples, policy['selected_native_size_xyz'])
        samples_by_split[split], patients_by_split[split] = samples, patients
        rows = [{'case': item['case'], 'side': item['side'], 'target_retention': item['metadata']['target_retention'],
                 'full_target_voxels': item['metadata']['full_target_voxels'],
                 'gt_label_roundtrip_dice': item['gt_label_roundtrip_dice']} for item in samples]
        write_csv(folder / 'coverage.csv', rows)
        if split == 'train' and any(row['target_retention'] != 1 for row in rows):
            raise ValueError('Frozen v4 ROI unexpectedly truncates a training target')
        coverage[split] = {'patients': len(patients), 'side_rois': len(samples),
                           'minimum_target_retention': min(row['target_retention'] for row in rows)}
    if args.resume and json.loads((run / 'source_provenance.json').read_text()) != provenance:
        raise ValueError('Source images/final labels changed during this run')
    save_json(run / 'source_provenance.json', provenance)
    if args.stage == 'cohort':
        gate_sources = json.loads((args.tiny_run_dir / 'source_provenance.json').read_text())['train']
        actual = {row['patient_id']: row for row in provenance['train']}
        expected_ids = select_patients(manifest, args.seed)
        if sorted(row['patient_id'] for row in gate_sources) != expected_ids:
            raise ValueError('Gate diagnostic did not use the frozen two training patients')
        for row in gate_sources:
            if any(row[key] != actual[row['patient_id']][key] for key in
                   ('source_image', 'source_label', 'source_image_sha256', 'source_label_sha256')):
                raise ValueError('Gate diagnostic source changed before cohort training')
    # Reset before initialization so tiny/cohort start from the same seeded random model.
    torch.manual_seed(args.seed)
    model = build_model().to(device)
    initial_hash = state_hash(model.state_dict())
    if args.stage == 'cohort':
        gate_summary = json.loads((args.tiny_run_dir / 'summary.json').read_text())
        if gate_summary['initial_state_sha256'] != initial_hash:
            raise ValueError('Cohort random initialization differs from the tiny diagnostic')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0. if args.stage == 'tiny' else 1e-4)
    def schedule(epoch):
        if args.stage == 'tiny':
            return 1.
        warmup = min(5, args.epochs)
        if epoch < warmup:
            return (epoch + 1) / warmup
        fraction = min(1., (epoch - warmup) / max(1, args.epochs - warmup))
        return .05 + .95 * .5 * (1 + math.cos(math.pi * fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    criterion = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    start, best_epoch, best, history, early = 0, 0, -1., [], {'anchor': -1., 'last_improvement_epoch': 0}
    initial = None
    if args.resume:
        source = torch.load(run / 'unet_latest.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(source['model_state_dict'], strict=True)
        optimizer.load_state_dict(source['optimizer_state_dict'])
        scheduler.load_state_dict(source['scheduler_state_dict'])
        scaler.load_state_dict(source['scaler_state_dict'])
        start, best_epoch, best, history, early = [source[key] for key in ('epoch', 'best_epoch', 'best_score', 'history', 'early_state')]
        initial = source['initial_metrics']
        if source['initial_state_sha256'] != initial_hash:
            raise ValueError('Random initialization differs from the frozen run')
        torch.set_rng_state(source['torch_rng_state'])
        np.random.set_state(source['numpy_rng_state'])
        random.setstate(source['python_rng_state'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(source['cuda_rng_state'])
        del source
    def state(epoch):
        return {'epoch': epoch, 'best_epoch': best_epoch, 'best_score': best, 'history': history,
                'initial_metrics': initial, 'initial_state_sha256': initial_hash, 'early_state': early, 'config': config,
                'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(),
                'torch_rng_state': torch.get_rng_state(), 'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate(), 'cuda_rng_state': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
    selected_split = 'train' if args.stage == 'tiny' else 'val'
    selected_samples, selected_patients = samples_by_split[selected_split], patients_by_split[selected_split]
    if not args.resume:
        initial, _, _ = evaluate(model, selected_samples, selected_patients, device, progress_label='initial_' + selected_split)
        best = selection_score(initial, args.stage)
        early = {'anchor': best, 'last_improvement_epoch': 0}
        atomic_save(run / 'unet_best.pth', state(0))
        atomic_save(run / 'unet_latest.pth', state(0))
        save_json(run / 'preflight.json', {'passed': True, 'stage': args.stage, 'coverage': coverage,
            'initial_metrics': initial, 'initial_state_sha256': initial_hash, 'oracle_clicks': 0,
            'validation_images_read': args.stage == 'cohort', 'test_images_read': False})
    stopped = args.stage == 'cohort' and args.patience_epochs > 0 and start - early['last_improvement_epoch'] >= args.patience_epochs
    passed = args.stage == 'tiny' and best >= .90
    for epoch in range(start, args.epochs) if not stopped and not passed else []:
        lr = optimizer.param_groups[0]['lr']
        metrics = train_epoch(model, samples_by_split['train'], optimizer, scaler, criterion, device, epoch,
                              args.seed, 1 if args.stage == 'tiny' else args.accumulation, args.stage == 'cohort')
        scheduler.step()
        row = {'epoch': epoch + 1, 'lr': lr, **metrics, 'selection_score': None}
        measured = epoch == 0 or (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs
        improved = False
        if measured:
            means, _, _ = evaluate(model, selected_samples, selected_patients, device, progress_label=f'{selected_split}_epoch_{epoch + 1}')
            score = selection_score(means, args.stage)
            row.update(selection_score=score, mean_bilateral_patient_dice=means['mean_bilateral_patient_dice'],
                       minimum_side_dice=means['minimum_side_dice'])
            if args.stage == 'cohort':
                early, stopped = early_stop_update(early, score, epoch + 1, args.patience_epochs, args.min_delta)
            improved = score > best
            if improved:
                best, best_epoch = score, epoch + 1
            passed = args.stage == 'tiny' and best >= .90
        history.append(row)
        write_csv(run / 'history.csv', history)
        save_json(run / 'progress.json', {'stage': args.stage, 'completed_epochs': epoch + 1, 'planned_epochs': args.epochs,
            'best_epoch': best_epoch, 'best_selection_score': best, 'diagnostic_passed': passed if args.stage == 'tiny' else None,
            'stopped_early': stopped, 'latest': row})
        if improved:
            atomic_save(run / 'unet_best.pth', state(epoch + 1))
        if measured or improved or stopped or passed:
            atomic_save(run / 'unet_latest.pth', state(epoch + 1))
        print(json.dumps({'epoch_result': row, 'best_score': best}), flush=True)
        if stopped or passed:
            break
    chosen = torch.load(run / 'unet_best.pth', map_location='cpu', weights_only=False)
    if chosen['epoch'] != best_epoch or not np.isclose(chosen['best_score'], best, atol=1e-8, rtol=0):
        raise ValueError('Best/latest U-Net checkpoint metadata are inconsistent')
    model.load_state_dict(chosen['model_state_dict'], strict=True)
    del chosen
    means, sides, bilateral = evaluate(model, selected_samples, selected_patients, device, progress_label='selected_' + selected_split)
    if not np.isclose(selection_score(means, args.stage), best, atol=1e-4, rtol=0):
        raise ValueError('Reloaded U-Net best checkpoint does not reproduce its selection score')
    export = run / 'best_exports'
    number = 0
    while export.exists():
        number += 1
        export = run / f'best_exports_resume_{number}'
    export.mkdir()
    write_csv(export / 'sides.csv', sides)
    write_csv(export / 'patients.csv', bilateral)
    ranked = sorted(bilateral, key=lambda row: (row['dice_full_target'], row['case']))
    aliases = {ranked[0]['case'], ranked[len(ranked) // 2]['case'], ranked[-1]['case']}
    subset = [sample for sample in selected_samples if sample['case'] in aliases]
    _, exported, _ = evaluate(model, subset, {alias: selected_patients[alias] for alias in aliases}, device,
                              export=export / 'representatives')
    lookup = {(row['case'], row['side']): row['dice_full_target'] for row in sides}
    if any(abs(row['dice_full_target'] - lookup[(row['case'], row['side'])]) > 1e-4 for row in exported):
        raise ValueError('Exported predictions do not reproduce selected scores')
    training = means
    if args.stage == 'cohort':
        training, train_sides, train_patients = evaluate(model, samples_by_split['train'], patients_by_split['train'], device,
                                                       progress_label='selected_frozen_training')
        write_csv(export / 'frozen_training_sides.csv', train_sides)
        write_csv(export / 'frozen_training_patients.csv', train_patients)
    completed = history[-1]['epoch'] if history else start
    summary = {'complete': True, 'stage': args.stage, 'completed_epochs': completed, 'planned_epochs': args.epochs,
               'stopped_early': stopped, 'best_epoch': best_epoch, 'best_selection_score': best,
               'diagnostic_passed': best >= .90 if args.stage == 'tiny' else None,
               'train_patients': len(patients_by_split['train']), 'validation_patients': len(patients_by_split.get('val', {})),
               'initial_metrics': initial, 'best_selected_metrics': means, 'best_frozen_training_metrics': training,
               'train_validation_gap': training['mean_bilateral_patient_dice'] - means['mean_bilateral_patient_dice'] if args.stage == 'cohort' else None,
               'v4_validation_dice_5_per_side': v4['best_validation_dice']['5']['mean_bilateral_patient_dice'],
               'validation_dice_change_vs_v4': means['mean_bilateral_patient_dice'] - v4['best_validation_dice']['5']['mean_bilateral_patient_dice'] if args.stage == 'cohort' else None,
               'best_checkpoint_sha256': file_hash(run / 'unet_best.pth'), 'initial_state_sha256': initial_hash,
               'native_roi_size_xyz': policy['selected_native_size_xyz'], 'coverage': coverage, 'oracle_clicks': 0,
               'validation_images_read': args.stage == 'cohort', 'test_images_read': False,
               'protocol': config['protocol'], 'comparison_note': 'U-Net receives no oracle clicks; SAM gets1/5 per side. Both use GT-derived crops. Different architecture/initialization/recipe; pipeline benchmark, not a one-variable ablation or automatic full-scan inference.',
               'artifacts': str(run), 'export_directory': str(export)}
    save_json(run / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    if args.stage == 'tiny' and not summary['diagnostic_passed']:
        print('Tiny diagnostic finished below minimum Dice0.90; cohort stage remains blocked. Inspect metrics/overlays before further training.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'data-dir', 'v4-run-dir', 'run-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--tiny-run-dir', type=Path)
    parser.add_argument('--stage', choices=('tiny', 'cohort'), default='tiny')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--patience-epochs', type=int, default=50)
    parser.add_argument('--min-delta', type=float, default=.001)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
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
        if not (args.run_dir / 'unet_latest.pth').is_file():
            parser.error('Resume requires the original U-Net run and latest checkpoint')
    elif args.run_dir.exists():
        parser.error('Run exists; choose a new run or --resume')
    else:
        args.run_dir.mkdir(parents=True)
    try:
        execute(args)
    except Exception as error:
        save_json(args.run_dir / 'failure.json', {'error_type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
