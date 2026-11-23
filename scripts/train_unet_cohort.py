#!/usr/bin/env python3
"""Exploratory 54/10-patient U-Net training with train-calibrated threshold."""

import argparse
import inspect
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

import train_private_unet as base
from audit_unet_tiny import threshold_sweep
from evaluate_private_validation import file_hash, save_crop, scores, write_csv
from roi_zoom import overlay_native, restore_logits, zoom_samples
from train_private_long import atomic_save, early_stop_update
from train_private_tiny import native_prediction_indices, prepare_samples, select_patients, sparse_scores


def read(path):
    return json.loads(path.read_text())


def verify_calibration(args, versions, policy):
    from monai.networks.nets import UNet
    tiny, audit = args.tiny_run_dir, args.audit_dir
    summary, old = read(tiny / 'summary.json'), read(tiny / 'config.json')
    report, thresholds = read(audit / 'summary.json'), read(audit / 'thresholds.json')
    for result in (summary, report, thresholds):
        if not result['complete'] or result['validation_images_read'] or result['test_images_read']:
            raise ValueError('Calibration must be completed and training-only')
    if (summary['stage'] != 'tiny' or summary['train_patients'] != 2
            or thresholds['training_patients'] != 2 or thresholds['training_sides'] != 4
            or report['run'] != tiny.name or thresholds['run'] != tiny.name):
        raise ValueError('Calibration must refer to this two-patient tiny run')
    expected = {'manifest_sha256': file_hash(args.data_dir / 'manifest.json'),
                'script_sha256': file_hash(Path(base.__file__)), 'architecture': base.ARCHITECTURE,
                'versions': versions, 'seed': args.seed, 'native_roi_size_xyz': policy['selected_native_size_xyz'],
                'monai_source_sha256': file_hash(Path(inspect.getfile(UNet))),
                'v4_control_sha256': {name: file_hash(args.v4_run_dir / name) for name in
                    ('summary.json', 'config.json', 'roi_policy.json', 'source_provenance.json')}}
    for key, value in expected.items():
        if old[key] != value:
            raise ValueError(f'Tiny calibration source/config differs: {key}')
    for name, checksum in old['shared_code_sha256'].items():
        if file_hash(Path(base.__file__).with_name(name)) != checksum:
            raise ValueError(f'Calibration helper changed: {name}')
    if report['audit_script_sha256'] != file_hash(Path(__file__).with_name('audit_unet_tiny.py')):
        raise ValueError('Calibration audit implementation changed')
    for name in ('unet_best', 'unet_latest'):
        if file_hash(tiny / f'{name}.pth') != report['checkpoints'][name]['sha256']:
            raise ValueError('Calibration checkpoint differs from audit')
    if report['checkpoints']['unet_best']['sha256'] != summary['best_checkpoint_sha256']:
        raise ValueError('Tiny best checkpoint differs from summary')
    chosen = {}
    for precision in ('amp', 'fp32'):
        values = thresholds['checkpoints']['unet_latest'][precision]
        candidate = max(values['curve'], key=lambda row: (row['minimum_diagnostic_dice'], -abs(row['threshold'] - .5)))
        if candidate != values['best_train_only_candidate'] or candidate['threshold'] != args.threshold:
            raise ValueError('Fixed threshold must match the train-only latest-checkpoint candidate')
        if (len(candidate['per_side']) != 4 or len(candidate['per_patient']) != 2
                or not all(np.isfinite(candidate[key]) for key in
                           ('mean_bilateral_patient_dice', 'minimum_diagnostic_dice'))):
            raise ValueError('Incomplete threshold calibration')
        chosen[precision] = candidate
    return {'original_tiny_diagnostic_passed': summary['diagnostic_passed'],
            'original_tiny_minimum_target': .90, 'exploratory_cohort_authorized': True,
            'threshold': args.threshold, 'threshold_source': 'latest tiny checkpoint; two training patients only',
            'candidate': chosen, 'tiny_initial_state_sha256': summary['initial_state_sha256'],
            'tiny_artifacts': {name: file_hash(tiny / name) for name in
                ('summary.json', 'config.json', 'source_provenance.json', 'unet_best.pth', 'unet_latest.pth')},
            'audit_artifacts': {name: file_hash(audit / name) for name in ('summary.json', 'thresholds.json')}}


@torch.inference_mode()
def evaluate(model, samples, patients, device, threshold, export=None, label='evaluation'):
    model.eval()
    sides, parts, records = [], {}, []
    cutoff = math.log(threshold / (1 - threshold))
    if export is not None:
        export.mkdir(parents=True)
    for index, sample in enumerate(samples, 1):
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits = base.checked_logits(model, sample['image'].to(device))
        native = restore_logits(logits[0, 0].float().cpu().numpy(), sample['native_target'].shape)
        prediction = (native > cutoff) & sample['valid']
        meta = sample['metadata']
        sides.append({'case': sample['case'], 'side': sample['side'], 'threshold': threshold, 'oracle_clicks': 0,
                      'target_retention': meta['target_retention'],
                      **scores(prediction, sample['native_target'], meta['full_target_voxels'], sample['valid'])})
        parts.setdefault(sample['case'], []).append(native_prediction_indices(
            prediction, sample['valid'], meta, patients[sample['case']]['shape']))
        records.append({'sample': sample, 'amp': native, 'fp32': native})
        if export is not None:
            save_crop(export / f'{sample["name"]}_prediction_crop.nii.gz', prediction.astype(np.uint8), meta)
            save_crop(export / f'{sample["name"]}_target_crop.nii.gz', sample['native_target'].astype(np.uint8), meta)
            overlay_native(export / f'{sample["name"]}_overlay.png', sample, prediction)
        print(f'{label}: {sample["name"]} complete ({index}/{len(samples)})', flush=True)
    if (set(parts) != set(patients) or len(sides) != 2 * len(patients)
            or any(sorted(s['side'] for s in samples if s['case'] == alias) != ['LFSDP', 'RFSDP'] for alias in patients)):
        raise ValueError('Evaluation requires both sides for every patient')
    bilateral = [{'case': alias, 'threshold': threshold, 'oracle_clicks': 0,
                  **sparse_scores(np.concatenate(values), patients[alias]['target_indices'])}
                 for alias, values in sorted(parts.items())]
    means = {'mean_side_dice': float(np.mean([r['dice_full_target'] for r in sides])),
             'minimum_side_dice': min(r['dice_full_target'] for r in sides),
             'mean_bilateral_patient_dice': float(np.mean([r['dice_full_target'] for r in bilateral])),
             'minimum_bilateral_patient_dice': min(r['dice_full_target'] for r in bilateral),
             'mean_precision': float(np.mean([r['precision'] for r in bilateral])),
             'mean_recall': float(np.mean([r['recall_full_target'] for r in bilateral]))}
    # The 0.5 control uses the same logits; it never participates in checkpoint selection.
    control = threshold_sweep(records, patients, thresholds=(.5,))['amp']['curve'][0]
    return means, sides, bilateral, control


def execute(args):
    from monai.losses import DiceCELoss
    if not args.exploratory_cohort:
        raise ValueError('Explicit exploratory cohort authorization is required; original tiny gate remains unmet')
    run = args.run_dir
    manifest = read(args.data_dir / 'manifest.json')
    selected_ids = select_patients(manifest, args.seed)
    v4, policy, versions = base.v4_control(args, args.data_dir / 'manifest.json')
    calibration = verify_calibration(args, versions, policy)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != 'resume'}
    config.update(experiment='unet_cohort_train_calibrated', architecture=base.ARCHITECTURE,
                  calibration=calibration, versions=versions,
                  code_sha256={name: file_hash(Path(__file__).with_name(name)) for name in
                    ('train_unet_cohort.py', 'train_private_unet.py', 'audit_unet_tiny.py',
                     'train_private_tiny.py', 'roi_zoom.py', 'evaluate_private_validation.py', 'train_private_long.py')},
                  initialization='fresh random weights with the tiny seed; no tiny/SAM learned weights',
                  loss='DiceCELoss sigmoid=True squared_pred=True', optimizer='AdamW; weight_decay=1e-4',
                  schedule='5 epoch warmup; cosine to 0.05 of LR', augmentation='independent XYZ flips',
                  selection='native full-target patient-macro bilateral validation Dice at fixed train-calibrated threshold',
                  moving_training_meter_threshold=.5, native_roi_size_xyz=policy['selected_native_size_xyz'],
                  protocol='GT-side-centered fixed v4 ROI; 128 cube input; native logits restored before threshold; no clicks',
                  manifest_sha256=file_hash(args.data_dir / 'manifest.json'), test_images_read=False)
    if args.resume:
        if read(run / 'config.json') != config:
            raise ValueError('Resume config/code/calibration differs from frozen run')
    else:
        base.save_json(run / 'config.json', config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    folder = run / 'preparation'
    number = 0
    while folder.exists():
        number += 1
        folder = run / f'preparation_resume_{number}'
    folder.mkdir()
    splits, patients_by_split, provenance, coverage = {}, {}, {}, {}
    v4_sources = read(args.v4_run_dir / 'source_provenance.json')
    for split, count in (('train', 54), ('val', 10)):
        output = folder / split
        output.mkdir()
        samples, patients = prepare_samples(args.source_root, args.data_dir, manifest, output, args.seed,
                                            split=split, all_training_patients=split == 'train')
        if len(patients) != count or len(samples) != 2 * count:
            raise ValueError('Cohort must contain 54 train and 10 validation patients')
        name = 'selected_training_patients.json' if split == 'train' else 'selected_validation_patients.json'
        provenance[split] = read(output / name)
        reference = {row['patient_id']: row for row in v4_sources[split]}
        for row in provenance[split]:
            if any(row[key] != reference[row['patient_id']][key] for key in
                   ('source_image', 'source_label', 'source_image_sha256', 'source_label_sha256')):
                raise ValueError('Original source CT/final annotations differ from v4')
        for sample in samples:
            sample['target'] = sample['target'].to(torch.uint8)
        splits[split] = zoom_samples(samples, policy['selected_native_size_xyz'])
        patients_by_split[split] = patients
        retention = [s['metadata']['target_retention'] for s in splits[split]]
        if min(retention) != 1:
            raise ValueError('Frozen crop must retain every train/validation target')
        coverage[split] = {'patients': count, 'side_rois': 2 * count, 'minimum_target_retention': min(retention)}
    tiny_sources = read(args.tiny_run_dir / 'source_provenance.json')['train']
    actual = {row['patient_id']: row for row in provenance['train']}
    if (sorted(row['patient_id'] for row in tiny_sources) != selected_ids
            or any(row[key] != actual[row['patient_id']][key] for row in tiny_sources for key in
                   ('source_image', 'source_label', 'source_image_sha256', 'source_label_sha256'))):
        raise ValueError('Calibration sources do not match the selected training patients')
    if args.resume and read(run / 'source_provenance.json') != provenance:
        raise ValueError('Source CT/final labels changed on resume')
    base.save_json(run / 'source_provenance.json', provenance)
    torch.manual_seed(args.seed)
    model = base.build_model().to(device)
    initial_hash = base.state_hash(model.state_dict())
    if initial_hash != calibration['tiny_initial_state_sha256']:
        raise ValueError('Random initialization does not match frozen tiny architecture/seed')
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
    start, best_epoch, best, history, early = 0, 0, -1., [], {'anchor': -1., 'last_improvement_epoch': 0}
    initial = None
    if args.resume:
        checkpoint = torch.load(run / 'unet_latest.pth', map_location='cpu', weights_only=False)
        if checkpoint['config'] != config or checkpoint['initial_state_sha256'] != initial_hash:
            raise ValueError('Resume checkpoint/config/initialization mismatch')
        for obj, key in ((model, 'model'), (optimizer, 'optimizer'), (scheduler, 'scheduler'), (scaler, 'scaler')):
            obj.load_state_dict(checkpoint[key + '_state_dict'])
        start, best_epoch, best, history, early, initial = [checkpoint[key] for key in
            ('epoch', 'best_epoch', 'best_score', 'history', 'early_state', 'initial_metrics')]
        torch.set_rng_state(checkpoint['torch_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['python_rng_state'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        del checkpoint
    def state(epoch):
        return {'epoch': epoch, 'best_epoch': best_epoch, 'best_score': best, 'history': history,
                'early_state': early, 'initial_metrics': initial, 'initial_state_sha256': initial_hash, 'config': config,
                'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(),
                'torch_rng_state': torch.get_rng_state(), 'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate(), 'cuda_rng_state': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
    def validation(label):
        return evaluate(model, splits['val'], patients_by_split['val'], device, args.threshold, label=label)
    if not args.resume:
        initial, _, _, control = validation('initial_validation')
        best = initial['mean_bilateral_patient_dice']
        early = {'anchor': best, 'last_improvement_epoch': 0}
        atomic_save(run / 'unet_best.pth', state(0))
        atomic_save(run / 'unet_latest.pth', state(0))
        base.save_json(run / 'preflight.json', {'passed': True, 'coverage': coverage, 'initial_metrics': initial,
            'fixed_threshold': args.threshold, 'initial_control_0_5': control, 'calibration': calibration,
            'initial_state_sha256': initial_hash, 'validation_images_read': True, 'test_images_read': False})
    stopped = args.patience_epochs > 0 and start - early['last_improvement_epoch'] >= args.patience_epochs
    for epoch in range(start, args.epochs) if not stopped else []:
        lr = optimizer.param_groups[0]['lr']
        metrics = base.train_epoch(model, splits['train'], optimizer, scaler, criterion, device,
                                   epoch, args.seed, args.accumulation, True)
        if metrics['optimizer_updates'] + metrics['amp_skipped_updates'] != metrics['expected_updates']:
            raise ValueError('Incomplete optimizer accumulation windows')
        scheduler.step()
        metrics['moving_model_grid_train_dice_at_0_5'] = metrics.pop('moving_model_grid_train_dice')
        row = {'epoch': epoch + 1, 'lr': lr, **metrics, 'validation_dice': None, 'validation_control_dice_0_5': None}
        improved = False
        if epoch == 0 or (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            means, _, _, control = validation(f'validation_epoch_{epoch + 1}')
            score = means['mean_bilateral_patient_dice']
            row.update(validation_dice=score, validation_control_dice_0_5=control['mean_bilateral_patient_dice'],
                       validation_precision=means['mean_precision'], validation_recall=means['mean_recall'])
            early, stopped = early_stop_update(early, score, epoch + 1, args.patience_epochs, args.min_delta)
            improved = score > best
            if improved:
                best, best_epoch = score, epoch + 1
        history.append(row)
        write_csv(run / 'history.csv', history)
        base.save_json(run / 'progress.json', {'complete': False, 'completed_epochs': epoch + 1,
            'planned_epochs': args.epochs, 'best_epoch': best_epoch, 'best_validation_dice': best,
            'threshold': args.threshold, 'stopped_early': stopped, 'latest': row})
        if improved:
            atomic_save(run / 'unet_best.pth', state(epoch + 1))
        atomic_save(run / 'unet_latest.pth', state(epoch + 1))
        print(json.dumps({'epoch_result': row, 'best_validation_dice': best}), flush=True)
        if stopped:
            break
    chosen = torch.load(run / 'unet_best.pth', map_location='cpu', weights_only=False)
    if chosen['epoch'] != best_epoch or chosen['best_score'] != best or chosen['config'] != config:
        raise ValueError('Selected checkpoint metadata differs')
    model.load_state_dict(chosen['model_state_dict'], strict=True)
    del chosen
    means, sides, bilateral, control = validation('selected_validation')
    if not np.isclose(means['mean_bilateral_patient_dice'], best, atol=1e-4, rtol=0):
        raise ValueError('Selected checkpoint failed score reproduction')
    export, number = run / 'best_exports', 0
    while export.exists():
        number += 1
        export = run / f'best_exports_resume_{number}'
    export.mkdir()
    write_csv(export / 'sides.csv', sides)
    write_csv(export / 'patients.csv', bilateral)
    ranked = sorted(bilateral, key=lambda r: (r['dice_full_target'], r['case']))
    aliases = {ranked[0]['case'], ranked[len(ranked) // 2]['case'], ranked[-1]['case']}
    _, exported, _, _ = evaluate(model, [s for s in splits['val'] if s['case'] in aliases],
                                 {a: patients_by_split['val'][a] for a in aliases}, device, args.threshold,
                                 export=export / 'representatives', label='export_validation')
    expected = {(r['case'], r['side']): r['dice_full_target'] for r in sides}
    if any(abs(r['dice_full_target'] - expected[(r['case'], r['side'])]) > 1e-4 for r in exported):
        raise ValueError('Representative exports do not reproduce selected validation scores')
    train, train_sides, train_patients, train_control = evaluate(model, splits['train'], patients_by_split['train'],
                                                              device, args.threshold, label='selected_frozen_training')
    write_csv(export / 'frozen_training_sides.csv', train_sides)
    write_csv(export / 'frozen_training_patients.csv', train_patients)
    summary = {'complete': True, 'experiment': config['experiment'], 'completed_epochs': history[-1]['epoch'] if history else start,
               'planned_epochs': args.epochs, 'stopped_early': stopped, 'best_epoch': best_epoch,
               'train_patients': 54, 'validation_patients': 10, 'reserved_test_patients': 20,
               'threshold': args.threshold, 'best_validation_metrics': means, 'best_frozen_training_metrics': train,
               'control_0_5_at_selected_checkpoint': {'val': control, 'train': train_control},
               'train_validation_gap': train['mean_bilateral_patient_dice'] - means['mean_bilateral_patient_dice'],
               'v4_validation_dice_5_per_side': v4['best_validation_dice']['5']['mean_bilateral_patient_dice'],
               'best_checkpoint_sha256': file_hash(run / 'unet_best.pth'), 'coverage': coverage,
               'calibration': calibration, 'initial_state_sha256': initial_hash, 'protocol': config['protocol'],
               'oracle_clicks': 0, 'test_images_read': False, 'validation_images_read': True,
               'interpretation': 'Exploratory GT-assisted validation; original tiny 90% gate remains unmet. Not automatic full-scan/test performance.',
               'artifacts': str(run), 'export_directory': str(export)}
    base.save_json(run / 'summary.json', summary)
    base.save_json(run / 'progress.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'data-dir', 'v4-run-dir', 'tiny-run-dir', 'audit-dir', 'run-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--exploratory-cohort', action='store_true', required=True,
                        help='Explicitly proceed despite the original unmet tiny minimum Dice0.90')
    parser.add_argument('--threshold', type=float, default=.3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--patience-epochs', type=int, default=50)
    parser.add_argument('--min-delta', type=float, default=.001)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (min(args.epochs, args.val_every, args.accumulation) < 1 or not np.isfinite(args.lr) or args.lr <= 0
            or not 0 < args.threshold < 1 or args.patience_epochs < 0 or not np.isfinite(args.min_delta) or args.min_delta < 0):
        parser.error('Invalid training/threshold/early-stop settings')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    if args.resume:
        if not (args.run_dir / 'unet_latest.pth').is_file() or (args.run_dir / 'summary.json').exists():
            parser.error('Resume requires an unfinished run and its latest checkpoint')
    elif args.run_dir.exists():
        parser.error('Run already exists')
    else:
        args.run_dir.mkdir(parents=True)
    try:
        execute(args)
    except Exception as error:
        base.save_json(args.run_dir / 'failure.json', {'error_type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
