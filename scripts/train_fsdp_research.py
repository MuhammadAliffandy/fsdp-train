#!/usr/bin/env python3
"""Independent FSDP research trainer; existing resolution workflow files stay unchanged."""
import argparse
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import tarfile
import time

import numpy as np
import torch
from monai.losses import DiceCELoss

import physical_fsdp as data
import train_sam_refine32 as base
import train_fsdp_foundation as foundation
from evaluate_private_validation import file_hash, write_csv
from fsdp_research_model import ResearchModel, VARIANTS, prepare_distances, run_sample
from train_private_long import accumulation_window, atomic_save, early_stop_update
from train_private_unet import state_hash


EXTRA_FILES = ('train_fsdp_foundation.py', 'audit_foundation_sources.py', 'fsdp_verified_cache.py',
               'sam_refine_resolution.py', 'multiclass_model.py', 'fsdp_research_model.py',
               'train_fsdp_research.py', 'run_fsdp_research.py', 'fsdp_research_gpu.py', 'fsdp_research_metrics.py',
               'fsdp_research_resources.py')


def code_signature():
    return {name: file_hash(Path(__file__).with_name(name)) for name in (*base.CODE_FILES, *EXTRA_FILES)}


def prepare(args):
    old, previous = foundation.inspect_foundation(args.foundation_run_dir, args.source_root, args.verified_cache)
    args.source_best_epoch = previous['best_epoch']
    args.sam_dir, args.checkpoint, args.cohort_run_dir = [Path(old[k]).resolve() for k in ('sam_dir', 'checkpoint', 'cohort_run_dir')]
    for key, value in foundation.recipe_values(old).items():
        setattr(args, key, value)
    args.sam_lr, args.head_lr = old['sam_lr'] * args.lr_multiplier, old['head_lr'] * args.lr_multiplier
    args.head_warmup_epochs = 0
    inputs = args.run_dir / 'foundation_inputs'
    inputs.mkdir(exist_ok=True)
    source_args = copy.copy(args)
    source_args.run_dir = inputs
    prepared, control = foundation.prepare_foundation(source_args)
    config = {'experiment': 'fsdp_research_screening', 'variant': args.variant,
        'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'},
        'foundation': control['foundation'], 'cache_sha256': control['cache_sha256'],
        'manifest_sha256': control['manifest_sha256'], 'physical_policy': prepared['policy'],
        'versions': {**control['versions'], 'scipy': importlib.metadata.version('scipy')},
        'code_sha256': code_signature(), 'target_recovery': control.get('target_recovery'),
        'variant_protocol': {'control': 'original last point and original objective',
            'history': 'all sparse points1..k to SAM; original objective and CNN history',
            'boundary': 'original prompts; DiceCE +0.25context + boundary_weight*physical signed-distance term',
            'refiner': 'original prompts/objective plus zero-output-initialized native-half residual branch'}[args.variant],
        'refiner_protocol': {'enabled': args.variant == 'refiner', 'feature_grid': 'native half-resolution',
            'input_channels': '16half +8pooled full +1mask prior +2physical point maps',
            'layers': 'depthwise27x5x5x5;pointwise27to16;GroupNorm4;SiLU;zero-output16to1',
            'initialization': 'all foundation tensors identical; only new output projection initialized to zero'},
        'boundary_protocol': {'enabled': args.variant == 'boundary', 'weight': args.boundary_weight, 'clip_mm': args.boundary_clip_mm,
            'normalization': 'signed physical distance clipped to +/-clip_mm and divided by clip_mm',
            'support': 'invalid CT padding is not a background reference and has zero loss gradient',
            'augmentation': 'same LR reflection as paired CT/GT/context; intensity gain/shift do not change distances'},
        'evaluation_protocol': 'same cached GT-side ROI;5oracle clicks/side; train-only threshold; native full-target patient-macro Dice',
        'test_images_read': False}
    if args.resume:
        if data.read(args.run_dir / 'config.json') != config:
            raise ValueError('Research resume config/code/source changed')
    else:
        data.save(args.run_dir / 'config.json', config)
    if args.variant == 'boundary':
        prepare_distances(prepared, args.boundary_clip_mm)
    return prepared, config


def train_epoch(model, prepared, optimizer, scaler, criterion, device, epoch, args):
    source_epoch = epoch + args.source_best_epoch
    stage = base.set_stage(model, source_epoch, 0)
    samples = prepared['samples']['train']
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(args.seed + source_epoch)).tolist()
    optimizer.zero_grad(set_to_none=True)
    losses, updates, skipped = [], 0, 0
    for step, index in enumerate(order):
        loss, _ = run_sample(model, samples[index], device, args.seed + source_epoch * 1000 + step,
                             criterion, args.augmentation == 'light', args.auxiliary_weight)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite research loss')
        window, should_step = accumulation_window(step, len(samples), args.accumulation)
        scaler.scale(loss / window).backward()
        losses.append(loss.item())
        if should_step:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm) and not scaler.is_enabled():
                raise ValueError('Nonfinite research gradients')
            previous = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            successful = scaler.get_scale() >= previous
            updates += int(successful); skipped += int(not successful)
            optimizer.zero_grad(set_to_none=True)
        print(f'research={args.variant} epoch={epoch + 1}/{args.epochs} roi={step + 1}/{len(samples)} '
              f'loss={loss.item():.6f} updates={updates} skipped={skipped}', flush=True)
    expected = math.ceil(len(samples) / args.accumulation)
    if updates + skipped != expected:
        raise ValueError('Research optimizer update count differs')
    return {'loss': float(np.mean(losses)), 'stage': stage, 'optimizer_updates': updates,
            'amp_skipped_updates': skipped, 'expected_updates': expected}


def gradient_probe(model, prepared, args, criterion, device):
    before = state_hash(model.state_dict())
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    base.set_stage(model, args.source_best_epoch, 0)
    model.zero_grad(set_to_none=True)
    shapes = {}
    modules = [('tokens', model.sam.image_encoder), ('full', model.full), ('half', model.half_features),
               ('fusion32', model.detail32)]
    if args.variant == 'refiner':
        modules.append(('local_refiner', model.research_refiner.output))
    hooks = [module.register_forward_hook(lambda module, inputs, output, name=name:
             shapes.update({name: list(output.shape)})) for name, module in modules]
    try:
        loss, _ = run_sample(model, prepared['samples']['train'][0], device, args.seed, criterion,
                             auxiliary_weight=args.auxiliary_weight)
        loss.backward()
    finally:
        for hook in hooks:
            hook.remove()
    if shapes['tokens'][-3:] != [8] * 3 or shapes['fusion32'][-3:] != [32] * 3:
        raise ValueError('Research base feature grids differ from frozen8/32foundation')
    if args.variant == 'refiner' and shapes['local_refiner'][-3:] != shapes['half'][-3:]:
        raise ValueError('Research refiner did not use native-half features')
    prefixes = ['sam.image_encoder.blocks.10.', 'sam.image_encoder.blocks.11.', 'sam.mask_decoder.', 'residual.']
    if args.variant == 'refiner':
        prefixes.append('research_refiner.output.')
    groups = {}
    for prefix in prefixes:
        gradients = [p.grad for name, p in model.named_parameters() if name.startswith(prefix) and p.grad is not None]
        if not gradients or any(not g.isfinite().all() for g in gradients) or sum(g.abs().sum().item() for g in gradients) == 0:
            raise ValueError(f'Research gradient probe failed: {prefix}')
        groups[prefix] = sum(g.float().square().sum().item() for g in gradients) ** .5
    model.zero_grad(set_to_none=True); model.eval()
    if state_hash(model.state_dict()) != before:
        raise ValueError('Research gradient-only probe changed model state')
    return {'passed': True, 'optimizer_updates': 0, 'gradient_l2': groups, 'loss': float(loss.detach()),
            'model_state_unchanged': True, 'feature_shapes': shapes,
            'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None,
            'refiner_note': 'zero output preserves initial prediction; upstream new-branch gradients may start at zero'}


def metric_map(bundle, threshold):
    return {str(k): bundle['refined'][k][threshold]['metrics'] for k in (1, 5)}


def allocator_limit(device):
    value = os.environ.get('FSDP_RESEARCH_GPU_MEMORY_GIB')
    if value is None or device.type != 'cuda':
        return None
    budget = float(value)
    total = torch.cuda.get_device_properties(device).total_memory
    if not math.isfinite(budget) or budget <= 0 or budget * 2**30 > total:
        raise ValueError('Invalid per-process research GPU memory budget')
    torch.cuda.set_per_process_memory_fraction(budget * 2**30 / total, device)
    return budget


def verify_exports(prepared, validation, threshold, output):
    from train_private_tiny import native_prediction_indices, sparse_scores
    from fsdp_research_metrics import surface_metrics
    bundle = validation['refined'][5][threshold]
    sides = {(r['case'], r['side']): r for r in bundle['sides']}
    patients = {r['case']: r for r in bundle['patients']}
    parts = {alias: [] for alias in patients}
    spacing = {}
    for sample in prepared['samples']['val']:
        path = output / f'{sample["name"]}_t{threshold:g}_prediction.nii.gz'
        saved = data.sitk.ReadImage(str(path))
        prediction = data.xyz(saved)
        for key, value in data.geometry(saved).items():
            if not np.allclose(value, sample['native'][key], rtol=2e-7, atol=1e-5):
                raise ValueError('Exported prediction physical geometry differs')
        if not np.isin(prediction, (0, 1)).all():
            raise ValueError('Exported research prediction is not binary')
        prediction = prediction.astype(bool)
        reference = data.xyz(data.sitk.ReadImage(str(output / f'{sample["name"]}_target.nii.gz')))
        if not np.array_equal(reference, sample['native_target']):
            raise ValueError('Exported research target differs from original cache')
        shape = prepared['patients']['val'][sample['case']]['shape']
        valid = np.ones(prediction.shape, bool)
        predicted = native_prediction_indices(prediction, valid, sample['metadata'], shape)
        target = native_prediction_indices(sample['native_target'], valid, sample['metadata'], shape)
        score = sparse_scores(predicted, target)
        if abs(score['dice_full_target'] - sides[(sample['case'], sample['side'])]['dice_full_target']) > 1e-4:
            raise ValueError('Exported research side Dice differs')
        parts[sample['case']].append(predicted)
        spacing[sample['case']] = sample['native']['spacing']
        # The shared exporter may also write a fixed0.5control; the main PNG must show the selected threshold.
        base.overlay(sample, prediction, score['dice_full_target'], threshold, output)
    surfaces = []
    for alias, predictions in parts.items():
        predicted = np.concatenate(predictions)
        target = prepared['patients']['val'][alias]['target_indices']
        score = sparse_scores(predicted, target)
        if abs(score['dice_full_target'] - patients[alias]['dice_full_target']) > 1e-4:
            raise ValueError('Exported research bilateral Dice differs')
        surfaces.append({'case': alias, **surface_metrics(predicted, target,
            prepared['patients']['val'][alias]['shape'], spacing[alias])})
    write_csv(output / 'val_5_boundary_metrics.csv', surfaces)
    hd95 = [r['hd95_mm'] for r in surfaces if r['hd95_mm'] is not None]
    return {'mean_surface_dice_0_5mm': float(np.mean([r['surface_dice_0_5mm'] for r in surfaces])),
            'mean_surface_dice_1mm': float(np.mean([r['surface_dice_1mm'] for r in surfaces])),
            'median_hd95_mm_defined_patients': float(np.median(hd95)) if hd95 else None,
            'hd95_defined_patients': len(hd95), 'empty_prediction_patients': sum(r['empty_prediction'] for r in surfaces),
            'patients': len(surfaces), 'method': 'MONAI symmetric HD95; subvoxel area-weighted surface Dice; original spacing',
            'scope': 'diagnostic tolerances0.5/1mm; not clinical acceptance thresholds'}


def execute(args):
    started = time.monotonic()
    prepared, config = prepare(args)
    args.source_best_epoch = config['foundation']['source_best_epoch']
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    memory_budget = allocator_limit(device)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    properties = torch.cuda.get_device_properties(device) if device.type == 'cuda' else None
    hardware = {'device': str(device), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_name': properties.name if properties else None, 'precision': args.precision,
        'pytorch_allocator_budget_gib': memory_budget}
    data.save(args.run_dir / ('resume_hardware.json' if args.resume else 'hardware.json'), hardware)
    original, _ = base.build(args)
    foundation.load_foundation_weights(original, args, config)
    original = original.to(device)
    original.amp_dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16
    if not args.resume:
        threshold, training, validation = base.calibrate_validate(original, prepared, device, args.seed)
        foundation.verify_reproduction(threshold, training, validation, config['foundation'])
        data.save(args.run_dir / 'foundation_reproduction.json', {'passed': True, 'optimizer_updates': 0,
            'threshold': threshold, 'train': metric_map(training, threshold), 'val': metric_map(validation, threshold),
            'checkpoint_sha256': config['foundation']['checkpoint_sha256']})
    else:
        proof = data.read(args.run_dir / 'foundation_reproduction.json')
        if not proof['passed'] or proof['checkpoint_sha256'] != config['foundation']['checkpoint_sha256']:
            raise ValueError('Research resume requires verified foundation reproduction')
    original_state = {k: v.detach().cpu().clone() for k, v in original.state_dict().items()}
    rng = torch.get_rng_state()
    model = ResearchModel.from_foundation(original, args.variant, args.boundary_weight).to(device)
    torch.set_rng_state(rng)
    model.amp_dtype = original.amp_dtype
    for key, value in original_state.items():
        if not torch.equal(model.state_dict()[key].cpu(), value):
            raise ValueError('Research adaptation changed a learned foundation tensor')
    del original, original_state
    initial_hash = state_hash(model.state_dict())
    optimizer = torch.optim.AdamW([
        {'params': model.sam.parameters(), 'lr': args.sam_lr, 'weight_decay': .01},
        {'params': [p for name, p in model.named_parameters() if not name.startswith('sam.')],
         'lr': args.head_lr, 'weight_decay': 1e-4}])
    def schedule(epoch):
        warmup = min(5, args.epochs)
        if epoch < warmup:
            return (epoch + 1) / warmup
        return .05 + .95 * .5 * (1 + math.cos(math.pi * min(1., (epoch - warmup) / max(1, args.epochs - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and args.precision == 'fp16')
    criterion = DiceCELoss(sigmoid=True, squared_pred=True)
    start, best, best_epoch, best_threshold, history, initial, stopped = 0, -1., 0, .5, [], None, False
    early = {'anchor': -1., 'last_improvement_epoch': 0}
    if args.resume:
        saved = torch.load(args.run_dir / 'research_latest.pth', map_location='cpu', weights_only=False)
        if saved['config'] != config or saved['initial_state_sha256'] != initial_hash:
            raise ValueError('Research checkpoint initialization/config differs')
        for obj, key in ((model, 'model'), (optimizer, 'optimizer'), (scheduler, 'scheduler'), (scaler, 'scaler')):
            obj.load_state_dict(saved[key + '_state_dict'])
        start, best, best_epoch, best_threshold, history, initial, early = [saved[key] for key in
            ('epoch', 'best_score', 'best_epoch', 'best_threshold', 'history', 'initial_metrics', 'early_state')]
        torch.set_rng_state(saved['torch_rng_state']); np.random.set_state(saved['numpy_rng_state'])
        random.setstate(saved['python_rng_state'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
        del saved
    def checkpoint(epoch):
        return {'epoch': epoch, 'config': config, 'initial_state_sha256': initial_hash, 'best_score': best,
            'best_epoch': best_epoch, 'best_threshold': best_threshold, 'history': history, 'initial_metrics': initial,
            'early_state': early, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(),
            'torch_rng_state': torch.get_rng_state(), 'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(), 'cuda_rng_state': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
    if not args.resume:
        probe = gradient_probe(model, prepared, args, criterion, device)
        threshold, training, validation = base.calibrate_validate(model, prepared, device, args.seed)
        if args.variant in ('control', 'boundary', 'refiner'):
            foundation.verify_reproduction(threshold, training, validation, config['foundation'])
        best_threshold = threshold
        best = validation['refined'][5][threshold]['metrics']['mean_bilateral_patient_dice']
        initial = {'train_threshold': threshold, 'train': metric_map(training, threshold), 'val': metric_map(validation, threshold)}
        early = {'anchor': best, 'last_improvement_epoch': 0}
        data.save(args.run_dir / 'preflight.json', {'passed': True, 'gradient_probe': probe,
            'foundation_weights_preserved': True, 'initial_metrics': initial, 'model_state_sha256': initial_hash,
            'parameter_counts': {'total': sum(p.numel() for p in model.parameters()),
                                 'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                                 'new_local_refiner': sum(p.numel() for p in model.research_refiner.parameters())
                                     if args.variant == 'refiner' else 0},
            'detail_grid': [32] * 3, 'token_grid': [8] * 3, 'test_images_read': False})
        atomic_save(args.run_dir / 'research_best.pth', checkpoint(0))
        atomic_save(args.run_dir / 'research_latest.pth', checkpoint(0))
    stopped = args.patience_epochs > 0 and start - early['last_improvement_epoch'] >= args.patience_epochs
    for epoch in range(start, args.epochs) if not stopped else []:
        lrs = [g['lr'] for g in optimizer.param_groups]
        measured = train_epoch(model, prepared, optimizer, scaler, criterion, device, epoch, args)
        scheduler.step()
        row = {'epoch': epoch + 1, 'source_plus_additional_epoch': args.source_best_epoch + epoch + 1,
            'sam_lr': lrs[0], 'head_lr': lrs[1], **measured, 'train_threshold': None, 'train_dice_5': None,
            'val_dice_1': None, 'val_dice_5': None, 'val_control_0_5': None}
        improved = False
        if epoch == 0 or (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            threshold, training, validation = base.calibrate_validate(model, prepared, device, args.seed)
            score = validation['refined'][5][threshold]['metrics']['mean_bilateral_patient_dice']
            row.update(train_threshold=threshold, train_dice_5=training['refined'][5][threshold]['metrics']['mean_bilateral_patient_dice'],
                val_dice_1=validation['refined'][1][threshold]['metrics']['mean_bilateral_patient_dice'], val_dice_5=score,
                val_control_0_5=validation['refined'][5][.5]['metrics']['mean_bilateral_patient_dice'])
            early, stopped = early_stop_update(early, score, epoch + 1, args.patience_epochs, args.min_delta)
            improved = score > best
            if improved:
                best, best_epoch, best_threshold = score, epoch + 1, threshold
        history.append(row)
        write_csv(args.run_dir / 'history.csv', history)
        data.save(args.run_dir / 'progress.json', {'complete': False, 'variant': args.variant,
            'completed_epochs': epoch + 1, 'planned_epochs': args.epochs, 'best_epoch': best_epoch,
            'best_validation_dice_5_per_side': best, 'latest': row, 'test_images_read': False})
        if improved:
            atomic_save(args.run_dir / 'research_best.pth', checkpoint(epoch + 1))
        atomic_save(args.run_dir / 'research_latest.pth', checkpoint(epoch + 1))
        print(json.dumps({'variant': args.variant, 'epoch_result': row, 'best_validation_dice_5_per_side': best}), flush=True)
        if stopped:
            break
    selected = torch.load(args.run_dir / 'research_best.pth', map_location='cpu', weights_only=False)
    if selected['config'] != config or selected['best_score'] != best or selected['best_epoch'] != best_epoch:
        raise ValueError('Selected research checkpoint metadata differs')
    model.load_state_dict(selected['model_state_dict'], strict=True)
    output = args.run_dir / 'best_validation'; output.mkdir(exist_ok=True)
    validation = base.evaluate(model, prepared, 'val', device, sorted(set((best_threshold, .5))), args.seed,
                               controls=True, export=output)
    actual = validation['refined'][5][best_threshold]['metrics']['mean_bilateral_patient_dice']
    if abs(actual - best) > 1e-4:
        raise ValueError('Research native exports do not reproduce selected checkpoint')
    boundary_metrics = verify_exports(prepared, validation, best_threshold, output)
    training = base.evaluate(model, prepared, 'train', device, [best_threshold], args.seed)
    base.plot_results(history, training, validation, best_threshold, output)
    for split, bundle in (('train', training), ('val', validation)):
        for suffix in ('patients', 'sides'):
            write_csv(output / f'{split}_5_{suffix}.csv', bundle['refined'][5][best_threshold][suffix])
    summary = {'complete': True, 'variant': args.variant, 'completed_epochs': history[-1]['epoch'] if history else start,
        'planned_epochs': args.epochs, 'stopped_early': stopped, 'best_epoch': best_epoch, 'threshold': best_threshold,
        'initial_candidate_metrics': initial, 'best_validation_metrics': metric_map(validation, best_threshold),
        'best_frozen_training_metrics': metric_map(training, best_threshold),
        'native_boundary_metrics': boundary_metrics,
        'validation_control_fixed_0_5': validation['refined'][5][.5]['metrics'],
        'validation_change_vs_foundation': best - config['foundation']['expected_validation']['5']['mean_bilateral_patient_dice'],
        'validation_change_vs_candidate_initial': best - initial['val']['5']['mean_bilateral_patient_dice'],
        'best_checkpoint_sha256': file_hash(args.run_dir / 'research_best.pth'), 'foundation': config['foundation'],
        'cache_sha256': config['cache_sha256'], 'target_recovery': config['target_recovery'],
        'test_images_read': False, 'validation_images_read': True, 'oracle_clicks_per_side': [1, 5],
        'initial_candidate_in_selection': True, 'elapsed_seconds_this_invocation': time.monotonic() - started,
        'train_patients': len(prepared['patients']['train']), 'validation_patients': len(prepared['patients']['val']),
        'train_side_rois': len(prepared['samples']['train']), 'validation_side_rois': len(prepared['samples']['val']),
        'protocol': config['evaluation_protocol'], 'detail_grid': [32] * 3, 'token_grid': [8] * 3,
        'interpretation': 'Controlled research screen; history changes prompt information. GT-assisted validation, not automatic test performance.',
        'artifacts': str(args.run_dir), 'archive': str(args.run_dir / 'results.tar.gz')}
    data.save(args.run_dir / 'report.json', summary)
    with tarfile.open(args.run_dir / 'results.tar.gz', 'w:gz') as archive:
        archive.add(output, arcname='results/best_validation')
        for name in ('report.json', 'config.json', 'history.csv', 'preflight.json', 'foundation_reproduction.json', 'hardware.json'):
            archive.add(args.run_dir / name, arcname='results/' + name)
        if (args.run_dir / 'resume_hardware.json').exists():
            archive.add(args.run_dir / 'resume_hardware.json', arcname='results/resume_hardware.json')
        for name in ('source_verification.json', 'cache_target_verification.json'):
            path = args.run_dir / 'foundation_inputs' / name
            if path.exists():
                archive.add(path, arcname='results/' + name)
    data.save(args.run_dir / 'summary.json', summary)
    data.save(args.run_dir / 'progress.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foundation-run-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--verified-cache', action='store_true')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr-multiplier', type=float, default=.2)
    parser.add_argument('--patience-epochs', type=int, default=50)
    parser.add_argument('--boundary-weight', type=float, default=.05)
    parser.add_argument('--boundary-clip-mm', type=float, default=5.)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (args.epochs < 1 or args.patience_epochs < 0 or not math.isfinite(args.lr_multiplier)
            or not 0 < args.lr_multiplier < 1 or not math.isfinite(args.boundary_weight) or args.boundary_weight <= 0
            or not math.isfinite(args.boundary_clip_mm) or args.boundary_clip_mm <= 0):
        parser.error('Invalid research budget or objective')
    args.foundation_run_dir = args.foundation_run_dir.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    if args.source_root:
        args.source_root = args.source_root.expanduser().resolve()
    old, _ = foundation.inspect_foundation(args.foundation_run_dir, args.source_root, args.verified_cache)
    control = data.read(Path(old['cohort_run_dir']) / 'config.json')
    protected = [args.foundation_run_dir, Path(old['sam_dir']).resolve(), Path(old['cohort_run_dir']).resolve(),
                 Path(control['data_dir']).resolve(), Path(control['source_root']).resolve()]
    if args.source_root:
        protected.append(args.source_root)
    if any(args.run_dir.is_relative_to(p) for p in protected):
        parser.error('Research outputs must be outside original data and foundation')
    if args.resume:
        if not (args.run_dir / 'research_latest.pth').exists() or (args.run_dir / 'summary.json').exists():
            parser.error('Resume requires unfinished research latest checkpoint')
    elif args.run_dir.exists():
        parser.error('Choose a new research run directory')
    args.run_dir.mkdir(parents=True, exist_ok=args.resume)
    if args.device.startswith('cuda') and (not torch.cuda.is_available() or (old['precision'] == 'bf16' and not torch.cuda.is_bf16_supported())):
        parser.error('Requested CUDA/BF16 device unavailable')
    try:
        data.save(args.run_dir / 'progress.json', {'complete': False, 'phase': 'verifying_foundation', 'test_images_read': False})
        execute(args)
    except Exception as error:
        data.save(args.run_dir / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
