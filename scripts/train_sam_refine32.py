#!/usr/bin/env python3
"""Fine-tune original Turbo with a physically aligned32³CNN residual refiner."""

import argparse
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys
import tarfile

import numpy as np
import torch
from torch.nn import functional as F

import physical_fsdp as data
import sam_refine_data as views
from sam_refine32 import SamRefine32, TURBO_SHA256, interactions
from evaluate_private_validation import file_hash, save_crop, write_csv
from train_private_long import accumulation_window, atomic_save, early_stop_update
from train_private_tiny import native_prediction_indices, select_patients, sparse_scores
from train_private_unet import state_hash


CODE_FILES = ('train_sam_refine32.py', 'sam_refine32.py', 'sam_refine_data.py', 'physical_fsdp.py',
              'train_private_long.py', 'train_private_tiny.py', 'train_private_unet.py',
              'evaluate_private_validation.py', 'audit_expert_agreement.py')


def build(args):
    if file_hash(args.checkpoint) != TURBO_SHA256:
        raise ValueError('Expected the original sam_med3d_turbo.pth SHA256; checkpoint is different')
    sys.path.insert(0, str(args.sam_dir))
    implementation = importlib.import_module('segment_anything.build_sam3D')
    if not Path(implementation.__file__).resolve().is_relative_to(args.sam_dir):
        raise ValueError('Loaded SAM implementation is not from the requested source directory')
    sam = implementation.sam_model_registry3D['vit_b_ori'](checkpoint=None)
    encoder = sam.image_encoder
    if (encoder.img_size != 128 or tuple(encoder.patch_embed.proj.stride) != (16,) * 3
            or tuple(encoder.pos_embed.shape) != (1, 8, 8, 8, 768)
            or sam.prompt_encoder.embed_dim != 384):
        raise ValueError('Expected original Turbo vit_b_ori with128input/8³embedding')
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sam.load_state_dict(source.get('model_state_dict', source), strict=True)
    del source
    before = state_hash(sam.state_dict())
    model = SamRefine32(sam)
    if state_hash(model.sam.state_dict()) != before:
        raise ValueError('Adding the refiner unexpectedly changed Turbo weights')
    return model, before


def verify_sources(provenance):
    for rows in provenance.values():
        for row in rows:
            for key in ('source_image', 'source_label', 'color_table'):
                if file_hash(Path(row[key])) != row[key + '_sha256']:
                    raise ValueError('Original training/validation data changed')


def prepare(args):
    prior = args.cohort_run_dir
    old, previous = data.read(prior / 'config.json'), data.read(prior / 'summary.json')
    if (not previous['complete'] or previous['experiment'] != 'unet_cohort_train_calibrated'
            or previous['train_patients'] != 54 or previous['validation_patients'] != 10 or previous['test_images_read']):
        raise ValueError('Completed prior54/10cohort is required as data/provenance control')
    manifest_path = Path(old['data_dir']) / 'manifest.json'
    manifest = data.read(manifest_path)
    select_patients(manifest, args.seed)
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'torchio', 'monai', 'numpy', 'SimpleITK')}
    if file_hash(manifest_path) != old['manifest_sha256'] or args.seed != old['seed'] or versions != old['versions']:
        raise ValueError('Frozen manifest/seed/package versions differ')
    if file_hash(args.checkpoint) != TURBO_SHA256:
        raise ValueError('Original Turbo checkpoint checksum differs')
    requested = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != 'resume'}
    requested.update(experiment='sam_turbo_cnn_refine32',
        code_sha256={name: file_hash(Path(__file__).with_name(name)) for name in CODE_FILES},
        sam_source_sha256=views.source_hashes(args.sam_dir), turbo_sha256=TURBO_SHA256, versions=versions,
        control_sha256={name: file_hash(prior / name) for name in ('config.json', 'summary.json', 'source_provenance.json')},
        manifest_sha256=file_hash(manifest_path),
        architecture='Turbo128context/8³tokens/32³coarse mask + raw-detail CNN32³ + physical feature fusion + full/half-resolution skips + residual logits',
        initialization='all Turbo image/prompt/mask decoder weights loaded strictly; new CNN/projection/refiner random; residual output zero initialized',
        trainability='new CNN/refiner only for warmup; then image blocks10/11 and mask decoder at sam_lr plus new modules at head_lr; frozen prompt/neck/earlier image blocks remain differentiable where needed',
        augmentation='shared physical LR reflection and shared gain0.9-1.1/shift+-0.1 only' if args.augmentation == 'light' else 'none',
        loss='FP32 valid-voxel squared DiceCE on refined detail + auxiliary_weight*valid-voxel squared DiceCE on SAM context; average across5clicks',
        prompts='fixed-seed oracle error clicks at probability0.5, valid detail voxels only; native-coordinate SAM convention; latest click and detached coarse feedback for SAM; accumulated physical click Gaussians for CNN',
        threshold_protocol='select threshold by frozen54train bilateral Dice@5; apply to validation@1/@5; val never selects threshold',
        threshold_candidates=list(data.THRESHOLDS),
        selection='native full-target patient-macro bilateral validation Dice@5per side (10total bilateral clicks)',
        protocol='GT-side-centered physically aligned SAM context/detail; shared train-derived normalization; original native grid scored',
        test_images_read=False)
    cache = args.run_dir / 'hybrid_data.pt'
    if args.resume:
        config = data.read(args.run_dir / 'config.json')
        if any(config[key] != value for key, value in requested.items()) or file_hash(cache) != config['cache_sha256']:
            raise ValueError('Resume configuration/code/source/cache differs')
        prepared = torch.load(cache, map_location='cpu', weights_only=False)
        if config['physical_policy'] != prepared['policy']:
            raise ValueError('Physical cache policy differs')
        verify_sources(prepared['source_provenance'])
    else:
        data.save(args.run_dir / 'requested_config.json', requested)
        sources = data.read(prior / 'source_provenance.json')
        prepared = data.prepare_dataset(manifest, Path(old['source_root']).resolve(), sources, args.run_dir / 'preprocessing')
        prepared = views.add_contexts(prepared, manifest, Path(old['source_root']).resolve(), sources)
        data.save(args.run_dir / 'preprocessing' / 'policy.json', prepared['policy'])
        atomic_save(cache, prepared)
        config = {**requested, 'cache_sha256': file_hash(cache), 'physical_policy': prepared['policy']}
        data.save(args.run_dir / 'config.json', config)
        data.save(args.run_dir / 'source_provenance.json', prepared['source_provenance'])
    return prepared, config


def run_sample(model, sample, device, seed, criterion=None, augment=False, auxiliary_weight=.25):
    tensors = views.paired_inputs(sample, device, seed if augment else None)
    with torch.autocast(device_type=device.type, enabled=device.type == 'cuda', dtype=getattr(model, 'amp_dtype', torch.bfloat16)):
        return interactions(model, tensors['image'], tensors['sam_image'], tensors['ratio'], tensors['spacing'],
            tensors['target'], tensors['valid'], (1, 5), seed, criterion,
            tensors['sam_target'], tensors['sam_valid'], auxiliary_weight)


def metrics(bundle):
    rows, sides = bundle['patients'], bundle['sides']
    return {'mean_bilateral_patient_dice': float(np.mean([r['dice_full_target'] for r in rows])),
            'minimum_bilateral_patient_dice': min(r['dice_full_target'] for r in rows),
            'mean_side_dice': float(np.mean([r['dice_full_target'] for r in sides])),
            'minimum_side_dice': min(r['dice_full_target'] for r in sides),
            'mean_precision': float(np.mean([r['precision'] for r in rows])),
            'mean_recall': float(np.mean([r['recall_full_target'] for r in rows]))}


@torch.inference_mode()
def evaluate(model, prepared, split, device, thresholds, seed, controls=False, export=None):
    model.eval()
    samples, patients = prepared['samples'][split], prepared['patients'][split]
    components = ('refined', 'coarse_on_detail') if controls else ('refined',)
    results = {component: {click: {float(t): {'sides': [], 'patients': []} for t in thresholds} for click in (1, 5)} for component in components}
    for alias in sorted(patients):
        collection = [s for s in samples if s['case'] == alias]
        if sorted(s['side'] for s in collection) != ['LFSDP', 'RFSDP']:
            raise ValueError('Expected two side ROIs per patient')
        records = []
        for index, sample in enumerate(collection):
            sample_seed = seed + int(alias.rsplit('_', 1)[1]) * 10 + (0 if sample['side'] == 'RFSDP' else 1)
            _, outputs = run_sample(model, sample, device, sample_seed)
            records.append({component: {click: data.restore_logits(outputs[click][component][0, 0].float().cpu().numpy(), sample)
                for click in (1, 5)} for component in components})
        for component in components:
            for click in (1, 5):
                for threshold, bundle in results[component][click].items():
                    parts = []
                    for sample, record in zip(collection, records):
                        prediction = record[component][click] > math.log(threshold / (1 - threshold))
                        valid = np.ones(prediction.shape, bool)
                        indices = native_prediction_indices(prediction, valid, sample['metadata'], patients[alias]['shape'])
                        target = native_prediction_indices(sample['native_target'], valid, sample['metadata'], patients[alias]['shape'])
                        score = sparse_scores(indices, target)
                        bundle['sides'].append({'case': alias, 'side': sample['side'], 'clicks_per_side': click,
                            'threshold': threshold, 'target_retention': sample['metadata']['target_retention'], **score})
                        parts.append(indices)
                        if export is not None and component == 'refined' and click == 5:
                            export.mkdir(exist_ok=True, parents=True)
                            save_crop(export / f'{sample["name"]}_t{threshold:g}_prediction.nii.gz', prediction.astype(np.uint8), sample['metadata'])
                            save_crop(export / f'{sample["name"]}_target.nii.gz', sample['native_target'].astype(np.uint8), sample['metadata'])
                            overlay(sample, prediction, score['dice_full_target'], threshold, export)
                    bundle['patients'].append({'case': alias, 'clicks_per_side': click, 'threshold': threshold,
                        **sparse_scores(np.concatenate(parts), patients[alias]['target_indices'])})
        print(f'{split}: {alias} hybrid inference complete', flush=True)
    for component in results.values():
        for clicks in component.values():
            for bundle in clicks.values():
                bundle['metrics'] = metrics(bundle)
    return results


def overlay(sample, prediction, dice, threshold, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
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
    fig.suptitle(f'{sample["name"]}; GT-assisted ROI; 5 oracle clicks/side; threshold{threshold:g}; Dice{dice:.4f}')
    fig.tight_layout()
    fig.savefig(output / f'{sample["name"]}.png', dpi=140)
    plt.close(fig)


def plot_results(history, training, validation, threshold, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    measured = [r for r in history if r['val_dice_5'] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, label in (('train_dice_5', 'Frozen train'), ('val_dice_5', 'Validation')):
        axes[0].plot([r['epoch'] for r in measured], [r[key] for r in measured], label=label)
    axes[0].set(xlabel='Epoch', ylabel='Native bilateral Dice', ylim=(0, 1))
    axes[0].legend()
    for split, bundle in (('Train', training), ('Validation', validation)):
        rows = bundle['refined'][5][threshold]['patients']
        axes[1].scatter(range(len(rows)), [r['dice_full_target'] for r in rows], s=15, label=split)
    axes[1].set(xlabel='Patient alias index within split', ylabel='Selected checkpoint Dice', ylim=(0, 1))
    axes[1].legend()
    fig.suptitle('GT-assisted SAM Turbo + CNN32; 5 oracle clicks per side')
    fig.tight_layout()
    fig.savefig(output / 'training_and_patient_dice.png', dpi=160)
    plt.close(fig)


def calibrate_validate(model, prepared, device, seed):
    training = evaluate(model, prepared, 'train', device, data.THRESHOLDS, seed)
    threshold = data.choose_threshold(training['refined'][5])
    validation = evaluate(model, prepared, 'val', device, sorted(set((threshold, .5))), seed, controls=True)
    return threshold, training, validation


def gradient_groups(model):
    result = {}
    for name, sam in (('turbo', True), ('new_refiner', False)):
        values = [p.grad.detach().float() for key, p in model.named_parameters()
                  if p.grad is not None and (key.startswith('sam.') == sam)]
        nonfinite = sum(int((~value.isfinite()).sum().item()) for value in values)
        result[name] = {'nonfinite_elements': nonfinite,
                        'l2_norm': sum(v.square().sum().item() for v in values) ** .5 if not nonfinite else None}
    return result


def set_stage(model, epoch, warmup):
    model.train()
    active = epoch >= warmup
    for name, parameter in model.sam.named_parameters():
        parameter.requires_grad_(active and name.startswith(('image_encoder.blocks.10.', 'image_encoder.blocks.11.', 'mask_decoder.')))
    model.sam.eval()
    if active:
        model.sam.image_encoder.blocks[10].train()
        model.sam.image_encoder.blocks[11].train()
        model.sam.mask_decoder.train()
    return 'partial_turbo_and_refiner' if active else 'refiner_warmup'


def train_epoch(model, prepared, optimizer, scaler, criterion, device, epoch, args):
    stage = set_stage(model, epoch, args.head_warmup_epochs)
    samples = prepared['samples']['train']
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
    optimizer.zero_grad(set_to_none=True)
    total, updates, skipped, probes = [], 0, 0, []
    for step, index in enumerate(order):
        loss, _ = run_sample(model, samples[index], device, args.seed + epoch * 1000 + step, criterion,
                             args.augmentation == 'light', args.auxiliary_weight)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite hybrid segmentation loss')
        window, should_step = accumulation_window(step, len(samples), args.accumulation)
        scaler.scale(loss / window).backward()
        total.append(loss.item())
        if should_step:
            scaler.unscale_(optimizer)
            if epoch == 0 and updates + skipped < 2:
                probes.append(gradient_groups(model))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm) and not scaler.is_enabled():
                raise ValueError('Nonfinite hybrid gradients')
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            success = scaler.get_scale() >= old_scale
            updates += int(success)
            skipped += int(not success)
            optimizer.zero_grad(set_to_none=True)
            print(f'epoch={epoch + 1} roi={step + 1}/{len(samples)} loss={loss.item():.6f} updates={updates} skipped={skipped}', flush=True)
    return {'stage': stage, 'loss': float(np.mean(total)), 'optimizer_updates': updates, 'amp_skipped_updates': skipped,
            'expected_updates': math.ceil(len(samples) / args.accumulation), 'gradient_probes_first_epoch': probes}


def execute(args):
    from monai.losses import DiceCELoss
    prepared, config = prepare(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    hardware = {'device': str(device), 'precision': args.precision}
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        hardware.update(gpu=properties.name, total_vram_gib=properties.total_memory / 2**30)
    data.save(args.run_dir / 'hardware.json', hardware)
    print(json.dumps({'hardware': hardware}), flush=True)
    model, turbo_hash = build(args)
    model = model.to(device)
    model.amp_dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16
    initial_hash = state_hash(model.state_dict())
    sam_parameters = list(model.sam.parameters())
    new_parameters = [p for name, p in model.named_parameters() if not name.startswith('sam.')]
    optimizer = torch.optim.AdamW([{'params': sam_parameters, 'lr': args.sam_lr, 'weight_decay': .01},
                                  {'params': new_parameters, 'lr': args.head_lr, 'weight_decay': 1e-4}])
    def schedule(epoch):
        warmup = min(5, args.epochs)
        if epoch < warmup:
            return (epoch + 1) / warmup
        fraction = min(1., (epoch - warmup) / max(1, args.epochs - warmup))
        return .05 + .95 * .5 * (1 + math.cos(math.pi * fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and args.precision == 'fp16')
    criterion = DiceCELoss(sigmoid=True, squared_pred=True)
    run = args.run_dir
    start, best, best_epoch, best_threshold, history, initial = 0, -1., 0, .5, [], None
    early = {'anchor': -1., 'last_improvement_epoch': 0}
    if args.resume:
        saved = torch.load(run / 'hybrid_latest.pth', map_location='cpu', weights_only=False)
        if saved['config'] != config or saved['initial_state_sha256'] != initial_hash:
            raise ValueError('Resume checkpoint/config/initialization mismatch')
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
                'config': config, 'initial_state_sha256': initial_hash, 'initial_turbo_state_sha256': turbo_hash,
                'history': history, 'initial_metrics': initial, 'early_state': early, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(), 'torch_rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(), 'python_rng_state': random.getstate(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
    if not args.resume:
        threshold, training, validation = calibrate_validate(model, prepared, device, args.seed)
        # Zero-initialized residual head must reproduce mapped original SAM logits initially.
        for clicks in (1, 5):
            a = validation['refined'][clicks][threshold]['metrics']['mean_bilateral_patient_dice']
            b = validation['coarse_on_detail'][clicks][threshold]['metrics']['mean_bilateral_patient_dice']
            if abs(a - b) > 1e-6:
                raise ValueError('Initial zero residual does not reproduce original SAM component')
        best_threshold = threshold
        best = validation['refined'][5][threshold]['metrics']['mean_bilateral_patient_dice']
        early = {'anchor': best, 'last_improvement_epoch': 0}
        initial = {'threshold_from_train': threshold, 'train': training['refined'][5][threshold]['metrics'],
                   'val': validation['refined'][5][threshold]['metrics']}
        data.save(run / 'preflight.json', {'passed': True, 'original_turbo_weights_unchanged_on_wrap': True,
            'initial_turbo_state_sha256': turbo_hash, 'detail_feature_grid': [32] * 3, 'sam_token_grid': [8] * 3,
            'initial_component_reproduces': True, 'initial_metrics': initial, 'coverage': prepared['coverage'],
            'parameter_counts': {'turbo': sum(p.numel() for p in sam_parameters), 'new': sum(p.numel() for p in new_parameters)},
            'test_images_read': False})
        atomic_save(run / 'hybrid_best.pth', state(0))
        atomic_save(run / 'hybrid_latest.pth', state(0))
    stopped = args.patience_epochs > 0 and start - early['last_improvement_epoch'] >= args.patience_epochs
    for epoch in range(start, args.epochs) if not stopped else []:
        lrs = [g['lr'] for g in optimizer.param_groups]
        measured = train_epoch(model, prepared, optimizer, scaler, criterion, device, epoch, args)
        if measured['optimizer_updates'] + measured['amp_skipped_updates'] != measured['expected_updates']:
            raise ValueError('Incomplete optimizer accumulation')
        scheduler.step()
        row = {'epoch': epoch + 1, 'sam_lr': lrs[0], 'head_lr': lrs[1], **measured,
               'train_threshold': None, 'train_dice_5': None, 'val_dice_1': None, 'val_dice_5': None, 'val_control_0_5': None}
        improved = False
        if epoch == 0 or (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            threshold, training, validation = calibrate_validate(model, prepared, device, args.seed)
            score = validation['refined'][5][threshold]['metrics']['mean_bilateral_patient_dice']
            row.update(train_threshold=threshold, train_dice_5=training['refined'][5][threshold]['metrics']['mean_bilateral_patient_dice'],
                val_dice_1=validation['refined'][1][threshold]['metrics']['mean_bilateral_patient_dice'], val_dice_5=score,
                val_control_0_5=validation['refined'][5][.5]['metrics']['mean_bilateral_patient_dice'])
            data.save(run / 'train_threshold_curve.json', {str(t): r['metrics'] for t, r in training['refined'][5].items()})
            write_csv(run / 'validation_latest_patients.csv', validation['refined'][5][threshold]['patients'])
            early, stopped = early_stop_update(early, score, epoch + 1, args.patience_epochs, args.min_delta)
            improved = score > best
            if improved:
                best, best_epoch, best_threshold = score, epoch + 1, threshold
        history.append(row)
        write_csv(run / 'history.csv', history)
        data.save(run / 'progress.json', {'complete': False, 'completed_epochs': epoch + 1, 'planned_epochs': args.epochs,
            'best_epoch': best_epoch, 'best_validation_dice_5_per_side': best, 'best_train_threshold': best_threshold,
            'stopped_early': stopped, 'latest': row})
        if improved:
            atomic_save(run / 'hybrid_best.pth', state(epoch + 1))
        atomic_save(run / 'hybrid_latest.pth', state(epoch + 1))
        print(json.dumps({'epoch_result': row, 'best_validation_dice_5_per_side': best}), flush=True)
        if stopped:
            break
    saved = torch.load(run / 'hybrid_best.pth', map_location='cpu', weights_only=False)
    if saved['epoch'] != best_epoch or saved['best_score'] != best or saved['best_threshold'] != best_threshold or saved['config'] != config:
        raise ValueError('Selected hybrid checkpoint metadata differs')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    del saved
    threshold, training, validation = calibrate_validate(model, prepared, device, args.seed)
    if threshold != best_threshold or abs(validation['refined'][5][threshold]['metrics']['mean_bilateral_patient_dice'] - best) > 1e-4:
        raise ValueError('Selected hybrid checkpoint/threshold does not reproduce')
    output, number = run / 'best_exports', 0
    while output.exists():
        number += 1
        output = run / f'best_exports_resume_{number}'
    output.mkdir()
    for split, result in (('train', training), ('val', validation)):
        for click in (1, 5):
            bundle = result['refined'][click][threshold]
            write_csv(output / f'{split}_{click}_sides.csv', bundle['sides'])
            write_csv(output / f'{split}_{click}_patients.csv', bundle['patients'])
    plot_results(history, training, validation, threshold, output)
    data.save(output / 'selected_train_threshold_curve.json', {str(t): r['metrics'] for t, r in training['refined'][5].items()})
    exported = evaluate(model, prepared, 'val', device, [threshold], args.seed, export=output / 'overlays')
    original_rows = {r['case']: r['dice_full_target'] for r in validation['refined'][5][threshold]['patients']}
    if any(abs(r['dice_full_target'] - original_rows[r['case']]) > 1e-4 for r in exported['refined'][5][threshold]['patients']):
        raise ValueError('Exported predictions do not reproduce selected patient scores')
    summary = {'complete': True, 'experiment': config['experiment'], 'completed_epochs': history[-1]['epoch'] if history else start,
        'planned_epochs': args.epochs, 'stopped_early': stopped, 'best_epoch': best_epoch, 'threshold': threshold,
        'train_patients': 54, 'validation_patients': 10, 'reserved_test_patients': 20,
        'best_validation_metrics': {str(k): validation['refined'][k][threshold]['metrics'] for k in (1, 5)},
        'best_frozen_training_metrics': {str(k): training['refined'][k][threshold]['metrics'] for k in (1, 5)},
        'sam_component_same_refined_click_trace': {str(k): validation['coarse_on_detail'][k][threshold]['metrics'] for k in (1, 5)},
        'validation_control_fixed_0_5': validation['refined'][5][.5]['metrics'],
        'train_validation_gap_5': training['refined'][5][threshold]['metrics']['mean_bilateral_patient_dice'] - best,
        'turbo_checkpoint_sha256': TURBO_SHA256, 'initial_turbo_state_sha256': turbo_hash,
        'best_checkpoint_sha256': file_hash(run / 'hybrid_best.pth'), 'detail_feature_grid': [32] * 3, 'sam_token_grid': [8] * 3,
        'physical_policy': prepared['policy'], 'coverage': prepared['coverage'],
        'training_stages': {'head_only_epochs': args.head_warmup_epochs, 'then': 'image_encoder.blocks.10/11 and mask_decoder plus CNN refiner'},
        'oracle_clicks_per_side': [1, 5], 'bilateral_total_clicks': {'1': 2, '5': 10},
        'validation_images_read': True, 'test_images_read': False, 'protocol': config['protocol'],
        'interpretation': 'GT-assisted interactive hybrid validation.32³is learned CNN detail fusion, not32768SAM transformer tokens. Coarse component uses the refined model click trace and is not an independently trained/queried vanilla-SAM control. No80% generalization claim.',
        'artifacts': str(run), 'export_directory': str(output)}
    archive = run / 'results.tar.gz'
    summary['archive'] = str(archive)
    data.save(output / 'summary.json', summary)
    with tarfile.open(archive, 'w:gz') as package:
        package.add(output, arcname='results')
        for name in ('history.csv', 'config.json', 'preflight.json'):
            package.add(run / name, arcname='results/' + name)
    data.save(run / 'summary.json', summary)
    data.save(run / 'progress.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sam-dir', 'checkpoint', 'cohort-run-dir', 'run-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--head-warmup-epochs', type=int, default=5)
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--sam-lr', type=float, default=1e-5)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--auxiliary-weight', type=float, default=.25)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--patience-epochs', type=int, default=50)
    parser.add_argument('--min-delta', type=float, default=.001)
    parser.add_argument('--augmentation', choices=('none', 'light'), default='light')
    parser.add_argument('--precision', choices=('bf16', 'fp16'), default='bf16')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    if (args.head_warmup_epochs < 0 or min(args.epochs, args.val_every, args.accumulation) < 1 or args.patience_epochs < 0
            or not all(np.isfinite(x) and x > 0 for x in (args.sam_lr, args.head_lr))
            or not np.isfinite(args.auxiliary_weight) or args.auxiliary_weight < 0
            or not np.isfinite(args.min_delta) or args.min_delta < 0):
        parser.error('Invalid training settings')
    old = data.read(args.cohort_run_dir / 'config.json')
    if any(args.run_dir.is_relative_to(p) for p in (args.sam_dir, args.cohort_run_dir, Path(old['source_root']).resolve(), Path(old['data_dir']).resolve())):
        parser.error('Use an output directory outside original source/data/control')
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    if torch.device(args.device).type == 'cuda' and args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        parser.error('Selected GPU does not support BF16; use--precision fp16 in a new run')
    if args.resume:
        if not (args.run_dir / 'hybrid_latest.pth').exists() or (args.run_dir / 'summary.json').exists():
            parser.error('Resume requires an unfinished hybrid run/latest checkpoint')
    elif args.run_dir.exists():
        parser.error('Run directory already exists')
    else:
        args.run_dir.mkdir(parents=True)
    try:
        execute(args)
    except Exception as error:
        data.save(args.run_dir / 'failure.json', {'error_type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
