#!/usr/bin/env python3
"""Full physical-ROI ResUNet baseline, zero clicks, train-only threshold calibration."""
import argparse
import fcntl
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import tarfile

import numpy as np
import torch

import physical_fsdp as data
from evaluate_private_validation import file_hash, save_crop, write_csv
from fsdp_verified_cache import validate_targets, image_inventory, require_images
from physical_resunet import PhysicalResUNet, ARCHITECTURE, supervised_loss, training_input
from train_fsdp_foundation import inspect_foundation
from train_fsdp_research import allocator_limit
from train_private_long import atomic_save, accumulation_window, early_stop_update
from train_private_tiny import native_prediction_indices, sparse_scores
from train_private_unet import state_hash


def prepare(args):
    old, _ = inspect_foundation(args.foundation_run_dir, args.source_root, verified_cache=True)
    prior = data.read(Path(old['cohort_run_dir']) / 'config.json')
    comparison, sam = [data.read(args.sam_comparison_dir / name) for name in ('config.json', 'summary.json')]
    if (not sam['complete'] or sam['test_images_read'] or sam['train_patients'] != 54 or sam['validation_patients'] != 10
            or any(comparison[k] != old[k] for k in ('cache_sha256', 'manifest_sha256', 'physical_policy'))
            or file_hash(args.sam_comparison_dir / 'research_best.pth') != sam['best_checkpoint_sha256']):
        raise ValueError('SAM benchmark must use the same complete54/10physical cache and unchanged checkpoint')
    protected = [args.foundation_run_dir, args.sam_comparison_dir, Path(old['sam_dir']), Path(old['cohort_run_dir']),
                 Path(prior['data_dir']), Path(prior['source_root'])]
    if args.source_root:
        protected.append(args.source_root)
    if any(args.run_dir.is_relative_to(p.resolve()) for p in protected):
        raise ValueError('Outputs must be outside data/model/source directories')
    ancestor = args.run_dir.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 8 * 2**30:
        raise ValueError('Keep at least8GiBfree on the output filesystem')
    prepared = torch.load(args.foundation_run_dir / 'hybrid_data.pt', map_location='cpu', weights_only=False)
    integrity = validate_targets(prepared, data.read(Path(prior['data_dir']) / 'manifest.json'))
    inventory = image_inventory(prepared['source_provenance'], args.source_root, verify=False)
    require_images(inventory)
    code_files = ('physical_resunet.py', 'train_physical_resunet.py', 'physical_fsdp.py', 'fsdp_verified_cache.py',
                  'train_private_tiny.py', 'train_private_long.py', 'evaluate_private_validation.py')
    use25d = getattr(args, 'neighbor_slices', None) is not None
    if use25d:
        prepared['patients']['train'].pop('train_28')
        prepared['samples']['train'] = [s for s in prepared['samples']['train'] if s['case'] != 'train_28']
        for split in ('train', 'val'):
            for sample in prepared['samples'][split]:
                raw = np.asarray(sample['raw_native'], dtype=np.float32)
                if not np.isfinite(raw).all() or raw.std() < 1e-6:
                    raise ValueError('Invalid native CT ROI')
                sample['image'] = torch.from_numpy((raw - raw.mean()) / raw.std())[None, None]
                sample['target'] = torch.from_numpy(sample['native_target'].astype(np.uint8))[None, None]
                sample['valid'] = torch.ones_like(sample['target'], dtype=torch.bool)
                sample['grid'] = sample['native']
        code_files = (*code_files, 'physical_unet25d.py', 'fsdp_research_metrics.py')
        if getattr(args, 'slice_loss', None):
            code_files = (*code_files, 'fsdp_brain_recipe.py')
        if getattr(args,'rotation_backbone',None):
            code_files = (*code_files, 'fsdp_c8_model.py')
    config = {'experiment': 'physical_resunet_zero_clicks', 'architecture': ARCHITECTURE,
        'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'},
        'cache_sha256': old['cache_sha256'], 'manifest_sha256': old['manifest_sha256'], 'physical_policy': prepared['policy'],
        'versions': {k: importlib.metadata.version(k) for k in ('torch', 'numpy', 'SimpleITK')},
        'code_sha256': {name: file_hash(Path(__file__).with_name(name)) for name in code_files},
        'sam_comparison': {'directory': str(args.sam_comparison_dir), 'summary_sha256': file_hash(args.sam_comparison_dir / 'summary.json'),
            'checkpoint_sha256': sam['best_checkpoint_sha256'],
            'dice_5_per_side': sam['best_validation_metrics']['5']['mean_bilateral_patient_dice']},
        'protocol': 'Same GT-assisted physical ROI/cache and54/10patient split;0oracle clicks; full-target native Dice; train-only threshold.',
        'loss': 'soft Dice +0.5class-balanced BCE; deep outputs1/0.3/0.15upsampled as logits; GT never downsampled',
        'augmentation': '50%full ROI/50%patch;67%patches include sampled train foreground; paired LR reflection, mild gain/shift and noise; no sharpening',
        'patch_scoring': 'Train patches may truncate targets; validation always full unchanged ROI and full original target.',
        'comparison_note': 'New architecture/random initialization/loss/recipe and0clicks versus SAM5/side; benchmark, not one-variable ablation or full-scan automatic localization.',
        'cache_integrity': integrity, 'test_images_read': False}
    if use25d:
        config.update(experiment='native_unet25d_zero_clicks', architecture={
            'channels': [16,32,64,128], 'neighbors': args.neighbor_slices, 'axis': 'native voxel Z; may be oblique',
            'edge_policy': 'repeat boundary slices', 'initialization': 'random; no SAM weights',
            'batch_slice_chunks': 8, 'deep_supervision': [1.,.3,.15]},
            protocol='53train/10val; native GT ROI; zero clicks; native volume reconstruction; train-only threshold',
            normalization='per native ROI zscore without foreground clipping; per-case unsupervised statistics',
            slice_spacing_mm={s['name']: s['native']['spacing'][2] for split in ('train','val') for s in prepared['samples'][split]},
            comparison_note='1vs5slice paired architecture experiment; different data/recipe/prompts from SAM; GTROI is not automatic localization')
    if getattr(args, 'slice_loss', None):
        config.update(experiment='brain_recipe_five_slice', slice_loss=args.slice_loss,
            training_sampler='All native center slices shuffled within exact XY-shape groups; batches shuffled; no cross-shape padding; negatives retained',
            loss=('existing volume Dice+balancedBCE on slice minibatches' if args.slice_loss == 'control' else
                  'per-slice0.5focal_gamma2+2Dice+0.5normalized_edgeBCE; physical1mmrectangular half-width band; no10xpositiveweight'),
            paired_design='Same random initialization/seed/slices/optimizer/schedule/budget; only loss changes between new arms',
            scope='Brain-inspired sampling/loss; residualConv2Dbackbone; C8not enabled')
    if getattr(args,'rotation_backbone',None):
        config.update(experiment='brain_c8_five_slice',
            architecture={'backbone':args.rotation_backbone,'group_order':8 if args.rotation_backbone=='c8' else None,
                          'regular_fields':[args.c8_base_fields*2**i for i in range(5)],
                          'scalar_channels':[8*args.c8_base_fields*2**i for i in range(5)],
                          'downsampling_stages':4,'normalization':'InnerBatchNorm(C8) or BatchNorm2D(plain)',
                          'deep_supervision':[1.,.3,.15],'initialization':'random; noSAMorBraincheckpoint'},
            scope='Four-level Brain-style plain/C8comparison; same five-slice inputs/loss; noSAMprompts',
            comparison_note='Matched four-level plain/C8study;five slices;GTROI;0clicks;random2Dweights;notSAMfinetuning;different group constraints/parametercounts',
            paired_design='Same scalar feature widths/topology/data/loss/budget; different group constraints and parameter counts')
        if args.rotation_backbone=='c8':
            config['versions']['escnn']=importlib.metadata.version('escnn')
            for package in ('scipy','lie-learn','pymanopt','autograd','py3nj'):
                try:
                    config['versions'][package]=importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    config['versions'][package]=None
    if args.resume:
        if data.read(args.run_dir / 'config.json') != config or (args.run_dir / 'summary.json').exists():
            raise ValueError('Resume requires unchanged unfinished baseline configuration')
    else:
        args.run_dir.mkdir(parents=True, exist_ok=False)
        data.save(args.run_dir / 'config.json', config)
    return prepared, config


def metrics(bundle):
    patients, sides = bundle['patients'], bundle['sides']
    return {'mean_bilateral_patient_dice': float(np.mean([p['dice_full_target'] for p in patients])),
        'minimum_bilateral_patient_dice': min(p['dice_full_target'] for p in patients),
        'mean_side_dice': float(np.mean([p['dice_full_target'] for p in sides])),
        'minimum_side_dice': min(p['dice_full_target'] for p in sides),
        'mean_precision': float(np.mean([p['precision'] for p in patients])),
        'mean_recall': float(np.mean([p['recall_full_target'] for p in patients]))}


def overlay(sample, prediction, dice, threshold, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    for row, axis in enumerate((2, 1, 0)):
        index = int(sample['native_target'].sum(axis=tuple(i for i in range(3) if i != axis)).argmax())
        ct, gt, pred = [np.take(value, index, axis=axis).T for value in (sample['raw_native'], sample['native_target'], prediction)]
        remaining = [i for i in range(3) if i != axis]
        aspect = sample['native']['spacing'][remaining[1]] / sample['native']['spacing'][remaining[0]]
        for col, ax in enumerate(axes[row]):
            ax.imshow(ct, cmap='gray', vmin=-1000, vmax=1000, origin='lower', aspect=aspect)
            for mask, color, enabled in ((gt, 'cyan', col != 1), (pred, 'magenta', col != 0)):
                if enabled and mask.any() and not mask.all():
                    ax.contour(mask, levels=[.5], colors=[color], linewidths=.7, origin='lower')
                elif enabled and mask.all():
                    from matplotlib.colors import ListedColormap
                    ax.imshow(mask, cmap=ListedColormap([color]), alpha=.3, origin='lower', aspect=aspect)
            ax.set_title(f'Axis{axis}, slice{index}: ' + ('GT', 'prediction', 'GT+prediction')[col], fontsize=8)
            ax.axis('off')
    fig.suptitle(f'{sample["name"]}; physical ROI; GT-assisted ROI;0clicks; t={threshold:g}; Dice={dice:.4f}')
    fig.tight_layout(); fig.savefig(output / f'{sample["name"]}.png', dpi=130); plt.close(fig)


@torch.inference_mode()
def evaluate(model, prepared, split, device, thresholds, precision, export=None):
    model.eval()
    result = {float(t): {'patients': [], 'sides': []} for t in thresholds}
    if export:
        export.mkdir(parents=True, exist_ok=True)
    for case, patient in sorted(prepared['patients'][split].items()):
        collection = [s for s in prepared['samples'][split] if s['case'] == case]
        if sorted(s['side'] for s in collection) != ['LFSDP', 'RFSDP']:
            raise ValueError('Two sides are required per patient')
        parts = {t: [] for t in result}
        for sample in collection:
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda', dtype=precision):
                logits = model(sample['image'].to(device))[0]
            if logits.shape != sample['image'].shape or not torch.isfinite(logits).all():
                raise ValueError('Invalid full ROI output')
            restored = data.restore_logits(logits[0, 0].float().cpu().numpy(), sample)
            for threshold, bundle in result.items():
                prediction = restored > math.log(threshold / (1 - threshold))
                indices = native_prediction_indices(prediction, np.ones(prediction.shape, bool), sample['metadata'], patient['shape'])
                target = native_prediction_indices(sample['native_target'], np.ones(prediction.shape, bool), sample['metadata'], patient['shape'])
                score = sparse_scores(indices, target)
                bundle['sides'].append({'case': case, 'side': sample['side'], 'threshold': threshold, 'oracle_clicks': 0, **score})
                parts[threshold].append(indices)
                if export:
                    path = export / f'{sample["name"]}_t{threshold:g}_prediction.nii.gz'
                    save_crop(path, prediction.astype(np.uint8), sample['metadata'])
                    saved = data.sitk.ReadImage(str(path))
                    if not np.array_equal(data.xyz(saved), prediction):
                        raise ValueError('Saved native prediction differs')
                    for key, value in data.geometry(saved).items():
                        if not np.allclose(value, sample['native'][key], atol=1e-5, rtol=2e-7):
                            raise ValueError('Saved native geometry differs')
                    save_crop(export / f'{sample["name"]}_target.nii.gz', sample['native_target'].astype(np.uint8), sample['metadata'])
                    overlay(sample, prediction, score['dice_full_target'], threshold, export)
        for threshold, bundle in result.items():
            prediction = np.concatenate(parts[threshold])
            record = {'case': case, 'threshold': threshold, 'oracle_clicks': 0,
                **sparse_scores(prediction, patient['target_indices'])}
            if hasattr(model, 'neighbor_slices') and split == 'val':
                from fsdp_research_metrics import surface_metrics
                record.update(surface_metrics(prediction, patient['target_indices'], patient['shape'], collection[0]['native']['spacing']))
            bundle['patients'].append(record)
        label = f'UNet25D[{model.neighbor_slices}]' if hasattr(model, 'neighbor_slices') else 'ResUNet'
        print(f'{split}: {case} {label} native inference complete', flush=True)
    for threshold, bundle in result.items():
        bundle['metrics'] = metrics(bundle)
        tp = sum(p['true_positive_voxels'] for p in bundle['patients'])
        denominator = sum(p['predicted_voxels_valid']+p['full_target_voxels'] for p in bundle['patients'])
        bundle['metrics']['global_dice'] = 2*tp/denominator
        if hasattr(model, 'neighbor_slices') and split == 'val':
            for key in ('surface_dice_0_5mm', 'surface_dice_1mm'):
                bundle['metrics']['mean_' + key] = float(np.mean([p[key] for p in bundle['patients']]))
            hd = [p['hd95_mm'] for p in bundle['patients'] if p['hd95_mm'] is not None]
            bundle['metrics']['median_hd95_mm'] = float(np.median(hd)) if hd else None
        if export:
            write_csv(export / f'{split}_t{threshold:g}_patients.csv', bundle['patients'])
            write_csv(export / f'{split}_t{threshold:g}_sides.csv', bundle['sides'])
    return result


def train_epoch(model, samples, optimizer, scaler, device, precision, epoch, args):
    if getattr(args, 'slice_loss', None):
        from fsdp_brain_recipe import train_slices
        return train_slices(model,samples,optimizer,scaler,device,precision,epoch,args)
    model.train(); optimizer.zero_grad(set_to_none=True)
    order = np.random.default_rng(args.seed + epoch).permutation(len(samples))
    losses, norms, counts, updates, skips = [], [], {}, 0, 0
    for step, index in enumerate(order):
        (image, target, valid), record = training_input(samples[index], device, args.seed + epoch * 1000 + step)
        counts[record['view']] = counts.get(record['view'], 0) + 1
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda', dtype=precision):
            outputs = model(image)
        loss = supervised_loss(outputs, target, valid)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite ResUNet loss')
        window, update = accumulation_window(step, len(samples), args.accumulation)
        scaler.scale(loss / window).backward(); losses.append(float(loss.detach()))
        if update:
            scaler.unscale_(optimizer)
            length = torch.nn.utils.clip_grad_norm_(model.parameters(), 12.)
            if not torch.isfinite(length) and not scaler.is_enabled():
                raise ValueError('Nonfinite ResUNet gradient')
            norms.append(float(length)); old_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            success = scaler.get_scale() >= old_scale
            updates += int(success); skips += int(not success)
            optimizer.zero_grad(set_to_none=True)
        label = f'UNet25D[{model.neighbor_slices}]' if hasattr(model, 'neighbor_slices') else 'ResUNet'
        print(f'{label} epoch={epoch + 1}/{args.epochs} roi={step + 1}/{len(samples)} loss={loss.item():.5f} updates={updates}', flush=True)
    return {'loss': float(np.mean(losses)), 'optimizer_updates': updates, 'amp_skipped_updates': skips,
            'view_counts': counts, 'gradient_norm_mean': float(np.mean(norms))}


def execute(args):
    prepared, config = prepare(args)
    with (args.run_dir / 'writer.lock').open('a') as writer_lock:
        try:
            fcntl.flock(writer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another baseline process is writing this run')
        fit(args, prepared, config)


def fit(args, prepared, config):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device); allocator_limit(device)
    precision = torch.bfloat16 if args.precision == 'bf16' else torch.float16
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    if getattr(args,'rotation_backbone',None):
        from fsdp_c8_model import BrainUNet25D
        model=BrainUNet25D(args.rotation_backbone,args.c8_base_fields).to(device)
    elif getattr(args, 'neighbor_slices', None) is not None:
        from physical_unet25d import PhysicalUNet25D
        model = PhysicalUNet25D(args.neighbor_slices).to(device)
    else:
        model = PhysicalResUNet().to(device)
    initial_hash = state_hash(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    def schedule(epoch):
        warmup = min(10, args.epochs)
        if epoch < warmup:
            return (epoch + 1) / warmup
        return .01 + .99 * .5 * (1 + math.cos(math.pi * min(1., (epoch - warmup) / max(1, args.epochs - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and args.precision == 'fp16')
    start, best, best_epoch, threshold, history = 0, -1., 0, .5, []
    early = {'anchor': -1., 'last_improvement_epoch': 0}
    if args.resume:
        saved = torch.load(args.run_dir / 'resunet_latest.pth', map_location='cpu', weights_only=False)
        if saved['config'] != config or saved['initial_hash'] != initial_hash:
            raise ValueError('Checkpoint config or initialization differs')
        for obj, key in ((model, 'model'), (optimizer, 'optimizer'), (scheduler, 'scheduler'), (scaler, 'scaler')):
            obj.load_state_dict(saved[key])
        start, best, best_epoch, threshold, history, early = [saved[k] for k in ('epoch', 'best', 'best_epoch', 'threshold', 'history', 'early')]
        torch.set_rng_state(saved['torch_rng']); np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    else:
        model.train(); sample = prepared['samples']['train'][0]
        initial_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        (image, target, valid), _ = training_input(sample, device, args.seed, augment=False)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda', dtype=precision):
            outputs = model(image)
        loss = supervised_loss(outputs, target, valid); loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or any(not g.isfinite().all() for g in gradients) or sum(float(g.abs().sum()) for g in gradients) == 0:
            raise ValueError('Native-shape backward preflight failed')
        model.zero_grad(set_to_none=True)
        model.load_state_dict(initial_state,strict=True)
        del initial_state
        if state_hash(model.state_dict()) != initial_hash:
            raise ValueError('Gradient preflight changed initialization')
        data.save(args.run_dir / 'preflight.json', {'passed': True, 'optimizer_updates': 0, 'native_shape': list(image.shape),
            'parameters': sum(p.numel() for p in model.parameters()), 'initial_hash': initial_hash})
        del image, target, valid, outputs, loss, gradients
    def checkpoint(epoch):
        return {'config': config, 'initial_hash': initial_hash, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch, 'best': best, 'best_epoch': best_epoch,
            'threshold': threshold, 'history': history, 'early': early, 'torch_rng': torch.get_rng_state(),
            'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
    stopped = start - early['last_improvement_epoch'] >= args.patience
    for epoch in range(start, args.epochs) if not stopped else []:
        row = {'epoch': epoch + 1, 'lr': optimizer.param_groups[0]['lr'],
               **train_epoch(model, prepared['samples']['train'], optimizer, scaler, device, precision, epoch, args),
               'train_dice': None, 'val_dice': None, 'threshold': None}
        scheduler.step(); improved = False
        if epoch == 0 or (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            train = evaluate(model, prepared, 'train', device, data.THRESHOLDS, precision)
            candidate = data.choose_threshold(train)
            val = evaluate(model, prepared, 'val', device, sorted(set((candidate, .5))), precision)
            score = val[candidate]['metrics']['mean_bilateral_patient_dice']
            row.update(train_dice=train[candidate]['metrics']['mean_bilateral_patient_dice'], val_dice=score, threshold=candidate)
            row['val_global_dice'] = val[candidate]['metrics']['global_dice']
            if hasattr(model, 'neighbor_slices'):
                row.update({key: value for key, value in val[candidate]['metrics'].items() if 'surface' in key or 'hd95' in key})
            early, stopped = early_stop_update(early, score, epoch + 1, args.patience, .001)
            improved = score > best
            if improved:
                best, best_epoch, threshold = score, epoch + 1, candidate
        history.append(row); write_csv(args.run_dir / 'history.csv', history)
        progress = {'complete': False, 'completed_epochs': epoch + 1, 'best_epoch': best_epoch, 'best_validation_dice': best, 'latest': row}
        data.save(args.run_dir / 'progress.json', progress)
        if improved:
            atomic_save(args.run_dir / 'resunet_best.pth', checkpoint(epoch + 1))
        atomic_save(args.run_dir / 'resunet_latest.pth', checkpoint(epoch + 1))
        print(json.dumps(progress), flush=True)
        if stopped:
            break
    saved = torch.load(args.run_dir / 'resunet_best.pth', map_location='cpu', weights_only=False)
    if saved['config'] != config or saved['best'] != best:
        raise ValueError('Selected checkpoint differs')
    model.load_state_dict(saved['model']); output = args.run_dir / 'best_exports'
    val = evaluate(model, prepared, 'val', device, [threshold], precision, output)
    if abs(val[threshold]['metrics']['mean_bilateral_patient_dice'] - best) > 1e-4:
        raise ValueError('Native selected exports fail score reproduction')
    fixed = evaluate(model, prepared, 'val', device, [.5], precision)
    train = evaluate(model, prepared, 'train', device, [threshold], precision)
    for split, bundle in (('train', train), ('val', val)):
        write_csv(output / f'{split}_patients.csv', bundle[threshold]['patients'])
        write_csv(output / f'{split}_sides.csv', bundle[threshold]['sides'])
    plot(history, train[threshold], val[threshold], output)
    summary = {'complete': True, 'experiment': config['experiment'], 'completed_epochs': len(history), 'planned_epochs': args.epochs,
        'stopped_early': stopped, 'best_epoch': best_epoch, 'threshold_train_only': threshold,
        'best_validation_metrics': val[threshold]['metrics'], 'best_frozen_training_metrics': train[threshold]['metrics'],
        'validation_fixed_0_5': fixed[.5]['metrics'], 'oracle_clicks': 0,
        'train_patients': len(prepared['patients']['train']), 'validation_patients': len(prepared['patients']['val']),
        'test_images_read': False, 'checkpoint_sha256': file_hash(args.run_dir / 'resunet_best.pth'),
        'sam_comparison': config['sam_comparison'], 'delta_vs_sam_5_click_benchmark': best - config['sam_comparison']['dice_5_per_side'],
        'protocol': config['protocol'], 'interpretation': config['comparison_note'], 'artifacts': str(args.run_dir)}
    if getattr(args, 'slice_loss', None):
        summary.update(slice_loss=args.slice_loss, backbone=getattr(args,'rotation_backbone',None) or 'ResidualConv2D; noC8', target_validation_dice=.8)
    data.save(args.run_dir / 'report.json', summary)
    with tarfile.open(args.run_dir / 'results.tar.gz', 'w:gz') as archive:
        for name in ('report.json', 'config.json', 'history.csv', 'preflight.json', 'best_exports'):
            archive.add(args.run_dir / name, arcname='resunet/' + name)
    data.save(args.run_dir / 'summary.json', summary); data.save(args.run_dir / 'progress.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def plot(history, training, validation, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    measured = [r for r in history if r['val_dice'] is not None]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot([r['epoch'] for r in history], [r['loss'] for r in history])
    axes[0].set(xlabel='Epoch', ylabel='Moving augmented training loss')
    for key, label in (('train_dice', 'Frozen train'), ('val_dice', 'Validation')):
        axes[1].plot([r['epoch'] for r in measured], [r[key] for r in measured], label=label)
    axes[1].set(xlabel='Epoch', ylabel='Native patient-macro Dice', ylim=(0, 1)); axes[1].legend()
    for bundle, label in ((training, 'Train'), (validation, 'Validation')):
        axes[2].scatter(range(len(bundle['patients'])), [p['dice_full_target'] for p in bundle['patients']], label=label, s=15)
    axes[2].set(xlabel='Patient index within split', ylabel='Selected checkpoint Dice', ylim=(0, 1)); axes[2].legend()
    fig.suptitle('Physical ROI segmentation; GT-assisted ROI;0oracle clicks')
    fig.tight_layout(); fig.savefig(output / 'training_and_patient_dice.png', dpi=140); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('foundation-run-dir', 'sam-comparison-dir', 'run-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--rotation-backbone',choices=('plain4','c8'))
    parser.add_argument('--c8-base-fields',type=int,default=2)
    parser.add_argument('--slice-loss', choices=('control','focal_boundary'), help='Enable Brain-inspired native slice minibatches')
    parser.add_argument('--slice-batch-size', type=int, default=4)
    parser.add_argument('--neighbor-slices', type=int, choices=(1, 5), help='Enable native 2.5D (5) or matched slice control (1)')
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=60)
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--accumulation', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--precision', choices=('bf16', 'fp16'), default='bf16')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.device.startswith('cuda') and (not torch.cuda.is_available() or (args.precision == 'bf16' and not torch.cuda.is_bf16_supported())):
        parser.error('Requested CUDA/BF16 device unavailable')
    if min(args.epochs, args.patience, args.val_every, args.accumulation) < 1 or not math.isfinite(args.lr) or args.lr <= 0 or args.seed != 2026:
        parser.error('Invalid budget/LR or non-frozen patient seed')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if args.slice_loss and (args.neighbor_slices != 5 or args.slice_batch_size < 1):
        parser.error('Slice recipe requires five neighboring slices and positive batch size')
    if args.rotation_backbone and (not args.slice_loss or args.neighbor_slices!=5 or not 1<=args.c8_base_fields<=32):
        parser.error('Rotation study requires five-slice recipe and1..32base fields')
    existed = args.run_dir.exists()
    try:
        execute(args)
    except Exception as error:
        if not existed and args.run_dir.is_dir():
            data.save(args.run_dir / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
