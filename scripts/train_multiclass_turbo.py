#!/usr/bin/env python3
"""Sequential high-resolution all-structure experiments on the frozen private cohort."""
import argparse
from functools import lru_cache
import importlib.metadata
import json
import math
from pathlib import Path
import random
import tarfile

import numpy as np
import torch
from torch.nn import functional as F

import multiclass_data as data
import physical_fsdp as physical
from multiclass_model import MulticlassTurbo, configure_encoder_grid, load_encoder, segmentation_loss, set_stage
from evaluate_private_validation import file_hash, write_csv, save_crop
from sam_refine_data import source_hashes
from sam_refine32 import TURBO_SHA256
from train_private_long import atomic_save, accumulation_window, early_stop_update
from train_private_tiny import select_patients
from train_private_unet import state_hash


CODE = ('train_multiclass_turbo.py', 'multiclass_model.py', 'multiclass_data.py', 'physical_fsdp.py',
        'sam_refine32.py', 'sam_refine_data.py', 'train_private_long.py', 'train_private_unet.py',
        'train_private_tiny.py', 'evaluate_private_validation.py', 'audit_expert_agreement.py')


@lru_cache(maxsize=2)
def cached(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def control(args):
    prior = physical.read(args.cohort_run_dir / 'summary.json')
    old = physical.read(args.cohort_run_dir / 'config.json')
    if (not prior['complete'] or prior['experiment'] != 'unet_cohort_train_calibrated'
            or prior['train_patients'] != 54 or prior['validation_patients'] != 10 or prior['test_images_read']):
        raise ValueError('Expected completed54/10cohort control')
    manifest_path = Path(old['data_dir']) / 'manifest.json'
    manifest = physical.read(manifest_path)
    select_patients(manifest, args.seed)
    versions = {key: importlib.metadata.version(key) for key in ('torch', 'torchio', 'monai', 'numpy', 'SimpleITK')}
    if versions != old['versions'] or file_hash(manifest_path) != old['manifest_sha256'] or old['seed'] != args.seed:
        raise ValueError('Frozen manifest/seed/package versions differ from cohort')
    if file_hash(args.checkpoint) != TURBO_SHA256:
        raise ValueError('Original Turbo checksum differs')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'}
    config.update(experiment='multiclass_turbo_local_detail', turbo_sha256=TURBO_SHA256, versions=versions,
        code_sha256={key: file_hash(Path(__file__).with_name(key)) for key in CODE},
        sam_source_sha256=source_hashes(args.sam_dir), manifest_sha256=file_hash(manifest_path),
        control_sha256={key: file_hash(args.cohort_run_dir / key) for key in ('summary.json', 'config.json', 'source_provenance.json')},
        output_classes=18, detail_comparison='same adapted initialization/seed/physical field/local128cube/loss/schedule across requested CNN grids; encoder grid held fixed within run',
        encoder_adaptation='strict original Turbo load, then absolute position trilinear8to16 if requested; original patch kernel/attention/relative tables preserved; upstream relative positions resized at runtime; nonreentrant block checkpointing for16tokens',
        context_size=args.encoder_grid * 16,
        initialization='strict full Turbo load before extracting image encoder; new multiclass CNN decoder random, no SAM prompt/mask decoder used',
        selection='validation patient-macro mean Dice across evaluable foreground labels; empty/empty excluded, absent-GT false positives score0',
        protocol='GT-all17-union-centered broad field; physical LPS spacing/phase selected only on train; overlapping128cube local patches; whole-field softmax argmax; native-grid all-class evaluation',
        supervision='each label is one mutually exclusive class in final labelsTr; absent labels treated as background as supplied; full ColorTables verified but clinical annotation completeness not established',
        loss='mean squared soft Dice over foreground classes present in valid patch +0.25cross entropy; padding excluded; no threshold tuning',
        sampling='four local patches per train patient/epoch by default; 80%uniform-present-class center with+-16voxel jitter,20%uniform spatial center',
        augmentation='paired context/local gain0.9-1.1/shift+-0.1 and local noise0.02; no spatial augmentation in this grid-resolution comparison',
        oracle_clicks=0, test_images_read=False)
    sources = physical.read(args.cohort_run_dir / 'source_provenance.json')
    if args.resume:
        if physical.read(args.run_dir / 'config.json') != config:
            raise ValueError('Resume configuration/source/code differs')
        prepared = physical.read(args.run_dir / 'preprocessing/prepared.json')
        for path, digest in prepared['cache_sha256'].items():
            if file_hash(Path(path)) != digest:
                raise ValueError('Multiclass cache changed')
        for rows in prepared['source_provenance'].values():
            for row in rows:
                for key in ('source_image', 'source_label', 'color_table'):
                    if file_hash(Path(row[key])) != row[key + '_sha256']:
                        raise ValueError('Original source changed')
    else:
        physical.save(args.run_dir / 'config.json', config)
        prepared = data.prepare(manifest, Path(old['source_root']).resolve(), sources, args.run_dir / 'preprocessing')
        physical.save(args.run_dir / 'source_provenance.json', prepared['source_provenance'])
    return prepared, config


def amp(model, device):
    return torch.autocast(device_type=device.type, enabled=device.type == 'cuda', dtype=model.amp_dtype)


def confusion_metrics(confusion, mapping, case):
    rows = []
    for label in range(1, 18):
        truth, predicted, tp = int(confusion[label].sum()), int(confusion[:, label].sum()), int(confusion[label, label])
        rows.append({'case': case, 'label': label, 'name': mapping[str(label)], 'target_voxels': truth,
                     'predicted_voxels': predicted, 'true_positive': tp,
                     'dice': 2 * tp / (truth + predicted) if truth + predicted else None,
                     'precision': tp / predicted if predicted else (0. if truth else None),
                     'recall': tp / truth if truth else None})
    values = [r['dice'] for r in rows if r['dice'] is not None]
    ids = [1, 9]
    tp = int(confusion[np.ix_(ids, ids)].sum())
    truth, predicted = int(confusion[ids].sum()), int(confusion[:, ids].sum())
    return rows, {'case': case, 'macro_dice': float(np.mean(values)), 'evaluable_classes': len(values),
                  'fsdp_bilateral_dice': 2 * tp / (truth + predicted) if truth + predicted else None}


@torch.inference_mode()
def infer_case(model, case, device, stride=96):
    context = data.case_context(case, device, getattr(model, 'context_size', 128))
    with amp(model, device):
        embedding = model.encode(context)
    shape = tuple(case['target'].shape)
    sums = np.zeros((18,) + shape, np.float32)
    counts = np.zeros(shape, np.float32)
    starts = data.window_starts(shape, stride=stride)
    for index, start in enumerate(starts, 1):
        patch, _, _ = data.local_patch(case, np.asarray(start))
        with amp(model, device):
            logits = model.decode(patch.to(device), embedding, start, shape)
        probabilities = logits.float().softmax(1)[0].cpu().numpy()
        selection = tuple(slice(s, s + 128) for s in start)
        sums[(slice(None),) + selection] += probabilities
        counts[selection] += 1
        if index % 32 == 0:
            print(f'{case["case"]}: tile{index}/{len(starts)}', flush=True)
    if (counts == 0).any():
        raise ValueError('Sliding windows left voxels uncovered')
    sums /= counts[None]
    valid = case['valid'].numpy()
    sums[:, ~valid] = 0
    sums[0, ~valid] = 1
    native_shape = case['native_target'].shape
    prediction, best = np.zeros(native_shape, np.uint8), np.full(native_shape, -1., np.float32)
    # Restore each class probability, then argmax on the original native grid.
    for label in range(18):
        probability = physical.xyz(physical.resample(physical.image_xyz(sums[label], case['grid']), case['native'],
            physical.sitk.sitkLinear, 1. if label == 0 else 0.))
        choose = probability > best
        prediction[choose], best[choose] = label, probability[choose]
    return prediction


def export_overlay(case, prediction, score, path, mapping):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    gt = case['native_target']
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    color = plt.get_cmap('tab20', 18)
    norm = BoundaryNorm(np.arange(-.5, 18.5), 18)
    for row, axis in enumerate((2, 1, 0)):
        index = int(np.argmax((gt > 0).sum(axis=tuple(i for i in range(3) if i != axis))))
        raw, truth, pred = [np.take(v, index, axis=axis).T for v in (case['raw_native'], gt, prediction)]
        remaining = [i for i in range(3) if i != axis]
        aspect = case['native']['spacing'][remaining[1]] / case['native']['spacing'][remaining[0]]
        for col, ax in enumerate(axes[row]):
            ax.imshow(raw, cmap='gray', vmin=-1000, vmax=1000, origin='lower', aspect=aspect)
            values = truth if col == 0 else pred
            if col < 2:
                ax.imshow(np.ma.masked_where(values == 0, values), cmap=color, norm=norm, alpha=.55, origin='lower', aspect=aspect)
            else:
                errors = (truth != pred) & ((truth > 0) | (pred > 0))
                ax.imshow(np.ma.masked_where(~errors, errors), cmap='autumn', alpha=.55, origin='lower', aspect=aspect)
            ax.set_title(('GT classes', 'Predicted classes', 'Segmentation errors')[col] + f'; native axis{axis} slice{index}', fontsize=9)
            ax.axis('off')
    fig.suptitle(f'{case["case"]}; GT-all-class field; no clicks; macro Dice{score["macro_dice"]:.3f}; FSDP{score["fsdp_bilateral_dice"]:.3f}')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.inference_mode()
def evaluate(model, paths, mapping, device, stride=96, export=None):
    model.eval()
    rows, patients, total = [], [], np.zeros((18, 18), np.int64)
    if export:
        export.mkdir(parents=True, exist_ok=True)
    for path in paths:
        case = cached(path)
        prediction = infer_case(model, case, device, stride)
        truth = case['native_target'].astype(np.int64)
        confusion = np.bincount((truth.reshape(-1) * 18 + prediction.reshape(-1)), minlength=324).reshape(18, 18)
        class_rows, score = confusion_metrics(confusion, mapping, case['case'])
        rows.extend(class_rows)
        patients.append(score)
        total += confusion
        if export:
            save_crop(export / f'{case["case"]}_prediction.nii.gz', prediction, case['metadata'])
            save_crop(export / f'{case["case"]}_target.nii.gz', truth.astype(np.uint8), case['metadata'])
            export_overlay(case, prediction, score, export / f'{case["case"]}.png', mapping)
        print(f'{case["case"]}: macro={score["macro_dice"]:.4f} FSDP={score["fsdp_bilateral_dice"]:.4f}', flush=True)
    per_class = []
    for label in range(1, 18):
        collection = [r for r in rows if r['label'] == label]
        dice = [r['dice'] for r in collection if r['dice'] is not None]
        per_class.append({'label': label, 'name': mapping[str(label)], 'evaluable_patients': len(dice),
            'mean_dice': float(np.mean(dice)) if dice else None,
            'absent_gt_with_false_positives': sum(r['target_voxels'] == 0 and r['predicted_voxels'] > 0 for r in collection)})
    return {'metrics': {'mean_patient_macro_dice': float(np.mean([r['macro_dice'] for r in patients])),
                         'mean_fsdp_bilateral_dice': float(np.mean([r['fsdp_bilateral_dice'] for r in patients])),
                         'minimum_fsdp_bilateral_dice': min(r['fsdp_bilateral_dice'] for r in patients)},
            'classes': per_class, 'patients': patients, 'case_classes': rows, 'confusion': total.tolist()}


def preflight(model, case, device, spacing, output):
    start = data.choose_start(case, 2026)
    patch, target, valid = data.local_patch(case, start)
    model.train()
    set_stage(model, 0, 5)
    with amp(model, device):
        logits, shapes = model(patch.to(device), data.case_context(case, device, getattr(model, 'context_size', 128)), start, case['target'].shape, True)
    loss = segmentation_loss(logits, target.to(device), valid.to(device))
    loss.backward()
    grad = model.classifier.weight.grad
    if grad is None or not grad.isfinite().all() or grad.abs().sum() == 0:
        raise ValueError('Multiclass head failed gradient preflight')
    # Check encoder gradient flow through frozen blocks/neck without an optimizer step.
    del logits, loss
    model.zero_grad(set_to_none=True)
    set_stage(model, 5, 5)
    with amp(model, device):
        logits = model(patch.to(device), data.case_context(case, device, getattr(model, 'context_size', 128)), start, case['target'].shape)
    loss = segmentation_loss(logits, target.to(device), valid.to(device))
    loss.backward()
    probes = {}
    for prefix in ('image_encoder.blocks.10.', 'image_encoder.blocks.11.', 'detail.', 'classifier.'):
        grads = [p.grad for name, p in model.named_parameters() if name.startswith(prefix) and p.grad is not None]
        if not grads or any(not g.isfinite().all() for g in grads) or sum(g.abs().sum().item() for g in grads) == 0:
            raise ValueError(f'Gradient preflight failed for{prefix}')
        probes[prefix] = sum(g.float().square().sum().item() for g in grads) ** .5
    model.zero_grad(set_to_none=True)
    occupancy = []
    for label in range(1, 18):
        mask = (target == label).float()[None]
        occupancy.append({'label': label, 'patch_target_voxels': int(mask.sum()),
            'occupied_detail_cells_maxpool': int((F.adaptive_max_pool3d(mask, model.detail_grid) > 0).sum())})
    physical.save(output / 'preflight.json', {'passed': True, 'label_occupancy_probe': occupancy, 'feature_shapes': shapes,
        'detail_cell_mm_xyz': [float(s * 128 / model.detail_grid) for s in spacing],
        'encoder_token_grid': [getattr(model, 'encoder_grid', 8)] * 3,
        'context_shape': [getattr(model, 'context_size', 128)] * 3,
        'context_sampling': 'direct from physical field; no intermediate128cube; same physical field',
        'encoder_token_mm_xyz': [float(n * s / getattr(model, 'encoder_grid', 8)) for n, s in zip(case['target'].shape, spacing)],
        'gradient_l2': probes, 'optimizer_updates': 0,
        'gpu_peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None})


def train_epoch(model, prepared, optimizer, scaler, device, epoch, args):
    stage = set_stage(model, epoch, args.head_warmup_epochs)
    rng = np.random.default_rng(args.seed + epoch)
    # Shuffle cases while keeping their patches adjacent to avoid loading huge caches repeatedly.
    case_order = rng.permutation(len(prepared['paths']['train']))
    tasks = [(prepared['paths']['train'][i], k) for i in case_order for k in range(args.patches_per_patient)]
    losses, updates, skips = [], 0, 0
    optimizer.zero_grad(set_to_none=True)
    for index, (path, k) in enumerate(tasks):
        case = cached(path)
        sample_seed = args.seed + epoch * 100000 + int(case['case'].split('_')[-1]) * 10 + k
        start = data.choose_start(case, sample_seed)
        patch, target, valid = data.local_patch(case, start)
        context = data.case_context(case, device, getattr(model, 'context_size', 128))
        patch = patch.to(device)
        aug = np.random.default_rng(sample_seed)
        if aug.random() < .3:
            gain, shift = float(aug.uniform(.9, 1.1)), float(aug.uniform(-.1, .1))
            patch, context = patch * gain + shift, context * gain + shift
            patch.masked_fill_(~valid[:, None].to(device), 0)
        # Context padding must remain zero after the shared intensity transform.
        context_valid = F.interpolate(case['valid'][None, None].float(), size=context.shape[-3:], mode='nearest').to(device).bool()
        context.masked_fill_(~context_valid, 0)
        if aug.random() < .15:
            noise_rng = torch.Generator(device=device).manual_seed(sample_seed + 1)
            patch = patch + .02 * torch.randn(patch.shape, generator=noise_rng, device=device)
            patch.masked_fill_(~valid[:, None].to(device), 0)
        with amp(model, device):
            logits = model(patch, context, start, case['target'].shape)
        loss = segmentation_loss(logits, target.to(device), valid.to(device))
        if not loss.isfinite():
            raise ValueError('Nonfinite multiclass loss')
        window, step = accumulation_window(index, len(tasks), args.accumulation)
        scaler.scale(loss / window).backward()
        losses.append(float(loss.item()))
        if step:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm) and not scaler.is_enabled():
                raise ValueError('Nonfinite gradients')
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            success = scaler.get_scale() >= old_scale
            updates, skips = updates + int(success), skips + int(not success)
            optimizer.zero_grad(set_to_none=True)
            print(f'grid={model.detail_grid} epoch={epoch+1} patch={index+1}/{len(tasks)} loss={loss.item():.6f} updates={updates}', flush=True)
    return {'loss': float(np.mean(losses)), 'optimizer_updates': updates, 'amp_skipped_updates': skips,
            'expected_updates': math.ceil(len(tasks) / args.accumulation), 'stage': stage}


def plot_history(history, evaluation, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    measured = [r for r in history if r['val_macro_dice'] is not None]
    for key, label in (('val_macro_dice', 'Macro17'), ('val_fsdp_dice', 'FSDP union')):
        axes[0].plot([r['epoch'] for r in measured], [r[key] for r in measured], label=label)
    axes[0].set(xlabel='Epoch', ylabel='Native validation Dice', ylim=(0, 1))
    axes[0].legend()
    axes[1].bar([r['name'] for r in evaluation['classes']], [r['mean_dice'] or 0 for r in evaluation['classes']])
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].set(ylabel='Selected checkpoint validation Dice', ylim=(0, 1))
    fig.suptitle('18-class prediction in GT-all-class anatomical field; no oracle clicks')
    fig.tight_layout()
    fig.savefig(output / 'validation_curve_and_classes.png', dpi=150)
    plt.close(fig)
    confusion = np.asarray(evaluation['confusion'], float)
    fraction = confusion / np.maximum(confusion.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(fraction, vmin=0, vmax=1, cmap='Blues')
    ax.set(xlabel='Predicted label', ylabel='GT label', title='Voxel confusion; row-normalized; native GT-field domain')
    ax.set_xticks(range(18)); ax.set_yticks(range(18))
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=150)
    plt.close(fig)


def variant(args, prepared, config, grid):
    output = args.run_dir / f'detail{grid}'
    output.mkdir(exist_ok=True)
    if (output / 'summary.json').exists():
        summary = physical.read(output / 'summary.json')
        if file_hash(output / 'model_best.pth') != summary['best_checkpoint_sha256']:
            raise ValueError('Completed variant checkpoint changed')
        return summary
    device = torch.device(args.device)
    seed_all(args.seed)
    encoder, original_hash = load_encoder(args.sam_dir, args.checkpoint)
    token_grid = getattr(args, 'encoder_grid', 8)
    configure_encoder_grid(encoder, token_grid)
    adapted_hash = state_hash(encoder.state_dict())
    model = MulticlassTurbo(encoder, grid).to(device)
    model.encoder_grid, model.context_size = token_grid, token_grid * 16
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model.amp_dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16
    initial_hash = state_hash(model.state_dict())
    optimizer = torch.optim.AdamW([
        {'params': list(model.image_encoder.parameters()), 'lr': args.encoder_lr, 'weight_decay': .01},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('image_encoder.')], 'lr': args.head_lr, 'weight_decay': 1e-4}])
    def schedule(epoch):
        warmup = min(5, args.epochs)
        return (epoch + 1) / warmup if epoch < warmup else .05 + .475 * (1 + math.cos(math.pi * min(1., (epoch-warmup)/max(1,args.epochs-warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and args.precision == 'fp16')
    start, best, best_epoch, early, history = 0, -1., 0, {'anchor': -1., 'last_improvement_epoch': 0}, []
    latest = output / 'model_latest.pth'
    if latest.exists():
        if not args.resume:
            raise ValueError('Unexpected existing checkpoint')
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['config'] != config or saved['initial_state_sha256'] != initial_hash or saved['detail_grid'] != grid:
            raise ValueError('Variant resume configuration differs')
        for obj, key in ((model, 'model'), (optimizer, 'optimizer'), (scheduler, 'scheduler'), (scaler, 'scaler')):
            obj.load_state_dict(saved[key])
        start, best, best_epoch, early, history = [saved[k] for k in ('epoch', 'best', 'best_epoch', 'early', 'history')]
        torch.set_rng_state(saved['torch_rng'])
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved
    else:
        preflight(model, cached(prepared['paths']['train'][0]), device, prepared['policy']['selected']['spacing'], output)
    def state(epoch):
        return {'epoch': epoch, 'best': best, 'best_epoch': best_epoch, 'early': early, 'history': history,
                'config': config, 'initial_state_sha256': initial_hash, 'detail_grid': grid,
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
                'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
    stopped = args.patience_epochs > 0 and start - early['last_improvement_epoch'] >= args.patience_epochs
    for epoch in range(start, args.epochs) if not stopped else []:
        row = {'epoch': epoch + 1, **train_epoch(model, prepared, optimizer, scaler, device, epoch, args),
               'val_macro_dice': None, 'val_fsdp_dice': None}
        if row['optimizer_updates'] + row['amp_skipped_updates'] != row['expected_updates']:
            raise ValueError('Incomplete optimizer update windows')
        scheduler.step()
        improved = False
        if epoch == 0 or (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            evaluation = evaluate(model, prepared['paths']['val'], prepared['policy']['mapping'], device, args.stride)
            score = evaluation['metrics']['mean_patient_macro_dice']
            row.update(val_macro_dice=score, val_fsdp_dice=evaluation['metrics']['mean_fsdp_bilateral_dice'])
            physical.save(output / 'validation_latest.json', evaluation)
            early, stopped = early_stop_update(early, score, epoch + 1, args.patience_epochs, args.min_delta)
            improved = score > best
            if improved:
                best, best_epoch = score, epoch + 1
        history.append(row)
        write_csv(output / 'history.csv', history)
        if improved:
            atomic_save(output / 'model_best.pth', state(epoch + 1))
        atomic_save(latest, state(epoch + 1))
        physical.save(args.run_dir / 'progress.json', {'complete': False, 'detail_grid': grid, 'completed_epochs': epoch + 1,
            'planned_epochs': args.epochs, 'best_epoch': best_epoch, 'best_validation_macro_dice': best, 'latest': row})
        print(json.dumps({'detail_grid': grid, 'epoch_result': row, 'best_validation_macro': best}), flush=True)
        if stopped:
            break
    saved = torch.load(output / 'model_best.pth', map_location='cpu', weights_only=False)
    if saved['config'] != config or saved['best'] != best or saved['epoch'] != best_epoch:
        raise ValueError('Selected checkpoint metadata differs')
    model.load_state_dict(saved['model'])
    del saved
    exports = output / 'best_exports'
    evaluation = evaluate(model, prepared['paths']['val'], prepared['policy']['mapping'], device, args.stride, exports / 'validation')
    if abs(evaluation['metrics']['mean_patient_macro_dice'] - best) > 1e-4:
        raise ValueError('Selected checkpoint does not reproduce validation')
    training = evaluate(model, prepared['paths']['train'], prepared['policy']['mapping'], device, args.stride)
    for split, result in (('train', training), ('val', evaluation)):
        physical.save(exports / f'{split}_metrics.json', result)
        for key in ('patients', 'classes', 'case_classes'):
            write_csv(exports / f'{split}_{key}.csv', result[key])
    plot_history(history, evaluation, exports)
    summary = {'complete': True, 'detail_grid': grid, 'completed_epochs': history[-1]['epoch'], 'planned_epochs': args.epochs,
        'stopped_early': stopped, 'best_epoch': best_epoch, 'best_validation_metrics': evaluation['metrics'],
        'best_validation_per_class': evaluation['classes'], 'frozen_train_metrics': training['metrics'],
        'initial_encoder_sha256': original_hash, 'adapted_encoder_sha256': adapted_hash,
        'encoder_token_grid': token_grid, 'context_size': model.context_size,
        'initial_state_sha256': initial_hash,
        'best_checkpoint_sha256': file_hash(output / 'model_best.pth'), 'test_images_read': False,
        'validation_images_read': True, 'oracle_clicks': 0, 'protocol': config['protocol'],
        'macro_gap': training['metrics']['mean_patient_macro_dice'] - best,
        'export_directory': str(exports)}
    physical.save(output / 'summary.json', summary)
    del model, optimizer, scheduler, scaler
    cached.cache_clear()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return summary


def execute(args):
    physical.save(args.run_dir / 'progress.json', {'complete': False, 'phase': 'audit_and_preprocessing'})
    prepared, config = control(args)
    device = torch.device(args.device)
    physical.save(args.run_dir / 'hardware.json', {'device': str(device), 'precision': args.precision,
        'gpu': torch.cuda.get_device_properties(device).name if device.type == 'cuda' else None})
    summaries = []
    for grid in args.detail_grids:
        print(f'Starting learned detail{grid}³ experiment', flush=True)
        summaries.append(variant(args, prepared, config, grid))
    if len({r['initial_state_sha256'] for r in summaries}) != 1:
        raise ValueError('Resolution comparison did not share initial weights')
    report = {'complete': True, 'experiment': config['experiment'], 'comparison': summaries,
              'encoder_token_grid': getattr(args, 'encoder_grid', 8),
              'train_patients': 54, 'validation_patients': 10, 'reserved_test_patients': 20,
              'test_images_read': False, 'oracle_clicks': 0, 'protocol': config['protocol'],
              'preprocessing_coverage': {split: {'minimum_roundtrip_dice': min(r['roundtrip_dice'] for r in rows if r['roundtrip_dice'] is not None),
                 'roundtrip_below_0_95': sum(r['roundtrip_dice'] is not None and r['roundtrip_dice'] < .95 for r in rows)}
                 for split, rows in prepared['coverage'].items()},
              'interpretation': 'All-structure GT-field-assisted semantic segmentation; not automatic full-scan localization. Macro17 score is not FSDP Dice. Architecture comparison differs from prior binary interactive training.'}
    archive = args.run_dir / 'results.tar.gz'
    report['archive'] = str(archive)
    physical.save(args.run_dir / 'report.json', report)
    with tarfile.open(archive, 'w:gz') as package:
        for name in ('report.json', 'config.json', 'hardware.json', 'preprocessing/policy.json', 'preprocessing/audit_train.json',
                     'preprocessing/train_coverage.csv', 'preprocessing/val_coverage.csv'):
            package.add(args.run_dir / name, arcname='results/' + name)
        for grid in args.detail_grids:
            for name in ('summary.json', 'preflight.json', 'history.csv', 'best_exports'):
                package.add(args.run_dir / f'detail{grid}' / name, arcname=f'results/detail{grid}/' + name)
    physical.save(args.run_dir / 'summary.json', report)
    physical.save(args.run_dir / 'progress.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('sam-dir', 'checkpoint', 'cohort-run-dir', 'run-dir'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--encoder-grid', type=int, choices=(8, 16), default=8)
    parser.add_argument('--detail-grids', type=int, nargs='+', default=[64, 32])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--head-warmup-epochs', type=int, default=5)
    parser.add_argument('--val-every', type=int, default=10)
    parser.add_argument('--patches-per-patient', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--stride', type=int, default=96)
    parser.add_argument('--encoder-lr', type=float, default=1e-5)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--patience-epochs', type=int, default=50)
    parser.add_argument('--min-delta', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--precision', choices=('bf16', 'fp16'), default='bf16')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if (not args.detail_grids or len(set(args.detail_grids)) != len(args.detail_grids) or any(g not in (32, 64) for g in args.detail_grids)
            or min(args.epochs, args.val_every, args.patches_per_patient, args.accumulation) < 1 or not 1 <= args.stride <= 128
            or min(args.head_warmup_epochs, args.patience_epochs) < 0
            or not all(math.isfinite(v) and v > 0 for v in (args.encoder_lr, args.head_lr))
            or not math.isfinite(args.min_delta) or args.min_delta < 0):
        parser.error('Invalid experiment settings')
    old = physical.read(args.cohort_run_dir / 'config.json')
    if any(args.run_dir.is_relative_to(p) for p in (args.sam_dir, args.cohort_run_dir, Path(old['source_root']).resolve(), Path(old['data_dir']).resolve())):
        parser.error('Output must be outside source/data/control folders')
    device = torch.device(args.device)
    if device.type == 'cuda' and (not torch.cuda.is_available() or (args.precision == 'bf16' and not torch.cuda.is_bf16_supported())):
        parser.error('Requested CUDA/BF16 device unavailable')
    if args.resume:
        if not (args.run_dir / 'preprocessing/prepared.json').exists() or (args.run_dir / 'summary.json').exists():
            parser.error('Resume requires unfinished run with completed preprocessing')
    elif args.run_dir.exists():
        parser.error('Run exists; use a new timestamped name')
    else:
        args.run_dir.mkdir(parents=True)
    try:
        execute(args)
    except Exception as error:
        physical.save(args.run_dir / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
