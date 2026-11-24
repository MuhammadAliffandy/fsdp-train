"""Adapt and probe the successful FSDP hybrid before controlled fine-tuning."""
import copy
from pathlib import Path

import numpy as np
import torch

import physical_fsdp as data
import train_sam_refine32 as trainer
from sam_refine_resolution import SamRefineResolution, VARIANTS
from train_fsdp_foundation import verify_reproduction
from train_private_unet import state_hash
from audit_foundation_sources import bind_record, load_bound_case


def configuration(args, old):
    grid, mode = VARIANTS[args.resolution_variant]
    detail = old['physical_policy']['selected']
    extent = old['physical_policy']['sam_context']['physical_extent_mm']
    return {'variant': args.resolution_variant, 'token_grid': grid, 'detail_mode': mode,
            'detail_feature_shape': [n // 2 for n in detail['shape']] if mode == 'native_half' else [32] * 3,
            'effective_context_policy': {'size': [grid * 16] * 3, 'physical_extent_mm': extent,
                'spacing_mm': [float(n / (grid * 16)) for n in extent],
                'source': 'original CT resampled directly; same physical center and field as foundation'},
            'coarse_mask_size': grid * 4, 'checkpoint_initialization': 'strict full trained foundation load before adaptation',
            'positional_adaptation': 'absolute8to16trilinear; retain original relative tables with runtime interpolation' if grid == 16 else 'unchanged',
            'prompt_adaptation': 'same physical points/budget/convention; context-normalized coordinates, dense PE and feedback grids resized together',
            'scope': 'binary FSDP1/9; same detail ROI, normalization, loss,5clicks/side and frozen patient split'}


def replace_contexts(prepared, args):
    """Change context sampling only; leave every cached detail tensor and grid intact."""
    old = data.read(args.foundation_run_dir / 'config.json')
    control = data.read(Path(old['cohort_run_dir']) / 'config.json')
    manifest = data.read(Path(control['data_dir']) / 'manifest.json')
    source_root = getattr(args, 'source_root', None)
    verified_cache = getattr(args, 'verified_cache', False)
    root = Path(source_root or control['source_root']).resolve()
    norm = prepared['policy']['normalization']
    for split in ('train', 'val'):
        prior = {row['patient_id']: row for row in prepared['source_provenance'][split]}
        for index, record in enumerate(data.records_for(manifest, split), 1):
            alias = f'{split}_{index:02d}'
            samples = [s for s in prepared['samples'][split] if s['case'] == alias]
            if verified_cache:
                from fsdp_verified_cache import read_image, source_label
                image = read_image(record, root, prior[record['patient_id']], samples)
            else:
                if source_root:
                    record = bind_record(record, source_root)
                loader = load_bound_case if source_root else data.load_case
                image, _, _, labels, _ = loader(record, root, prior[record['patient_id']], alias)
            for sample in samples:
                previous = sample['sam_grid']
                size, step = np.asarray(previous['size']), np.asarray(previous['spacing'])
                center = np.asarray(previous['origin']) + (size - 1) * step / 2
                spacing = size * step / 256
                grid = {'size': [256] * 3, 'spacing': spacing.tolist(),
                        'origin': (center - 255 * spacing / 2).tolist(), 'direction': previous['direction']}
                value = 1 if sample['side'] == 'RFSDP' else 9
                binary = source_label(sample, image) if verified_cache else data.image_xyz((labels == value).astype(np.uint8), data.geometry(image))
                support = data.sitk.Image(image.GetSize(), data.sitk.sitkUInt8) + 1
                support.CopyInformation(image)
                target = data.xyz(data.resample(binary, grid, data.sitk.sitkNearestNeighbor, pixel=data.sitk.sitkUInt8))
                valid = data.xyz(data.resample(support, grid, data.sitk.sitkNearestNeighbor, pixel=data.sitk.sitkUInt8)).astype(bool)
                raw = data.xyz(data.resample(image, grid, data.sitk.sitkLinear, norm['clip_lower']))
                normalized = (np.clip(raw, norm['clip_lower'], norm['clip_upper']) - norm['mean']) / norm['std']
                normalized[~valid] = 0
                if not target.any() or not valid[target.astype(bool)].all() or not np.isfinite(normalized).all():
                    raise ValueError(f'{sample["name"]}: invalid256context')
                sample.update(sam_image=torch.from_numpy(normalized.astype(np.float32))[None, None],
                    sam_target=torch.from_numpy(target.astype(np.uint8))[None, None],
                    sam_valid=torch.from_numpy(valid)[None, None], sam_grid=grid)
            print(f'{alias}:256context sampled from original CT; cached detail ROI unchanged', flush=True)


def setup(model, prepared, args, config):
    device = next(model.parameters()).device
    if not args.resume:
        threshold, training, validation = trainer.calibrate_validate(model, prepared, device, args.seed)
        verify_reproduction(threshold, training, validation, config['foundation'])
        data.save(args.run_dir / 'foundation_reproduction.json', {'passed': True, 'optimizer_updates': 0,
            'threshold': threshold, 'train': {str(k): training['refined'][k][threshold]['metrics'] for k in (1, 5)},
            'val': {str(k): validation['refined'][k][threshold]['metrics'] for k in (1, 5)},
            'foundation_checkpoint_sha256': config['foundation']['checkpoint_sha256']})
    elif not (args.run_dir / 'foundation_reproduction.json').exists():
        raise ValueError('Resume requires completed foundation reproduction')
    else:
        proof = data.read(args.run_dir / 'foundation_reproduction.json')
        if not proof['passed'] or proof['foundation_checkpoint_sha256'] != config['foundation']['checkpoint_sha256']:
            raise ValueError('Foundation reproduction metadata differs')
    source = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    adapted = SamRefineResolution(model.sam)
    adapted.load_state_dict(source, strict=True)
    adapted.adapt(args.resolution_variant)
    adapted = adapted.to(device)
    adapted.amp_dtype = model.amp_dtype
    position = 'sam.image_encoder.pos_embed'
    for name, value in adapted.state_dict().items():
        if name == position and adapted.token_grid == 16:
            continue
        if not torch.equal(value.cpu(), source[name]):
            raise ValueError(f'Learned foundation weights changed during adaptation: {name}')
    preservation = {'passed': True, 'foundation_full_state_sha256': state_hash(source),
        'preserved_non_position_sha256': state_hash({k: v for k, v in source.items() if k != position}),
        'adapted_full_state_sha256': state_hash(adapted.state_dict()), 'resolution': config['resolution'],
        'only_parameter_adapted': position if adapted.token_grid == 16 else None}
    data.save(args.run_dir / 'weight_preservation.json', preservation)
    if adapted.token_grid == 16:
        replace_contexts(prepared, args)
    return adapted, prepared


def gradient_probe(model, prepared, args, config, criterion):
    sample = prepared['samples']['train'][0]
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    before = state_hash(model.state_dict())
    trainer.set_stage(model, config['foundation']['source_best_epoch'], 0)
    model.zero_grad(set_to_none=True)
    loss, _ = trainer.run_sample(model, sample, device, args.seed, criterion=criterion,
        auxiliary_weight=args.auxiliary_weight)
    loss.backward()
    probes = {}
    for prefix in ('sam.image_encoder.blocks.10.', 'sam.image_encoder.blocks.11.', 'sam.mask_decoder.', 'detail32.', 'residual.'):
        grads = [p.grad for name, p in model.named_parameters() if name.startswith(prefix) and p.grad is not None]
        if not grads or any(not g.isfinite().all() for g in grads) or sum(g.abs().sum().item() for g in grads) == 0:
            raise ValueError(f'Resolution gradient probe failed: {prefix}')
        probes[prefix] = sum(g.float().square().sum().item() for g in grads) ** .5
    model.zero_grad(set_to_none=True)
    model.eval()
    tensors = trainer.views.paired_inputs(sample, device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == 'cuda', dtype=model.amp_dtype):
        cache = model.encode(tensors['image'], tensors['sam_image'], tensors['ratio'], tensors['spacing'])
        shapes = {key: list(cache[key].shape) for key in ('embedding', 'full', 'half', 'detail32', 'semantic32')}
    cells = np.asarray(shapes['detail32'][-3:])
    if state_hash(model.state_dict()) != before:
        raise ValueError('Gradient-only audit changed model state')
    report = {'passed': True, 'optimizer_updates': 0, 'training_side': sample['name'],
        'model_state_unchanged': True, 'model_sha256': before,
        'five_interaction_loss': float(loss.detach()), 'gradient_l2': probes, 'feature_shapes': shapes,
        'feedback_shape': [model.coarse_size] * 3,
        'detail_cell_mm_xyz': (np.asarray(sample['grid']['size']) * sample['grid']['spacing'] / cells).tolist(),
        'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None}
    data.save(args.run_dir / 'resolution_probe.json', report)
    return report


def finish_audit(model, prepared, args, config, threshold, training, validation, probe):
    score = validation['refined'][5][threshold]['metrics']['mean_bilateral_patient_dice']
    summary = {'complete': True, 'stage': 'resolution_audit', 'experiment': 'FSDP_foundation_resolution',
        'target_recovery': config.get('target_recovery'),
        'resolution': config['resolution'], 'foundation_reproduced_before_adaptation': True,
        'candidate_train_threshold': threshold,
        'candidate_train': {str(k): training['refined'][k][threshold]['metrics'] for k in (1, 5)},
        'candidate_val': {str(k): validation['refined'][k][threshold]['metrics'] for k in (1, 5)},
        'candidate_validation_fixed_0_5': validation['refined'][5][.5]['metrics'],
        'candidate_validation_delta_before_training': score - config['foundation']['expected_validation']['5']['mean_bilateral_patient_dice'],
        'gradient_probe': probe, 'optimizer_updates': 0, 'validation_images_read': True, 'test_images_read': False,
        'interpretation': 'Readiness and initial prediction audit; no training or performance improvement established. Validation never changes ROI or selects resolution automatically.'}
    trainer.write_csv(args.run_dir / 'train_5_sides.csv', training['refined'][5][threshold]['sides'])
    trainer.write_csv(args.run_dir / 'val_5_sides.csv', validation['refined'][5][threshold]['sides'])
    exported = trainer.evaluate(model, prepared, 'val', next(model.parameters()).device, [threshold], args.seed,
        export=args.run_dir / 'overlays')
    expected = {r['case']: r['dice_full_target'] for r in validation['refined'][5][threshold]['patients']}
    actual = {r['case']: r['dice_full_target'] for r in exported['refined'][5][threshold]['patients']}
    if set(expected) != set(actual) or any(abs(actual[key] - expected[key]) > 1e-4 for key in expected):
        raise ValueError('Audit overlays do not reproduce initial candidate scores')
    data.save(args.run_dir / 'summary.json', summary)
    data.save(args.run_dir / 'progress.json', summary)
    return summary


def update_summary(summary, model, config, prepared):
    summary['resolution'] = config['resolution']
    summary['detail_feature_grid'] = config['resolution']['detail_feature_shape']
    summary['sam_token_grid'] = [model.token_grid] * 3
    summary['foundation_physical_policy'] = summary['physical_policy']
    summary['physical_policy'] = copy.deepcopy(prepared['policy'])
    summary['physical_policy']['sam_context'] = config['resolution']['effective_context_policy']
    summary['selection_includes_unchanged_foundation'] = config['resolution']['variant'] == 'baseline'
    summary['optimizer_protocol'] = 'fresh lower-LR optimizer; additional epoch0is the inherited model after the declared resolution adaptation'
    summary['improves_foundation_validation'] = summary['validation_change_vs_foundation'] > 0
    summary['interpretation'] = ('GT-assisted binary FSDP; original trained hybrid weights inherited before adaptation. '
        'Changed-resolution epoch0is an adapted model, not unchanged45.63%foundation. Original foundation is retained externally; '
        'a worse variant is not promoted automatically. Five oracle clicks per side; no automatic full-scan performance claim.')
