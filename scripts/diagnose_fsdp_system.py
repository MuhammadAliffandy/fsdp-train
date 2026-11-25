#!/usr/bin/env python3
"""CPU-only experiment/cache diagnosis and optional train-derived normalization candidate."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch

import physical_fsdp as physical
from evaluate_private_validation import file_hash, write_csv
from fsdp_verified_cache import validate_targets, read_image
from train_fsdp_foundation import inspect_foundation
from train_private_long import atomic_save


def distribution(array):
    values = np.asarray(array, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        return {'count': int(values.size), 'finite': bool(np.isfinite(values).all())}
    return {'count': int(values.size), 'finite': True, 'min': float(values.min()),
            'p05': float(np.percentile(values, 5)), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'max': float(values.max()), 'std': float(values.std())}


def audit_sample(sample, normalization):
    image, target, valid = [sample[k][0, 0].numpy() for k in ('image', 'target', 'valid')]
    native = sample['native_target'].astype(bool)
    raw = sample['raw_native']
    restored = physical.restore_logits(target.astype(np.float32) * 2 - 1, sample) > 0
    denominator = int(restored.sum()) + int(native.sum())
    lo, hi = physical.world_bounds(native, sample['native'])
    low = (normalization['clip_lower'] - normalization['mean']) / normalization['std']
    high = (normalization['clip_upper'] - normalization['mean']) / normalization['std']
    row = {'case': sample['case'], 'side': sample['side'], 'name': sample['name'],
        'full_target_voxels': int(native.sum()), 'recorded_target_voxels': sample['metadata']['full_target_voxels'],
        'cached_target_retention': sample['metadata']['target_retention'],
        'roundtrip_recomputed': 2 * int((restored & native).sum()) / denominator,
        'physical_target_extent_lps_mm': (hi - lo).tolist(),
        'foreground_fraction_detail': float(target.sum() / valid.sum()), 'padding_fraction': float(1 - valid.mean()),
        'normalized_valid': distribution(image[valid.astype(bool)]), 'normalized_target': distribution(image[target.astype(bool)]),
        'native_ct_target': distribution(raw[native]), 'native_ct_context': distribution(raw),
        'context_at_or_below_clip_lower_fraction': float((raw <= normalization['clip_lower']).mean()),
        'context_at_or_above_clip_upper_fraction': float((raw >= normalization['clip_upper']).mean()),
        'detail_spacing_mm': sample['grid']['spacing'], 'native_spacing_mm': sample['native']['spacing'],
        'native_direction': sample['native']['direction'],
        'detail_at_clip_lower_fraction': float((image[valid.astype(bool)] <= low + 1e-5).mean()),
        'detail_at_clip_upper_fraction': float((image[valid.astype(bool)] >= high - 1e-5).mean())}
    if 'sam_image' in sample:
        context = sample['sam_image'][0, 0].numpy()[sample['sam_valid'][0, 0].numpy().astype(bool)]
        row.update(sam_context_at_clip_lower_fraction=float((context <= low + 1e-5).mean()),
                   sam_context_at_clip_upper_fraction=float((context >= high - 1e-5).mean()))
    return row


def experiment_inventory(project):
    records, errors, incomplete = [], [], []
    for path in sorted((project / 'runs').rglob('summary.json')):
        try:
            summary = physical.read(path)
            if 'entries' in summary or not summary.get('complete'):
                continue
            if summary.get('test_images_read'):
                errors.append({'path': str(path), 'error': 'test access reported; excluded from development comparisons'})
                continue
            config_path = path.parent / 'config.json'
            config = physical.read(config_path) if config_path.exists() else {}
            row = {'directory': str(path.parent), 'summary_sha256': file_hash(path),
                'experiment': summary.get('experiment', config.get('experiment')), 'variant': summary.get('loss_arm', summary.get('variant')),
                'best_epoch': summary.get('best_epoch'), 'completed_epochs': summary.get('completed_epochs'),
                'train_patients': summary.get('train_patients'), 'validation_patients': summary.get('validation_patients'),
                'test_images_read_reported': summary.get('test_images_read'),
                'oracle_clicks': summary.get('oracle_clicks', summary.get('oracle_clicks_per_side')),
                'selected_validation': summary.get('best_validation_metrics', summary.get('best_validation_dice')),
                'selected_train': summary.get('best_frozen_training_metrics', summary.get('best_checkpoint_frozen_training_dice')),
                'threshold': summary.get('threshold', summary.get('threshold_train_only')),
                'manifest_sha256': config.get('manifest_sha256'), 'cache_sha256': config.get('cache_sha256'),
                'physical_policy': config.get('physical_policy'), 'parameters': config.get('parameters'),
                'architecture': config.get('architecture'), 'preflight': None}
            if (path.parent / 'preflight.json').exists():
                row['preflight'] = physical.read(path.parent / 'preflight.json')
            history = path.parent / 'history.csv'
            if history.exists():
                with history.open(newline='') as stream:
                    measured = [r for r in csv.DictReader(stream) if r.get('val_dice_5') or r.get('val_dice')]
                if measured:
                    last = measured[-1]
                    row['last_measured'] = {k: last.get(k) for k in ('epoch', 'loss', 'train_dice_5', 'val_dice_5', 'train_dice', 'val_dice',
                         'sam_lr', 'head_lr', 'lr', 'gradient_norm_before_clipping_mean', 'gradient_clip_fraction')}
            rows = []
            for relative in ('best_exports/train_patients.csv', 'best_validation/train_5_patients.csv'):
                csv_path = path.parent / relative
                if csv_path.exists():
                    with csv_path.open(newline='') as stream:
                        rows = list(csv.DictReader(stream))
                    break
            if rows:
                row['worst_selected_train'] = sorted(rows, key=lambda r: float(r['dice_full_target']))[:6]
            records.append(row)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({'path': str(path), 'error': str(error)})
    for path in sorted((project / 'runs').rglob('progress.json')):
        if (path.parent / 'summary.json').exists():
            continue
        try:
            progress = physical.read(path)
            if any(k in progress for k in ('completed_epochs', 'active_variant', 'phase')):
                row = {'directory': str(path.parent), 'progress': progress}
                if (path.parent / 'failure.json').exists():
                    row['failure'] = physical.read(path.parent / 'failure.json')
                incomplete.append(row)
        except (OSError, ValueError) as error:
            errors.append({'path': str(path), 'error': str(error)})
    return {'runs': records, 'incomplete': incomplete, 'unreadable_or_unsupported': errors,
            'scope': 'Summary/config/history/CSV only; no old checkpoints or images loaded. Different protocols are not pooled.'}


def source_header(record, source_root):
    path = Path(source_root) / 'imagesTr' / Path(record['source_image']).name
    reader = physical.sitk.ImageFileReader(); reader.SetFileName(str(path)); reader.ReadImageInformation()
    direction = list(reader.GetDirection())
    return {'case': record['case'], 'size': list(reader.GetSize()), 'spacing_mm': list(reader.GetSpacing()),
            'direction': direction, 'origin_mm': list(reader.GetOrigin()),
            'scaling_metadata': {k: reader.GetMetaData(k) for k in ('scl_slope', 'scl_inter', 'qform_code', 'sform_code')
                                 if k in reader.GetMetaDataKeys()},
            'orientation': physical.sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(direction)}


def resunet_probe(run, prepared, output, cache_hash):
    from physical_resunet import PhysicalResUNet
    from train_physical_resunet import overlay
    from train_private_tiny import native_prediction_indices, sparse_scores
    from evaluate_private_validation import save_crop
    config, summary = [physical.read(run / name) for name in ('config.json', 'summary.json')]
    if (not summary['complete'] or summary['test_images_read'] or summary['oracle_clicks'] != 0
            or config['cache_sha256'] != cache_hash
            or file_hash(run / 'resunet_best.pth') != summary['checkpoint_sha256']):
        raise ValueError('ResUNet probe needs unchanged completed weights and matching target cache')
    if any(file_hash(Path(__file__).with_name(name)) != value for name, value in config['code_sha256'].items()):
        raise ValueError('ResUNet scientific implementation changed')
    checkpoint = torch.load(run / 'resunet_best.pth', map_location='cpu', weights_only=False)
    if checkpoint['config'] != config:
        raise ValueError('ResUNet checkpoint/config differs')
    model = PhysicalResUNet().eval(); model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    captures, handles, records = {}, [], []
    def capture(name):
        def hook(module, inputs, value):
            values = value.detach().float()
            captures[name] = {'shape': list(values.shape), 'mean': float(values.mean()),
                'std': float(values.std(unbiased=False)), 'spatial_std_mean': float(values.flatten(2).std(-1, unbiased=False).mean()),
                'finite': bool(values.isfinite().all())}
        return hook
    for prefix in ('encoder', 'decoder', 'heads'):
        for i, layer in enumerate(getattr(model, prefix)):
            handles.append(layer.register_forward_hook(capture(f'{prefix}.{i}')))
    output.mkdir()
    try:
        with torch.inference_mode():
            for sample in prepared['samples']['train']:
                if sample['case'] not in ('train_01', 'train_28', 'train_53'):
                    continue
                print(f'{sample["name"]}: CPU probability/layer probe', flush=True)
                logits = model(sample['image'])[0][0, 0].numpy()
                if not np.isfinite(logits).all():
                    raise ValueError('Nonfinite ResUNet CPU logits')
                native = physical.restore_logits(logits, sample)
                probability = 1 / (1 + np.exp(-np.clip(native, -60, 60)))
                gt = sample['native_target'].astype(bool)
                row = {'case': sample['case'], 'side': sample['side'], 'foreground_probability': distribution(probability[gt]),
                    'background_probability': distribution(probability[~gt]), 'layers': copy.deepcopy(captures), 'thresholds': []}
                for threshold in sorted(set((.5, summary['threshold_train_only']))):
                    prediction = probability > threshold
                    shape = prepared['patients']['train'][sample['case']]['shape']
                    ids = native_prediction_indices(prediction, np.ones(prediction.shape, bool), sample['metadata'], shape)
                    truth = native_prediction_indices(gt, np.ones(gt.shape, bool), sample['metadata'], shape)
                    scores = sparse_scores(ids, truth)
                    row['thresholds'].append({'threshold': threshold, **scores})
                    folder = output / f't{threshold:g}'; folder.mkdir(exist_ok=True)
                    overlay(sample, prediction, scores['dice_full_target'], threshold, folder)
                save_crop(output / f'{sample["name"]}_probability.nii.gz', probability.astype(np.float32), sample['metadata'])
                records.append(row)
    finally:
        for handle in handles:
            handle.remove()
    return {'source': str(run), 'optimizer_updates': 0, 'precision': 'CPU FP32; small differences from BF16 source are possible',
            'scope': 'Only train_01/28/53, both sides; activation statistics do not establish causal feature usefulness', 'records': records}


def build_normalization_candidate(prepared, source_root, output, source_hash):
    """Preserve all grids/targets; replace CT normalization from verified raw CT and TRAIN context statistics."""
    draws = []
    generator = np.random.default_rng(2026)
    for record in prepared['source_provenance']['train']:
        samples = [s for s in prepared['samples']['train'] if s['case'] == record['case']]
        image = read_image(record, source_root, record, samples)
        for sample in samples:
            grid = sample.get('sam_grid', sample['grid'])
            raw = physical.xyz(physical.resample(image, grid, physical.sitk.sitkLinear, -1000.))
            support = sample.get('sam_valid', sample['valid'])[0, 0].numpy().astype(bool)
            values = np.clip(raw[support], -1000., 2000.)
            take = min(4096, len(values))
            draws.append(values[generator.choice(len(values), take, replace=False)].astype(np.float64))
        del image
    values = np.concatenate(draws)
    norm = {'clip_lower': -1000., 'clip_upper': 2000., 'mean': float(values.mean()), 'std': float(values.std()),
            'source': 'fixed CT window; deterministic up-to4096valid context voxels/side from54TRAINpatients only',
            'training_samples': len(values), 'seed': 2026}
    if norm['std'] <= 1e-6:
        raise ValueError('Degenerate train context normalization')
    result = {**prepared, 'policy': copy.deepcopy(prepared['policy']), 'samples': {'train': [], 'val': []}}
    result['policy']['normalization'] = norm
    for split in ('train', 'val'):
        for record in prepared['source_provenance'][split]:
            samples = [s for s in prepared['samples'][split] if s['case'] == record['case']]
            image = read_image(record, source_root, record, samples)
            for sample in samples:
                changed = dict(sample)
                for prefix, grid_key in (('', 'grid'), ('sam_', 'sam_grid')):
                    if grid_key not in sample:
                        continue
                    raw = physical.xyz(physical.resample(image, sample[grid_key], physical.sitk.sitkLinear, -1000.))
                    normalized = (np.clip(raw, -1000., 2000.) - norm['mean']) / norm['std']
                    normalized[~sample[prefix + 'valid'][0, 0].numpy().astype(bool)] = 0
                    changed[prefix + 'image'] = torch.from_numpy(normalized.astype(np.float32))[None, None]
                    if not torch.isfinite(changed[prefix + 'image']).all():
                        raise ValueError('Nonfinite normalization candidate')
                result['samples'][split].append(changed)
            del image
    # Native masks, grids, support, manifest membership and patient targets stay the same objects/values.
    unchanged = {'splits': True, 'patient_targets': True, 'sample_count': {},
                 'native_targets': 0, 'model_targets': 0, 'support_masks': 0, 'grids': 0}
    if set(result['patients']) != set(prepared['patients']):
        raise ValueError('Candidate changed patient/split membership')
    for split in ('train', 'val'):
        if set(result['patients'][split]) != set(prepared['patients'][split]):
            raise ValueError(f'Candidate changed {split} patient membership')
        for alias, old_patient in prepared['patients'][split].items():
            new_patient = result['patients'][split][alias]
            if (old_patient['shape'] != new_patient['shape']
                    or not np.array_equal(old_patient['target_indices'], new_patient['target_indices'])):
                raise ValueError('Candidate changed patient target geometry or native target indices')
    for split in ('train', 'val'):
        before, after = prepared['samples'][split], result['samples'][split]
        if len(before) != len(after):
            raise ValueError(f'Candidate changed {split} ROI count')
        unchanged['sample_count'][split] = len(before)
        for old, new in zip(before, after):
            if (old['case'], old['side'], old['name']) != (new['case'], new['side'], new['name']):
                raise ValueError('Candidate changed patient/side identity or order')
            if old['grid'] != new['grid'] or old['native'] != new['native'] or old['metadata'] != new['metadata']:
                raise ValueError('Candidate changed model/native geometry or target metadata')
            for key, counter in (('native_target', 'native_targets'), ('target', 'model_targets'),
                                 ('sam_target', 'model_targets'), ('valid', 'support_masks'),
                                 ('sam_valid', 'support_masks')):
                if key in old:
                    if key not in new or not np.array_equal(np.asarray(old[key]), np.asarray(new[key])):
                        raise ValueError(f'Candidate changed authoritative {key}')
                    unchanged[counter] += 1
    proof = {'source_cache_sha256': source_hash, 'normalization': norm, 'target_policy': 'identical authoritative binary cache targets',
        'target_geometry_equivalence': unchanged,
             'status': 'candidate_for_review_not_automatically_promoted', 'training_launched': False,
             'limitations': 'CT window/context normalization assumes HU-like source intensities and is a predefined hypothesis, not a proven repair; '
                            'source intensity units and pretrained input compatibility must be reviewed.'}
    path = output / 'candidate_context_norm.pt'
    atomic_save(path, result)
    proof['candidate_cache_sha256'] = file_hash(path)
    physical.save(output / 'candidate_provenance.json', proof)
    return proof


def execute(args):
    project = args.project.resolve()
    old, _ = inspect_foundation(args.foundation_run_dir, args.source_root, verified_cache=True)
    cohort = physical.read(Path(old['cohort_run_dir']) / 'config.json')
    source_root = args.source_root or Path(cohort['source_root'])
    protected = [args.foundation_run_dir, source_root, Path(cohort['data_dir']), Path(old['sam_dir']), project / 'runs']
    if any(args.output_dir.is_relative_to(path.resolve()) for path in protected):
        raise ValueError('Diagnostic output must be outside original data/model/run directories')
    prepared = torch.load(args.foundation_run_dir / 'hybrid_data.pt', map_location='cpu', weights_only=False)
    manifest = physical.read(Path(cohort['data_dir']) / 'manifest.json')
    integrity = validate_targets(prepared, manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.build_candidate:
        minimum = 2 * (args.foundation_run_dir / 'hybrid_data.pt').stat().st_size + 2 * 2**30
        if shutil.disk_usage(args.output_dir).free < minimum:
            raise ValueError('Insufficient output space for a separate normalization candidate')
    normalization = prepared['policy']['normalization']
    rows, headers, warnings = [], [], []
    for split in ('train', 'val'):
        for record in prepared['source_provenance'][split]:
            try:
                headers.append({'split': split, **source_header(record, source_root)})
            except (RuntimeError, OSError) as error:
                warnings.append({'case': record['case'], 'header_error': str(error)})
        for sample in prepared['samples'][split]:
            row = {'split': split, **audit_sample(sample, normalization)}
            rows.append(row)
            print(f'{sample["name"]}: roundtrip={row["roundtrip_recomputed"]:.4f}; upper clipping={row["context_at_or_above_clip_upper_fraction"]:.3f}', flush=True)
    report = {'complete': False, 'test_images_read': False, 'gpu_used': False, 'optimizer_updates': 0,
        'source_cache_sha256': old['cache_sha256'], 'manifest_sha256': old['manifest_sha256'],
        'target_integrity': integrity, 'normalization': normalization, 'roi_rows': rows, 'source_headers': headers,
        'experiment_inventory': experiment_inventory(project), 'warnings': warnings,
        'review_flags': {'roundtrip_below_0_95': [r['name'] for r in rows if r['roundtrip_recomputed'] < .95],
            'detail_upper_saturation_above_half': [r['name'] for r in rows if r['detail_at_clip_upper_fraction'] > .5],
            'meaning': 'Screening flags for review, not causal diagnoses or automatic patient exclusions.'},
        'scope': 'All available local run metadata, all54train/10valcached ROIs and their CT headers; no test images/headers.',
        'limitations': 'No activation/gradient inference or anatomical annotation validation. Clipping/orientation associations do not prove causation.',
        'new_set_plan': {'split': 'retain frozen54train/10val/20unread test; future folds need inner-train-only preprocessing',
            'control': 'retain SAM49.47%and original cache as immutable references',
            'N1': 'train-derived context normalization only; exact same grids/targets; paired matched control required',
            'L1': 'loss component/auxiliary/gradient conflict from existing audit before choosing loss changes',
            'A1': 'encoder blocks8–11versus10–11only after input/label checks; same source and update budget',
            'A2': 'tokens16study remains unproven after SIGTERM; input256interpolation is not new acquired CT resolution',
            'E1': 'untouched pretrained inference under explicit prompt protocol before another foundation-model fine-tune',
            'release_gate': 'train_28/53visual+probability review and cache integrity before any new cohort training; no exclusions or label edits automated'}}
    for path in sorted((project / 'outputs/private/fsdp_loss_workflow').glob('*/audit/summary.json')):
        report.setdefault('existing_gradient_reports', []).append({'path': str(path), 'report': physical.read(path)})
    for name in ('prompt_audit', 'roundtrip_audit', 'unet_audit', 'expert_agreement', 'dataset_profile'):
        for path in sorted((project / 'outputs/private' / name).rglob('summary.json')):
            try:
                report.setdefault('existing_audits', []).append({'kind': name, 'path': str(path), 'report': physical.read(path)})
            except (OSError, ValueError) as error:
                warnings.append({'path': str(path), 'error': str(error)})
    physical.save(args.output_dir / 'summary.json', report)
    write_csv(args.output_dir / 'roi_diagnostics.csv', rows)
    physical.save(args.output_dir / 'new_set_plan.json', report['new_set_plan'])
    if args.build_candidate:
        candidate = build_normalization_candidate(prepared, source_root, args.output_dir, old['cache_sha256'])
        report['candidate'] = candidate
    if args.include_probe:
        report['resunet_probe'] = resunet_probe(args.resunet_run, prepared, args.output_dir / 'resunet_probe', old['cache_sha256'])
    report['complete'] = True
    physical.save(args.output_dir / 'summary.json', report)
    print(f'Report: {args.output_dir / "summary.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--foundation-run-dir', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--build-candidate', action='store_true')
    parser.add_argument('--include-probe', action='store_true')
    parser.add_argument('--resunet-run', type=Path)
    args = parser.parse_args()
    args.project = args.project.expanduser().resolve()
    args.foundation_run_dir = (args.foundation_run_dir or args.project / 'runs/fsdp_private_sam_refine32_20261123T144159Z_3745292').resolve()
    args.resunet_run = (args.resunet_run or args.project / 'runs/physical_resunet_20261125T130044Z_641541').resolve()
    args.output_dir = (args.output_dir or args.project / 'outputs/private/system_diagnosis' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    if args.source_root:
        args.source_root = args.source_root.expanduser().resolve()
    existed = args.output_dir.exists()
    try:
        execute(args)
    except Exception as error:
        if not existed and args.output_dir.is_dir():
            physical.save(args.output_dir / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
