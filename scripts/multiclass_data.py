"""Train-only planning of a broad GT-assisted anatomical field and local patches."""
import csv
import itertools
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import physical_fsdp as physical
from evaluate_private_validation import file_hash


PHASES = ((0., 0., 0.), (.5, .5, .5))
SPACINGS = (.25, .20)
# Inference keeps an 18-channel float32 probability accumulator on the host
# (76 bytes per field voxel, before cached inputs and temporary arrays).
MAX_GRID_VOXELS = 16_000_000


def color_mapping(root, record):
    path = root / 'ColorTable' / f'{record["patient_id"]}_ColorTable.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    mapping = {int(r['LabelValue']): r['Name'].strip().upper() for r in rows}
    if len(mapping) != len(rows) or not set(range(1, 18)) <= set(mapping):
        raise ValueError('ColorTable must define every class1..17 without duplicate IDs')
    mapping = {str(i): mapping[i] for i in range(1, 18)}
    if any(not name for name in mapping.values()) or len(set(mapping.values())) != 17:
        raise ValueError('Foreground class names must be distinct and nonempty')
    return {'0': 'BACKGROUND', **mapping}


def load(record, root, expected, alias, mapping=None):
    image, _, ct, labels, provenance = physical.load_case(record, root, expected, alias)
    current = color_mapping(root, record)
    if mapping is not None and current != mapping:
        raise ValueError(f'{alias}: full ColorTable differs from training mapping')
    return image, ct, labels, provenance, current


def audit_train(manifest, root, sources):
    mapping, samples, extents, provenance, rows = None, [], [], [], []
    expected = {r['patient_id']: r for r in sources}
    for index, record in enumerate(physical.records_for(manifest, 'train'), 1):
        alias = f'train_{index:02d}'
        image, ct, labels, origin, current = load(record, root, expected[record['patient_id']], alias, mapping)
        mapping = current
        foreground = labels > 0
        lo, hi = physical.world_bounds(foreground, physical.geometry(image))
        extents.append(hi - lo)
        values = ct[foreground].astype(np.float64)
        rng = np.random.default_rng(2026 + index)
        samples.append(values if len(values) <= 65536 else rng.choice(values, 65536, replace=False))
        rows.extend({'case': alias, 'label': value, 'voxels': int((labels == value).sum()),
                     'volume_mm3': float((labels == value).sum() * np.prod(image.GetSpacing()))} for value in range(1, 18))
        provenance.append(origin)
        print(f'{alias}: all-class mapping and foreground counts checked', flush=True)
    counts = {i: sum(r['voxels'] for r in rows if r['label'] == i) for i in range(1, 18)}
    values = np.concatenate(samples)
    lower, upper = np.percentile(values, [.5, 99.5])
    clipped = np.clip(values, lower, upper)
    norm = {'clip_lower': float(lower), 'clip_upper': float(upper), 'mean': float(clipped.mean()), 'std': float(clipped.std()),
            'source': 'train foreground of all17classes; at most65536 randomly sampled voxels/case; seed2026; no val/test'}
    if norm['std'] <= 1e-6:
        raise ValueError('Degenerate training intensities')
    extent = np.max(extents, 0) + 10.
    return {'mapping': mapping, 'normalization': norm, 'required_field_mm': extent.tolist(),
            'margin_mm': 5., 'minimum_roundtrip_gate': .95, 'per_case_train_counts': rows, 'train_voxel_counts_by_class': counts,
            'missing_class_interpretation': 'Absent classes are treated as annotated background as supplied; counts do not establish anatomical absence or annotation completeness',
            'protocol': 'GT-union-of-all17-centered physical LPS field; no FSDP-centered crop or oracle clicks; not automatic full-scan localization'}, provenance


def make_case(image, ct, labels, alias, plan, norm):
    source = physical.geometry(image)
    lo, hi = physical.world_bounds(labels > 0, source)
    spacing, shape = np.asarray(plan['spacing']), np.asarray(plan['shape'])
    origin = (lo + hi) / 2 - (shape - 1) * spacing / 2
    if plan.get('anchor') == 'source_origin':
        anchor = np.asarray(source['origin'])
        origin = anchor + np.round((origin - anchor) / spacing) * spacing
    origin = origin + np.asarray(plan['origin_shift_fraction_xyz']) * spacing
    grid = {'size': shape.tolist(), 'spacing': spacing.tolist(), 'origin': origin.tolist(), 'direction': np.eye(3).ravel().tolist()}
    label_image = physical.image_xyz(labels.astype(np.uint8), source)
    target = physical.xyz(physical.resample(label_image, grid, physical.sitk.sitkNearestNeighbor, pixel=physical.sitk.sitkUInt8))
    ones = physical.sitk.Image(image.GetSize(), physical.sitk.sitkUInt8) + 1
    ones.CopyInformation(image)
    valid = physical.xyz(physical.resample(ones, grid, physical.sitk.sitkNearestNeighbor, pixel=physical.sitk.sitkUInt8)).astype(bool)
    raw = physical.xyz(physical.resample(image, grid, physical.sitk.sitkLinear, norm['clip_lower']))
    normalized = (np.clip(raw, norm['clip_lower'], norm['clip_upper']) - norm['mean']) / norm['std']
    normalized[~valid] = 0
    if not np.isfinite(normalized).all() or not valid[target > 0].all():
        raise ValueError(f'{alias}: invalid target/support/intensities')
    native, selection, offset = physical.native_reference(grid, source, labels > 0)
    native_target = labels[selection].copy()
    sample = {'case': alias, 'grid': grid, 'native': native,
              'image': torch.from_numpy(normalized.astype(np.float16))[None, None],
              'target': torch.from_numpy(target), 'valid': torch.from_numpy(valid),
              'native_target': native_target, 'raw_native': ct[selection].copy(),
              'metadata': {'crop_offset_xyz': offset.tolist(), 'crop_origin_mm': native['origin'],
                           'spacing_mm': native['spacing'], 'direction': native['direction']}}
    lower, upper = origin - spacing / 2, origin + (shape - .5) * spacing
    measured = []
    for value in range(1, 18):
        original = labels == value
        count = int(original.sum())
        if not count:
            measured.append({'case': alias, 'label': value, 'target_voxels': 0, 'roundtrip_dice': None, 'retention': None})
            continue
        a, b = physical.world_bounds(original, source)
        retention = bool(np.all(a >= lower - 1e-6) and np.all(b <= upper + 1e-6))
        prediction = physical.xyz(physical.resample(physical.image_xyz((target == value).astype(np.float32) * 2 - 1, grid),
            native, physical.sitk.sitkLinear, -30.)) > 0
        truth = native_target == value
        dice = 2 * int((prediction & truth).sum()) / (int(prediction.sum()) + count)
        measured.append({'case': alias, 'label': value, 'target_voxels': count, 'roundtrip_dice': dice, 'retention': retention})
    generator = np.random.default_rng(2026)
    sample['sampling_points'] = {}
    for value in range(1, 18):
        points = np.argwhere(target == value)
        if len(points):
            sample['sampling_points'][value] = points if len(points) <= 2048 else points[generator.choice(len(points), 2048, replace=False)]
    sample['coverage'] = measured
    sample['minimum_margin_mm'] = float(np.minimum(lo - lower, upper - hi).min())
    return sample


def prepare(manifest, root, sources, output):
    output.mkdir()
    policy, inspected = audit_train(manifest, root, sources['train'])
    physical.save(output / 'audit_train.json', policy)
    missing = [value for value, count in policy['train_voxel_counts_by_class'].items() if count == 0]
    if missing:
        raise ValueError(f'Requested classes have no training examples: {missing}; inspect audit_train.json')
    attempts = []
    selected = None
    candidates = [(spacing, anchor, phase) for spacing in SPACINGS
                  for anchor, phase in (('source_origin', (0., 0., 0.)), ('bbox_center', (0., 0., 0.)), ('bbox_center', (.5, .5, .5)))]
    for spacing, anchor, phase in candidates:
        shape = np.maximum(128, np.ceil((np.asarray(policy['required_field_mm']) + 2 * spacing) / spacing / 16).astype(int) * 16)
        plan = {'shape': shape.tolist(), 'spacing': [spacing] * 3, 'anchor': anchor, 'origin_shift_fraction_xyz': list(phase)}
        attempt = {'plan': plan, 'eligible': False, 'failed': []}
        if int(np.prod(shape)) > MAX_GRID_VOXELS:
            attempt['error'] = 'physical field exceeds16million-voxel CPU accumulation guard'
            attempts.append(attempt)
            physical.save(output / 'policy.json', {**policy, 'attempts': attempts, 'selected': None})
            continue
        print(f'All-class physical candidate: {plan}', flush=True)
        folder = output / f'attempt_{len(attempts):02d}'
        folder.mkdir()
        prior = {r['patient_id']: r for r in sources['train']}
        train_coverage = []
        for index, record in enumerate(physical.records_for(manifest, 'train'), 1):
            alias = f'train_{index:02d}'
            image, ct, labels, origin, _ = load(record, root, prior[record['patient_id']], alias, policy['mapping'])
            if origin != inspected[index - 1]:
                raise ValueError('Source changed during planning')
            sample = make_case(image, ct, labels, alias, plan, policy['normalization'])
            bad = [r for r in sample['coverage'] if r['target_voxels'] and (not r['retention'] or r['roundtrip_dice'] < .95)]
            if sample['minimum_margin_mm'] < 5 - 1e-5:
                bad.append({'case': alias, 'error': 'margin below5mm', 'margin_mm': sample['minimum_margin_mm']})
            attempt['failed'].extend(bad)
            train_coverage.extend(sample['coverage'])
            torch.save(sample, folder / f'{alias}.pt')
            print(f'{alias}: all-class preparation; failing_checks={len(bad)}', flush=True)
        attempt['eligible'] = not attempt['failed']
        attempts.append(attempt)
        physical.save(output / 'policy.json', {**policy, 'attempts': attempts, 'selected': plan if attempt['eligible'] else None})
        if attempt['eligible']:
            selected = folder
            break
        # Failed caches are reproducible from policy/source; release their disk space.
        for cache in folder.glob('*.pt'):
            cache.unlink()
    if selected is None:
        raise ValueError('No all-class training grid passes retention/margin/roundtrip checks; inspect preprocessing/policy.json')
    policy.update(selected=plan, attempts=attempts)
    physical.save(output / 'policy.json', policy)
    paths = {'train': sorted(selected.glob('*.pt')), 'val': []}
    provenance = {'train': inspected, 'val': []}
    val_coverage = []
    prior = {r['patient_id']: r for r in sources['val']}
    print('All-class policy frozen from train; preparing validation without refitting', flush=True)
    for index, record in enumerate(physical.records_for(manifest, 'val'), 1):
        alias = f'val_{index:02d}'
        image, ct, labels, origin, _ = load(record, root, prior[record['patient_id']], alias, policy['mapping'])
        sample = make_case(image, ct, labels, alias, plan, policy['normalization'])
        # Validation does not choose phase. Full target retention is necessary to score this protocol.
        if any(r['target_voxels'] and not r['retention'] for r in sample['coverage']):
            raise ValueError(f'{alias}: validation target exceeds train-frozen field; no case removed')
        path = selected / f'{alias}.pt'
        torch.save(sample, path)
        paths['val'].append(path)
        val_coverage.extend(sample['coverage'])
        provenance['val'].append(origin)
        print(f'{alias}: all-class validation prepared', flush=True)
    for split, rows in (('train', train_coverage), ('val', val_coverage)):
        physical.write_csv(output / f'{split}_coverage.csv', rows)
    result = {'coverage': {'train': train_coverage, 'val': val_coverage}, 'policy': policy, 'paths': {s: [str(p.resolve()) for p in v] for s, v in paths.items()},
              'cache_sha256': {str(p.resolve()): file_hash(p) for v in paths.values() for p in v},
              'source_provenance': provenance, 'test_images_read': False}
    physical.save(output / 'prepared.json', result)
    return result


def case_context(case, device):
    return F.interpolate(case['image'].float(), size=(128,) * 3, mode='trilinear', align_corners=False).to(device)


def local_patch(case, start, size=128):
    # Starts may be negative at training boundaries; jointly pad CT/target/support.
    shape = case['target'].shape
    stop = np.asarray(start) + size
    lower, upper = np.maximum(start, 0), np.minimum(stop, shape)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    patch = torch.zeros((1, 1, size, size, size), dtype=torch.float32)
    target = torch.zeros((1, size, size, size), dtype=torch.long)
    valid = torch.zeros((1, size, size, size), dtype=torch.bool)
    destination = tuple(slice(int(a), int(b)) for a, b in zip(lower - start, upper - start))
    patch[(0, 0) + destination] = case['image'][(0, 0) + slices].float()
    target[(0,) + destination] = case['target'][slices].long()
    valid[(0,) + destination] = case['valid'][slices]
    return patch, target, valid


def choose_start(case, seed):
    rng = np.random.default_rng(seed)
    labels = list(case['sampling_points'])
    if rng.random() < .8 and labels:
        value = int(rng.choice(labels))
        # One foreground coordinate via flat indexing; deterministic and class-balanced.
        points = case['sampling_points'][value]
        point = points[int(rng.integers(len(points)))]
    else:
        point = np.asarray([rng.integers(n) for n in case['target'].shape])
    return np.clip(point - 64 + rng.integers(-16, 17, size=3), 0, np.asarray(case['target'].shape) - 128).astype(int)


def window_starts(shape, size=128, stride=96):
    axes = []
    for n in shape:
        if n < size:
            raise ValueError('Anatomical grid dimensions must be at least128')
        values = list(range(0, n - size + 1, stride))
        if values[-1] != n - size:
            values.append(n - size)
        axes.append(values)
    return list(itertools.product(*axes))
