"""Train-derived physical FSDP grids with original-grid reconstruction."""

import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from audit_expert_agreement import same_geometry
from evaluate_private_validation import file_hash, write_csv
from train_private_tiny import native_prediction_indices, select_patients, sparse_scores


SPACING_CANDIDATES = (.35, .30, .25)
MINIMUM_FOV_MM = (22.4, 39.2, 50.4)
MARGIN_MM = 5.
ROUNDTRIP_GATE = .95
THRESHOLDS = (.05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .7, .8, .9, .95)


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def geometry(image):
    return {'size': list(image.GetSize()), 'spacing': list(image.GetSpacing()),
            'origin': list(image.GetOrigin()), 'direction': list(image.GetDirection())}


def reference(meta, pixel=sitk.sitkFloat32):
    image = sitk.Image([int(n) for n in meta['size']], pixel)
    image.SetSpacing(meta['spacing'])
    image.SetOrigin(meta['origin'])
    image.SetDirection(meta['direction'])
    return image


def xyz(image):
    return sitk.GetArrayFromImage(image).transpose(2, 1, 0).copy()


def image_xyz(array, meta):
    image = sitk.GetImageFromArray(np.asarray(array).transpose(2, 1, 0).copy())
    image.SetSpacing(meta['spacing'])
    image.SetOrigin(meta['origin'])
    image.SetDirection(meta['direction'])
    return image


def resample(image, grid, interpolation, outside=0., pixel=sitk.sitkFloat32):
    return sitk.Resample(image, reference(grid), sitk.Transform(3, sitk.sitkIdentity),
                         interpolation, float(outside), pixel)


def world_points(indices, meta):
    return np.asarray(meta['origin']) + (np.asarray(indices) * meta['spacing']) @ np.asarray(meta['direction']).reshape(3, 3).T


def source_points(points, meta):
    return ((np.asarray(points) - meta['origin']) @ np.linalg.inv(np.asarray(meta['direction']).reshape(3, 3)).T) / meta['spacing']


def world_bounds(mask, meta):
    points = world_points(np.argwhere(mask), meta)
    if not len(points):
        raise ValueError('Empty original FSDP target')
    half_cell = np.abs(np.asarray(meta['direction']).reshape(3, 3)) @ (np.asarray(meta['spacing']) / 2)
    return points.min(0) - half_cell, points.max(0) + half_cell


def load_case(record, source_root, expected, alias):
    paths = [Path(record[key]).resolve() for key in ('source_image', 'source_label')]
    if (not paths[0].is_relative_to(source_root / 'imagesTr')
            or not paths[1].is_relative_to(source_root / 'labelsTr')):
        raise ValueError('Expected original private image/final label paths')
    provenance = {'case': alias, 'patient_id': record['patient_id'],
                  'source_image': str(paths[0]), 'source_label': str(paths[1]),
                  'source_image_sha256': file_hash(paths[0]), 'source_label_sha256': file_hash(paths[1])}
    if any(provenance[key] != expected[key] for key in
           ('source_image', 'source_label', 'source_image_sha256', 'source_label_sha256')):
        raise ValueError('Original CT/final annotation differs from completed cohort')
    color_path = source_root / 'ColorTable' / f'{record["patient_id"]}_ColorTable.csv'
    with color_path.open(encoding='utf-8-sig', newline='') as stream:
        colors = {int(row['LabelValue']): row['Name'].strip().upper() for row in csv.DictReader(stream)}
    if colors.get(1) != 'RFSDP' or colors.get(9) != 'LFSDP':
        raise ValueError('Final ColorTable no longer confirms FSDP1/9')
    provenance.update(color_table=str(color_path), color_table_sha256=file_hash(color_path))
    image, label = sitk.ReadImage(str(paths[0])), sitk.ReadImage(str(paths[1]))
    if not same_geometry(image, label):
        raise ValueError('Original CT/label physical geometry differs')
    ct, labels = xyz(image), xyz(label)
    if (not np.isfinite(ct).all() or not np.isin(labels, range(18)).all()
            or int((labels == 1).sum()) != record['right_voxels']
            or int((labels == 9).sum()) != record['left_voxels']):
        raise ValueError('Original intensities/label values or FSDP counts changed')
    return image, label, ct, labels, provenance


def records_for(manifest, split):
    select_patients(manifest, 2026)
    if split not in ('train', 'val'):
        raise ValueError('Test image access is not supported')
    by_id = {row['patient_id']: row for row in manifest['cases']}
    if len(by_id) != len(manifest['cases']):
        raise ValueError('Duplicate manifest records')
    rows = [by_id[key] for key in sorted(manifest['splits'][split])]
    if any(row['split'] != split for row in rows):
        raise ValueError('Manifest case/split mismatch')
    return rows


def inspect_training(manifest, source_root, prior_sources):
    expected = {row['patient_id']: row for row in prior_sources}
    intensities, extents, provenance = [], [], []
    for i, row in enumerate(records_for(manifest, 'train'), 1):
        image, label, ct, labels, origin = load_case(row, source_root, expected[row['patient_id']], f'train_{i:02d}')
        meta = geometry(image)
        for value in (1, 9):
            mask = labels == value
            lo, hi = world_bounds(mask, meta)
            extents.append(hi - lo)
            intensities.append(ct[mask].astype(np.float64))
        provenance.append(origin)
        print(f'train_{i:02d}: physical extent and normalization source checked', flush=True)
    values = np.concatenate(intensities)
    low, high = np.percentile(values, [.5, 99.5])
    norm = {'clip_lower': float(low), 'clip_upper': float(high), 'mean': float(values.mean()),
            'std': float(values.std()), 'source': 'raw final FSDP1/9 voxels from all54 training patients; no validation/test',
            'target_voxels': len(values)}
    if not np.isfinite(list(norm[key] for key in ('clip_lower', 'clip_upper', 'mean', 'std'))).all() or norm['std'] <= 1e-6:
        raise ValueError('Degenerate training-derived intensity normalization')
    maximum = np.max(extents, axis=0)
    desired = np.maximum(maximum + 2 * MARGIN_MM, MINIMUM_FOV_MM)
    candidates = [{'spacing': [spacing] * 3,
                   'shape': (np.ceil(desired / spacing / 16).astype(int) * 16).tolist()}
                  for spacing in SPACING_CANDIDATES]
    return {'normalization': norm, 'maximum_training_target_extent_lps_mm': maximum.tolist(),
            'requested_minimum_fov_mm': desired.tolist(), 'margin_mm': MARGIN_MM,
            'roundtrip_minimum_gate': ROUNDTRIP_GATE, 'candidates': candidates,
            'derived_from': '54training patients only; fixed before validation image preparation'}, provenance


def native_reference(grid, source, target):
    size, spacing = np.asarray(grid['size']), np.asarray(grid['spacing'])
    low = np.asarray(grid['origin']) - spacing / 2
    high = np.asarray(grid['origin']) + (size - .5) * spacing
    corners = np.asarray(list(itertools.product(*zip(low, high))))
    indices = source_points(corners, source)
    points = np.argwhere(target)
    start = np.maximum(0, np.minimum(np.floor(indices.min(0)).astype(int) - 2, points.min(0)))
    stop = np.minimum(source['size'], np.maximum(np.ceil(indices.max(0)).astype(int) + 3, points.max(0) + 1))
    native = {**source, 'size': (stop - start).tolist(), 'origin': world_points(start, source).tolist()}
    return native, tuple(slice(int(a), int(b)) for a, b in zip(start, stop)), start


def restore_logits(logits, sample):
    values = np.where(sample['valid'][0, 0].numpy(), np.asarray(logits, np.float32), -30.)
    return xyz(resample(image_xyz(values, sample['grid']), sample['native'], sitk.sitkLinear, -30.))


def make_sample(image, ct, mask, alias, side, plan, normalization):
    source = geometry(image)
    lo, hi = world_bounds(mask, source)
    center = (lo + hi) / 2
    shape, spacing = np.asarray(plan['shape']), np.asarray(plan['spacing'])
    grid = {'size': shape.tolist(), 'spacing': spacing.tolist(),
            'origin': (center - (shape - 1) * spacing / 2).tolist(), 'direction': np.eye(3).ravel().tolist()}
    binary = image_xyz(mask.astype(np.uint8), source)
    target = xyz(resample(binary, grid, sitk.sitkNearestNeighbor, pixel=sitk.sitkUInt8)).astype(bool)
    ones = sitk.Image(image.GetSize(), sitk.sitkUInt8) + 1
    ones.CopyInformation(image)
    valid = xyz(resample(ones, grid, sitk.sitkNearestNeighbor, pixel=sitk.sitkUInt8)).astype(bool)
    raw = xyz(resample(image, grid, sitk.sitkLinear, normalization['clip_lower']))
    normalized = (np.clip(raw, normalization['clip_lower'], normalization['clip_upper']) - normalization['mean']) / normalization['std']
    normalized[~valid] = 0
    if not target.any() or not valid[target].all() or not np.isfinite(normalized).all():
        raise ValueError('Invalid physical model-grid image/target/validity')
    native, selection, offset = native_reference(grid, source, mask)
    points = world_points(np.argwhere(mask), source)
    lower, upper = np.asarray(grid['origin']) - spacing / 2, np.asarray(grid['origin']) + (shape - .5) * spacing
    retention = float(np.all((points >= lower - 1e-6) & (points <= upper + 1e-6), axis=1).mean())
    native_target = mask[selection].copy()
    if int(native_target.sum()) != int(mask.sum()):
        raise ValueError('Native scoring patch lost original target voxels')
    metadata = {'crop_offset_xyz': offset.tolist(), 'full_target_voxels': int(mask.sum()),
                'target_retention': retention, 'crop_origin_mm': native['origin'],
                'spacing_mm': native['spacing'], 'direction': native['direction']}
    sample = {'case': alias, 'side': side, 'name': f'{alias}_{side}',
              'image': torch.from_numpy(normalized.astype(np.float32))[None, None],
              'target': torch.from_numpy(target.astype(np.uint8))[None, None],
              'valid': torch.from_numpy(valid)[None, None], 'grid': grid, 'native': native,
              'native_target': native_target, 'metadata': metadata, 'raw_native': ct[selection].copy(),
              'original_shape': source['size']}
    roundtrip = restore_logits(target.astype(np.float32) * 2 - 1, sample) > 0
    denominator = int(roundtrip.sum()) + int(mask.sum())
    sample['roundtrip_dice'] = 2 * int((roundtrip & native_target).sum()) / denominator
    sample['physical_margin_mm'] = np.minimum(lo - lower, upper - hi).tolist()
    return sample


def prepare_split(manifest, source_root, prior_sources, split, plan, normalization):
    expected = {row['patient_id']: row for row in prior_sources}
    samples, patients, provenance = [], {}, []
    for i, record in enumerate(records_for(manifest, split), 1):
        alias = f'{split}_{i:02d}'
        image, label, ct, labels, origin = load_case(record, source_root, expected[record['patient_id']], alias)
        patients[alias] = {'shape': list(image.GetSize()), 'target_indices': np.flatnonzero(np.isin(labels, (1, 9)).reshape(-1))}
        for side, value in (('RFSDP', 1), ('LFSDP', 9)):
            sample = make_sample(image, ct, labels == value, alias, side, plan, normalization)
            samples.append(sample)
            print(f'{sample["name"]}: retention={sample["metadata"]["target_retention"]:.4f} '
                  f'roundtrip={sample["roundtrip_dice"]:.4f} padding={1 - sample["valid"].float().mean().item():.3f}', flush=True)
        provenance.append(origin)
    return samples, patients, provenance


def coverage(samples):
    return {'side_rois': len(samples), 'minimum_target_retention': min(s['metadata']['target_retention'] for s in samples),
            'minimum_target_margin_mm': min(min(s['physical_margin_mm']) for s in samples),
            'minimum_roundtrip_dice': min(s['roundtrip_dice'] for s in samples),
            'mean_roundtrip_dice': float(np.mean([s['roundtrip_dice'] for s in samples])),
            'roundtrip_below_gate_sides': sum(s['roundtrip_dice'] < ROUNDTRIP_GATE for s in samples),
            'mean_padding_fraction': float(np.mean([1 - s['valid'].float().mean().item() for s in samples])),
            'mean_model_foreground_fraction': float(np.mean([s['target'].float().mean().item() for s in samples]))}


def prepare_dataset(manifest, source_root, prior_sources, output):
    output.mkdir()
    policy, inspected = inspect_training(manifest, source_root, prior_sources['train'])
    attempts = []
    for plan in policy['candidates']:
        samples, patients, provenance = prepare_split(manifest, source_root, prior_sources['train'], 'train', plan, policy['normalization'])
        measured = coverage(samples)
        eligible = (measured['minimum_target_retention'] == 1
                    and measured['minimum_roundtrip_dice'] >= ROUNDTRIP_GATE
                    and measured['minimum_target_margin_mm'] >= MARGIN_MM - 1e-5)
        attempts.append({'plan': plan, 'eligible': eligible, **measured})
        save(output / 'policy.json', {**policy, 'attempts': attempts, 'selected': plan if eligible else None})
        if eligible:
            break
        del samples, patients
    else:
        raise ValueError('No training-only physical grid preserves labels at minimum roundtrip Dice0.95; inspect policy.json')
    if provenance != inspected:
        raise ValueError('Training sources changed during physical planning')
    policy.update(attempts=attempts, selected=plan)
    save(output / 'policy.json', policy)
    print('Physical policy frozen from training data; preparing validation without refitting.', flush=True)
    val_samples, val_patients, val_sources = prepare_split(manifest, source_root, prior_sources['val'], 'val', plan, policy['normalization'])
    result = {'samples': {'train': samples, 'val': val_samples}, 'patients': {'train': patients, 'val': val_patients},
              'source_provenance': {'train': provenance, 'val': val_sources}, 'policy': policy,
              'coverage': {'train': measured, 'val': coverage(val_samples)}}
    if len(patients) != 54 or len(val_patients) != 10:
        raise ValueError('Expected the frozen54/10patient cohort')
    for split, collection in result['samples'].items():
        write_csv(output / f'{split}_coverage.csv', [{'case': s['case'], 'side': s['side'],
            'target_retention': s['metadata']['target_retention'], 'roundtrip_dice': s['roundtrip_dice'],
            'padding_fraction': 1 - s['valid'].float().mean().item()} for s in collection])
    save(output / 'summary.json', {'complete': True, 'policy': policy, 'coverage': result['coverage'],
         'train_patients': 54, 'validation_patients': 10, 'test_images_read': False,
         'note': 'Validation never selects the grid/normalization; all native target voxels remain in evaluation.'})
    return result


def masked_loss(criterion, logits, target, valid):
    if valid.shape != logits.shape or target.shape != logits.shape or not valid.any():
        raise ValueError('Invalid loss validity/target shape')
    if target[~valid].any():
        raise ValueError('Target outside valid CT support')
    return criterion(logits.float()[valid].reshape(1, 1, 1, 1, -1), target.float()[valid].reshape(1, 1, 1, 1, -1))


def augment(sample, seed, enabled):
    image, target, valid = [sample[key][0].clone() for key in ('image', 'target', 'valid')]
    if not enabled:
        return image[None], target[None], valid[None], False
    from monai.transforms import RandAffined
    rng = np.random.default_rng(seed)
    transform = RandAffined(keys=('image', 'target', 'valid'), spatial_size=list(image.shape[1:]), prob=.5,
        rotate_range=[math.radians(5)] * 3, translate_range=[1 / x for x in sample['grid']['spacing']],
        scale_range=[.05] * 3, mode=('bilinear', 'nearest', 'nearest'), padding_mode='zeros')
    transform.set_random_state(seed=seed % (2**32))
    changed = transform({'image': image, 'target': target.float(), 'valid': valid.float()})
    changed = {key: value.as_tensor() if hasattr(value, 'as_tensor') else value for key, value in changed.items()}
    transformed_target = changed['target'] > .5
    coordinates = torch.nonzero(transformed_target[0])
    safe = len(coordinates) > 0 and bool(((coordinates > 1) & (coordinates < torch.tensor(image.shape[1:]) - 2)).all())
    rejected = not safe
    if safe:
        image = changed['image']
        target, valid = transformed_target.to(torch.uint8), changed['valid'] > .5
    if rng.random() < .5:
        image, target, valid = [torch.flip(value, (1,)) for value in (image, target, valid)]
    if rng.random() < .3:
        image = image * float(rng.uniform(.9, 1.1)) + float(rng.uniform(-.1, .1))
    if rng.random() < .15:
        generator = torch.Generator().manual_seed(seed + 17)
        image += torch.randn(image.shape, generator=generator) * .02
    image[~valid] = 0
    if target[~valid].any():
        raise ValueError('Augmentation moved a target outside transformed CT validity')
    return image[None], target[None], valid[None], rejected


@torch.inference_mode()
def evaluate(model, samples, patients, device, thresholds, label='evaluation'):
    from train_private_unet import checked_logits
    model.eval()
    curves = {float(t): {'sides': [], 'patients': []} for t in thresholds}
    for alias in sorted(patients):
        collection = [s for s in samples if s['case'] == alias]
        if sorted(s['side'] for s in collection) != ['LFSDP', 'RFSDP']:
            raise ValueError('Both FSDP sides required')
        native = []
        for sample in collection:
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = checked_logits(model, sample['image'].to(device))
            native.append(restore_logits(logits[0, 0].float().cpu().numpy(), sample))
        for threshold, result in curves.items():
            parts = []
            cutoff = math.log(threshold / (1 - threshold))
            for sample, logits in zip(collection, native):
                prediction = logits > cutoff
                indices = native_prediction_indices(prediction, np.ones(prediction.shape, bool), sample['metadata'], patients[alias]['shape'])
                target = native_prediction_indices(sample['native_target'], np.ones(prediction.shape, bool), sample['metadata'], patients[alias]['shape'])
                result['sides'].append({'case': alias, 'side': sample['side'], 'threshold': threshold,
                    'target_retention': sample['metadata']['target_retention'], 'oracle_clicks': 0,
                    **sparse_scores(indices, target)})
                parts.append(indices)
            result['patients'].append({'case': alias, 'threshold': threshold, 'oracle_clicks': 0,
                **sparse_scores(np.concatenate(parts), patients[alias]['target_indices'])})
        print(f'{label}: {alias} complete', flush=True)
    if len(samples) != 2 * len(patients):
        raise ValueError('Incomplete sample/patient coverage')
    for threshold, result in curves.items():
        rows, sides = result['patients'], result['sides']
        result['metrics'] = {'mean_bilateral_patient_dice': float(np.mean([r['dice_full_target'] for r in rows])),
            'minimum_bilateral_patient_dice': min(r['dice_full_target'] for r in rows),
            'mean_side_dice': float(np.mean([r['dice_full_target'] for r in sides])),
            'minimum_side_dice': min(r['dice_full_target'] for r in sides),
            'mean_precision': float(np.mean([r['precision'] for r in rows])),
            'mean_recall': float(np.mean([r['recall_full_target'] for r in rows]))}
    return curves


def choose_threshold(curves):
    return max(curves, key=lambda t: (curves[t]['metrics']['mean_bilateral_patient_dice'], -abs(t - .5), -t))
