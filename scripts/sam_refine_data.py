"""Physically aligned detail and128³SAM context views on the frozen private split."""

import numpy as np
import torch

import physical_fsdp as physical
from evaluate_private_validation import file_hash


def add_contexts(prepared, manifest, source_root, prior_sources):
    plan, norm = prepared['policy']['selected'], prepared['policy']['normalization']
    context_extent = float(np.max(np.asarray(plan['shape']) * plan['spacing']))
    for split in ('train', 'val'):
        expected = {row['patient_id']: row for row in prior_sources[split]}
        for i, record in enumerate(physical.records_for(manifest, split), 1):
            alias = f'{split}_{i:02d}'
            image, label, ct, labels, provenance = physical.load_case(record, source_root, expected[record['patient_id']], alias)
            for sample in [s for s in prepared['samples'][split] if s['case'] == alias]:
                detail = sample['grid']
                detail_extent = np.asarray(detail['size']) * detail['spacing']
                center = np.asarray(detail['origin']) + (np.asarray(detail['size']) - 1) * np.asarray(detail['spacing']) / 2
                step = context_extent / 128
                grid = {'size': [128] * 3, 'spacing': [step] * 3,
                        'origin': (center - 127 * step / 2).tolist(), 'direction': np.eye(3).ravel().tolist()}
                value = 1 if sample['side'] == 'RFSDP' else 9
                binary = physical.image_xyz((labels == value).astype(np.uint8), physical.geometry(image))
                ones = physical.sitk.Image(image.GetSize(), physical.sitk.sitkUInt8) + 1
                ones.CopyInformation(image)
                target = physical.xyz(physical.resample(binary, grid, physical.sitk.sitkNearestNeighbor, pixel=physical.sitk.sitkUInt8))
                valid = physical.xyz(physical.resample(ones, grid, physical.sitk.sitkNearestNeighbor, pixel=physical.sitk.sitkUInt8)).astype(bool)
                raw = physical.xyz(physical.resample(image, grid, physical.sitk.sitkLinear, norm['clip_lower']))
                normalized = (np.clip(raw, norm['clip_lower'], norm['clip_upper']) - norm['mean']) / norm['std']
                normalized[~valid] = 0
                if not target.any() or not valid[target.astype(bool)].all() or not np.isfinite(normalized).all():
                    raise ValueError('Invalid SAM context target or CT support')
                sample.update(sam_image=torch.from_numpy(normalized.astype(np.float32))[None, None],
                    sam_target=torch.from_numpy(target.astype(np.uint8))[None, None], sam_valid=torch.from_numpy(valid)[None, None],
                    sam_grid=grid, fov_ratio=torch.tensor(detail_extent / context_extent, dtype=torch.float32),
                    detail_spacing=torch.tensor(detail['spacing'], dtype=torch.float32))
            print(f'{alias}: SAM context and detail physical centers aligned', flush=True)
    prepared['policy']['sam_context'] = {'size': [128] * 3, 'physical_extent_mm': [context_extent] * 3,
        'spacing_mm': context_extent / 128, 'center': 'same physical side bbox center as detail',
        'detail_feature_grid': [32] * 3, 'source': 'context size derived from training-selected detail field only'}
    return prepared


def paired_inputs(sample, device, seed=None):
    keys = ('image', 'target', 'valid', 'sam_image', 'sam_target', 'sam_valid')
    result = {key: sample[key].clone() for key in keys}
    if seed is not None:
        generator = np.random.default_rng(seed)
        if generator.random() < .5:
            result = {key: torch.flip(value, (2,)) for key, value in result.items()}
        if generator.random() < .3:
            gain, shift = float(generator.uniform(.9, 1.1)), float(generator.uniform(-.1, .1))
            result['image'] = result['image'] * gain + shift
            result['sam_image'] = result['sam_image'] * gain + shift
    result['image'][~result['valid']] = 0
    result['sam_image'][~result['sam_valid']] = 0
    for key in keys:
        result[key] = result[key].to(device)
    result.update(ratio=sample['fov_ratio'].to(device), spacing=sample['detail_spacing'].to(device))
    return result


def source_hashes(sam_dir):
    package = sam_dir / 'segment_anything'
    files = sorted(package.rglob('*.py'))
    if not files or not (package / 'build_sam3D.py').is_file():
        raise ValueError('Official SAM-Med3D source directory is missing')
    return {str(path.relative_to(sam_dir)): file_hash(path) for path in files}
