"""Use the original FSDP cache as target authority when annotation files are unavailable."""
from pathlib import Path

import numpy as np
import torch

from audit_foundation_sources import digest


def image_inventory(provenance, source_root=None, verify=False):
    if set(provenance) != {'train', 'val'}:
        raise ValueError('Verified cache supports train/val only')
    rows = []
    for split, records in provenance.items():
        for record in records:
            path = Path(record['source_image'])
            if source_root is not None:
                path = Path(source_root).expanduser() / 'imagesTr' / path.name
            path = path.resolve()
            row = {'split': split, 'case': record['case'], 'path': str(path),
                   'expected_sha256': record['source_image_sha256'], 'status': 'missing'}
            if path.is_file():
                row['status'] = 'present_unchecked'
                if verify:
                    row['actual_sha256'] = digest(path)
                    row['status'] = 'ok' if row['actual_sha256'] == row['expected_sha256'] else 'changed'
            rows.append(row)
    return {'ready': all(r['status'] in ('ok', 'present_unchecked') for r in rows),
            'mode': 'verified_cache', 'rows': rows, 'ct_content_verified': verify,
            'annotation_files_verified': False, 'test_images_read': False,
            'target_authority': 'original checksum-verified hybrid_data.pt; no replacement annotations',
            'limitation': 'Original label NIfTI/ColorTable files are not recovered or independently reverified.'}


def require_images(report):
    if not report['ready']:
        problems = [f'{r["case"]}: {r["status"]}: {r["path"]}' for r in report['rows']
                    if r['status'] not in ('ok', 'present_unchecked')]
        raise ValueError('Verified-cache CT inputs unavailable or changed; no training started.\n' + '\n'.join(problems[:8]))


def validate_targets(prepared, manifest):
    """Check split identity, full native targets, bilateral union and cached input support."""
    from train_private_tiny import native_prediction_indices, select_patients
    select_patients(manifest, 2026)
    for key in ('samples', 'patients', 'source_provenance'):
        if set(prepared[key]) != {'train', 'val'}:
            raise ValueError('Cache contains unsupported split data')
    report = {'passed': True, 'splits': {}, 'test_images_read': False,
              'cached_validation_targets_read': True, 'original_annotation_files_read': False}
    for split in ('train', 'val'):
        ids = sorted(manifest['splits'][split])
        aliases = [f'{split}_{i:02d}' for i in range(1, len(ids) + 1)]
        sources = prepared['source_provenance'][split]
        patients, samples = prepared['patients'][split], prepared['samples'][split]
        if (len(sources) != len(ids) or {s['case']: s['patient_id'] for s in sources} != dict(zip(aliases, ids))
                or set(patients) != set(aliases) or len(samples) != 2 * len(ids)
                or {s['case'] for s in samples} != set(aliases)):
            raise ValueError('Cache patient identity/split/ROI membership differs from frozen manifest')
        for alias in aliases:
            parts = []
            sides = [s for s in samples if s['case'] == alias]
            if sorted(s['side'] for s in sides) != ['LFSDP', 'RFSDP']:
                raise ValueError('Cache requires exactly two distinct FSDP sides per patient')
            shape = patients[alias]['shape']
            for sample in sides:
                if list(sample['original_shape']) != list(shape):
                    raise ValueError('Cache original shape differs')
                for prefix, grid_name in (('', 'grid'), ('sam_', 'sam_grid')):
                    image, target, valid = [sample[prefix + key] for key in ('image', 'target', 'valid')]
                    wanted = (1, 1, *sample[grid_name]['size'])
                    if (tuple(image.shape) != wanted or target.shape != image.shape or valid.shape != image.shape
                            or not torch.isfinite(image).all() or not target.any()
                            or not ((target == 0) | (target == 1)).all()
                            or not ((valid == 0) | (valid == 1)).all()
                            or not valid[target.bool()].all()):
                        raise ValueError('Cache model/context tensors or target support invalid')
                target = sample['native_target']
                metadata = sample['metadata']
                if (list(target.shape) != sample['native']['size'] or not np.isin(target, (0, 1)).all()
                        or int(target.sum()) != metadata['full_target_voxels'] or not target.any()
                        or metadata['target_retention'] != 1.):
                    raise ValueError('Cache native target lost foreground or has invalid geometry')
                indices = native_prediction_indices(target, np.ones(target.shape, bool), metadata, shape)
                if len(indices) != metadata['full_target_voxels']:
                    raise ValueError('Cache native target contains out-of-volume voxels')
                parts.append(indices)
            if (np.intersect1d(*parts).size
                    or not np.array_equal(np.union1d(*parts), np.unique(patients[alias]['target_indices']))):
                raise ValueError('Cache side targets differ from original full bilateral target')
        report['splits'][split] = {'patients': len(ids), 'side_rois': len(samples),
                                  'native_targets_intact': len(samples), 'bilateral_targets_identical': True}
    return report


def read_image(record, source_root, expected, samples):
    import physical_fsdp as data
    path = Path(source_root).resolve() / 'imagesTr' / Path(expected['source_image']).name
    if (Path(record['source_image']).name != path.name or digest(path) != expected['source_image_sha256']):
        raise ValueError('Original CT differs from cached foundation source')
    image = data.sitk.ReadImage(str(path))
    for sample in samples:
        native = sample['native']
        offset = sample['metadata']['crop_offset_xyz']
        if (list(image.GetSize()) != list(sample['original_shape'])
                or not np.allclose(image.GetSpacing(), native['spacing'], rtol=0, atol=1e-6)
                or not np.allclose(image.GetDirection(), native['direction'], rtol=0, atol=1e-6)
                or not np.allclose(image.TransformIndexToPhysicalPoint([int(n) for n in offset]),
                                   native['origin'], rtol=0, atol=1e-5)):
            raise ValueError('Original CT geometry differs from cached native target')
    return image


def source_label(sample, image):
    import physical_fsdp as data
    offset = [int(n) for n in sample['metadata']['crop_offset_xyz']]
    target = sample['native_target']
    if any(a < 0 or a + n > full for a, n, full in zip(offset, target.shape, image.GetSize())):
        raise ValueError('Cached native target patch falls outside original CT')
    destination = data.sitk.Image(image.GetSize(), data.sitk.sitkUInt8)
    destination.CopyInformation(image)
    native = data.image_xyz(target.astype(np.uint8), sample['native'])
    return data.sitk.Paste(destination, native, native.GetSize(), [0, 0, 0], offset)
