#!/usr/bin/env python3
"""Profile frozen FSDP training geometry, intensities and preprocessing without training."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import ndimage
import SimpleITK as sitk
import torch

import train_private_unet as baseline
from evaluate_private_validation import file_hash, write_csv
from roi_zoom import zoom_samples
from train_private_tiny import prepare_samples, select_patients


def read(path):
    return json.loads(path.read_text())


def distribution(values):
    array = np.asarray(values, dtype=float)
    if not array.size or not np.isfinite(array).all():
        raise ValueError('Profile statistics require nonempty finite values')
    return {'minimum': float(array.min()), 'p25': float(np.percentile(array, 25)),
            'median': float(np.median(array)), 'p75': float(np.percentile(array, 75)),
            'maximum': float(array.max())}


def intensity_stats(values):
    values = np.asarray(values)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Image intensities must be nonempty and finite')
    return {'minimum': float(values.min()), 'p01': float(np.percentile(values, 1)),
            'p0_5': float(np.percentile(values, .5)), 'p99_5': float(np.percentile(values, 99.5)),
            'median': float(np.median(values)), 'p99': float(np.percentile(values, 99)),
            'maximum': float(values.max()), 'mean': float(values.mean()), 'std': float(values.std())}


def target_stats(mask, spacing):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3 or not mask.any():
        raise ValueError('Expected a nonempty three-dimensional target')
    points = np.where(mask)
    extent = [int(index.max() - index.min() + 1) for index in points]
    _, counts = np.unique(ndimage.label(mask, structure=np.ones((3, 3, 3)))[0][mask], return_counts=True)
    return {'voxels': int(mask.sum()), 'volume_mm3': float(mask.sum() * np.prod(spacing)),
            'bbox_extent_voxels_xyz': extent, 'bbox_extent_mm_xyz': (np.asarray(extent) * spacing).tolist(),
            'components_26_connected': len(counts), 'largest_component_fraction': float(counts.max() / mask.sum())}


def pixel_hash(array):
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def sample_profile(sample):
    valid, target = sample['valid'], sample['native_target']
    if not valid.any() or not np.all(valid[target]):
        raise ValueError('Native target must lie inside valid source voxels')
    normalized = sample['image'][0, 0].numpy()
    model_target = sample['target'][0, 0].numpy().astype(bool)
    row = {'case': sample['case'], 'side': sample['side'],
           'target_retention': sample['metadata']['target_retention'],
           'native_crop_foreground_fraction': float(target.sum() / valid.sum()),
           'model_grid_foreground_fraction': float(model_target.mean()),
           'gt_label_roundtrip_dice': sample['gt_label_roundtrip_dice'],
           'padding_fraction_native_roi': float(1 - valid.mean()),
           'normalized_positive_hu_voxels_in_original_128_cube': sample['metadata']['normalization_positive_voxels']}
    for prefix, values in (('clamped_native_roi', sample['raw'][valid]),
                           ('clamped_native_target', sample['raw'][target]),
                           ('normalized_model_grid_all_voxels', normalized)):
        row.update({f'{prefix}_{key}': value for key, value in intensity_stats(values).items()})
    if model_target.any():
        row.update({f'normalized_model_target_{key}': value for key, value in intensity_stats(normalized[model_target]).items()})
    else:
        raise ValueError('Resize removed the model-grid target')
    return row


def execute(args):
    run, output = args.run_dir, args.output_dir
    summary, config = read(run / 'summary.json'), read(run / 'config.json')
    if (not summary['complete'] or summary['experiment'] != 'unet_cohort_train_calibrated'
            or summary['train_patients'] != 54 or summary['validation_patients'] != 10 or summary['test_images_read']):
        raise ValueError('Profile requires a completed frozen 54/10-patient U-Net cohort run')
    for name, checksum in config['code_sha256'].items():
        if file_hash(Path(__file__).with_name(name)) != checksum:
            raise ValueError(f'Frozen training/preprocessing code differs: {name}')
    source, data = Path(config['source_root']), Path(config['data_dir'])
    manifest = read(data / 'manifest.json')
    select_patients(manifest, config['seed'])
    if file_hash(data / 'manifest.json') != config['manifest_sha256']:
        raise ValueError('Frozen manifest differs')
    _, policy, versions = baseline.v4_control(SimpleNamespace(v4_run_dir=Path(config['v4_run_dir']), seed=config['seed']), data / 'manifest.json')
    if versions != config['versions'] or policy['selected_native_size_xyz'] != config['native_roi_size_xyz']:
        raise ValueError('Preprocessing recipe/version differs from cohort')
    references = read(run / 'source_provenance.json')['train']
    if (len(references) != 54 or len({r['patient_id'] for r in references}) != 54
            or set(r['patient_id'] for r in references) != set(manifest['splits']['train'])):
        raise ValueError('Training source coverage differs')
    patients, side_native, by_content, orientation = [], {}, defaultdict(list), Counter()
    foreground_values = []
    for row in references:
        image_path, label_path = Path(row['source_image']), Path(row['source_label'])
        if file_hash(image_path) != row['source_image_sha256'] or file_hash(label_path) != row['source_label_sha256']:
            raise ValueError('Training CT/final annotation differs from completed cohort')
        image, label = sitk.ReadImage(str(image_path)), sitk.ReadImage(str(label_path))
        ct = sitk.GetArrayViewFromImage(image)
        labels = sitk.GetArrayViewFromImage(label)
        if not np.isfinite(ct).all():
            raise ValueError('Source CT contains nonfinite intensities')
        spacing = np.asarray(image.GetSpacing())
        code = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(image.GetDirection())
        orientation[code] += 1
        by_content[pixel_hash(ct)].append(row['case'])
        item = {'case': row['case'], 'orientation': code, 'spacing_anisotropy': float(spacing.max() / spacing.min()),
                'source_fraction_below_minus1000': float((ct < -1000).mean()),
                'source_fraction_above1000': float((ct > 1000).mean())}
        for index, axis in enumerate('xyz'):
            item[f'spacing_{axis}_mm'] = float(spacing[index])
            item[f'source_size_{axis}_voxels'] = image.GetSize()[index]
            item[f'roi_extent_{axis}_mm'] = float(policy['selected_native_size_xyz'][index] * spacing[index])
            item[f'model_grid_spacing_{axis}_mm'] = item[f'roi_extent_{axis}_mm'] / 128
        patients.append(item)
        for side, value in (('RFSDP', 1), ('LFSDP', 9)):
            mask = labels == value
            foreground_values.append(ct[mask].astype(np.float32))
            metrics = target_stats(mask.transpose(2, 1, 0), spacing)
            native = {'case': row['case'], 'side': side,
                      **{key: value for key, value in metrics.items() if not isinstance(value, list)},
                      **{f'source_target_{key}': val for key, val in intensity_stats(ct[mask]).items()}}
            for key in ('bbox_extent_voxels_xyz', 'bbox_extent_mm_xyz'):
                native.update({f'{key}_{axis}': val for axis, val in zip('xyz', metrics[key])})
            side_native[(row['case'], side)] = native
        print(f'{row["case"]}: source geometry/intensity/target profile complete', flush=True)
        del image, label, ct, labels, mask
    preparation = output / 'preparation'
    preparation.mkdir()
    samples, loaded_patients = prepare_samples(source, data, manifest, preparation, config['seed'], all_training_patients=True)
    if read(preparation / 'selected_training_patients.json') != references:
        raise ValueError('Reconstructed training sources differ from cohort')
    for sample in samples:
        sample['target'] = sample['target'].to(torch.uint8)
    samples = zoom_samples(samples, policy['selected_native_size_xyz'])
    if len(samples) != 108 or len(loaded_patients) != 54:
        raise ValueError('Training ROI coverage differs')
    sides = [{**side_native[(s['case'], s['side'])], **sample_profile(s)} for s in samples]
    write_csv(output / 'patients.csv', patients)
    write_csv(output / 'sides.csv', sides)
    patient_distributions = {key: distribution([row[key] for row in patients]) for key in patients[0]
                             if key not in ('case', 'orientation')}
    side_distributions = {key: distribution([row[key] for row in sides]) for key in sides[0]
                          if key not in ('case', 'side')}
    result = {'complete': True, 'cohort_run': run.name, 'training_patients': 54, 'training_sides': 108,
              'native_roi_size_xyz': policy['selected_native_size_xyz'], 'model_size_xyz': [128] * 3,
              'orientation_counts': dict(orientation), 'patient_distributions': patient_distributions,
              'side_distributions': side_distributions,
              'pooled_training_target_source_intensities': intensity_stats(np.concatenate(foreground_values)),
              'identical_source_pixel_groups_training_only': [aliases for aliases in by_content.values() if len(aliases) > 1],
              'cohort_reference': {'validation_dice': summary['best_validation_metrics']['mean_bilateral_patient_dice'],
                                   'frozen_training_dice': summary['best_frozen_training_metrics']['mean_bilateral_patient_dice'],
                                   'threshold': summary['threshold'], 'best_epoch': summary['best_epoch']},
              'implementation_findings': {
                  'orientation': 'Original physical affine discarded before tensor-only ToCanonical; source orientation is not harmonized by that call.',
                  'spacing': 'Fixed native voxel cuboid resized to128 cube; no common physical spacing across patients.',
                  'normalization': 'Clamp[-1000,1000]; per-128-cube ZNormalization uses values>0 for mean/std before tighter crop.',
                  'augmentation': 'Training currently uses independent XYZ flips only; anatomy-sensitive axes need review.'},
              'interpretation': 'Audit of training inputs, not a diagnosis of annotation error or proof of a Dice ceiling. Clinical correctness requires expert review; disconnected components are not automatically errors. Pixel-identical arrays require review and are not automatically excluded. Numerical CT ranges alone do not verify HU calibration.',
              'validation_images_read': False, 'test_images_read': False, 'optimizer_updates': 0,
              'training_artifacts_modified': False, 'source_data_modified': False,
              'input_artifact_sha256': {name: file_hash(run / name) for name in ('summary.json', 'config.json', 'source_provenance.json')},
              'profiler_sha256': file_hash(Path(__file__)), 'artifacts': str(output)}
    baseline.save_json(output / 'summary.json', result)
    compact = {key: value for key, value in result.items() if key not in ('patient_distributions', 'side_distributions')}
    compact.update(patient_distributions={key: patient_distributions[key] for key in
        ('spacing_x_mm', 'spacing_y_mm', 'spacing_z_mm', 'spacing_anisotropy', 'roi_extent_x_mm', 'roi_extent_y_mm', 'roi_extent_z_mm')},
        side_distributions={key: side_distributions[key] for key in
            ('voxels', 'volume_mm3', 'components_26_connected', 'largest_component_fraction', 'target_retention',
             'model_grid_foreground_fraction', 'gt_label_roundtrip_dice', 'normalized_model_target_mean', 'padding_fraction_native_roi')})
    baseline.save_json(output / 'overview.json', compact)
    print(json.dumps(compact, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.run_dir, args.output_dir = args.run_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    config = read(args.run_dir / 'config.json')
    if args.output_dir.exists() or any(args.output_dir.is_relative_to(folder) for folder in
            (args.run_dir, Path(config['data_dir']).resolve(), Path(config['source_root']).resolve())):
        parser.error('Use a new output directory outside training/source/prepared data')
    args.output_dir.mkdir(parents=True)
    try:
        execute(args)
    except Exception as error:
        baseline.save_json(args.output_dir / 'failure.json', {'error_type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
