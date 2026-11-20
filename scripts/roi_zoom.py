"""Training-selected ROI zoom with native-grid scoring and frozen v3 controls."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_private_validation import file_hash, infer_clicks, save_crop, scores, write_csv
from train_private_tiny import evaluate as evaluate_native, native_prediction_indices, sparse_scores


def choose_policy(samples, cubic_candidates=(64, 96), margin=4):
    if any(not sample['case'].startswith('train_') for sample in samples):
        raise ValueError('ROI size selection accepts training patients only')
    if len(samples) != 108 or len({sample['case'] for sample in samples}) != 54:
        raise ValueError('ROI policy requires all 54 training patients and 108 side ROIs')
    if margin < 0 or not cubic_candidates or tuple(sorted(set(cubic_candidates))) != tuple(cubic_candidates):
        raise ValueError('Cubic diagnostic candidates must be unique and ascending, with a nonnegative margin')
    if any([sample['side'] for sample in samples if sample['case'] == alias].count(side) != 1
           for alias in {sample['case'] for sample in samples} for side in ('RFSDP', 'LFSDP')):
        raise ValueError('Both training sides are required')
    cubic_report = []
    for size in cubic_candidates:
        if size <= 0 or size >= 128 or (128 - size) % 2:
            raise ValueError('Cubic diagnostic sizes must be smaller centered even cubes')
        start = (128 - size) // 2
        retention, borders = [], []
        for sample in samples:
            target = sample['target'][0, 0]
            if tuple(target.shape) != (128, 128, 128) or not target.any():
                raise ValueError('Expected a nonempty native 128-cube training target')
            points = torch.nonzero(target, as_tuple=False)
            cropped = target[start:start + size, start:start + size, start:start + size]
            retention.append(int(cropped.sum()) / int(target.sum()))
            low, high = points.min(dim=0).values, points.max(dim=0).values
            borders.append(int(torch.minimum(low - start, start + size - 1 - high).min()))
        eligible = all(value == 1. for value in retention) and min(borders) >= margin
        cubic_report.append({'native_size_xyz': [size, size, size], 'minimum_training_target_retention': min(retention),
                       'mean_training_target_retention': float(np.mean(retention)),
                       'fully_retained_sides': sum(value == 1. for value in retention),
                       'minimum_target_border_voxels': min(borders), 'eligible': eligible})
    axes = {}
    native_xyz = []
    for axis, name in enumerate(('X', 'Y', 'Z')):
        required = []
        for sample in samples:
            points = torch.nonzero(sample['target'][0, 0], as_tuple=False)
            low, high = int(points[:, axis].min()), int(points[:, axis].max())
            eligible_sizes = [size for size in range(2, 129, 2)
                              if (128 - size) // 2 + margin <= low
                              and high <= (128 - size) // 2 + size - 1 - margin]
            required.append(min(eligible_sizes) if eligible_sizes else None)
        valid = [size for size in required if size is not None]
        selected = max(valid) if len(valid) == len(samples) else None
        axes[name] = {'selected_native_size': selected, 'sides_with_valid_size': len(valid),
                      'required_size_min_median_max': [min(valid), float(np.median(valid)), max(valid)] if valid else None}
        native_xyz.append(selected)
    selected_box = native_xyz if all(size is not None for size in native_xyz) else None
    return {'selected_native_size_xyz': selected_box, 'model_size_xyz': [128, 128, 128],
            'axis_selection': axes, 'margin_voxels': margin, 'training_patients': 54, 'training_sides': 108,
            'cubic_diagnostics': cubic_report,
            'selection': 'Smallest centered crop independently per XYZ axis retaining every training target with the specified margin; no validation targets used'}


def resize_labels(target, size_xyz=(128, 128, 128)):
    # Pixel centers match trilinear image resizing with align_corners=False.
    result = target
    if len(size_xyz) != 3 or any(size <= 0 for size in size_xyz):
        raise ValueError('Model target size must contain three positive dimensions')
    for axis, (length, size) in enumerate(zip(target.shape[-3:], size_xyz), 2):
        index = torch.floor((torch.arange(size, device=target.device) + .5) * length / size).long().clamp(max=length - 1)
        result = result.index_select(axis, index)
    return result.to(torch.uint8)


def restore_logits(logits, native_shape):
    tensor = torch.as_tensor(logits, dtype=torch.float32)[None, None]
    return F.interpolate(tensor, size=native_shape, mode='trilinear', align_corners=False)[0, 0].numpy()


def zoom_samples(samples, native_size_xyz):
    if len(native_size_xyz) != 3 or any(size <= 0 or size > 128 or (128 - size) % 2 for size in native_size_xyz):
        raise ValueError('Native ROI must be a centered even XYZ cuboid no larger than 128³')
    starts = [(128 - size) // 2 for size in native_size_xyz]
    selection = tuple(slice(start, start + size) for start, size in zip(starts, native_size_xyz))
    result = []
    for sample in samples:
        if tuple(sample['target'].shape[-3:]) != (128, 128, 128):
            raise ValueError('Zoom expects the original native 128-cube preprocessing')
        target = sample['target'][(...,) + selection].clone().to(torch.uint8)
        metadata = dict(sample['metadata'])
        if int(sample['target'].sum()) != metadata['full_target_voxels']:
            raise ValueError('Original 128-cube does not retain the full target')
        direction = np.asarray(metadata['direction']).reshape(3, 3)
        metadata['crop_origin_mm'] = (np.asarray(metadata['crop_origin_mm']) + direction @ (np.asarray(starts) * metadata['spacing_mm'])).tolist()
        metadata['crop_offset_xyz'] = (np.asarray(metadata['crop_offset_xyz']) + starts).tolist()
        metadata['crop_target_voxels'] = int(target.sum())
        metadata['target_retention'] = int(target.sum()) / metadata['full_target_voxels']
        valid = sample['valid'][selection].copy()
        metadata['crop_padding_voxels'] = int((~valid).sum())
        image = F.interpolate(sample['image'][(...,) + selection], size=(128,) * 3, mode='trilinear', align_corners=False)
        model_target = resize_labels(target)
        roundtrip = restore_logits(model_target[0, 0].float().numpy() * 2 - 1, tuple(native_size_xyz)) > 0
        gt = target[0, 0].numpy().astype(bool)
        if not gt.any():
            raise ValueError(f"Empty native ROI for {sample['name']}; the chosen training policy is invalid")
        count = int(gt.sum()) + int(roundtrip.sum())
        item = {**sample, 'image': image, 'target': model_target,
                'raw': sample['raw'][selection].copy(), 'metadata': metadata, 'valid': valid,
                'native_target': gt, 'native_size_xyz': list(native_size_xyz),
                'gt_label_roundtrip_dice': 2 * int((gt & roundtrip).sum()) / count if count else None}
        result.append(item)
    return result


def overlay_native(path, sample, prediction):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    target, image = sample['native_target'], sample['raw']
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    for row, axis in enumerate((2, 1, 0)):
        reductions = tuple(i for i in range(3) if i != axis)
        index = int(np.argmax(target.sum(axis=reductions))) if target.any() else target.shape[axis] // 2
        ct, gt, pred = [np.take(value, index, axis=axis).T for value in (image, target, prediction)]
        for column in range(3):
            ax = axes[row, column]
            ax.imshow(ct, cmap='gray', origin='lower', vmin=-1000, vmax=1000)
            for mask, color, active in [(gt, 'cyan', column != 1), (pred, 'magenta', column != 0)]:
                if active and mask.any():
                    ax.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([color]), origin='lower', alpha=.2, vmin=0, vmax=1)
                if active and mask.any() and not mask.all():
                    ax.contour(mask, levels=[.5], colors=[color], linewidths=.8)
            ax.set_title(f'Native axis {axis}, slice {index}: ' + ['GT', 'prediction', 'GT + prediction'][column], fontsize=8)
            ax.axis('off')
    fig.suptitle(f"Native ROI XYZ {sample['native_size_xyz']} / model input 128³; target retained {sample['metadata']['target_retention']:.1%}; probability threshold 0.5")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


@torch.inference_mode()
def evaluate_zoom(model, samples, patients, device, export=None, progress_label=None):
    model.eval()
    sides, bilateral, restored = [], [], {}
    if export is not None:
        export.mkdir(parents=True)
    for index, sample in enumerate(samples, 1):
        outputs = infer_clicks(model, sample['image'].to(device), sample['target'].to(device), [1, 5], sample['prompt_seed'])
        for clicks, logits in outputs.items():
            prediction = restore_logits(logits, sample['native_target'].shape) > 0
            metadata = sample['metadata']
            sides.append({'case': sample['case'], 'side': sample['side'], 'clicks': clicks,
                          'target_retention': metadata['target_retention'],
                          **scores(prediction, sample['native_target'], metadata['full_target_voxels'], sample['valid'])})
            shape = patients[sample['case']]['shape']
            restored.setdefault((sample['case'], clicks), []).append(native_prediction_indices(prediction, sample['valid'], metadata, shape))
            if export is not None:
                name = f"{sample['name']}_{clicks}_clicks"
                save_crop(export / f'{name}_prediction_crop.nii.gz', prediction.astype(np.uint8), metadata)
                save_crop(export / f'{name}_target_crop.nii.gz', sample['native_target'].astype(np.uint8), metadata)
                overlay_native(export / f'{name}_overlay.png', sample, prediction)
        if progress_label:
            print(f'{progress_label}: {sample["name"]} complete ({index}/{len(samples)})', flush=True)
    for (alias, clicks), parts in sorted(restored.items()):
        bilateral.append({'case': alias, 'clicks': clicks, **sparse_scores(np.concatenate(parts), patients[alias]['target_indices'])})
    means = {str(clicks): {'mean_side_dice': float(np.mean([row['dice_full_target'] for row in sides if row['clicks'] == clicks])),
                          'minimum_side_dice': min(row['dice_full_target'] for row in sides if row['clicks'] == clicks),
                          'mean_bilateral_patient_dice': float(np.mean([row['dice_full_target'] for row in bilateral if row['clicks'] == clicks])),
                          'minimum_bilateral_patient_dice': min(row['dice_full_target'] for row in bilateral if row['clicks'] == clicks)} for clicks in (1, 5)}
    if export is not None:
        write_csv(export / 'sides.csv', sides)
        write_csv(export / 'patients.csv', bilateral)
    return means, sides, bilateral


def compare_saved(directory, sides, bilateral):
    maximum = 0.
    for name, rows, fields in [('sides', sides, ('case', 'side', 'clicks')), ('patients', bilateral, ('case', 'clicks'))]:
        with (directory / f'{name}.csv').open(newline='') as stream:
            saved = list(csv.DictReader(stream))
        key = lambda row: tuple(str(row[field]) for field in fields)
        lookup = {key(row): float(row['dice_full_target']) for row in saved}
        actual = {key(row): row['dice_full_target'] for row in rows}
        if len(saved) != len(rows) or len(actual) != len(rows) or set(lookup) != set(actual):
            raise ValueError('V3 control case/click coverage differs')
        for alias, value in actual.items():
            error = abs(value - lookup[alias])
            maximum = max(maximum, error)
            if error > 1e-4:
                raise ValueError('V3 checkpoint fails to reproduce its saved native validation')
    return maximum


class ZoomExperiment:
    evaluate = staticmethod(evaluate_zoom)

    def configuration(self, args, manifest, config):
        v3 = args.v3_run_dir
        self.v3_summary = json.loads((v3 / 'summary.json').read_text())
        old_config = json.loads((v3 / 'config.json').read_text())
        path = v3 / 'sam_model_side_val_best.pth'
        if not self.v3_summary['complete'] or self.v3_summary['test_images_read'] or self.v3_summary['train_patients'] != 54:
            raise ValueError('V4 requires the completed 54-patient v3 control')
        for name in ('epochs', 'val_every', 'accumulation', 'lr', 'seed', 'patience_epochs', 'min_delta'):
            if old_config[name] != getattr(args, name):
                raise ValueError(f'Keep {name} identical to v3 for this ROI-only experiment')
        frozen_fields = ('manifest_sha256', 'initial_checkpoint_sha256', 'baseline_sha256', 'versions',
                         'sam_builder_sha256', 'sam_source_sha256')
        if any(old_config.get(name) != config.get(name) for name in frozen_fields):
            raise ValueError('V3 control uses different data, baseline, software, SAM source or Turbo initialization')
        shared = ('train_private_tiny.py', 'evaluate_private_validation.py', 'train_private_long.py',
                  'audit_expert_agreement.py')
        if any(old_config['shared_code_sha256'][name] != config['shared_code_sha256'][name] for name in shared):
            raise ValueError('Preprocessing, inference, optimization or comparison helpers differ from v3')
        if file_hash(path) != self.v3_summary['best_checkpoint_sha256']:
            raise ValueError('V3 best checksum differs from its completed summary')
        self.v3_path = path
        return {'experiment': 'v4_anisotropic_roi_zoom', 'zoom_module_sha256': file_hash(Path(__file__)),
                'zoom_entrypoint_sha256': file_hash(Path(__file__).with_name('train_private_zoom.py')),
                'v3_control_sha256': {name: file_hash(v3 / name) for name in ('summary.json', 'config.json', 'source_provenance.json', 'sam_model_side_val_best.pth')},
                'native_roi_shape_selection': 'independent centered even crop size per XYZ axis, train only; four-voxel margin', 'native_roi_margin': 4,
                'normalization': 'Preserve positive-HU Z-normalization of original 128 cube before cropping/resizing',
                'protocol': 'Fixed anisotropic native XYZ ROI selected independently per axis from training-only masks with full retention and 4-voxel margin; CT trilinear to128 cube; labels nearest pixel centers; restore logits to native before threshold; full original target scored; 1/5 oracle clicks per side'}

    def prepare_split(self, split, samples, patients, directory, run, resume):
        if split == 'train':
            policy = choose_policy(samples)
            if resume:
                if json.loads((run / 'roi_policy.json').read_text()) != policy:
                    raise ValueError('Training-selected ROI policy changed since the frozen run')
            else:
                (run / 'roi_policy.json').write_text(json.dumps(policy, indent=2))
            if policy['selected_native_size_xyz'] is None:
                raise ValueError('No centered XYZ crop retains all training targets with margin; inspect roi_policy.json before another experiment')
            self.native_size_xyz = policy['selected_native_size_xyz']
            print(f'Training-only ROI policy selected: native XYZ {self.native_size_xyz} -> model 128³', flush=True)
        else:
            self.native_validation = samples
        zoomed = zoom_samples(samples, self.native_size_xyz)
        rows = [{'case': item['case'], 'side': item['side'], 'native_size_xyz': self.native_size_xyz,
                 'full_target_voxels': item['metadata']['full_target_voxels'],
                 'native_crop_target_voxels': item['metadata']['crop_target_voxels'],
                 'target_retention': item['metadata']['target_retention'],
                 'gt_label_roundtrip_dice': item['gt_label_roundtrip_dice']} for item in zoomed]
        if split == 'train' and any(row['target_retention'] != 1. for row in rows):
            raise ValueError('Selected ROI truncates a training target')
        write_csv(directory / 'zoom_coverage.csv', rows)
        self.retention = getattr(self, 'retention', {})
        self.retention[split] = {'minimum': min(row['target_retention'] for row in rows),
                                 'mean': float(np.mean([row['target_retention'] for row in rows])),
                                 'truncated_sides': sum(row['target_retention'] < 1 for row in rows),
                                 'empty_native_crops': sum(row['native_crop_target_voxels'] == 0 for row in rows)}
        return zoomed

    def initial_validation(self, model, samples, patients, device, args, run):
        from train_private_sides import reproduce_baseline
        current = json.loads((run / 'source_provenance.json').read_text())
        if current != json.loads((args.v3_run_dir / 'source_provenance.json').read_text()):
            raise ValueError('Source images/final labels differ from the completed v3 run')
        native, sides, bilateral = evaluate_native(model, self.native_validation, patients, device, progress_label='turbo_native_control')
        error = reproduce_baseline(args.baseline_dir, sides, bilateral)
        source = torch.load(self.v3_path, map_location='cpu', weights_only=False)
        if source['epoch'] != self.v3_summary['best_epoch']:
            raise ValueError('V3 checkpoint epoch differs from summary')
        model.load_state_dict(source['model_state_dict'], strict=True)
        del source
        v3_native, sides, bilateral = evaluate_native(model, self.native_validation, patients, device, progress_label='v3_native_control')
        saved = Path(self.v3_summary['export_directory']).resolve()
        if not saved.is_relative_to(args.v3_run_dir):
            raise ValueError('V3 exports are outside its run')
        v3_error = compare_saved(saved, sides, bilateral)
        v3_zoom, sides, bilateral = evaluate_zoom(model, samples, patients, device, progress_label='v3_zoom_control')
        control = run / 'v3_zoom_control'
        control.mkdir()
        write_csv(control / 'sides.csv', sides)
        write_csv(control / 'patients.csv', bilateral)
        source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(source.get('model_state_dict', source), strict=True)
        del source
        initial, sides, bilateral = evaluate_zoom(model, samples, patients, device, progress_label='turbo_zoom_initial')
        (run / 'roi_controls.json').write_text(json.dumps({'turbo_native': native, 'turbo_zoom': initial,
            'v3_native': v3_native, 'v3_zoom': v3_zoom, 'v3_maximum_reproduction_error': v3_error}, indent=2))
        return initial, sides, bilateral, error

    def finalize(self, model, patients_by_split, device, means, args, run, provenance):
        controls = json.loads((run / 'roi_controls.json').read_text())
        native, sides, bilateral = evaluate_native(model, self.native_validation, patients_by_split['val'], device, progress_label='selected_v4_native_control')
        write_csv(run / 'selected_v4_native_sides.csv', sides)
        write_csv(run / 'selected_v4_native_patients.csv', bilateral)
        matrix = [{'checkpoint': checkpoint, 'roi': roi, 'validation_dice': value} for checkpoint, roi, value in
                  [('v3', 'native128', controls['v3_native']), ('v3', f'zoomXYZ{self.native_size_xyz}', controls['v3_zoom']),
                   ('v4', 'native128', native), ('v4', f'zoomXYZ{self.native_size_xyz}', means)]]
        return {'native_roi_size_xyz': self.native_size_xyz, 'roi_coverage': self.retention,
                'v3_validation_same_zoom_protocol': controls['v3_zoom'],
                'validation_dice_change_vs_v3_same_zoom': means['5']['mean_bilateral_patient_dice'] - controls['v3_zoom']['5']['mean_bilateral_patient_dice'],
                'validation_roi_comparison': matrix,
                'roi_comparison_note': 'V4 selected by zoom validation only; crop context and model weights have separate controls. GT-assisted, not automatic inference.'}
