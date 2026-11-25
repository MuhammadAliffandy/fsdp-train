"""Train-only physical translations within the verified cached detail/context views."""
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from fsdp_research_model import ResearchModel


POLICY = {'probability': .5, 'maximum_translation_mm_per_axis': 1., 'target_margin_voxels': 1,
          'sampling': 'integer detail voxels; independent seed stream; same physical offset in both views',
          'interpolation': 'CT trilinear; binary target/support nearest; align_corners=False',
          'unsafe_transform': 'retain both original views if either target loses voxels or support',
          'scope': 'training only, within cached FOV; no new anatomy outside cache; evaluation unchanged'}


def _translate(image, target, valid, spacing, offset_mm):
    shape = image.shape[-3:]
    axes = [(torch.arange(size, device=image.device, dtype=torch.float32) + .5
             + float(offset_mm[i] / spacing[i])) * (2. / size) - 1 for i, size in enumerate(shape)]
    xyz = torch.meshgrid(*axes, indexing='ij')
    # Our tensors use N,C,X,Y,Z; grid_sample coordinates follow the reverse spatial order.
    grid = torch.stack((xyz[2], xyz[1], xyz[0]), -1)[None]
    with torch.autocast(device_type=image.device.type, enabled=False):
        shifted = [F.grid_sample(t.float(), grid, mode=mode, padding_mode='zeros', align_corners=False)
                   for t, mode in ((image, 'bilinear'), (target, 'nearest'), (valid, 'nearest'))]
    shifted[1] = shifted[1].to(target.dtype)
    shifted[2] = shifted[2].bool()
    shifted[0] = shifted[0].masked_fill(~shifted[2], 0).to(image.dtype)
    return tuple(shifted)


def translate_pair(image, context, ratio, spacing, target, valid, context_target, context_valid,
                   seed, probability=.5, maximum_mm=1., margin=1, offset_voxels=None):
    """Return both shifted views and a physical-offset record, or both originals on unsafe shifts."""
    original = (image, context, target, valid, context_target, context_valid)
    local_spacing = spacing.detach().float().cpu().numpy().reshape(-1)
    fov_ratio = ratio.detach().float().cpu().numpy().reshape(-1)
    if (local_spacing.shape != (3,) or fov_ratio.shape != (3,) or not np.isfinite(local_spacing).all()
            or not np.isfinite(fov_ratio).all() or (local_spacing <= 0).any() or (fov_ratio <= 0).any()
            or not math.isfinite(maximum_mm) or maximum_mm < 0 or not 0 <= probability <= 1
            or not isinstance(margin, int) or margin < 0):
        raise ValueError('Invalid paired translation geometry/policy')
    for view, mask, support in ((image, target, valid), (context, context_target, context_valid)):
        if (view.ndim != 5 or view.shape[:2] != (1, 1) or view.shape != mask.shape or mask.shape != support.shape
                or not mask.any() or not torch.isfinite(view).all() or not torch.all((mask == 0) | (mask == 1))
                or not support.bool()[mask.bool()].all()):
            raise ValueError('Invalid paired translation image/target/support')
    record = {'applied': False, 'sampling_offset_mm_xyz': [0., 0., 0.], 'reason': 'not_selected'}
    generator = np.random.default_rng(np.random.SeedSequence([int(seed), 91731]))
    if offset_voxels is None and generator.random() >= probability:
        return original, record
    points = torch.nonzero(target[0, 0], as_tuple=False)
    lo, hi = points.amin(0).cpu().numpy(), points.amax(0).cpu().numpy()
    size = np.asarray(image.shape[-3:])
    limit = np.floor(maximum_mm / local_spacing + 1e-6).astype(int)
    lower = np.maximum(-limit, hi - (size - 1 - margin))
    upper = np.minimum(limit, lo - margin)
    if (lower > upper).any():
        return original, {**record, 'reason': 'no_safe_offset'}
    if offset_voxels is None:
        offset = generator.integers(lower, upper + 1)
    else:
        offset = np.asarray(offset_voxels, dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all() or (offset != np.round(offset)).any():
            raise ValueError('Offset must contain three integer detail voxel offsets')
        if (offset < lower).any() or (offset > upper).any():
            return original, {**record, 'reason': 'unsafe_offset'}
    offset_mm = offset * local_spacing
    if not offset.any():
        return original, {**record, 'reason': 'zero_offset'}
    context_spacing = size * local_spacing / fov_ratio / np.asarray(context.shape[-3:])
    detail = _translate(image, target, valid, local_spacing, offset_mm)
    semantic = _translate(context, context_target, context_valid, context_spacing, offset_mm)
    for (_, shifted_mask, shifted_valid), source_mask in ((detail, target), (semantic, context_target)):
        if shifted_mask.count_nonzero() != source_mask.count_nonzero() or not shifted_valid[shifted_mask.bool()].all():
            return original, {**record, 'reason': 'target_or_support_loss'}
    return (detail[0], semantic[0], detail[1], detail[2], semantic[1], semantic[2]), {
        'applied': True, 'sampling_offset_mm_xyz': offset_mm.tolist(), 'reason': 'translated'}


class HistoryJitterModel(ResearchModel):
    @staticmethod
    def interaction_function(model, image, context, ratio, spacing, target, valid, milestones, seed,
                             criterion=None, context_target=None, context_valid=None, auxiliary_weight=.25, field=None):
        if criterion is not None:
            (image, context, target, valid, context_target, context_valid), record = translate_pair(
                image, context, ratio, spacing, target, valid, context_target, context_valid, seed)
            counts = getattr(model, 'jitter_counts', {})
            counts[record['reason']] = counts.get(record['reason'], 0) + 1
            model.jitter_counts = counts
        return ResearchModel.interaction_function(model, image, context, ratio, spacing, target, valid,
            milestones, seed, criterion, context_target, context_valid, auxiliary_weight, field)


def configure_experiment(args, config):
    """Require the completed unjittered continuation as a source- and recipe-matched control."""
    import physical_fsdp as data
    from evaluate_private_validation import file_hash
    folder = Path(args.matched_control_dir).resolve()
    previous, report = [data.read(folder / name) for name in ('config.json', 'summary.json')]
    keys = ('epochs', 'lr_multiplier', 'patience_epochs', 'seed', 'precision', 'augmentation',
            'accumulation', 'auxiliary_weight', 'val_every', 'min_delta', 'sam_lr', 'head_lr',
            'boundary_weight', 'boundary_clip_mm', 'head_warmup_epochs')
    if (not report['complete'] or report['test_images_read'] or report['variant'] != 'history'
            or previous['variant'] != 'history' or previous['experiment'] != config['experiment']
            or previous.get('roi_jitter') or 'research_source' not in config
            or report['train_patients'] != 54 or report['validation_patients'] != 10
            or report['planned_epochs'] != config['parameters']['epochs']
            or not 0 <= report['best_epoch'] <= report['completed_epochs'] <= report['planned_epochs']
            or any(previous.get(key) != config.get(key) for key in
                   ('research_source', 'foundation', 'cache_sha256', 'manifest_sha256', 'physical_policy', 'versions'))
            or any(previous['parameters'].get(key) != config['parameters'].get(key) for key in keys)):
        raise ValueError('ROI jitter requires the same source and recipe as the completed history control')
    if previous['parameters']['augmentation'] != 'light':
        raise ValueError('ROI jitter control requires the existing light paired augmentation')
    from train_sam_refine32 import CODE_FILES
    computational = tuple(dict.fromkeys((*CODE_FILES, 'fsdp_research_model.py', 'fsdp_research_metrics.py')))
    if any(previous['code_sha256'][name] != config['code_sha256'][name] for name in computational):
        raise ValueError('Matched control model/data/metric implementation differs')
    if file_hash(folder / 'research_best.pth') != report['best_checkpoint_sha256']:
        raise ValueError('Matched control checkpoint checksum differs')
    score = report['best_validation_metrics']['5']['mean_bilateral_patient_dice']
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Invalid matched control score')
    config.update(roi_jitter=POLICY, matched_control={'directory': str(folder),
        'config_sha256': file_hash(folder / 'config.json'), 'summary_sha256': file_hash(folder / 'summary.json'),
        'checkpoint_sha256': report['best_checkpoint_sha256'], 'validation_dice_5': score,
        'planned_epochs': report['planned_epochs'], 'completed_epochs': report['completed_epochs'],
        'scope': 'same initialization, seed, schedule, budget cap and stopping rule; only train ROI translation added'})


def comparison(config, score):
    from evaluate_private_validation import file_hash
    control = config['matched_control']
    for name, key in (('config.json', 'config_sha256'), ('summary.json', 'summary_sha256'),
                      ('research_best.pth', 'checkpoint_sha256')):
        if file_hash(Path(control['directory']) / name) != control[key]:
            raise ValueError('Matched control artifact changed during experiment')
    return {'roi_jitter': config['roi_jitter'], 'matched_control': control,
            'validation_change_vs_matched_control': score - control['validation_dice_5'],
            'interpretation': 'Single-seed train ROI translation ablation against a completed matched continuation; '
                              'GT-assisted repeatedly inspected validation, not automatic test performance.'}
