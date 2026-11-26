"""Paired physical affine augmentation for the learned SAM Turbo/CNN32 history model."""
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from evaluate_private_validation import file_hash
from fsdp_history_jitter import translate_pair
from fsdp_research_model import ResearchModel

POLICY = dict(probability=.5, rotation_degrees=5., scale=[.95, 1.05], translation_mm=1.,
              interpolation='CT trilinear; target and support nearest',
              safety='reject pair if foreground count changes >15%, touches border or leaves valid support',
              evaluation='unchanged cached GT-side ROI and 1/5 oracle clicks; no automatic localization claim')


def affine_pair(image, context, ratio, spacing, target, valid, context_target, context_valid, seed):
    original = (image, context, target, valid, context_target, context_valid)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 28116]))
    if rng.random() >= POLICY['probability']:
        return original, 'affine_not_selected'
    angles = np.deg2rad(rng.uniform(-5, 5, 3))
    cx, cy, cz = np.cos(angles); sx, sy, sz = np.sin(angles)
    rotation = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotation = rotation @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotation = rotation @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    scale = float(rng.uniform(.95, 1.05))
    matrix = torch.as_tensor(rotation / scale, device=image.device, dtype=torch.float32)
    detail_extent = torch.as_tensor(image.shape[-3:], device=image.device) * spacing
    context_extent = detail_extent / ratio
    outputs = []
    for view, mask, support, extent in ((image, target, valid, detail_extent),
                                        (context, context_target, context_valid, context_extent)):
        shape = view.shape[-3:]
        axes = [(torch.arange(n, device=view.device, dtype=torch.float32) + .5) / n * 2 - 1 for n in shape]
        xyz = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1)
        physical = xyz * extent / 2
        coordinates = (physical @ matrix.T) * 2 / extent
        grid = coordinates.flip(-1)[None]
        with torch.autocast(device_type=view.device.type, enabled=False):
            values = [F.grid_sample(t.float(), grid, mode=mode, padding_mode='zeros', align_corners=False)
                      for t, mode in ((view, 'bilinear'), (mask, 'nearest'), (support, 'nearest'))]
        shifted_mask, shifted_support = values[1].to(mask.dtype), values[2] > .5
        count = int(shifted_mask.count_nonzero())
        expected = float(mask.count_nonzero()) * scale**3
        if not count or abs(count / expected - 1) > .15 or not shifted_support[shifted_mask.bool()].all():
            return original, 'affine_rejected'
        points = torch.nonzero(shifted_mask[0, 0], as_tuple=False)
        if (points.amin(0) < 1).any() or (points.amax(0) >= torch.tensor(shape, device=view.device) - 1).any():
            return original, 'affine_rejected'
        outputs.append((values[0].masked_fill(~shifted_support, 0).to(view.dtype), shifted_mask, shifted_support))
    a, b = outputs
    return (a[0], b[0], a[1], a[2], b[1], b[2]), 'affine_applied'


class AugmentedSAM(ResearchModel):
    @staticmethod
    def interaction_function(model, image, context, ratio, spacing, target, valid, milestones, seed,
                             criterion=None, context_target=None, context_valid=None, auxiliary_weight=.25, field=None):
        if criterion is not None:
            pair, record = translate_pair(image, context, ratio, spacing, target, valid, context_target, context_valid, seed)
            model.jitter_counts[record['reason']] = model.jitter_counts.get(record['reason'], 0) + 1
            if model.augmentation_arm == 'affine':
                pair, reason = affine_pair(pair[0], pair[1], ratio, spacing, *pair[2:], seed)
                model.jitter_counts[reason] = model.jitter_counts.get(reason, 0) + 1
            image, context, target, valid, context_target, context_valid = pair
        return ResearchModel.interaction_function(model, image, context, ratio, spacing, target, valid,
            milestones, seed, criterion, context_target, context_valid, auxiliary_weight, field)


def configure(args, config):
    if args.variant != 'history' or not args.research_source_dir:
        raise ValueError('Requires a completed history checkpoint as initialization')
    score = config['research_source']['expected_validation']['5']['mean_bilateral_patient_dice']
    if score < .49:
        raise ValueError(f'Selected source is below the requested 49% benchmark: {score:.4%}')
    config.update(training_experiment='sam_paired_affine', augmentation_arm=args.augmentation_arm,
                  paired_affine_policy=POLICY, excluded_training=['train_28'], source_validation_dice=score,
                  source_exposure='Source weights saw train_28; reproduction uses original cohort; continuation uses53patients',
                  augmentation_code_sha256=file_hash(Path(__file__)))


def adapt_model(original, args, config):
    model = AugmentedSAM.from_foundation(original, 'history')
    model.augmentation_arm = args.augmentation_arm
    model.jitter_counts = {}
    return model


def finish_summary(summary, config):
    summary.update(experiment='sam_paired_affine', augmentation_arm=config['augmentation_arm'],
                   paired_affine_policy=POLICY, excluded_training=['train_28'],
                   delta_vs_source=summary['best_validation_metrics']['5']['mean_bilateral_patient_dice'] - config['source_validation_dice'])


def extend_parser(parser):
    parser.set_defaults(record_jitter_counts=True, record_gradient_stats=True, exclude_invalid_train28=True)
    parser.add_argument('--augmentation-arm', choices=('control', 'affine'), required=True)


if __name__ == '__main__':
    from train_fsdp_research import main
    main(configure, adapt_model, finish_summary, extend_parser)
