"""Controlled prompt, boundary-objective and local-refiner variants of the trained hybrid."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sam_refine32 import SamRefine32, interactions, next_click
from sam_refine_resolution import SamRefineResolution


VARIANTS = ('control', 'history', 'boundary', 'refiner')


def signed_distance(target, valid, spacing, clip_mm=5.):
    from scipy.ndimage import distance_transform_edt
    foreground = np.asarray(target, dtype=bool)
    support = np.asarray(valid, dtype=bool)
    spacing = np.asarray(spacing, dtype=float)
    if (foreground.shape != support.shape or foreground.ndim != 3 or not foreground.any()
            or foreground[~support].any() or not (support & ~foreground).any()
            or spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any()
            or not np.isfinite(clip_mm) or clip_mm <= 0):
        raise ValueError('Invalid physical boundary target/support/spacing')
    # Padding is neither positive foreground nor a negative boundary reference.
    to_foreground = distance_transform_edt(~foreground, sampling=spacing)
    to_background = distance_transform_edt(~(support & ~foreground), sampling=spacing)
    field = np.where(foreground, -to_background, to_foreground)
    field = np.clip(field / clip_mm, -1., 1.).astype(np.float32)
    field[~support] = 0
    return field


def boundary_loss(logits, field, valid):
    if logits.shape != field.shape or logits.shape != valid.shape or not valid.any():
        raise ValueError('Boundary tensors/support differ')
    return (logits.float().sigmoid()[valid] * field.float()[valid]).mean()


class LocalRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv3d(27, 27, 5, padding=2, groups=27),
            nn.Conv3d(27, 16, 1), nn.GroupNorm(4, 16), nn.SiLU())
        self.output = nn.Conv3d(16, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, cache, logits, history):
        shape = cache['half'].shape[-3:]
        full = F.adaptive_avg_pool3d(cache['full'], shape)
        prior = F.interpolate(logits, size=shape, mode='trilinear', align_corners=False)
        points = SamRefineResolution.point_maps({**cache, 'detail32': cache['half']}, history)
        features = torch.cat((cache['half'], full, prior, points), 1)
        delta = self.output(self.features(features))
        return F.interpolate(delta, size=logits.shape[-3:], mode='trilinear', align_corners=False)


class ResearchModel(SamRefine32):
    @classmethod
    def from_foundation(cls, model, variant, boundary_weight=.05):
        if variant not in VARIANTS or not np.isfinite(boundary_weight) or boundary_weight <= 0:
            raise ValueError('Invalid research variant/boundary weight')
        result = cls(model.sam)
        result.load_state_dict(model.state_dict(), strict=True)
        result.variant = variant
        result.boundary_weight = boundary_weight
        if variant == 'refiner':
            result.research_refiner = LocalRefiner()
        return result

    def decode(self, cache, points, labels, previous_coarse, history):
        if self.variant == 'history':
            points = torch.cat([p for p, _ in history], dim=1)
            labels = torch.cat([label for _, label in history], dim=1)
        logits, coarse, baseline = super().decode(cache, points, labels, previous_coarse, history)
        if self.variant == 'refiner':
            logits = logits + self.research_refiner(cache, logits, history)
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite research logits')
        return logits, coarse, baseline

    @staticmethod
    def interaction_function(model, image, context, ratio, spacing, target, valid, milestones, seed,
                             criterion=None, context_target=None, context_valid=None, auxiliary_weight=.25, field=None):
        if model.variant != 'boundary' or criterion is None:
            return interactions(model, image, context, ratio, spacing, target, valid, milestones, seed,
                                criterion, context_target, context_valid, auxiliary_weight)
        from physical_fsdp import masked_loss
        if field is None:
            raise ValueError('Boundary training requires a paired physical signed-distance field')
        generator = np.random.default_rng(seed)
        cache = model.encode(image, context, ratio, spacing)
        logits = torch.zeros_like(target, dtype=torch.float32)
        coarse = torch.zeros((1, 1, 32, 32, 32), device=image.device)
        history, outputs, loss = [], {}, 0.
        for click in range(1, max(milestones) + 1):
            points, labels = next_click(logits.detach(), target, valid, generator)
            history.append((points, labels))
            logits, coarse, baseline = model.decode(cache, points, labels, coarse.detach(), history)
            with torch.autocast(device_type=image.device.type, enabled=False):
                context_logits = F.interpolate(coarse.float(), size=context_target.shape[-3:], mode='trilinear', align_corners=False)
                loss = loss + masked_loss(criterion, logits, target, valid)
                loss = loss + model.boundary_weight * boundary_loss(logits, field, valid)
                loss = loss + auxiliary_weight * masked_loss(criterion, context_logits, context_target, context_valid)
            if click in milestones:
                outputs[click] = {'refined': logits, 'coarse_on_detail': baseline}
        return loss / max(milestones), outputs


def prepare_distances(prepared, clip_mm):
    for sample in prepared['samples']['train']:
        field = signed_distance(sample['target'][0, 0].numpy(), sample['valid'][0, 0].numpy(),
                                sample['grid']['spacing'], clip_mm)
        sample['research_distance'] = torch.from_numpy(field)[None, None]
        print(f'{sample["name"]}: train-only physical boundary field prepared', flush=True)


def run_sample(model, sample, device, seed, criterion=None, augment=False, auxiliary_weight=.25):
    from sam_refine_data import paired_inputs
    tensors = paired_inputs(sample, device, seed if augment else None)
    field = None
    if model.variant == 'boundary' and criterion is not None:
        field = sample['research_distance']
        if augment and np.random.default_rng(seed).random() < .5:
            field = torch.flip(field, (2,))
        field = field.to(device)
    with torch.autocast(device_type=device.type, enabled=device.type == 'cuda', dtype=model.amp_dtype):
        return model.interaction_function(model, tensors['image'], tensors['sam_image'], tensors['ratio'],
            tensors['spacing'], tensors['target'], tensors['valid'], (1, 5), seed, criterion,
            tensors['sam_target'], tensors['sam_valid'], auxiliary_weight, field=field)
