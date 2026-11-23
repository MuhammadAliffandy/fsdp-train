"""Turbo SAM context plus an independently learned32-cube detail branch."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


TURBO_SHA256 = '899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c'


def block(inc, outc):
    return nn.Sequential(nn.Conv3d(inc, outc, 3, padding=1), nn.GroupNorm(4, outc), nn.PReLU(outc),
                         nn.Conv3d(outc, outc, 3, padding=1), nn.GroupNorm(4, outc), nn.PReLU(outc))


def context_to_detail(feature, ratio, shape):
    """Map centered LPS context bins to a centered detail field, not a shape-only resize."""
    if ratio.shape != (3,) or (ratio <= 0).any() or (ratio > 1.00001).any():
        raise ValueError('Detail/context physical extent ratios must lie in(0,1]')
    axes = [(2 * (torch.arange(n, device=feature.device, dtype=torch.float32) + .5) / n - 1) * ratio[i]
            for i, n in enumerate(shape)]
    coordinates = torch.meshgrid(*axes, indexing='ij')
    # Torch grid_sample names its last spatial dimension x; our arrays are XYZ.
    grid = torch.stack(coordinates[::-1], -1)[None].expand(feature.shape[0], -1, -1, -1, -1)
    return F.grid_sample(feature.float(), grid, mode='bilinear', padding_mode='border', align_corners=False)


class SamRefine32(nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.sam = sam
        channels = sam.prompt_encoder.embed_dim
        self.full = block(1, 8)
        self.half_features = nn.Sequential(nn.AvgPool3d(2), nn.Conv3d(8, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.PReLU(16))
        self.detail32 = block(16, 32)
        self.semantic_projection = nn.Conv3d(channels, 32, 1)
        self.fusion = block(67, 64)
        self.up_half = block(80, 32)
        self.up_full = block(40, 16)
        self.residual = nn.Conv3d(16, 1, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def encode(self, image, context, ratio, spacing):
        if tuple(context.shape) != (1, 1, 128, 128, 128) or min(image.shape[-3:]) < 64:
            raise ValueError('SAM context must be128³ and detail dimensions at least64')
        embedding = self.sam.image_encoder(context)
        if tuple(embedding.shape[-3:]) != (8, 8, 8):
            raise ValueError('Turbo image embedding must remain8³')
        full = self.full(image)
        half = self.half_features(full)
        if min(half.shape[-3:]) < 32:
            raise ValueError('The detail32 branch must not be obtained by upsampling a smaller CNN feature')
        detail = self.detail32(F.interpolate(half, size=(32,) * 3, mode='trilinear', align_corners=False))
        return {'embedding': embedding, 'full': full, 'half': half, 'detail32': detail,
                'semantic32': context_to_detail(self.semantic_projection(embedding), ratio, (32,) * 3),
                'ratio': ratio, 'spacing': spacing, 'shape': tuple(image.shape[-3:])}

    @staticmethod
    def map_points(cache, points):
        shape = torch.tensor(cache['shape'], device=points.device, dtype=torch.float32)
        return (((points + .5) / shape - .5) * cache['ratio'] + .5) * 128 - .5

    @staticmethod
    def point_maps(cache, history):
        feature = cache['detail32']
        axes = [torch.arange(32, device=feature.device, dtype=torch.float32) for _ in range(3)]
        grid = torch.stack(torch.meshgrid(*axes, indexing='ij'))
        shape = torch.tensor(cache['shape'], device=feature.device, dtype=torch.float32)
        millimetres = shape * cache['spacing'] / 32
        maps = torch.zeros((1, 2, 32, 32, 32), device=feature.device)
        for point, label in history:
            center = (point.reshape(3).float() + .5) / shape * 32 - .5
            distance = ((grid - center[:, None, None, None]) * millimetres[:, None, None, None]).square().sum(0)
            value = torch.exp(-distance / 2)  # One-millimetre Gaussian sigma in physical space.
            channel = int(label.item())
            maps[0, channel] = torch.maximum(maps[0, channel], value)
        return maps

    def decode(self, cache, points, labels, previous_coarse, history):
        sparse, dense = self.sam.prompt_encoder(points=(self.map_points(cache, points), labels), boxes=None, masks=previous_coarse)
        coarse, _ = self.sam.mask_decoder(image_embeddings=cache['embedding'], image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
        if tuple(coarse.shape) != (1, 1, 32, 32, 32):
            raise ValueError('Turbo coarse mask must remain32³')
        coarse32 = context_to_detail(coarse, cache['ratio'], (32,) * 3)
        fused = self.fusion(torch.cat((cache['detail32'], cache['semantic32'], coarse32, self.point_maps(cache, history)), 1))
        half = self.up_half(torch.cat((F.interpolate(fused, size=cache['half'].shape[-3:], mode='trilinear', align_corners=False), cache['half']), 1))
        full = self.up_full(torch.cat((F.interpolate(half, size=cache['shape'], mode='trilinear', align_corners=False), cache['full']), 1))
        baseline = context_to_detail(coarse, cache['ratio'], cache['shape'])
        refined = baseline + self.residual(full)
        if not torch.isfinite(refined).all():
            raise ValueError('Nonfinite refined mask logits')
        return refined, coarse, baseline


def next_click(logits, target, valid, generator):
    points = torch.nonzero((((logits > 0) != target.bool()) & valid)[0, 0])
    if not len(points):
        points = torch.nonzero((target.bool() & valid)[0, 0])
    if not len(points):
        raise ValueError('No valid FSDP click target')
    point = points[int(generator.integers(len(points)))].reshape(1, 1, 3).float()
    p = point.reshape(3).long()
    label = target[0, 0, p[0], p[1], p[2]].reshape(1, 1).long()
    return point, label


def interactions(model, image, context, ratio, spacing, target, valid, milestones, seed, criterion=None,
                 context_target=None, context_valid=None, auxiliary_weight=.25):
    from physical_fsdp import masked_loss
    generator = np.random.default_rng(seed)
    cache = model.encode(image, context, ratio, spacing)
    logits = torch.zeros_like(target, dtype=torch.float32)
    coarse = torch.zeros((1, 1, 32, 32, 32), device=image.device)
    history, outputs, loss = [], {}, 0.
    for click in range(1, max(milestones) + 1):
        points, labels = next_click(logits.detach(), target, valid, generator)
        history.append((points, labels))
        logits, coarse, baseline = model.decode(cache, points, labels, coarse.detach(), history)
        if criterion is not None:
            with torch.autocast(device_type=image.device.type, enabled=False):
                refined_loss = masked_loss(criterion, logits, target, valid)
                context_logits = F.interpolate(coarse.float(), size=(128,) * 3, mode='trilinear', align_corners=False)
                loss = loss + refined_loss + auxiliary_weight * masked_loss(criterion, context_logits, context_target, context_valid)
        if click in milestones:
            outputs[click] = {'refined': logits, 'coarse_on_detail': baseline}
    return loss / max(milestones) if criterion is not None else None, outputs
