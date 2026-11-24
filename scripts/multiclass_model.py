"""Turbo image encoder with local CNN detail at 32³ or 64³ and 18-class logits."""
import importlib
import sys
from types import MethodType
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from evaluate_private_validation import file_hash
from sam_refine32 import TURBO_SHA256, block
from train_private_unet import state_hash


def load_encoder(sam_dir, checkpoint):
    if file_hash(checkpoint) != TURBO_SHA256:
        raise ValueError('Original Turbo checkpoint checksum differs')
    sys.path.insert(0, str(sam_dir))
    implementation = importlib.import_module('segment_anything.build_sam3D')
    if not Path(implementation.__file__).resolve().is_relative_to(sam_dir):
        raise ValueError('SAM imported from a different directory')
    sam = implementation.sam_model_registry3D['vit_b_ori'](checkpoint=None)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    sam.load_state_dict(state.get('model_state_dict', state), strict=True)
    encoder = sam.image_encoder
    if tuple(encoder.pos_embed.shape) != (1, 8, 8, 8, 768):
        raise ValueError('Expected original Turbo image encoder')
    return encoder, state_hash(encoder.state_dict())


def _higher_resolution_forward(self, image):
    if tuple(image.shape[-3:]) != (256,) * 3:
        raise ValueError('16-cube tokens require a directly sampled256-cube context')
    features = self.patch_embed(image)
    if tuple(features.shape[1:4]) != (16,) * 3:
        raise ValueError('Expected16-cube patch tokens before position encoding')
    features = features + self.pos_embed
    for layer in self.blocks:
        needs_grad = features.requires_grad or any(p.requires_grad for p in layer.parameters())
        if torch.is_grad_enabled() and needs_grad:
            features = checkpoint(layer, features, use_reentrant=False)
        else:
            features = layer(features)
    return self.neck(features.permute(0, 4, 1, 2, 3))


def configure_encoder_grid(encoder, grid):
    """Adapt absolute positions after strict Turbo loading; retain patch/attention weights."""
    if grid == 8:
        return
    if grid != 16 or tuple(encoder.pos_embed.shape[1:4]) != (8,) * 3:
        raise ValueError('Encoder adaptation supports original8-cube to16-cube only')
    projection = encoder.patch_embed.proj
    if projection.kernel_size != (16,) * 3 or projection.stride != (16,) * 3:
        raise ValueError('Expected original16-cube patch kernel and stride')
    for layer in encoder.blocks:
        if not layer.attn.use_rel_pos:
            raise ValueError('Expected Turbo relative positions with upstream runtime resizing')
    with torch.no_grad():
        position = F.interpolate(encoder.pos_embed.permute(0, 4, 1, 2, 3).float(),
                                 size=(16,) * 3, mode='trilinear', align_corners=False)
    encoder.pos_embed = nn.Parameter(position.permute(0, 2, 3, 4, 1).contiguous())
    encoder.img_size = 256
    # Upstream get_rel_pos interpolates relative tables to each attention grid.
    # Keep original table parameters, global layers and window sizes unchanged.
    encoder.forward = MethodType(_higher_resolution_forward, encoder)


def spatial_crop(feature, offset, shape, output_size):
    # Arrays and metadata use XYZ. grid_sample's coordinate order is reversed.
    device = feature.device
    axes = [2 * (offset[i] + (torch.arange(output_size, device=device).float() + .5) * 128 / output_size) / shape[i] - 1
            for i in range(3)]
    coords = torch.meshgrid(*axes, indexing='ij')
    grid = torch.stack(coords[::-1], -1)[None]
    return F.grid_sample(feature.float(), grid, align_corners=False, padding_mode='border')


class MulticlassTurbo(nn.Module):
    def __init__(self, encoder, detail_grid=64, semantic_channels=384):
        super().__init__()
        if detail_grid not in (32, 64):
            raise ValueError('Detail grid must be32 or64')
        self.image_encoder, self.detail_grid = encoder, detail_grid
        self.full = block(1, 8)
        self.half_features = nn.Sequential(nn.Conv3d(8, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.PReLU(16))
        self.detail = block(16, 32)
        self.projection = nn.Conv3d(semantic_channels, 32, 1)
        self.fusion = block(64, 48)
        self.half_decoder = block(64, 24)
        self.full_decoder = block(32, 16)
        self.classifier = nn.Conv3d(16, 18, 1)

    def encode(self, context):
        result = self.image_encoder(context)
        grid = getattr(self, 'encoder_grid', 8)
        if tuple(context.shape[-3:]) != (getattr(self, 'context_size', 128),) * 3:
            raise ValueError('Context size differs from configured encoder')
        if tuple(result.shape[-3:]) != (grid,) * 3:
            raise ValueError('Turbo embedding differs from configured token grid')
        return self.projection(result)

    def decode(self, image, embedding, offset, shape, return_shapes=False):
        if tuple(image.shape) != (1, 1, 128, 128, 128):
            raise ValueError('Local detail patch must be128³')
        full = self.full(image)
        half = self.half_features(full)
        local = half if self.detail_grid == 64 else F.avg_pool3d(half, 2)
        detail = self.detail(local)
        semantic = spatial_crop(embedding, offset, shape, self.detail_grid)
        fusion = self.fusion(torch.cat((detail, semantic), 1))
        half_output = self.half_decoder(torch.cat((F.interpolate(fusion, size=half.shape[-3:], mode='trilinear', align_corners=False), half), 1))
        decoded = self.full_decoder(torch.cat((F.interpolate(half_output, size=image.shape[-3:], mode='trilinear', align_corners=False), full), 1))
        logits = self.classifier(decoded)
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite multiclass logits')
        if return_shapes:
            return logits, {key: list(value.shape) for key, value in
                (('full_skip', full), ('half_skip', half), ('local_detail', detail), ('semantic_detail', semantic), ('logits', logits))}
        return logits

    def forward(self, image, context, offset, shape, return_shapes=False):
        return self.decode(image, self.encode(context), offset, shape, return_shapes)


def set_stage(model, epoch, warmup):
    model.train()
    active = epoch >= warmup
    for name, parameter in model.image_encoder.named_parameters():
        parameter.requires_grad_(active and name.startswith(('blocks.10.', 'blocks.11.')))
    model.image_encoder.eval()
    if active:
        model.image_encoder.blocks[10].train()
        model.image_encoder.blocks[11].train()
    return 'partial_encoder_and_decoder' if active else 'decoder_warmup'


def segmentation_loss(logits, target, valid):
    if logits.shape[1] != 18 or target.shape != valid.shape or target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError('Multiclass tensor shapes differ')
    if not valid.any() or target.min() < 0 or target.max() > 17:
        raise ValueError('Invalid valid mask or target classes')
    x, y = logits.float().movedim(1, -1)[valid], target.long()[valid]
    ce = F.cross_entropy(x, y)
    probabilities = x.softmax(-1)
    terms = []
    for label in range(1, 18):
        truth = y == label
        if truth.any():
            pred = probabilities[:, label]
            terms.append(1 - (2 * pred[truth].sum() + 1e-5) / (pred.square().sum() + truth.sum() + 1e-5))
    dice = torch.stack(terms).mean() if terms else x.sum() * 0
    return dice + .25 * ce
