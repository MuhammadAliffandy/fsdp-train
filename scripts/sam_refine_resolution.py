"""Resolution variants of the trained binary FSDP hybrid; preserve parameter names."""
import numpy as np
import torch
from torch.nn import functional as F

from multiclass_model import configure_encoder_grid
from sam_refine32 import SamRefine32, context_to_detail, next_click


VARIANTS = {'baseline': (8, '32'), 'detail': (8, 'native_half'),
            'tokens': (16, '32'), 'combined': (16, 'native_half')}


class SamRefineResolution(SamRefine32):
    def adapt(self, variant):
        self.token_grid, self.detail_mode = VARIANTS[variant]
        self.context_size, self.coarse_size = self.token_grid * 16, self.token_grid * 4
        configure_encoder_grid(self.sam.image_encoder, self.token_grid)
        if self.token_grid == 16:
            prompt = self.sam.prompt_encoder
            prompt.input_image_size = (256,) * 3
            prompt.image_embedding_size = (16,) * 3
            # Match upstream PromptEncoder3D metadata; actual feedback is64³,
            # reduced to16³by its two original stride2convolutions.
            prompt.mask_input_size = (16,) * 3

    def encode(self, image, context, ratio, spacing):
        if self.token_grid == 8 and self.detail_mode == '32':
            return super().encode(image, context, ratio, spacing)
        if tuple(context.shape) != (1, 1) + (self.context_size,) * 3 or min(image.shape[-3:]) < 64:
            raise ValueError('Unexpected SAM context or local detail dimensions')
        embedding = self.sam.image_encoder(context)
        if tuple(embedding.shape[-3:]) != (self.token_grid,) * 3:
            raise ValueError('Actual encoder tokens differ from requested grid')
        full = self.full(image)
        half = self.half_features(full)
        shape = tuple(half.shape[-3:]) if self.detail_mode == 'native_half' else (32,) * 3
        local = half if self.detail_mode == 'native_half' else F.interpolate(half, size=shape, mode='trilinear', align_corners=False)
        detail = self.detail32(local)
        return {'embedding': embedding, 'full': full, 'half': half, 'detail32': detail,
                'semantic32': context_to_detail(self.semantic_projection(embedding), ratio, shape),
                'ratio': ratio, 'spacing': spacing, 'shape': tuple(image.shape[-3:]), 'context_size': self.context_size}

    @staticmethod
    def map_points(cache, points):
        shape = torch.tensor(cache['shape'], device=points.device, dtype=torch.float32)
        return (((points + .5) / shape - .5) * cache['ratio'] + .5) * cache.get('context_size', 128) - .5

    @staticmethod
    def point_maps(cache, history):
        feature = cache['detail32']
        dimensions = tuple(feature.shape[-3:])
        axes = [torch.arange(n, device=feature.device, dtype=torch.float32) for n in dimensions]
        grid = torch.stack(torch.meshgrid(*axes, indexing='ij'))
        shape = torch.tensor(cache['shape'], device=feature.device, dtype=torch.float32)
        cells = torch.tensor(dimensions, device=feature.device, dtype=torch.float32)
        millimetres = shape * cache['spacing'] / cells
        maps = torch.zeros((1, 2) + dimensions, device=feature.device)
        for point, label in history:
            center = (point.reshape(3).float() + .5) / shape * cells - .5
            distance = ((grid - center[:, None, None, None]) * millimetres[:, None, None, None]).square().sum(0)
            value = torch.exp(-distance / 2)
            channel = int(label.item())
            maps[0, channel] = torch.maximum(maps[0, channel], value)
        return maps

    def decode(self, cache, points, labels, previous_coarse, history):
        if self.token_grid == 8 and self.detail_mode == '32':
            return super().decode(cache, points, labels, previous_coarse, history)
        sparse, dense = self.sam.prompt_encoder(points=(self.map_points(cache, points), labels), boxes=None, masks=previous_coarse)
        if tuple(dense.shape[-3:]) != (self.token_grid,) * 3:
            raise ValueError('Prompt mask-feedback grid does not match encoder')
        coarse, _ = self.sam.mask_decoder(image_embeddings=cache['embedding'], image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
        if tuple(coarse.shape) != (1, 1) + (self.coarse_size,) * 3:
            raise ValueError('SAM coarse mask differs from expected4xembedding grid')
        shape = cache['detail32'].shape[-3:]
        coarse_local = context_to_detail(coarse, cache['ratio'], shape)
        fused = self.fusion(torch.cat((cache['detail32'], cache['semantic32'], coarse_local, self.point_maps(cache, history)), 1))
        half = self.up_half(torch.cat((F.interpolate(fused, size=cache['half'].shape[-3:], mode='trilinear', align_corners=False), cache['half']), 1))
        full = self.up_full(torch.cat((F.interpolate(half, size=cache['shape'], mode='trilinear', align_corners=False), cache['full']), 1))
        baseline = context_to_detail(coarse, cache['ratio'], cache['shape'])
        refined = baseline + self.residual(full)
        if not torch.isfinite(refined).all():
            raise ValueError('Nonfinite resolution-refined logits')
        return refined, coarse, baseline

    @staticmethod
    def interaction_function(model, image, context, ratio, spacing, target, valid, milestones, seed, criterion=None,
                             context_target=None, context_valid=None, auxiliary_weight=.25):
        from physical_fsdp import masked_loss
        generator = np.random.default_rng(seed)
        cache = model.encode(image, context, ratio, spacing)
        logits = torch.zeros_like(target, dtype=torch.float32)
        coarse = torch.zeros((1, 1) + (model.coarse_size,) * 3, device=image.device)
        history, outputs, loss = [], {}, 0.
        for click in range(1, max(milestones) + 1):
            points, labels = next_click(logits.detach(), target, valid, generator)
            history.append((points, labels))
            logits, coarse, baseline = model.decode(cache, points, labels, coarse.detach(), history)
            if criterion is not None:
                with torch.autocast(device_type=image.device.type, enabled=False):
                    context_logits = F.interpolate(coarse.float(), size=context_target.shape[-3:], mode='trilinear', align_corners=False)
                    loss = loss + masked_loss(criterion, logits, target, valid)
                    loss = loss + auxiliary_weight * masked_loss(criterion, context_logits, context_target, context_valid)
            if click in milestones:
                outputs[click] = {'refined': logits, 'coarse_on_detail': baseline}
        return loss / max(milestones) if criterion is not None else None, outputs
