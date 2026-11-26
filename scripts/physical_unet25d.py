"""Predict native central slices from ordered adjacent slices, then reconstruct XYZ logits."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def slice_stack(image, centers, neighbors):
    if image.ndim != 5 or image.shape[:2] != (1, 1) or neighbors < 1 or neighbors % 2 != 1:
        raise ValueError('Expected one XYZ volume and positive odd neighbor count')
    centers = torch.as_tensor(centers, device=image.device, dtype=torch.long)
    offsets = torch.arange(neighbors, device=image.device) - neighbors // 2
    indices = (centers[:, None] + offsets).clamp(0, image.shape[-1] - 1)
    # Neighbor channels preserve low-to-high native slice order; edge slices repeat.
    return image[0, 0].permute(2, 0, 1)[indices]


class Residual2D(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False),
            nn.GroupNorm(8, outgoing), nn.LeakyReLU(.01),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False), nn.GroupNorm(8, outgoing))
        self.skip = nn.Conv2d(incoming, outgoing, 1, stride, bias=False)

    def forward(self, value):
        return F.leaky_relu(self.layers(value) + self.skip(value), .01)


class PhysicalUNet25D(nn.Module):
    def __init__(self, neighbors=5, chunk_size=8, channels=(16, 32, 64, 128)):
        super().__init__()
        if neighbors not in (1, 5) or chunk_size < 1:
            raise ValueError('Use one-slice control or five-slice context and positive chunk size')
        self.neighbor_slices, self.chunk_size = neighbors, chunk_size
        self.encoder = nn.ModuleList([Residual2D(neighbors, channels[0]),
            *[Residual2D(channels[i-1], channels[i], 2) for i in range(1, 4)]])
        self.decoder = nn.ModuleList([Residual2D(channels[i]+channels[i-1], channels[i-1]) for i in range(3,0,-1)])
        self.heads = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in channels[:3]])

    def plane(self, value):
        original = value.shape[-2:]
        value = F.pad(value, (0, (-original[1]) % 8, 0, (-original[0]) % 8))
        features = []
        for block in self.encoder:
            value = block(value); features.append(value)
        result = {}
        for index, block in zip(range(2,-1,-1), self.decoder):
            value = F.interpolate(value, size=features[index].shape[-2:], mode='bilinear', align_corners=False)
            value = block(torch.cat((value, features[index]), 1))
            logits = F.interpolate(self.heads[index](value), size=features[0].shape[-2:], mode='bilinear', align_corners=False)
            result[index] = logits[..., :original[0], :original[1]]
        return tuple(result[i] for i in range(3))

    def forward(self, image):
        collected = [[], [], []]
        for start in range(0, image.shape[-1], self.chunk_size):
            planes = slice_stack(image, range(start, min(start+self.chunk_size, image.shape[-1])), self.neighbor_slices)
            outputs = checkpoint(self.plane, planes, use_reentrant=False) if self.training else self.plane(planes)
            for chunks, logits in zip(collected, outputs):
                chunks.append(logits)
        return tuple(torch.cat(chunks, 0).permute(1, 2, 3, 0)[None] for chunks in collected)
