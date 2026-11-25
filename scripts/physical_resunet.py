"""Native-detail residual3D U-Net with full-resolution supervised outputs."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE = {'channels': [16, 32, 64, 128, 192], 'normalization': 'GroupNorm8',
                'downsampling': 'four stride2 convolutions', 'decoder': 'trilinear + residual skip blocks',
                'deep_supervision': [1., .3, .15], 'input_channels': 1, 'output_channels': 1,
                'initialization': 'random Kaiming; all parameters trained; no SAM weights or oracle prompts'}


class Residual(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv3d(incoming, outgoing, 3, stride, 1, bias=False),
            nn.GroupNorm(8, outgoing), nn.LeakyReLU(.01), nn.Conv3d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(8, outgoing))
        self.skip = nn.Identity() if incoming == outgoing and stride == 1 else nn.Conv3d(incoming, outgoing, 1, stride, bias=False)
        self.activation = nn.LeakyReLU(.01)

    def forward(self, value):
        return self.activation(self.layers(value) + self.skip(value))


class PhysicalResUNet(nn.Module):
    def __init__(self, channels=(16, 32, 64, 128, 192)):
        super().__init__()
        if len(channels) != 5 or any(c < 8 or c % 8 for c in channels):
            raise ValueError('Five feature widths divisible by8are required')
        self.encoder = nn.ModuleList([Residual(1, channels[0]),
            *[Residual(channels[i - 1], channels[i], 2) for i in range(1, 5)]])
        self.decoder = nn.ModuleList([Residual(channels[i] + channels[i - 1], channels[i - 1]) for i in range(4, 0, -1)])
        self.heads = nn.ModuleList([nn.Conv3d(channels[i], 1, 1) for i in range(3)])
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, a=.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        if image.ndim != 5 or image.shape[1] != 1 or min(image.shape[-3:]) < 16:
            raise ValueError('Expected N,1,X,Y,Z with each spatial dimension at least16')
        features = []
        value = image
        for layer in self.encoder:
            value = layer(value)
            features.append(value)
        decoded = {}
        for index, layer in zip(range(3, -1, -1), self.decoder):
            value = F.interpolate(value, size=features[index].shape[-3:], mode='trilinear', align_corners=False)
            value = layer(torch.cat((value, features[index]), 1))
            if index < 3:
                decoded[index] = self.heads[index](value)
        return tuple(decoded[i] for i in range(3))


def supervised_loss(outputs, target, valid):
    if target.shape != valid.shape or not valid.any() or target[~valid.bool()].any():
        raise ValueError('Invalid residual U-Net target/support')
    weights = ARCHITECTURE['deep_supervision']
    if len(outputs) != len(weights):
        raise ValueError('Expected three supervised decoder outputs')
    total = 0.
    with torch.autocast(device_type=target.device.type, enabled=False):
        truth = target.float()[valid.bool()]
        for weight, logits in zip(weights, outputs):
            logits = F.interpolate(logits.float(), size=target.shape[-3:], mode='trilinear', align_corners=False)
            selected = logits[valid.bool()]
            probability = selected.sigmoid()
            dice = 1 - (2 * (probability * truth).sum() + 1e-5) / (probability.sum() + truth.sum() + 1e-5)
            element = F.binary_cross_entropy_with_logits(selected, truth, reduction='none')
            foreground = truth.bool()
            if foreground.any() and (~foreground).any():
                balanced = .5 * (element[foreground].mean() + element[~foreground].mean())
            else:
                balanced = element.mean()
            total = total + weight * (dice + .5 * balanced)
    return total / sum(weights)


def training_input(sample, device, seed, augment=True, patch=(64, 112, 144)):
    generator = np.random.default_rng(seed)
    image, target, valid = [sample[key].clone() for key in ('image', 'target', 'valid')]
    record = {'view': 'full_roi', 'foreground_centered': False}
    if augment and generator.random() < .5:
        shape = np.asarray(image.shape[-3:])
        wanted = np.minimum(np.asarray(patch), shape)
        origin = np.array([generator.integers(0, int(n - p) + 1) for n, p in zip(shape, wanted)])
        if generator.random() < .67:
            points = torch.nonzero(target[0, 0], as_tuple=False)
            point = points[int(generator.integers(len(points)))].numpy()
            lower = np.maximum(0, point - wanted + 1)
            upper = np.minimum(point, shape - wanted)
            origin = generator.integers(lower, upper + 1)
            record['foreground_centered'] = True
        selection = (slice(None), slice(None), *[slice(int(a), int(a + p)) for a, p in zip(origin, wanted)])
        image, target, valid = [value[selection].clone() for value in (image, target, valid)]
        record['view'] = 'patch'
    if not valid.any():
        image, target, valid = [sample[key].clone() for key in ('image', 'target', 'valid')]
        record = {'view': 'fallback_full_roi', 'foreground_centered': False}
    if augment:
        if generator.random() < .5:
            image, target, valid = [torch.flip(value, (2,)) for value in (image, target, valid)]
        if generator.random() < .3:
            image = image * float(generator.uniform(.9, 1.1)) + float(generator.uniform(-.1, .1))
        if generator.random() < .25:
            noise = torch.from_numpy(generator.normal(0, .01, size=tuple(image.shape)).astype(np.float32))
            image = image + noise
    image[~valid.bool()] = 0
    return tuple(value.to(device) for value in (image, target, valid)), record
