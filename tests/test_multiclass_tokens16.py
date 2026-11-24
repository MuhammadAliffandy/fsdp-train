import copy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import multiclass_data as data
import multiclass_model as model_code


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(1, 4, 16, stride=16)

    def forward(self, image):
        return self.proj(image).permute(0, 2, 3, 4, 1)


class TokenBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(4, 4)
        self.attn.use_rel_pos = True

    def forward(self, tokens):
        return tokens + .1 * self.attn(tokens).tanh()


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.pos_embed = nn.Parameter(torch.randn(1, 8, 8, 8, 4))
        self.blocks = nn.ModuleList([TokenBlock() for _ in range(12)])
        self.neck = nn.Conv3d(4, 4, 1)


class TokenGridTests(unittest.TestCase):
    def test_adaptation_preserves_pretrained_weights_except_absolute_positions(self):
        encoder = Encoder()
        original = {k: v.clone() for k, v in encoder.state_dict().items()}
        model_code.configure_encoder_grid(encoder, 16)
        expected = F.interpolate(original['pos_embed'].permute(0, 4, 1, 2, 3),
                                 (16,) * 3, mode='trilinear', align_corners=False)
        torch.testing.assert_close(encoder.pos_embed, expected.permute(0, 2, 3, 4, 1))
        for name, value in encoder.state_dict().items():
            if name != 'pos_embed':
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, '256-cube'):
            encoder(torch.zeros(1, 1, 128, 128, 128))

    def test_checkpointed_forward_and_gradients_match_direct_computation(self):
        encoder = Encoder()
        model_code.configure_encoder_grid(encoder, 16)
        model = model_code.MulticlassTurbo(encoder, 64, semantic_channels=4)
        model.encoder_grid, model.context_size = 16, 256
        model_code.set_stage(model, 5, 5)
        direct = copy.deepcopy(encoder)
        image = torch.randn(1, 1, 256, 256, 256)
        output = model.encode(image)
        tokens = direct.patch_embed(image) + direct.pos_embed
        for layer in direct.blocks:
            tokens = layer(tokens)
        reference = direct.neck(tokens.permute(0, 4, 1, 2, 3))
        self.assertEqual(tuple(output.shape), (1, 32, 16, 16, 16))
        # Compare the encoder itself before the independently learned projection.
        actual = encoder(image)
        torch.testing.assert_close(actual, reference)
        actual.square().mean().backward()
        reference.square().mean().backward()
        for (name, parameter), other in zip(encoder.named_parameters(), direct.parameters()):
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                torch.testing.assert_close(parameter.grad, other.grad)
            else:
                self.assertIsNone(parameter.grad, name)

    def test_context_samples_physical_field_directly_at256(self):
        image = torch.randn(1, 1, 144, 160, 192)
        case = {'image': image}
        actual = data.case_context(case, torch.device('cpu'), 256)
        expected = F.interpolate(image, (256,) * 3, mode='trilinear', align_corners=False)
        torch.testing.assert_close(actual, expected)
        two_step = F.interpolate(data.case_context(case, torch.device('cpu')), (256,) * 3,
                                 mode='trilinear', align_corners=False)
        self.assertGreater((actual - two_step).abs().mean().item(), .01)


if __name__ == '__main__':
    unittest.main()
