import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from partial_experiment import PartialExperiment, TRAINABLE_PREFIXES
from train_private_sides import train_epoch
from test_decoder_training import MiniSAM


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Conv3d(2, 2, 1), torch.nn.BatchNorm3d(2), torch.nn.Dropout(.2))

    def forward(self, x):
        return x + .1 * self.layers(x)


class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = torch.nn.Conv3d(1, 2, 1)
        self.pos_embed = torch.nn.Parameter(torch.tensor(.1))
        self.blocks = torch.nn.ModuleList([Block() for _ in range(12)])
        self.neck = torch.nn.Sequential(torch.nn.Conv3d(2, 1, 1), torch.nn.BatchNorm3d(1))

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.neck(x)


def model():
    result = MiniSAM()
    result.image_encoder = ImageEncoder()
    return result


class PartialTests(unittest.TestCase):
    def test_gradient_through_frozen_neck_updates_exact_regions_and_preserves_other_state(self):
        from monai.losses import DiceCELoss
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            torch.manual_seed(3)
            sam = model()
            initial = {name: value.clone() for name, value in sam.state_dict().items()}
            path = root / 'turbo.pth'
            torch.save({'model_state_dict': initial}, path)
            experiment = PartialExperiment()
            parameters = experiment.configure_model(sam, root)
            optimizer = torch.optim.AdamW(parameters, lr=.02, weight_decay=.1)
            samples = [{'image': torch.ones((1, 1, 4, 4, 4)) * .7,
                        'target': torch.ones((1, 1, 4, 4, 4), dtype=torch.uint8)} for _ in range(3)]
            metrics = train_epoch(sam, samples, optimizer, torch.amp.GradScaler('cuda', enabled=False),
                                  DiceCELoss(sigmoid=True, squared_pred=True), torch.device('cpu'),
                                  0, 2026, 2, train_mode=experiment.train_mode)
            self.assertEqual(metrics['optimizer_updates'], 2)
            changed = [name for name, value in sam.state_dict().items() if not torch.equal(value, initial[name])]
            for prefix in TRAINABLE_PREFIXES:
                self.assertTrue(any(name.startswith(prefix) and name.endswith('weight') for name in changed), prefix)
            self.assertTrue(all(name.startswith(TRAINABLE_PREFIXES) for name in changed))
            self.assertFalse(sam.image_encoder.neck.training)
            self.assertFalse(sam.image_encoder.patch_embed.training)
            self.assertFalse(sam.prompt_encoder.training)
            self.assertTrue(all(not block.training for block in sam.image_encoder.blocks[:10]))
            self.assertTrue(all(block.training for block in sam.image_encoder.blocks[10:]))
            experiment.verify_frozen(sam, types.SimpleNamespace(checkpoint=path), root)
            self.assertTrue(json.loads((root / 'frozen_regions_check.json').read_text())['matches_original_turbo'])
            with torch.no_grad():
                sam.image_encoder.neck[1].running_mean.add_(1)
            with self.assertRaisesRegex(ValueError, 'buffers differ'):
                experiment.verify_frozen(sam, types.SimpleNamespace(checkpoint=path), root)

    def test_optimizer_and_rng_resume_match_with_only_two_encoder_blocks_trainable(self):
        from monai.losses import DiceCELoss
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            torch.manual_seed(11)
            uninterrupted = model()
            experiment = PartialExperiment()
            optimizer = torch.optim.AdamW(experiment.configure_model(uninterrupted, root), lr=.01)
            samples = [{'image': torch.full((1, 1, 4, 4, 4), .3),
                        'target': torch.ones((1, 1, 4, 4, 4), dtype=torch.uint8)} for _ in range(3)]
            criterion = DiceCELoss(sigmoid=True, squared_pred=True)
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            train_epoch(uninterrupted, samples, optimizer, scaler, criterion, torch.device('cpu'), 0, 2026, 2,
                        train_mode=experiment.train_mode)
            saved = copy.deepcopy({'model': uninterrupted.state_dict(), 'optimizer': optimizer.state_dict(),
                                   'rng': torch.get_rng_state()})
            train_epoch(uninterrupted, samples, optimizer, scaler, criterion, torch.device('cpu'), 1, 2026, 2,
                        train_mode=experiment.train_mode)
            resumed = model()
            resumed_experiment = PartialExperiment()
            resumed_optimizer = torch.optim.AdamW(resumed_experiment.configure_model(resumed, root), lr=.01)
            resumed.load_state_dict(saved['model'])
            resumed_optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['rng'])
            train_epoch(resumed, samples, resumed_optimizer, scaler, criterion, torch.device('cpu'), 1, 2026, 2,
                        train_mode=resumed_experiment.train_mode)
            for name, tensor in resumed.state_dict().items():
                torch.testing.assert_close(tensor, uninterrupted.state_dict()[name], rtol=0, atol=0)

    def test_refuses_an_unknown_encoder_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'exactly12 blocks'):
                PartialExperiment().configure_model(MiniSAM(), Path(directory))


if __name__ == '__main__':
    unittest.main()
