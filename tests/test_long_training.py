import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_private_long as trainer


class Samples(Dataset):
    def __len__(self):
        return 54

    def __getitem__(self, index):
        return torch.full((1, 2, 2, 2), (index + 1) / 54), torch.ones((1, 2, 2, 2)), index


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))

    def forward(self, image):
        return image * self.weight


class AccumulationTests(unittest.TestCase):
    def test_all_54_samples_match_explicit_window_mean_updates(self):
        model, reference = ScalarModel(), ScalarModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.05)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.05)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        criterion = torch.nn.MSELoss()
        seen = []
        def loss(model, image, target, criterion, clicks, seed):
            seen.append(float(image.flatten()[0]))
            prediction = model(image)
            return criterion(prediction, target.float()), prediction.detach()
        with patch.object(trainer, 'interactive_loss', loss), contextlib.redirect_stdout(io.StringIO()):
            result = trainer.train_epoch(model, Samples(), optimizer, scaler, criterion, torch.device('cpu'), 0, 2026, 8, 5)
        batches = list(DataLoader(Samples(), batch_size=1, shuffle=True, num_workers=0,
                                  generator=torch.Generator().manual_seed(2026)))
        for start in range(0, 54, 8):
            window = batches[start:start + 8]
            reference_optimizer.zero_grad()
            mean_loss = sum(criterion(reference(image), target.float()) for image, target, _ in window) / len(window)
            mean_loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
            reference_optimizer.step()
        self.assertEqual(result['optimizer_updates'], 7)
        self.assertEqual(result['amp_skipped_updates'], 0)
        self.assertEqual(len(set(seen)), 54)
        torch.testing.assert_close(model.weight, reference.weight, rtol=1e-6, atol=1e-7)

    def test_accumulation_one_updates_every_batch(self):
        self.assertEqual([trainer.accumulation_window(i, 3, 1) for i in range(3)], [(1, True)] * 3)


if __name__ == '__main__':
    unittest.main()
