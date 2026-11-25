import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from monai.losses import DiceCELoss
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fsdp_loss_diagnostics import RecordedLoss, inspect_forward, virtual_adamw, norm
from audit_fsdp_loss_gradients import snapshot, plan


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.sam = nn.Module()
        self.sam.image_encoder = nn.Linear(1, 1)
        self.sam.mask_decoder = nn.Linear(1, 1)
        self.detail = nn.Linear(1, 1)

    def forward(self, value):
        context = self.sam.mask_decoder(self.sam.image_encoder(value))
        return context + self.detail(value), context


def optimizer(model):
    return torch.optim.AdamW([
        {'params': model.sam.parameters(), 'lr': 1e-4, 'weight_decay': .01},
        {'params': model.detail.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4}])


class LossDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_control_matches_original_dicece_and_balancing_is_capped_normalized(self):
        logits = torch.linspace(-2., 2., 100).reshape(1, 1, 1, 1, -1).requires_grad_()
        target = torch.zeros_like(logits); target[..., 40] = 1
        recorded = RecordedLoss()
        actual = recorded(logits, target)
        expected = DiceCELoss(sigmoid=True, squared_pred=True)(logits, target)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(recorded.terms[0]['positive_weight'], 8.)
        zero = torch.zeros_like(logits, requires_grad=True)
        recorded(zero, target)
        self.assertAlmostEqual(float(recorded.terms[-1]['balanced'].detach()), float(nn.functional.binary_cross_entropy_with_logits(zero, target).detach()), places=6)

    def test_component_gradients_sum_to_exact_original_backward_and_cosine_is_defined(self):
        torch.manual_seed(2026)
        model = Tiny()
        criterion = RecordedLoss()
        target = torch.zeros(1, 1, 1, 1, 32); target[..., 16:18] = 1
        total = 0.
        for click in range(5):
            detail, context = model(torch.linspace(-1., 1., 32).reshape(-1, 1))
            detail, context = [v.reshape_as(target) for v in (detail, context)]
            total = total + criterion(detail, target) + .25 * criterion(context, target)
        report, names, candidates = inspect_forward(model, criterion)
        actual = torch.autograd.grad(total / 5, [p for p in model.parameters()])
        self.assertEqual(list(names), [n for n, _ in model.named_parameters()])
        for a, b in zip(actual, candidates['control']):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertIsNone(report['detail_context_cosine']['detail_cnn'])
        self.assertTrue(-1.000001 <= report['detail_context_cosine']['image_encoder'] <= 1.000001)
        self.assertGreater(report['component_logit_gradients']['detail_bce']['foreground_abs_sum'], 0)

    def test_virtual_adamw_matches_real_step_with_saved_moments_and_clipping_without_mutation(self):
        torch.manual_seed(2026)
        model = Tiny(); opt = optimizer(model)
        for p in model.parameters():
            p.grad = torch.full_like(p, .2)
        opt.step(); opt.zero_grad(set_to_none=True)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        saved = copy.deepcopy(opt.state_dict())
        names = [n for n, _ in model.named_parameters()]
        gradients = [torch.full_like(p, 2.) for p in model.parameters()]
        report = virtual_adamw(model, names, gradients, saved)
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)
        for pid, state in saved['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(value, opt.state_dict()['state'][pid][key], rtol=0, atol=0)
        for p, g in zip(model.parameters(), gradients):
            p.grad = g.clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        from fsdp_loss_diagnostics import group
        for key, row in report.items():
            pairs = [(n, p) for n, p in model.named_parameters() if group(n) == key]
            delta = norm([p.detach() - before[n] for n, p in pairs])
            self.assertAlmostEqual(row['delta_l2'], delta, delta=1e-7)

    def test_virtual_step_with_fresh_optimizer_and_zero_parameter_norm(self):
        model = Tiny(); opt = optimizer(model)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        report = virtual_adamw(model, [n for n, _ in model.named_parameters()],
                               [torch.ones_like(p) for p in model.parameters()], opt.state_dict())
        self.assertTrue(all(row['update_to_weight_ratio'] is None for row in report.values()))

    def test_snapshot_is_immutable_and_plan_does_not_start_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'best.pth'; destination = root / 'snapshot.pth'
            source.write_bytes(b'first checkpoint')
            snapshot(source, destination)
            source.write_bytes(b'new checkpoint')
            self.assertEqual(destination.read_bytes(), b'first checkpoint')
            with self.assertRaises(FileExistsError):
                snapshot(source, destination)
        self.assertIn('no training automatically launched', plan()['status'])
        self.assertEqual(len(plan()['arms']), 4)


if __name__ == '__main__':
    unittest.main()
