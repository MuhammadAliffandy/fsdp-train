import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fsdp_research_model as study
from sam_refine32 import SamRefine32, interactions
from test_sam_refine32 import FakeSAM, FakeEncoder, tensors


class ResearchModelTests(unittest.TestCase):
    def test_gradient_preflight_checks_actual_shapes_and_does_not_change_model(self):
        from monai.losses import DiceCELoss
        import train_fsdp_research as training
        from train_private_unet import state_hash
        class Encoder(FakeEncoder):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Conv3d(8, 8, 1) for _ in range(12)])
            def forward(self, value):
                value = super().forward(value)
                for layer in self.blocks:
                    value = value + .1 * layer(value)
                return value
        torch.manual_seed(2026)
        sam = FakeSAM(); sam.image_encoder = Encoder()
        original = SamRefine32(sam)
        nn.init.normal_(original.residual.weight, std=.01)
        model = study.ResearchModel.from_foundation(original, 'refiner')
        model.amp_dtype = torch.bfloat16
        image, context, target, context_target = tensors()
        sample = {'image': image, 'sam_image': context, 'target': target,
            'valid': torch.ones_like(target, dtype=torch.bool), 'sam_target': context_target,
            'sam_valid': torch.ones_like(context_target, dtype=torch.bool),
            'fov_ratio': torch.ones(3), 'detail_spacing': torch.full((3,), .25)}
        args = SimpleNamespace(source_best_epoch=30, seed=2026, auxiliary_weight=.25, variant='refiner')
        before = state_hash(model.state_dict())
        report = training.gradient_probe(model, {'samples': {'train': [sample]}}, args,
                                        DiceCELoss(sigmoid=True, squared_pred=True), torch.device('cpu'))
        self.assertTrue(report['passed'])
        self.assertEqual(report['feature_shapes']['tokens'][-3:], [8] * 3)
        self.assertEqual(report['feature_shapes']['local_refiner'][-3:], [32] * 3)
        self.assertEqual(before, state_hash(model.state_dict()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_native_surface_metrics_are_physical_and_empty_predictions_are_counted(self):
        from fsdp_research_metrics import surface_metrics
        shape = (16, 16, 16)
        mask = np.zeros(shape, bool); mask[4:8, 5:9, 6:10] = True
        target = np.flatnonzero(mask)
        exact = surface_metrics(target, target, shape, (.25, .5, 1.))
        self.assertEqual(exact['hd95_mm'], 0.)
        self.assertEqual(exact['surface_dice_0_5mm'], 1.)
        moved = np.flatnonzero(np.roll(mask, 2, axis=2))
        shifted = surface_metrics(moved, target, shape, (.25, .5, 1.))
        self.assertAlmostEqual(shifted['hd95_mm'], 2.)
        self.assertLess(shifted['surface_dice_0_5mm'], 1.)
        empty = surface_metrics(np.array([], dtype=int), target, shape, (.25, .5, 1.))
        self.assertTrue(empty['empty_prediction'])
        self.assertIsNone(empty['hd95_mm'])
        self.assertEqual(empty['surface_dice_1mm'], 0.)

    def test_control_is_identical_and_prompt_history_has_exactly_one_to_five_points(self):
        torch.manual_seed(2026)
        original = SamRefine32(FakeSAM()).eval()
        nn.init.normal_(original.residual.weight, std=.01)
        image, context, target, _ = tensors()
        valid = torch.ones_like(target, dtype=torch.bool)
        call = (image, context, torch.ones(3), torch.full((3,), .25), target, valid, (1, 5), 2026)
        with torch.inference_mode():
            _, expected = interactions(original, *call)
            for variant in study.VARIANTS:
                candidate = study.ResearchModel.from_foundation(original, variant).eval()
                for name, value in original.state_dict().items():
                    torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)
                counts = []
                hook = candidate.sam.prompt_encoder.register_forward_pre_hook(
                    lambda module, args, kwargs: counts.append(kwargs['points'][0].shape[1]), with_kwargs=True)
                _, actual = candidate.interaction_function(candidate, *call)
                hook.remove()
                self.assertEqual(counts, [1, 2, 3, 4, 5] if variant == 'history' else [1] * 5)
                torch.testing.assert_close(actual[1]['refined'], expected[1]['refined'], rtol=0, atol=0)
                if variant != 'history':
                    torch.testing.assert_close(actual[5]['refined'], expected[5]['refined'], rtol=0, atol=0)

    def test_boundary_distance_is_physical_and_does_not_use_padding_as_background(self):
        target = np.zeros((9, 9, 9), bool); target[4, 4, 4] = True
        valid = np.ones_like(target)
        field = study.signed_distance(target, valid, (.25, .5, 1.), 5.)
        self.assertAlmostEqual(float(field[5, 4, 4]), .25 / 5.)
        self.assertAlmostEqual(float(field[4, 4, 5]), 1. / 5.)
        self.assertLess(float(field[4, 4, 4]), 0)
        valid = np.zeros((7,) * 3, bool); valid[1:6, 1:6, 1:6] = True
        target = np.zeros_like(valid); target[1:4, 1:6, 1:6] = True
        field = study.signed_distance(target, valid, (1., 1., 1.), 5.)
        self.assertAlmostEqual(float(field[1, 3, 3]), -.6, places=6)
        padded = study.signed_distance(np.pad(target, 3), np.pad(valid, 3), (1., 1., 1.), 5.)
        np.testing.assert_array_equal(field, padded[3:-3, 3:-3, 3:-3])
        self.assertTrue((field[~valid] == 0).all())

    def test_boundary_gradients_have_correct_sign_and_zero_padding_gradient(self):
        target = np.zeros((7,) * 3, bool); target[3, 3, 3] = True
        valid = np.zeros_like(target); valid[1:6, 1:6, 1:6] = True
        field = torch.tensor(study.signed_distance(target, valid, (1.,) * 3))[None, None]
        logits = torch.zeros_like(field, requires_grad=True)
        mask = torch.tensor(valid)[None, None]
        study.boundary_loss(logits, field, mask).backward()
        self.assertLess(float(logits.grad[0, 0, 3, 3, 3]), 0)
        self.assertGreater(float(logits.grad[0, 0, 2, 2, 2]), 0)
        self.assertTrue((logits.grad[~mask] == 0).all())

    def test_zero_initialized_refiner_can_learn_beyond_its_output_projection(self):
        torch.manual_seed(2026)
        model = study.ResearchModel.from_foundation(SamRefine32(FakeSAM()), 'refiner').eval()
        image, context, target, _ = tensors()
        optimizer = torch.optim.AdamW(model.research_refiner.parameters(), lr=.01)
        for step in range(2):
            model.zero_grad(set_to_none=True)
            cache = model.encode(image, context, torch.ones(3), torch.full((3,), .25))
            point, label = torch.tensor([[[30., 30., 30.]]]), torch.ones((1, 1), dtype=torch.long)
            logits, _, _ = model.decode(cache, point, label, torch.zeros(1, 1, 32, 32, 32), [(point, label)])
            (logits - 1).square().mean().backward()
            self.assertGreater(model.research_refiner.output.weight.grad.abs().sum().item(), 0)
            if step == 1:
                self.assertGreater(model.research_refiner.features[0].weight.grad.abs().sum().item(), 0)
            optimizer.step()

    def test_boundary_field_uses_exact_same_lr_reflection_as_target(self):
        shape = (8, 10, 12)
        target = torch.zeros(1, 1, *shape, dtype=torch.uint8); target[..., 1:3, 2:4, 3:5] = 1
        valid = torch.ones_like(target, dtype=torch.bool)
        field = torch.tensor(study.signed_distance(target[0, 0].numpy(), valid[0, 0].numpy(), (.25,) * 3))[None, None]
        sample = {'image': target.float(), 'target': target, 'valid': valid, 'sam_image': target.float().clone(),
            'sam_target': target.clone(), 'sam_valid': valid.clone(), 'fov_ratio': torch.ones(3),
            'detail_spacing': torch.full((3,), .25), 'research_distance': field}
        captured = []
        def interaction(*args, **kwargs):
            captured.append((args[5], kwargs['field']))
            return None, {}
        model = SimpleNamespace(variant='boundary', amp_dtype=torch.bfloat16, interaction_function=interaction)
        seed = next(i for i in range(100) if np.random.default_rng(i).random() < .5)
        study.run_sample(model, sample, torch.device('cpu'), seed, criterion=object(), augment=True)
        torch.testing.assert_close(captured[0][0], torch.flip(target, (2,)), rtol=0, atol=0)
        torch.testing.assert_close(captured[0][1], torch.flip(field, (2,)), rtol=0, atol=0)
        torch.testing.assert_close(sample['research_distance'], field, rtol=0, atol=0)

    def test_boundary_interactions_reach_existing_sam_and_refiner(self):
        from monai.losses import DiceCELoss
        torch.manual_seed(2026)
        model = study.ResearchModel.from_foundation(SamRefine32(FakeSAM()), 'boundary')
        image, context, target, context_target = tensors()
        valid = torch.ones_like(target, dtype=torch.bool)
        field = torch.tensor(study.signed_distance(target[0, 0].numpy(), valid[0, 0].numpy(), (.25,) * 3))[None, None]
        loss, _ = model.interaction_function(model, image, context, torch.ones(3), torch.full((3,), .25),
            target, valid, (1, 5), 2026, DiceCELoss(sigmoid=True, squared_pred=True),
            context_target, torch.ones_like(context_target, dtype=torch.bool), field=field)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in (model.residual.weight, model.sam.image_encoder.project.weight, model.sam.mask_decoder.project.weight):
            self.assertTrue(parameter.grad.isfinite().all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
