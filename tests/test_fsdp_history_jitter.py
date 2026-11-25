import argparse
import contextlib
import copy
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fsdp_history_jitter as jitter
import physical_fsdp as data
import run_fsdp_research as workflow
from evaluate_private_validation import file_hash
from fsdp_research_model import ResearchModel
from sam_refine32 import SamRefine32, next_click
from test_fsdp_research_workflow import arguments, completed_child
from test_sam_refine32 import FakeSAM, tensors


def paired():
    shape, context_shape = (16, 20, 24), (32, 32, 32)
    spacing = torch.tensor([.25, .5, .25])
    ratio = torch.tensor([.5, 1., .75])
    context_spacing = np.array(shape) * spacing.numpy() / ratio.numpy() / context_shape
    def view(size, steps):
        axes = [(torch.arange(n) - (n - 1) / 2) * float(s) for n, s in zip(size, steps)]
        xyz = torch.meshgrid(*axes, indexing='ij')
        image = (xyz[0] + 10 * xyz[1] + 100 * xyz[2])[None, None]
        mask = torch.zeros_like(image, dtype=torch.uint8)
        mid = np.array(size) // 2
        mask[..., mid[0]-1:mid[0]+1, mid[1]-1:mid[1]+1, mid[2]-1:mid[2]+1] = 1
        return image, mask, torch.ones_like(mask, dtype=torch.bool)
    image, target, valid = view(shape, spacing)
    context, context_target, context_valid = view(context_shape, context_spacing)
    return [image, context, ratio, spacing, target, valid, context_target, context_valid]


class JitterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_physical_xyz_translation_aligns_both_different_grids_and_preserves_masks(self):
        inputs = paired()
        before = [value.clone() for value in inputs]
        shifted, report = jitter.translate_pair(*inputs, seed=10, offset_voxels=(2, -1, 4))
        self.assertTrue(report['applied'])
        np.testing.assert_allclose(report['sampling_offset_mm_xyz'], [.5, -.5, 1.])
        # The linear physical-coordinate image must change by the same amount in each view.
        for original, actual in ((inputs[0], shifted[0]), (inputs[1], shifted[1])):
            difference = actual[..., 5:-5, 5:-5, 5:-5] - original[..., 5:-5, 5:-5, 5:-5]
            torch.testing.assert_close(difference, torch.full_like(difference, 95.5), atol=1e-4, rtol=0)
        for original, actual, valid in ((inputs[4], shifted[2], shifted[3]), (inputs[6], shifted[4], shifted[5])):
            self.assertEqual(actual.count_nonzero(), original.count_nonzero())
            self.assertTrue(valid[actual.bool()].all())
            self.assertTrue(torch.all((actual == 0) | (actual == 1)))
        torch.testing.assert_close(torch.nonzero(shifted[2][0, 0]).float().mean(0),
                                   torch.nonzero(inputs[4][0, 0]).float().mean(0) - torch.tensor([2., -1., 4.]))
        self.assertFalse(shifted[3][..., -1, :, :].any())
        self.assertFalse(shifted[0][~shifted[3]].any())
        for a, b in zip(inputs, before):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        click, label = next_click(torch.zeros_like(shifted[0]), shifted[2], shifted[3], np.random.default_rng(10))
        x, y, z = click.long()[0, 0]
        self.assertEqual(label.item(), 1)
        self.assertTrue(shifted[2][0, 0, x, y, z])

    def test_seed_is_reproducible_independent_and_has_real_offset_variation(self):
        inputs = paired()
        torch_state, np_state = torch.get_rng_state().clone(), np.random.get_state()
        records = []
        for seed in range(50):
            first, report = jitter.translate_pair(*inputs, seed=seed)
            second, again = jitter.translate_pair(*inputs, seed=seed)
            self.assertEqual(report, again)
            for a, b in zip(first, second):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            records.append(report)
        self.assertGreater(sum(r['applied'] for r in records), 10)
        self.assertGreater(sum(r['reason'] == 'not_selected' for r in records), 10)
        self.assertGreater(len({tuple(r['sampling_offset_mm_xyz']) for r in records}), 10)
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
        self.assertEqual(np.random.get_state()[2:], np_state[2:])

    def test_context_clipping_rejects_both_views_and_edge_target_is_never_dropped(self):
        inputs = paired()
        inputs[6].zero_(); inputs[6][..., 0, 16, 16] = 1
        outputs, record = jitter.translate_pair(*inputs, seed=1, offset_voxels=(4, 0, 0))
        self.assertEqual(record['reason'], 'target_or_support_loss')
        for a, b in zip(outputs, (inputs[i] for i in (0, 1, 4, 5, 6, 7))):
            self.assertIs(a, b)
        inputs = paired(); inputs[4].zero_(); inputs[4][..., 0, 0, 0] = 1
        outputs, record = jitter.translate_pair(*inputs, seed=1, offset_voxels=(4, 0, 0))
        self.assertEqual(record['reason'], 'unsafe_offset')
        self.assertEqual(outputs[2].count_nonzero(), 1)
        inputs[2][0] = 0
        with self.assertRaisesRegex(ValueError, 'geometry'):
            jitter.translate_pair(*inputs, seed=1)

    def test_model_weights_eval_predictions_and_click_history_are_unchanged(self):
        torch.manual_seed(2026)
        original = SamRefine32(FakeSAM()).eval()
        torch.nn.init.normal_(original.residual.weight, std=.01)
        control = ResearchModel.from_foundation(original, 'history').eval()
        model = jitter.HistoryJitterModel.from_foundation(original, 'history').eval()
        for key, value in control.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        image, context, target, context_target = tensors()
        args = (image, context, torch.ones(3), torch.full((3,), .25), target,
                torch.ones_like(target, dtype=torch.bool), (1, 5), 2026)
        with torch.inference_mode(), patch.object(jitter, 'translate_pair', side_effect=AssertionError('eval augmented')):
            _, expected = control.interaction_function(control, *args)
            _, actual = model.interaction_function(model, *args)
        for click in (1, 5):
            torch.testing.assert_close(actual[click]['refined'], expected[click]['refined'], rtol=0, atol=0)
        from monai.losses import DiceCELoss
        model.train()
        with patch.object(jitter, 'translate_pair', wraps=jitter.translate_pair) as transform:
            loss, _ = model.interaction_function(model, *args, DiceCELoss(sigmoid=True, squared_pred=True),
                context_target, torch.ones_like(context_target, dtype=torch.bool))
            loss.backward()
        transform.assert_called_once()
        for parameter in (model.residual.weight, model.sam.mask_decoder.project.weight, model.full[0].weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum(), 0)

    def test_matched_control_checks_source_recipe_and_artifact_integrity(self):
        import train_fsdp_research as trainer
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {'variant': 'history', 'experiment': 'fsdp_research_screening',
                      'research_source': {'checkpoint_sha256': 'source49'},
                      'parameters': {'epochs': 100, 'lr_multiplier': .1, 'patience_epochs': 25, 'augmentation': 'light'},
                      'code_sha256': trainer.code_signature()}
            data.save(root / 'config.json', config)
            (root / 'research_best.pth').write_bytes(b'control model')
            report = {'complete': True, 'test_images_read': False, 'variant': 'history',
                'train_patients': 54, 'validation_patients': 10, 'planned_epochs': 100, 'completed_epochs': 25,
                'best_epoch': 0,
                'best_checkpoint_sha256': file_hash(root / 'research_best.pth'),
                'best_validation_metrics': {'5': {'mean_bilateral_patient_dice': .4918}}}
            data.save(root / 'summary.json', report)
            args = argparse.Namespace(matched_control_dir=root)
            for key, changed in (('epochs', 20), ('lr_multiplier', .2), ('patience_epochs', 50)):
                wrong = copy.deepcopy(config); wrong['parameters'][key] = changed
                with self.assertRaisesRegex(ValueError, 'same source and recipe'):
                    jitter.configure_experiment(args, wrong)
            wrong = copy.deepcopy(config); wrong['research_source']['checkpoint_sha256'] = 'other source'
            with self.assertRaisesRegex(ValueError, 'same source and recipe'):
                jitter.configure_experiment(args, wrong)
            jitter.configure_experiment(args, config)
            self.assertAlmostEqual(jitter.comparison(config, .51)['validation_change_vs_matched_control'], .0182)
            (root / 'research_best.pth').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                jitter.comparison(config, .51)

    def test_parent_forwards_opt_in_runs_history_only_and_reports_control_delta(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder); args = arguments(root); args.run_dir.mkdir()
            args.research_source_dir = root / 'source'
            args.roi_jitter, args.matched_control_dir = True, root / 'control'
            def child(command, record):
                self.assertIn('--roi-jitter', command)
                self.assertEqual(command[command.index('--matched-control-dir') + 1], str(args.matched_control_dir))
                completed_child(command, record)
                path = args.run_dir / 'history/summary.json'
                report = data.read(path)
                report.update(research_source={}, validation_change_vs_research_source=.01,
                              roi_jitter=jitter.POLICY, matched_control={}, validation_change_vs_matched_control=.01)
                data.save(path, report)
                return 0
            with patch.object(workflow, 'inspect_foundation'), patch.object(workflow, 'run_child', side_effect=child) as call:
                workflow.execute(args)
            call.assert_called_once()
            report = data.read(args.run_dir / 'summary.json')
            self.assertEqual(report['entries'][0]['delta_vs_matched_control'], .01)

    def test_real_training_path_replays_identically_after_checkpoint_resume(self):
        import train_fsdp_research as trainer
        class Model(torch.nn.Module):
            interaction_function = staticmethod(jitter.HistoryJitterModel.interaction_function)
            variant, amp_dtype = 'history', torch.bfloat16
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.1))
        def proxy(model, image, context, ratio, spacing, target, valid, milestones, seed,
                  criterion=None, context_target=None, context_valid=None, auxiliary_weight=.25, field=None):
            logits = model.weight * image
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[valid], target.float()[valid])
            return loss, {}
        inputs = paired()
        sample = dict(zip(('image', 'sam_image', 'fov_ratio', 'detail_spacing', 'target', 'valid',
                           'sam_target', 'sam_valid'), inputs))
        prepared = {'samples': {'train': [copy.deepcopy(sample) for _ in range(4)]}}
        args = argparse.Namespace(roi_jitter=True, source_best_epoch=35, seed=2026, augmentation='light',
                                  accumulation=2, auxiliary_weight=.25, variant='history', epochs=2)
        def setup():
            model = Model()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            return model, optimizer, scaler
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(ResearchModel, 'interaction_function', side_effect=proxy), \
             patch.object(trainer.base, 'set_stage', return_value='test'):
            uninterrupted, optimizer, scaler = setup()
            expected = [trainer.train_epoch(uninterrupted, prepared, optimizer, scaler,
                        object(), torch.device('cpu'), epoch, args) for epoch in range(2)]
            partial, optimizer, scaler = setup()
            first = trainer.train_epoch(partial, prepared, optimizer, scaler, object(), torch.device('cpu'), 0, args)
            path = Path(folder) / 'latest.pth'
            torch.save({'model': partial.state_dict(), 'optimizer': optimizer.state_dict()}, path)
            saved = torch.load(path, weights_only=False)
            resumed, optimizer, scaler = setup()
            resumed.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            second = trainer.train_epoch(resumed, prepared, optimizer, scaler, object(), torch.device('cpu'), 1, args)
            self.assertEqual(expected, [first, second])
            self.assertEqual(sum(second['roi_jitter_counts'].values()), 4)
            torch.testing.assert_close(resumed.weight, uninterrupted.weight, rtol=0, atol=0)

    def test_launcher_keeps_history_source_and_matched_control_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); scripts = root / 'scripts'; scripts.mkdir()
            for name in ('launch_fsdp_history_jitter_nohup.sh', 'launch_fsdp_history_nohup.sh'):
                shutil.copy(Path(jitter.__file__).with_name(name), scripts)
            (scripts / 'launch_fsdp_research_nohup.sh').write_text('''#!/bin/sh
printf '%s\\n' "$FSDP_RESEARCH_SOURCE_DIR" "$FSDP_MATCHED_CONTROL_DIR" "$FSDP_HISTORY_ROI_JITTER" "$ADDITIONAL_EPOCHS" "$LR_MULTIPLIER" "$PATIENCE_EPOCHS" "$FSDP_RESEARCH_RUN_NAME" > "$TASK_CAPTURE"
''')
            env = dict(os.environ, TASK_CAPTURE=str(root / 'capture'))
            for key in ('FSDP_RESEARCH_SOURCE_DIR', 'FSDP_MATCHED_CONTROL_DIR', 'FSDP_RESEARCH_RUN_NAME',
                        'ADDITIONAL_EPOCHS', 'LR_MULTIPLIER', 'PATIENCE_EPOCHS'):
                env.pop(key, None)
            result = subprocess.run(['bash', str(scripts / 'launch_fsdp_history_jitter_nohup.sh')],
                                    env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            settings = (root / 'capture').read_text().splitlines()
            self.assertIn('fsdp_research_gpu7_20261124T132503Z_3250370/history', settings[0])
            self.assertIn('fsdp_research_history_20261125T022714Z_2766483/history', settings[1])
            self.assertEqual(settings[2:6], ['1', '100', '0.1', '25'])
            self.assertTrue(settings[6].startswith('fsdp_research_jitter_'))


if __name__ == '__main__':
    unittest.main()
