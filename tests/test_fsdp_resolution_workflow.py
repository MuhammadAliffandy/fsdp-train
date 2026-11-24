import argparse
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import signal
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_fsdp_resolution as workflow
import physical_fsdp as data
import train_sam_refine32 as trainer
from train_private_unet import state_hash


class WorkflowTests(unittest.TestCase):
    def test_missing_label_stops_before_loading_foundation_or_launching_any_child(self):
        from test_audit_foundation_sources import fixture as source_fixture
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            provenance = source_fixture(root / 'data')
            (root / 'base').mkdir()
            data.save(root / 'base/source_provenance.json', provenance)
            Path(provenance['train'][0]['source_label']).unlink()
            args = argparse.Namespace(foundation_run_dir=root / 'base', run_dir=root, train=True)
            with patch.object(workflow, 'inspect_foundation') as foundation, patch.object(workflow, 'run_child') as child:
                with self.assertRaisesRegex(ValueError, 'Foundation inputs unavailable'):
                    workflow.execute(args)
            foundation.assert_not_called(); child.assert_not_called()
            self.assertFalse(data.read(root / 'source_precheck.json')['ready'])

    def test_stop_signal_is_forwarded_to_active_child_and_exits_workflow(self):
        from unittest.mock import MagicMock
        child = MagicMock(pid=12345)
        child.poll.return_value = None
        handlers = {}
        def install(sig, handler):
            handlers[sig] = handler
            return signal.SIG_DFL
        def wait(*unused, **kwargs):
            if 'timeout' not in kwargs:
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            return -signal.SIGTERM
        child.wait.side_effect = wait
        with patch.object(workflow.subprocess, 'Popen', return_value=child), \
             patch.object(workflow.signal, 'signal', side_effect=install), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                workflow.run_child(['synthetic_child'])
        self.assertEqual(stopped.exception.code, 128 + signal.SIGTERM)
        child.send_signal.assert_called_once_with(signal.SIGTERM)

    def test_audit_only_exits_before_any_optimizer_step_checkpoint_or_training(self):
        from sam_refine32 import SamRefine32
        from test_sam_refine32 import FakeSAM
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            args = SimpleNamespace(run_dir=root, resume=False, seed=2026, device='cpu', precision='bf16',
                sam_lr=2e-6, head_lr=2e-5, epochs=100, audit_only=True)
            model = SamRefine32(FakeSAM())
            before = state_hash(model.state_dict())
            config = {'foundation': {'source_best_epoch': 30}, 'resolution': {'variant': 'detail'}}
            import fsdp_resolution
            import train_fsdp_foundation
            with patch.object(trainer, 'prepare', return_value=({}, config)), \
                 patch.object(trainer, 'build', return_value=(model, 'synthetic')), \
                 patch.object(train_fsdp_foundation, 'load_foundation_weights'), \
                 patch.object(fsdp_resolution, 'setup', return_value=(model, {})), \
                 patch.object(trainer, 'calibrate_validate', return_value=(.35, {}, {})), \
                 patch.object(fsdp_resolution, 'gradient_probe', return_value={'passed': True}), \
                 patch.object(fsdp_resolution, 'finish_audit', return_value={'complete': True}), \
                 patch.object(trainer, 'train_epoch') as train, \
                 patch('torch.optim.AdamW.step') as step:
                trainer.execute(args)
            train.assert_not_called(); step.assert_not_called()
            self.assertEqual(state_hash(model.state_dict()), before)
            self.assertFalse(list(root.glob('*.pth')))

    def test_failed_audit_prevents_every_training_variant(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            (root / 'base').mkdir()
            data.save(root / 'base/source_provenance.json', {'train': [], 'val': []})
            args = argparse.Namespace(foundation_run_dir=root / 'base', run_dir=root, train=True, epochs=100,
                lr_multiplier=.2, patience_epochs=50, device='cpu')
            calls = []
            def run(command):
                calls.append(command)
                return argparse.Namespace(returncode=1)
            with patch.object(workflow, 'inspect_foundation'), patch.object(workflow, 'run_child', run):
                with self.assertRaisesRegex(ValueError, 'no subsequent stage'):
                    workflow.execute(args)
            self.assertEqual(len(calls), 4)
            self.assertTrue(all('--audit-only' in command for command in calls))

    def test_matching_audits_allow_four_independent_training_runs_and_control_comparison(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            (root / 'base').mkdir()
            data.save(root / 'base/source_provenance.json', {'train': [], 'val': []})
            args = argparse.Namespace(foundation_run_dir=root / 'base', run_dir=root, train=True, epochs=100,
                lr_multiplier=.2, patience_epochs=50, device='cpu')
            def run(command):
                target = Path(command[command.index('--run-dir') + 1]); target.mkdir(parents=True)
                if '--audit-only' in command:
                    summary = {'complete': True, 'candidate_val': {}, 'gradient_probe': {'passed': True},
                        'optimizer_updates': 0, 'foundation_reproduced_before_adaptation': True,
                        'candidate_validation_delta_before_training': -.1}
                else:
                    summary = {'complete': True, 'best_epoch': 1, 'best_validation_metrics': {
                        '5': {'mean_bilateral_patient_dice': .5 if target.name == 'baseline' else .55}},
                        'validation_change_vs_foundation': .04, 'best_checkpoint_sha256': 'synthetic'}
                data.save(target / 'summary.json', summary)
                return argparse.Namespace(returncode=0)
            args.source_root = root / 'actual data'
            child_run = run
            def run(command):
                self.assertEqual(command[command.index('--source-root') + 1], str(args.source_root))
                return child_run(command)
            with patch.object(workflow, 'inspect_foundation'), patch.object(workflow, 'run_child', run):
                workflow.execute(args)
            summary = data.read(root / 'summary.json')
            self.assertEqual(len(summary['entries']), 8)
            train = [r for r in summary['entries'] if r['stage'] == 'train']
            self.assertAlmostEqual(train[1]['delta_vs_finetuned_baseline'], .05)
            self.assertFalse(summary['test_images_read'])


if __name__ == '__main__':
    unittest.main()
