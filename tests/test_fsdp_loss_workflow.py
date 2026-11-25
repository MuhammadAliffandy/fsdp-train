import argparse
import contextlib
import copy
import io
import os
from pathlib import Path
import sys
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import physical_fsdp as data
import train_fsdp_loss_tiny as tiny
import run_fsdp_loss_workflow as workflow
from test_fsdp_loss_diagnostics import Tiny
import audit_fsdp_loss_gradients as audit
from evaluate_private_validation import file_hash


class LossWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_candidates_change_detail_only_and_auxiliary_uses_original_loss(self):
        logits = torch.linspace(-1, 1, 32).reshape(1, 1, 1, 1, -1).requires_grad_()
        target = torch.zeros_like(logits); target[..., 16] = 1
        candidate, control = tiny.CandidateLoss(True), tiny.CandidateLoss(False)
        first = candidate(logits, target)
        self.assertFalse(torch.equal(first, control(logits, target)))
        torch.testing.assert_close(candidate(logits, target), control(logits, target), rtol=0, atol=0)

    def test_loader_snapshots_full_weights_validates_real_cache_and_selects_train_sides(self):
        from test_fsdp_verified_cache import fixture
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, manifest, prepared, _ = fixture(root)
            foundation = root / 'foundation'; foundation.mkdir()
            cohort = root / 'cohort'; cohort.mkdir()
            frozen_data = root / 'data'; frozen_data.mkdir()
            data.save(frozen_data / 'manifest.json', manifest)
            data.save(cohort / 'config.json', {'data_dir': str(frozen_data), 'source_root': str(dataset)})
            torch.save(prepared, foundation / 'hybrid_data.pt')
            old = {'cache_sha256': file_hash(foundation / 'hybrid_data.pt'), 'physical_policy': prepared['policy']}
            config = {**old, 'variant': 'history', 'test_images_read': False, 'versions': {},
                'parameters': {'foundation_run_dir': str(foundation), 'cohort_run_dir': str(cohort),
                    'sam_dir': str(root / 'sam'), 'checkpoint': str(root / 'turbo.pth'),
                    'auxiliary_weight': .25, 'seed': 2026, 'precision': 'bf16', 'source_best_epoch': 35},
                'code_sha256': audit.trainer.code_signature()}
            source = root / 'source'; source.mkdir()
            data.save(source / 'config.json', config)
            trained = Tiny()
            with torch.no_grad():
                for parameter in trained.parameters():
                    parameter.fill_(.75)
            torch.save({'config': config, 'model_state_dict': trained.state_dict(), 'best_score': .4938,
                        'epoch': 5}, source / 'research_best.pth')
            before = file_hash(source / 'research_best.pth')
            args = argparse.Namespace(run_dir=source, output_dir=root / 'audit', device='cpu', source_root=None,
                                      cases=['train_01', 'train_02'])
            with patch.object(audit.foundation, 'inspect_foundation', return_value=(old, {})), \
                 patch.object(audit.trainer.base, 'build', return_value=(Tiny(), 'proxy')), \
                 patch.object(audit.ResearchModel, 'from_foundation', side_effect=lambda original, *args: original), \
                 patch.object(audit.trainer.base, 'set_stage', side_effect=lambda model, *args: model.train()):
                model, _, _, saved, checksum, selected = audit.load_source(args)
            self.assertEqual(len(selected), 4)
            self.assertEqual({s['case'] for s in selected}, set(args.cases))
            self.assertEqual(checksum, before)
            self.assertEqual(file_hash(source / 'research_best.pth'), before)
            for n, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, trained.state_dict()[n], rtol=0, atol=0)
            self.assertNotIn('model_state_dict', saved)

    def test_tiny_training_updates_copies_scores_native_training_only_and_retains_initial_candidate(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            image = data.sitk.GetImageFromArray(np.zeros((16,) * 3, np.int16))
            ct = np.zeros((16,) * 3, np.float32); ct[6:10, 6:10, 6:10] = 1
            mask = ct > 0
            prepared = {'samples': {'train': [], 'val': []}, 'patients': {'train': {}, 'val': {}}}
            for case in ('train_01', 'train_02'):
                prepared['patients']['train'][case] = {'shape': [16] * 3, 'target_indices': np.flatnonzero(mask)}
                for side in ('RFSDP', 'LFSDP'):
                    sample = data.make_sample(image, ct, mask, case, side, {'shape': [16] * 3, 'spacing': [1.] * 3},
                        {'clip_lower': -1000., 'clip_upper': 1000., 'mean': 0., 'std': 1.})
                    prepared['samples']['train'].append(sample)
            torch.manual_seed(2026)
            source = Tiny()
            before = {n: p.detach().clone() for n, p in source.named_parameters()}
            def load(args):
                args.output_dir.mkdir()
                return copy.deepcopy(source), prepared, argparse.Namespace(seed=2026, precision='bf16', source_best_epoch=35), \
                    {'best_threshold': .5}, 'synthetic hash', prepared['samples']['train']
            def run(model, sample, device, seed, criterion=None, augment=False, auxiliary_weight=.25):
                self.assertFalse(augment)
                shape = sample['image'].shape
                detail, context = model(sample['image'].reshape(-1, 1))
                detail, context = detail.reshape(shape), context.reshape(shape)
                loss = None
                if criterion is not None:
                    loss = sum(criterion(detail, sample['target'].float())
                               + auxiliary_weight * criterion(context, sample['target'].float()) for _ in range(5)) / 5
                return loss, {k: {'refined': detail, 'coarse_on_detail': context} for k in (1, 5)}
            args = argparse.Namespace(run_dir=root / 'source', output_dir=root / 'tiny', cases=['train_01', 'train_02'],
                device='cpu', source_root=None, loss='balanced', sam_lr=.01, head_lr=.01, epochs=2, eval_every=1, target=1.)
            with patch.object(tiny, 'load_source', side_effect=load), patch.object(tiny, 'run_sample', side_effect=run), \
                 patch.object(tiny.trainer.base, 'run_sample', side_effect=run), \
                 patch.object(tiny.trainer.base, 'set_stage', side_effect=lambda model, *args: model.train()):
                result = tiny.execute(args)
            self.assertTrue(result['complete'])
            self.assertFalse(result['validation_inference_performed'])
            self.assertFalse(result['test_images_read'])
            latest = torch.load(args.output_dir / 'tiny_latest.pth', weights_only=False)
            self.assertEqual(latest['epoch'], 2)
            self.assertTrue(all(row['optimizer_updates'] == 4 and row['amp_skipped_updates'] == 0 for row in latest['history']))
            self.assertTrue(any(not torch.equal(before[n], latest['model_state_dict'][n]) for n in before))
            for n, p in source.named_parameters():
                torch.testing.assert_close(p, before[n], rtol=0, atol=0)
            self.assertGreaterEqual(result['best_training_metrics']['minimum_side_dice'], result['initial_training_metrics']['minimum_side_dice'])

    def fixture(self, root):
        source = root / 'source'; source.mkdir()
        cohort = root / 'cohort'; cohort.mkdir()
        data.save(cohort / 'config.json', {'data_dir': str(root / 'dataset'), 'source_root': str(root / 'raw')})
        config = {'parameters': {'foundation_run_dir': str(root / 'foundation'), 'sam_dir': str(root / 'sam'),
                                  'cohort_run_dir': str(cohort)}}
        data.save(source / 'config.json', config)
        (source / 'research_best.pth').write_bytes(b'immutable original')
        return argparse.Namespace(run_dir=source, output_dir=root / 'workflow', source_root=None, device='cpu', epochs=2)

    def test_workflow_uses_one_snapshot_three_independent_tiny_arms_and_stops_on_failure(self):
        for failure in (False, True):
            with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
                root = Path(folder); args = self.fixture(root)
                seen = []
                def child(command, record):
                    source = Path(command[command.index('--run-dir') + 1])
                    self.assertEqual((source / 'research_best.pth').read_bytes(), b'immutable original')
                    output = Path(command[command.index('--output-dir') + 1]); output.mkdir()
                    seen.append(output.name)
                    if failure and output.name == 'tiny_balanced':
                        return -15
                    data.save(output / 'summary.json', {'complete': True, 'test_images_read': False,
                        'validation_inference_performed': False, 'best_epoch': 1, 'completed_epochs': 2,
                        'diagnostic_passed': False, 'best_training_metrics': {}})
                    return 0
                with patch.object(workflow, 'run_child', side_effect=child), \
                     patch.object(workflow.shutil, 'disk_usage', return_value=argparse.Namespace(free=100 * 2**30)):
                    if failure:
                        with self.assertRaisesRegex(ValueError, 'returncode=-15'):
                            workflow.execute(args)
                        self.assertNotIn('tiny_auxiliary', seen)
                    else:
                        workflow.execute(args)
                        self.assertEqual(seen, ['audit', 'tiny_control', 'tiny_balanced', 'tiny_auxiliary'])
                self.assertEqual((args.run_dir / 'research_best.pth').read_bytes(), b'immutable original')

    def test_output_space_and_dataset_guards_do_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); args = self.fixture(root)
            with patch.object(workflow.shutil, 'disk_usage', return_value=argparse.Namespace(free=1)):
                with self.assertRaisesRegex(ValueError, 'Insufficient output filesystem'):
                    workflow.execute(args)
            self.assertFalse(args.output_dir.exists())
            args.output_dir = root / 'raw' / 'forbidden'
            with self.assertRaisesRegex(ValueError, 'dataset/model/source'):
                workflow.execute(args)
            self.assertFalse(args.output_dir.exists())

    def test_launcher_gpu_failure_stops_before_starting_child_and_success_masks_one_uuid(self):
        for success in (False, True):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder); scripts = root / 'scripts'; scripts.mkdir()
                source = root / 'source'; source.mkdir()
                shutil.copy(Path(workflow.__file__).with_name('launch_fsdp_loss_workflow_nohup.sh'), scripts)
                (scripts / 'fsdp_research_gpu.py').write_text('print("GPU-test-uuid")\n' if success else 'raise SystemExit(3)\n')
                (scripts / 'run_fsdp_loss_workflow.py').write_text('''import os, pathlib, sys
pathlib.Path(os.environ['TASK_CAPTURE']).write_text(os.environ['CUDA_VISIBLE_DEVICES'] + '\\n' + '\\n'.join(sys.argv[1:]))
''')
                capture = root / 'capture'
                env = dict(os.environ, CONDA_DEFAULT_ENV='fsdp-med3d', FSDP_LOSS_SOURCE_DIR=str(source),
                           TASK_CAPTURE=str(capture), ALLOW_GPU_SHARING='1')
                result = subprocess.run(['bash', str(scripts / 'launch_fsdp_loss_workflow_nohup.sh')],
                                        env=env, capture_output=True, text=True, timeout=15)
                if success:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertTrue(capture.read_text().startswith('GPU-test-uuid\n'))
                    self.assertIn('--run-dir\n' + str(source.resolve()), capture.read_text())
                else:
                    self.assertEqual(result.returncode, 3)
                    self.assertFalse(capture.exists())


if __name__ == '__main__':
    unittest.main()
