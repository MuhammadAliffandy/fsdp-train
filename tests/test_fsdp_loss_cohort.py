import argparse
import contextlib
import copy
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fsdp_loss_cohort as cohort
import run_fsdp_loss_cohort as parent
import physical_fsdp as data
from evaluate_private_validation import file_hash
from fsdp_research_model import ResearchModel
from sam_refine32 import SamRefine32
from test_sam_refine32 import FakeSAM, tensors
from test_fsdp_history_jitter import paired


def fixture(root):
    source = root / 'source'; source.mkdir()
    (source / 'research_best.pth').write_bytes(b'full54patient weights')
    data.save(source / 'config.json', {'source': 'full54cohort'})
    diagnostic = root / 'diagnostic'; diagnostic.mkdir()
    frozen = diagnostic / 'frozen_source'; frozen.mkdir()
    shutil.copy(source / 'research_best.pth', frozen)
    shutil.copy(source / 'config.json', frozen)
    checksum = file_hash(source / 'research_best.pth')
    data.save(diagnostic / 'summary.json', {'complete': True, 'test_images_read': False,
        'validation_inference_performed': False, 'source_checkpoint_sha256': checksum})
    for arm in ('control', 'balanced'):
        folder = diagnostic / f'tiny_{arm}'; folder.mkdir()
        data.save(folder / 'summary.json', {'complete': True, 'diagnostic_passed': True, 'test_images_read': False,
            'validation_inference_performed': False, 'patients': ['train_01', 'train_02'],
            'source_checkpoint_snapshot_sha256': checksum, 'best_epoch': 25 if arm == 'balanced' else 35,
            'best_training_metrics': {'minimum_side_dice': .855, 'minimum_bilateral_patient_dice': .86}})
    old = {'sam_dir': str(root / 'sam'), 'cohort_run_dir': str(root / 'old_cohort')}
    Path(old['cohort_run_dir']).mkdir()
    data.save(Path(old['cohort_run_dir']) / 'config.json', {'data_dir': str(root / 'data'), 'source_root': str(root / 'raw')})
    args = argparse.Namespace(research_source_dir=source, diagnostic_dir=diagnostic, foundation_run_dir=root / 'foundation',
        run_dir=root / 'new', source_root=None, epochs=150, patience_epochs=30, accumulation=4, lr_multiplier=.3,
        device='cpu', resume=False)
    return args, old, checksum


class FullLossCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_source_and_tiny_proofs_are_required_not_tiny_checkpoint_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, checksum = fixture(Path(directory))
            proof = cohort.diagnostic(args.diagnostic_dir, args.research_source_dir)
            self.assertEqual(proof['source_checkpoint_sha256'], checksum)
            path = args.diagnostic_dir / 'tiny_balanced/summary.json'
            child = data.read(path); child['diagnostic_passed'] = False; data.save(path, child)
            with self.assertRaisesRegex(ValueError, 'tiny-fit proof'):
                cohort.diagnostic(args.diagnostic_dir, args.research_source_dir)
            child['diagnostic_passed'] = True; data.save(path, child)
            (args.research_source_dir / 'research_best.pth').write_bytes(b'tiny weights or changed checkpoint')
            with self.assertRaisesRegex(ValueError, 'must match'):
                cohort.diagnostic(args.diagnostic_dir, args.research_source_dir)

    def test_configuration_only_changes_explicit_cohort_recipe_and_keeps_source_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, checksum = fixture(Path(directory))
            args.variant, args.loss_arm, args.cohort_accumulation = 'history', 'balanced', 4
            args.roi_jitter = False
            config = {'parameters': {}, 'research_source': {'checkpoint_sha256': checksum}, 'test_images_read': False}
            cohort.configure(args, config)
            self.assertEqual(args.accumulation, 4)
            self.assertFalse(args.roi_jitter)
            self.assertTrue(args.record_jitter_counts)
            self.assertTrue(args.record_gradient_stats)
            self.assertEqual(config['loss_arm'], 'balanced')
            self.assertEqual(config['cohort_augmentation']['maximum_translation_mm_per_axis'], 1.)
            self.assertFalse(config['test_images_read'])

    def test_real_prepare_hook_restores_identical_config_and_rejects_changed_cohort_accumulation(self):
        import train_fsdp_research as trainer
        from test_fsdp_foundation import fixture as foundation_fixture
        from test_fsdp_verified_cache import fixture as cache_fixture
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, old, summary, _, _ = foundation_fixture(root)
            dataset, manifest, prepared, _ = cache_fixture(root)
            data.save(root / 'data/manifest.json', manifest)
            data.save(Path(old['cohort_run_dir']) / 'config.json', {'source_root': str(dataset), 'data_dir': str(root / 'data')})
            torch.save(prepared, source / 'hybrid_data.pt')
            old.update(cache_sha256=file_hash(source / 'hybrid_data.pt'), physical_policy=prepared['policy'],
                       manifest_sha256=file_hash(root / 'data/manifest.json'))
            summary['physical_policy'] = prepared['policy']
            for name, value in (('config.json', old), ('summary.json', summary), ('source_provenance.json', prepared['source_provenance'])):
                data.save(source / name, value)
            args = argparse.Namespace(foundation_run_dir=source, run_dir=root / 'arm', source_root=dataset,
                research_source_dir=root / 'trained_source', diagnostic_dir=root / 'diagnostic', variant='history',
                verified_cache=True, epochs=150, lr_multiplier=.3, patience_epochs=30, boundary_weight=.05,
                boundary_clip_mm=5., device='cpu', resume=False, loss_arm='balanced', cohort_accumulation=4,
                roi_jitter=False, matched_control_dir=None, record_jitter_counts=True, record_gradient_stats=True)
            args.run_dir.mkdir()
            metadata = {'checkpoint_sha256': 'cohort-best', 'source_plus_additional_epoch': 40}
            with patch.object(trainer.foundation, 'FOUNDATION_SHA256', summary['best_checkpoint_sha256']), \
                 patch.object(trainer, 'inspect_research_source', return_value=metadata), \
                 patch.object(cohort, 'diagnostic', return_value={'source_checkpoint_sha256': 'cohort-best'}):
                _, config = trainer.prepare(args, cohort.configure)
                self.assertEqual(config['parameters']['accumulation'], 4)
                args.resume = True
                _, resumed = trainer.prepare(args, cohort.configure)
                self.assertEqual(config, resumed)
                args.cohort_accumulation = 8
                with self.assertRaisesRegex(ValueError, 'changed'):
                    trainer.prepare(args, cohort.configure)

    def test_inherited_weights_and_eval_predictions_are_identical_to_history(self):
        torch.manual_seed(2026)
        original = SamRefine32(FakeSAM()).eval()
        torch.nn.init.normal_(original.residual.weight, std=.01)
        expected = ResearchModel.from_foundation(original, 'history').eval()
        model = cohort.adapt_model(original, argparse.Namespace(loss_arm='balanced'), {}).eval()
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        image, context, target, _ = tensors()
        call = (image, context, torch.ones(3), torch.full((3,), .25), target, torch.ones_like(target).bool(), (1, 5), 2026)
        with torch.inference_mode(), patch.object(cohort, 'translate_pair', side_effect=AssertionError('evaluation augmented')):
            _, wanted = expected.interaction_function(expected, *call)
            _, actual = model.interaction_function(model, *call)
        for clicks in (1, 5):
            torch.testing.assert_close(actual[clicks]['refined'], wanted[clicks]['refined'], rtol=0, atol=0)

    def test_opt_in_model_uses_existing_exact_resume_and_native_export_contract(self):
        import train_fsdp_research as trainer
        from test_train_fsdp_research import ResearchTrainerTests
        execute = trainer.execute
        def hooked(args):
            args.loss_arm = 'balanced'
            args.record_jitter_counts = True
            args.record_gradient_stats = True
            return execute(args, adapt_model=cohort.adapt_model,
                           finish_summary=lambda summary, config: summary.update(loss_arm='balanced'))
        contract = ResearchTrainerTests('test_interrupted_training_resumes_exactly_and_exports_native_masks_with_no_test_reads')
        with patch.object(trainer, 'execute', side_effect=hooked):
            contract.test_interrupted_training_resumes_exactly_and_exports_native_masks_with_no_test_reads()

    def test_training_injects_loss_per_roi_and_records_shared_jitter(self):
        image, context, ratio, spacing, target, valid, context_target, context_valid = paired()
        model = argparse.Namespace(loss_arm='balanced', jitter_counts={})
        with patch.object(ResearchModel, 'interaction_function', return_value=('loss', {})) as delegate:
            cohort.LossCohortModel.interaction_function(model, image, context, ratio, spacing, target, valid, (1, 5),
                2026, object(), context_target, context_valid, .25)
        self.assertIsInstance(delegate.call_args.args[9], cohort.CandidateLoss)
        self.assertTrue(delegate.call_args.args[9].balanced)
        self.assertEqual(sum(model.jitter_counts.values()), 1)

    def test_parent_sequencing_native_control_comparison_resume_and_failure(self):
        for fail in (False, True):
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                root = Path(directory); args, old, _ = fixture(root)
                seen = []
                def child(command, record):
                    arm = command[command.index('--loss-arm') + 1]
                    self.assertEqual(command[command.index('--cohort-accumulation') + 1], '4')
                    seen.append(arm)
                    if fail:
                        return -15
                    output = Path(command[command.index('--run-dir') + 1]); output.mkdir()
                    (output / 'research_best.pth').write_bytes(arm.encode())
                    score = .5 if arm == 'control' else .52
                    data.save(output / 'summary.json', {'complete': True, 'test_images_read': False, 'loss_arm': arm,
                        'train_patients': 54, 'validation_patients': 10, 'best_checkpoint_sha256': file_hash(output / 'research_best.pth'),
                        'completed_epochs': 35, 'best_epoch': 5,
                        'best_validation_metrics': {'5': {'mean_bilateral_patient_dice': score}},
                        'best_frozen_training_metrics': {}, 'validation_change_vs_research_source': score - .4938})
                    with tarfile.open(output / 'results.tar.gz', 'w:gz'):
                        pass
                    return 0
                with patch.object(parent, 'inspect_foundation', return_value=(old, {})), \
                     patch.object(parent.shutil, 'disk_usage', return_value=argparse.Namespace(free=100 * 2**30)), \
                     patch.object(parent, 'run_child', side_effect=child):
                    if fail:
                        with self.assertRaisesRegex(ValueError, 'returncode=-15'):
                            parent.execute(args)
                        self.assertEqual(seen, ['control'])
                    else:
                        parent.execute(args)
                        self.assertEqual(seen, ['control', 'balanced'])
                        result = data.read(args.run_dir / 'summary.json')
                        self.assertEqual(result['selected_arm'], 'balanced')
                        self.assertAlmostEqual(result['entries'][1]['validation_delta_vs_matched_control'], .02)
                        with tarfile.open(args.run_dir / 'results.tar.gz') as archive:
                            self.assertFalse(any(n.endswith('.pth') for n in archive.getnames()))
                        (args.run_dir / 'summary.json').unlink()
                        args.resume = True; seen.clear()
                        parent.execute(args)
                        self.assertEqual(seen, [])

    def test_cohort_launcher_defaults_and_failed_gpu_check_do_not_start_training(self):
        for success in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); scripts = root / 'scripts'; scripts.mkdir()
                shutil.copy(Path(parent.__file__).with_name('launch_fsdp_loss_cohort_nohup.sh'), scripts)
                (scripts / 'fsdp_research_gpu.py').write_text('print("GPU-only7")\n' if success else 'raise SystemExit(3)\n')
                (scripts / 'run_fsdp_loss_cohort.py').write_text('''import os, pathlib, sys
pathlib.Path(os.environ['TASK_CAPTURE']).write_text(os.environ['CUDA_VISIBLE_DEVICES'] + '\\n' + '\\n'.join(sys.argv[1:]))
''')
                capture = root / 'capture'
                env = dict(os.environ, TASK_CAPTURE=str(capture), CONDA_DEFAULT_ENV='fsdp-med3d', ALLOW_GPU_SHARING='1')
                for key in ('FSDP_COHORT_RUN_NAME', 'RESUME', 'COHORT_EPOCHS', 'LR_MULTIPLIER', 'PATIENCE_EPOCHS', 'ACCUMULATION_STEPS'):
                    env.pop(key, None)
                result = subprocess.run(['bash', str(scripts / 'launch_fsdp_loss_cohort_nohup.sh')], env=env,
                                        capture_output=True, text=True, timeout=15)
                if success:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    settings = capture.read_text()
                    self.assertTrue(settings.startswith('GPU-only7\n'))
                    for value in ('--epochs\n150', '--lr-multiplier\n0.3', '--patience-epochs\n30', '--accumulation\n4'):
                        self.assertIn(value, settings)
                else:
                    self.assertEqual(result.returncode, 3)
                    self.assertFalse(capture.exists())


if __name__ == '__main__':
    unittest.main()
