import argparse
import contextlib
import io
import json
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
import resume_fsdp_resolution as remaining
import run_fsdp_research as research_parent
import train_fsdp_foundation as foundation


def fixture(root):
    root = root.resolve(); scripts = root / 'scripts'; scripts.mkdir()
    for name in remaining.TRACKED:
        (scripts / name).write_text('# synthetic original ' + name)
    original = root / 'runs/fsdp_resolution_original'; original.mkdir(parents=True)
    config = {'run_dir': str(original), 'train': True, 'foundation_run_dir': str(root / 'foundation'),
              'epochs': 100, 'lr_multiplier': .2, 'patience_epochs': 50, 'device': 'cuda:0',
              'source_root': str(root / 'dataset'), 'verified_cache': True}
    remaining.save(original / 'workflow_config.json', config)
    remaining.save(original / 'workflow_code_sha256.json', {name: remaining.digest(scripts / name) for name in remaining.TRACKED})
    for variant in remaining.VARIANTS:
        remaining.save(original / 'audit' / variant / 'summary.json', {'complete': True, 'test_images_read': False,
            'optimizer_updates': 0, 'foundation_reproduced_before_adaptation': True, 'gradient_probe': {'passed': True},
            'resolution': {'variant': variant}})
    finish(original / 'train/baseline', 'baseline', config, .4651)
    detail = original / 'train/detail'; detail.mkdir(parents=True)
    detail_config = child_config(detail, 'detail', config)
    remaining.save(detail / 'config.json', detail_config)
    torch.save({'epoch': 3, 'config': detail_config}, detail / 'hybrid_latest.pth')
    return root, original, config


def child_config(folder, variant, original):
    return {**original, 'run_dir': str(folder), 'resolution': {'variant': variant}, 'audit_only': False,
            'foundation': {'checkpoint_sha256': 'same-original-foundation'}}


def finish(folder, variant, original, score=.48):
    folder.mkdir(parents=True, exist_ok=True)
    config = child_config(folder, variant, original)
    remaining.save(folder / 'config.json', config)
    (folder / 'hybrid_best.pth').write_bytes(variant.encode())
    remaining.save(folder / 'summary.json', {'complete': True, 'test_images_read': False,
        'resolution': config['resolution'], 'foundation': config['foundation'], 'completed_epochs': 70,
        'best_epoch': 20, 'best_checkpoint_sha256': remaining.digest(folder / 'hybrid_best.pth'),
        'best_validation_metrics': {'5': {'mean_bilateral_patient_dice': score}}})
    with tarfile.open(folder / 'results.tar.gz', 'w:gz'):
        pass


class RemainingResolutionTests(unittest.TestCase):
    def test_original_signatures_audits_and_resume_settings_are_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            root, original, config = fixture(Path(folder))
            self.assertEqual(remaining.verify_source(root, original), config)
            mode, state = remaining.action(original / 'train/detail', 'detail', config)
            self.assertEqual((mode, state['saved_epoch']), ('resume', 3))
            changed = {**config, 'epochs': 200}
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                remaining.action(original / 'train/detail', 'detail', changed)
            (root / 'scripts/sam_refine32.py').write_text('changed implementation')
            with self.assertRaisesRegex(ValueError, 'implementation changed'):
                remaining.verify_source(root, original)

    def test_skips_baseline_starts_tokens_resumes_detail_and_completes_combined(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root, original, config = fixture(Path(folder))
            baseline = remaining.digest(original / 'train/baseline/hybrid_best.pth')
            args = argparse.Namespace(project_dir=root, original_run_dir=original,
                run_dir=root / 'runs/fsdp_resolution_remaining_test', resume=False, gpu_record=None)
            commands = []
            def child(command, record):
                commands.append(command)
                variant = command[command.index('--resolution-variant') + 1]
                self.assertEqual(command[command.index('--epochs') + 1], '100')
                self.assertEqual(command[command.index('--lr-multiplier') + 1], '0.2')
                self.assertEqual('--resume' in command, variant == 'detail')
                finish(original / 'train' / variant, variant, config)
                return 0
            with patch.object(foundation, 'inspect_foundation'), patch.object(remaining, 'active_writers', return_value=[]), \
                 patch.object(research_parent, 'run_child', side_effect=child):
                remaining.execute(args)
            self.assertEqual([command[command.index('--resolution-variant') + 1] for command in commands], ['tokens', 'detail', 'combined'])
            self.assertEqual(remaining.digest(original / 'train/baseline/hybrid_best.pth'), baseline)
            report = remaining.read(args.run_dir / 'summary.json')
            self.assertTrue(report['complete'])
            self.assertFalse(report['test_images_read'])
            self.assertEqual([entry['mode'] for entry in report['entries']], ['skip', 'new', 'resume', 'new'])

    def test_active_writer_or_failed_child_prevents_later_training(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root, original, _ = fixture(Path(folder))
            args = argparse.Namespace(project_dir=root, original_run_dir=original,
                run_dir=root / 'runs/fsdp_resolution_remaining_test', resume=False, gpu_record=None)
            with patch.object(remaining, 'active_writers', return_value=[123]), \
                 patch.object(research_parent, 'run_child') as child:
                with self.assertRaisesRegex(ValueError, 'active writers'):
                    remaining.execute(args)
                child.assert_not_called()
            with patch.object(foundation, 'inspect_foundation'), patch.object(remaining, 'active_writers', return_value=[]), \
                 patch.object(research_parent, 'run_child', return_value=-15) as child:
                with self.assertRaisesRegex(ValueError, 'later variants were not launched'):
                    remaining.execute(args)
                child.assert_called_once()

    def test_gpu_protection_covers_research_parent_even_between_compute_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); (root / 'scripts').mkdir()
            parent = root / 'proc/111'; parent.mkdir(parents=True)
            (parent / 'stat').write_text('111 (python) R ' + ' '.join(['0'] * 18 + ['100']))
            (parent / 'cmdline').write_bytes(f'python\0-u\0{root}/scripts/run_fsdp_research.py\0'.encode())
            (parent / 'environ').write_bytes(b'CUDA_VISIBLE_DEVICES=GPU-history\0')
            tracked = root / 'logs/fsdp_research/latest.pid'; tracked.parent.mkdir(parents=True); tracked.write_text('111')
            responses = [subprocess.CompletedProcess([], 0, stdout='0, GPU-resolution\n7, GPU-history\n'),
                         subprocess.CompletedProcess([], 0, stdout='')]
            with patch.object(remaining.subprocess, 'run', side_effect=responses):
                self.assertEqual(remaining.research_gpus(root, root / 'proc'), ['7'])

    def test_child_allocator_limit_preserves_original_trainer_arguments(self):
        original = Path(remaining.__file__).with_name('train_fsdp_foundation.py').resolve()
        arguments = [str(original), '--run-dir', '/original/train/detail', '--resume', '--epochs', '100']
        with patch.dict(os.environ, FSDP_RESOLUTION_GPU_MEMORY_GIB='40'), \
             patch.object(torch.cuda, 'get_device_properties', return_value=argparse.Namespace(total_memory=80 * 2**30)), \
             patch.object(torch.cuda, 'set_per_process_memory_fraction') as limit, patch.object(sys, 'argv', ['wrapper']), \
             patch.object(remaining.runpy, 'run_path') as run, contextlib.redirect_stdout(io.StringIO()):
            remaining.child_main(arguments)
            limit.assert_called_once_with(.5, 0)
            self.assertEqual(sys.argv, arguments)
            run.assert_called_once_with(str(original), run_name='__main__')

    def test_second_parent_cannot_acquire_original_workflow_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            root, original, _ = fixture(Path(folder))
            with (original / '.remaining_workflow.lock').open('a') as lock:
                remaining.fcntl.flock(lock, remaining.fcntl.LOCK_EX | remaining.fcntl.LOCK_NB)
                command = ['script', '--project-dir', str(root), '--original-run-dir', str(original),
                           '--run-dir', str(root / 'runs/fsdp_resolution_remaining_test')]
                with patch.object(sys, 'argv', command), patch.object(remaining, 'execute') as execute, \
                     contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        remaining.main()
                    self.assertEqual(error.exception.code, 2)
                    execute.assert_not_called()

    def test_launcher_targets_gpu0_without_touching_existing_history_log(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); scripts = root / 'scripts'; scripts.mkdir()
            shutil.copy(Path(remaining.__file__).with_name('launch_fsdp_resolution_remaining_nohup.sh'), scripts)
            binary = root / 'bin'; binary.mkdir(); capture = root / 'capture'
            python = binary / 'python'
            python.write_text('''#!/bin/sh
case "$*" in
  *--gpu-check-only*)
    printf '%s\\n' "$*" > "$TASK_CHECK"
    if [ "$TASK_GPU_READY" != 1 ]; then exit 2; fi
    printf 'GPU-resolution-zero\\n' ;;
  *) printf '%s\\n' "$CUDA_VISIBLE_DEVICES" "$FSDP_RESOLUTION_GPU_MEMORY_GIB" "$@" > "$TASK_CAPTURE"; exit 3 ;;
esac
''')
            python.chmod(0o755)
            history = root / 'logs/fsdp_research/latest.log'; history.parent.mkdir(parents=True); history.write_text('active history GPU7')
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'], CONDA_DEFAULT_ENV='fsdp-med3d',
                GPU_ID='0', EXCLUDE_GPU_IDS='7', ALLOW_GPU_SHARING='1', TASK_CHECK=str(root / 'check'),
                TASK_CAPTURE=str(capture), TASK_GPU_READY='0', RESUME='0')
            command = ['bash', str(scripts / 'launch_fsdp_resolution_remaining_nohup.sh')]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(capture.exists())
            env['TASK_GPU_READY'] = '1'
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('--gpu-id 0', (root / 'check').read_text())
            self.assertEqual(capture.read_text().splitlines()[:2], ['GPU-resolution-zero', '40'])
            self.assertEqual(history.read_text(), 'active history GPU7')


if __name__ == '__main__':
    unittest.main()
