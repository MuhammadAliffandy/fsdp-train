import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fsdp_research_gpu as gpu
import run_fsdp_research as workflow
from evaluate_private_validation import file_hash
import physical_fsdp as data


def arguments(root):
    return argparse.Namespace(foundation_run_dir=root / 'base', run_dir=root / 'study', source_root=root / 'actual data',
        verified_cache=True, epochs=20, lr_multiplier=.2, patience_epochs=50, boundary_weight=.05,
        boundary_clip_mm=5., device='cpu', gpu_record=None, resume=False)


def completed_child(command, record):
    folder = Path(command[command.index('--run-dir') + 1]); folder.mkdir(parents=True)
    variant = command[command.index('--variant') + 1]
    (folder / 'research_best.pth').write_bytes(variant.encode())
    score = .46 if variant == 'control' else .47
    metrics = {'5': {'mean_bilateral_patient_dice': score}}
    data.save(folder / 'summary.json', {'complete': True, 'variant': variant, 'test_images_read': False,
        'best_checkpoint_sha256': file_hash(folder / 'research_best.pth'), 'best_epoch': 5, 'completed_epochs': 20,
        'best_validation_metrics': metrics, 'best_frozen_training_metrics': metrics,
        'validation_control_fixed_0_5': metrics['5'], 'validation_change_vs_foundation': score - .456,
        'validation_change_vs_candidate_initial': .01})
    with tarfile.open(folder / 'results.tar.gz', 'w:gz'):
        pass
    return 0


class ResearchWorkflowTests(unittest.TestCase):
    def test_four_independent_arms_forward_cache_and_compare_matched_control_without_touching_resolution(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder); args = arguments(root); args.run_dir.mkdir()
            old = root / 'resolution'; old.mkdir(); (old / 'progress.json').write_text('active_resolution_sentinel')
            calls = []
            def child(command, record):
                calls.append(command)
                self.assertIn('--verified-cache', command)
                self.assertEqual(command[command.index('--source-root') + 1], str(args.source_root))
                self.assertEqual(command[command.index('--epochs') + 1], '20')
                return completed_child(command, record)
            with patch.object(workflow, 'inspect_foundation'), patch.object(workflow, 'run_child', child):
                workflow.execute(args)
            self.assertEqual(len(calls), 4)
            summary = data.read(args.run_dir / 'summary.json')
            self.assertFalse(summary['test_images_read'])
            self.assertAlmostEqual(summary['entries'][1]['delta_vs_matched_control'], .01)
            self.assertEqual((old / 'progress.json').read_text(), 'active_resolution_sentinel')
            with tarfile.open(args.run_dir / 'results.tar.gz') as archive:
                self.assertEqual(sum(n.endswith('/results.tar.gz') for n in archive.getnames()), 4)
                self.assertFalse(any(n.endswith('.pth') or n.endswith('.pt') for n in archive.getnames()))

    def test_failure_stops_subsequent_arms_and_resume_checks_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder); args = arguments(root); args.run_dir.mkdir()
            calls = []
            def child(command, record):
                variant = command[command.index('--variant') + 1]
                calls.append(variant)
                return completed_child(command, record) if variant == 'control' else 7
            with patch.object(workflow, 'inspect_foundation'), patch.object(workflow, 'run_child', child):
                with self.assertRaisesRegex(ValueError, 'no subsequent arm'):
                    workflow.execute(args)
            self.assertEqual(calls, ['control', 'history'])
            args.resume = True
            (args.run_dir / 'control/research_best.pth').write_bytes(b'changed')
            with patch.object(workflow, 'inspect_foundation'), patch.object(workflow, 'run_child') as child:
                with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                    workflow.execute(args)
            child.assert_not_called()

    def test_parent_stop_signal_is_forwarded_only_to_its_active_child(self):
        child = MagicMock(pid=12345); child.poll.return_value = None
        handlers = {}
        def install(sig, handler):
            handlers[sig] = handler
            return signal.SIG_DFL
        def wait(*unused, **kwargs):
            if 'timeout' not in kwargs:
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            return -signal.SIGTERM
        child.wait.side_effect = wait
        with tempfile.TemporaryDirectory() as folder, patch.object(workflow.subprocess, 'Popen', return_value=child), \
             patch.object(workflow.signal, 'signal', side_effect=install):
            with self.assertRaises(SystemExit) as stopped:
                workflow.run_child(['synthetic_research_child'], Path(folder) / 'active.json')
        self.assertEqual(stopped.exception.code, 128 + signal.SIGTERM)
        child.send_signal.assert_called_once_with(signal.SIGTERM)


class GPUIsolationTests(unittest.TestCase):
    def test_guard_resolves_physical_gpu_uuid_and_detects_busy_device(self):
        calls = []
        def query(command, **kwargs):
            calls.append(command)
            output = '1, GPU-research-test, 80000, 0\n' if '--query-gpu=index,uuid,memory.total,memory.used' in command else ''
            return argparse.Namespace(stdout=output)
        with patch.object(gpu.subprocess, 'run', query):
            report = gpu.inspect_gpu('1')
        self.assertTrue(report['ready'])
        self.assertEqual(report['uuid'], 'GPU-research-test')
        self.assertEqual(calls[1][2], 'GPU-research-test')
        responses = [argparse.Namespace(stdout='1, GPU-research-test, 80000, 40000\n'),
                     argparse.Namespace(stdout='500, GPU-research-test\n')]
        with patch.object(gpu.subprocess, 'run', side_effect=responses):
            report = gpu.inspect_gpu('1')
        self.assertFalse(report['ready'])
        self.assertEqual(report['active_compute_pids'], [500])
        self.assertFalse(report['jobs_stopped'])
        with self.assertRaisesRegex(ValueError, 'one physical'):
            gpu.inspect_gpu('0,1')

    def test_launcher_pins_gpu_uuid_and_busy_gpu_never_starts_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'project with spaces'
            scripts = root / 'scripts'; scripts.mkdir(parents=True)
            original = Path(__file__).resolve().parents[1] / 'scripts'
            for name in ('launch_fsdp_research_nohup.sh', 'fsdp_research_gpu.py'):
                shutil.copyfile(original / name, scripts / name)
            base = root / 'base'; base.mkdir()
            for name in ('config.json', 'summary.json', 'hybrid_best.pth', 'hybrid_data.pt'):
                (base / name).write_text('synthetic')
            binary = root / 'bin'; binary.mkdir()
            python = binary / 'python'
            python.write_text('''#!/bin/sh
case "$*" in
  *fsdp_research_gpu.py*) exec "$TASK_REAL_PYTHON" "$@" ;;
  *) printf '%s\\n' "$CUDA_VISIBLE_DEVICES" "$@" > "$TASK_CAPTURE"; echo simulated-child-exit; exit 3 ;;
esac
''')
            python.chmod(0o755)
            nvidia = binary / 'nvidia-smi'
            nvidia.write_text('''#!/bin/sh
case "$*" in
  *--query-gpu=*) printf '1, GPU-research-test, 80000, 0\\n' ;;
  *) if [ "$TASK_GPU_BUSY" = 1 ]; then printf '777, GPU-research-test\\n'; fi ;;
esac
''')
            nvidia.chmod(0o755)
            old_log = root / 'logs/fsdp_resolution/latest.log'; old_log.parent.mkdir(parents=True)
            old_log.write_text('active GPU0 resolution log')
            capture = root / 'capture'
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'], CONDA_DEFAULT_ENV='fsdp-med3d',
                FSDP_FOUNDATION_DIR=str(base), FSDP_SOURCE_ROOT=str(root / 'actual data'), FSDP_VERIFIED_CACHE='1',
                GPU_ID='1', TASK_REAL_PYTHON=sys.executable, TASK_CAPTURE=str(capture), TASK_GPU_BUSY='1', RESUME='0',
                FSDP_RESEARCH_RUN_NAME='fsdp_research_unit')
            command = ['bash', str(scripts / 'launch_fsdp_research_nohup.sh')]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(capture.exists())
            env['TASK_GPU_BUSY'] = '0'
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            lines = capture.read_text().splitlines()
            self.assertEqual(lines[0], 'GPU-research-test')
            self.assertIn('--verified-cache', lines)
            self.assertEqual(lines[lines.index('--source-root') + 1], str(root / 'actual data'))
            self.assertEqual(lines[lines.index('--epochs') + 1], '20')
            self.assertEqual(old_log.read_text(), 'active GPU0 resolution log')


if __name__ == '__main__':
    unittest.main()
