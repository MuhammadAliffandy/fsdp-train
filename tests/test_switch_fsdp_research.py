import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stop_fsdp_resolution as stopper


def fixture(root):
    (root / 'scripts').mkdir()
    for name in ('run_fsdp_resolution.py', 'train_fsdp_foundation.py'):
        (root / 'scripts' / name).touch()
    channel = root / 'logs/fsdp_resolution'; channel.mkdir(parents=True)
    tracked = channel / 'fsdp_resolution_example.pid'; tracked.write_text('12345')
    (channel / 'latest.pid').symlink_to(tracked)
    run = root / 'runs/fsdp_resolution_example'; run.mkdir(parents=True)
    (run / 'workflow_config.json').write_text(json.dumps({'run_dir': str(run)}))
    proc = root / 'proc/12345'; proc.mkdir(parents=True)
    (proc / 'stat').write_text('12345 (python) R ' + ' '.join(['0'] * 18 + ['100']))
    command = [sys.executable, '-u', str(root / 'scripts/run_fsdp_resolution.py'), '--run-dir', str(run)]
    (proc / 'cmdline').write_bytes('\0'.join(command).encode())
    (proc / 'environ').write_bytes(b'CUDA_VISIBLE_DEVICES=GPU-original\0UNRELATED_SECRET=not-recorded\0')
    return command, run


class SwitchTests(unittest.TestCase):
    def test_inspection_requires_project_owner_and_matching_run_and_reads_only_gpu_env(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); command, run = fixture(root)
            target = stopper.inspect_workflow(root, root / 'proc')
            self.assertEqual(target['gpu_selector'], 'GPU-original')
            self.assertNotIn('not-recorded', json.dumps(target))
            with patch.object(stopper.os, 'getuid', return_value=os.getuid() + 1):
                with self.assertRaisesRegex(ValueError, 'another user'):
                    stopper.inspect_workflow(root, root / 'proc')
            command[-1] = str(root / 'unrelated_run')
            (root / 'proc/12345/cmdline').write_bytes('\0'.join(command).encode())
            with self.assertRaisesRegex(ValueError, 'does not match'):
                stopper.inspect_workflow(root, root / 'proc')

    def test_reused_pid_is_not_signalled(self):
        target = {'pid': 12345, 'command': ['python', 'ours'], 'start_time': '100'}
        with patch.object(stopper, 'process_info', return_value={'command': ['python', 'other'], 'start_time': '200'}), \
             patch.object(stopper.os, 'kill') as send:
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                stopper.stop_workflow(target, Path('/project'))
        send.assert_not_called()

    def test_only_parent_is_signalled_and_child_must_exit_before_success(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder).resolve(); fixture(root)
            target = stopper.inspect_workflow(root, root / 'proc')
            current = {key: target[key] for key in ('start_time', 'command')}
            with patch.object(stopper, 'process_info', side_effect=[current, None, None]), \
                 patch.object(stopper, 'active_children', side_effect=[[{'pid': 222}], []]), \
                 patch.object(stopper.time, 'sleep'), patch.object(stopper.os, 'kill') as send:
                stopper.stop_workflow(target, root, proc_root=root / 'proc')
            send.assert_called_once_with(12345, signal.SIGTERM)
            with patch.object(stopper, 'process_info', side_effect=[current, None]), \
                 patch.object(stopper, 'active_children', return_value=[{'pid': 222}]), \
                 patch.object(stopper.os, 'kill') as send:
                with self.assertRaisesRegex(TimeoutError, 'research launch blocked'):
                    stopper.stop_workflow(target, root, timeout=0, proc_root=root / 'proc')
            send.assert_called_once_with(12345, signal.SIGTERM)

    def test_gpu_mapping_and_child_scan_are_scoped_to_our_run(self):
        with patch.object(stopper.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0,
             stdout='0, GPU-original\n2, GPU-other\n')):
            self.assertEqual(stopper.resolve_gpu('GPU-original'), {'index': 0, 'uuid': 'GPU-original'})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); _, run = fixture(root)
            child = root / 'proc/222'; child.mkdir()
            (child / 'stat').write_text('222 (python) R ' + ' '.join(['0'] * 18 + ['101']))
            command = [sys.executable, '-u', str(root / 'scripts/train_fsdp_foundation.py'), '--run-dir', str(run / 'train/baseline')]
            (child / 'cmdline').write_bytes('\0'.join(command).encode())
            self.assertEqual([row['pid'] for row in stopper.active_children(root, run, root / 'proc')], [222])
            command[-1] = str(root / 'runs/unrelated')
            (child / 'cmdline').write_bytes('\0'.join(command).encode())
            self.assertEqual(stopper.active_children(root, run, root / 'proc'), [])

    def test_wrapper_never_launches_after_stop_failure_and_reuses_released_gpu_after_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); scripts = root / 'scripts'; scripts.mkdir()
            shutil.copy(Path(stopper.__file__).with_name('switch_fsdp_resolution_to_research.sh'), scripts)
            base = root / 'base'; base.mkdir()
            for name in ('config.json', 'summary.json', 'hybrid_best.pth', 'hybrid_data.pt'):
                (base / name).touch()
            binary = root / 'bin'; binary.mkdir()
            python = binary / 'python'
            python.write_text('''#!/bin/sh
case "$*" in
  *stop_fsdp_resolution.py*)
    printf 'stop\\n' >> "$TASK_CALLS"
    if [ "$TASK_STOP_STATUS" != 0 ]; then exit "$TASK_STOP_STATUS"; fi
    while [ "$1" != '--output' ]; do shift; done
    shift
    mkdir -p "$(dirname "$1")"
    printf '{"complete":true,"gpu":{"index":2}}' > "$1" ;;
  *) exec "$TASK_REAL_PYTHON" "$@" ;;
esac
''')
            python.chmod(0o755)
            (scripts / 'launch_fsdp_research_nohup.sh').write_text('''#!/bin/sh
printf 'launch GPU=%s exclude=%s resume=%s sharing=%s\\n' "$GPU_ID" "$EXCLUDE_GPU_IDS" "$RESUME" "$ALLOW_GPU_SHARING" >> "$TASK_CALLS"
''')
            calls = root / 'calls'
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'], CONDA_DEFAULT_ENV='fsdp-med3d',
                       FSDP_FOUNDATION_DIR=str(base), TASK_CALLS=str(calls), TASK_STOP_STATUS='1', TASK_REAL_PYTHON=sys.executable)
            command = ['bash', str(scripts / 'switch_fsdp_resolution_to_research.sh')]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(calls.read_text().splitlines(), ['stop'])
            calls.unlink(); env['TASK_STOP_STATUS'] = '0'; env['ALLOW_GPU_SHARING'] = '1'
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(calls.read_text().splitlines(), ['stop', 'launch GPU=2 exclude= resume=0 sharing=1'])


if __name__ == '__main__':
    unittest.main()
