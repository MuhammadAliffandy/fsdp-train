import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import signal
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stop_multiclass_training as stopper


class StopTests(unittest.TestCase):
    def test_only_matching_project_trainer_config_and_pid_are_selected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            script = root / 'scripts/train_multiclass_turbo.py'
            script.parent.mkdir(); script.touch()
            run = root / 'runs/fsdp_multiclass_example'; run.mkdir(parents=True)
            (run / 'config.json').write_text(json.dumps({'experiment': 'multiclass_turbo_local_detail', 'run_dir': str(run)}))
            logs = root / 'logs/multiclass'; logs.mkdir(parents=True)
            (logs / 'fsdp_multiclass_example.pid').write_text('12345')
            proc = root / 'proc/12345'; proc.mkdir(parents=True)
            (proc / 'stat').write_text('12345 (python) R ' + ' '.join(['0'] * 18 + ['100']))
            cmd = ['/usr/bin/python', '-u', str(script), '--run-dir', str(run)]
            (proc / 'cmdline').write_bytes('\0'.join(cmd).encode())
            self.assertEqual(stopper.find_targets(root, root / 'proc')[0]['pid'], 12345)
            cmd[2] = str(root / 'unrelated.py')
            (proc / 'cmdline').write_bytes('\0'.join(cmd).encode())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(stopper.find_targets(root, root / 'proc'), [])
            cmd[2] = str(script); cmd[-1] = str(root / 'other_run')
            (proc / 'cmdline').write_bytes('\0'.join(cmd).encode())
            with self.assertRaisesRegex(ValueError, 'does not match'):
                stopper.find_targets(root, root / 'proc')

    def test_reused_pid_or_changed_command_is_never_signalled(self):
        target = {'pid': 12345, 'start_time': '1', 'command': ['python', 'verified.py']}
        for current in ({'start_time': '2', 'command': target['command']},
                        {'start_time': '1', 'command': ['unrelated']}):
            with patch.object(stopper, 'process_info', return_value=current), patch.object(stopper.os, 'kill') as send:
                with self.assertRaisesRegex(ValueError, 'identity/command'):
                    stopper.stop_target(target)
                send.assert_not_called()

    def test_verified_identity_receives_sigterm_and_exit_is_checked(self):
        current = {'start_time': '1', 'command': ['python', 'verified.py']}
        target = {'pid': 12345, 'run_name': 'synthetic', **current}
        with patch.object(stopper, 'process_info', side_effect=[current, None]), \
             patch.object(stopper.os, 'kill') as send, contextlib.redirect_stdout(io.StringIO()):
            stopper.stop_target(target)
        send.assert_called_once_with(12345, signal.SIGTERM)


if __name__ == '__main__':
    unittest.main()
