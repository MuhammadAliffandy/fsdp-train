import contextlib
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stop_private_800 as stopper


class StopTests(unittest.TestCase):
    def test_run_pid_command_and_epoch_budget_must_match(self):
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder).resolve()
            (project / 'scripts').mkdir()
            script = project / 'scripts/train_private_long.py'
            script.touch()
            (project / 'logs/training').mkdir(parents=True)
            proc = project / 'proc'
            def fixture(name, pid, epochs):
                run = project / 'runs' / name
                run.mkdir(parents=True)
                (run / 'config.json').write_text(json.dumps({'epochs': epochs}))
                (project / 'logs/training' / f'{name}.pid').write_text(str(pid))
                process = proc / str(pid)
                process.mkdir(parents=True)
                (process / 'stat').write_text(f'{pid} (python) R ' + ' '.join(['0'] * 18 + ['1234']))
                (process / 'cmdline').write_bytes('\0'.join(['/usr/bin/python', str(script), '--run-dir', str(run)]).encode())
                return process
            process = fixture('fsdp_private_v2_old', 54321, 800)
            target = stopper.find_target(project, proc_root=proc)
            self.assertEqual(target['pid'], 54321)
            (process / 'cmdline').write_bytes(b'/usr/bin/python\0/unrelated/train.py')
            with self.assertRaises(ValueError):
                stopper.find_target(project, proc_root=proc)
            (process / 'cmdline').write_bytes('\0'.join(['/usr/bin/python', str(script), '--run-dir', str(project / 'runs/fsdp_private_v2_old')]).encode())
            fixture('fsdp_private_v2_second', 54322, 800)
            with self.assertRaises(ValueError):
                stopper.find_target(project, proc_root=proc)
            self.assertEqual(stopper.find_target(project, 'fsdp_private_v2_old', proc)['pid'], 54321)
            (project / 'runs/fsdp_private_v2_old/config.json').write_text('{"epochs": 200}')
            with self.assertRaises(ValueError):
                stopper.find_target(project, 'fsdp_private_v2_old', proc)

    def test_verified_target_receives_sigterm(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        try:
            with patch.object(stopper, 'process_info', side_effect=[{'start_time': '1'}, None]), contextlib.redirect_stdout(io.StringIO()):
                stopper.stop_target({'pid': child.pid, 'start_time': '1', 'run_name': 'test_child'})
            self.assertEqual(child.wait(timeout=5), -signal.SIGTERM)
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)

    def test_reused_pid_is_not_signalled(self):
        with patch.object(stopper, 'process_info', return_value={'start_time': '2'}), patch.object(stopper.os, 'kill') as send:
            with self.assertRaises(ValueError):
                stopper.stop_target({'pid': 54321, 'start_time': '1'})
            send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
