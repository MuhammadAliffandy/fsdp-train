import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class RecoveryTests(unittest.TestCase):
    def test_verification_precedes_stop_and_stop_failure_prevents_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'project with spaces'
            scripts = root / 'scripts'; scripts.mkdir(parents=True)
            source = Path(__file__).resolve().parents[1] / 'scripts'
            for name in ('recover_fsdp_foundation_nohup.sh', 'launch_fsdp_foundation_nohup.sh'):
                shutil.copyfile(source / name, scripts / name)
            base = root / 'base'; base.mkdir()
            for name in ('config.json', 'summary.json', 'hybrid_best.pth', 'hybrid_data.pt'):
                (base / name).write_text('synthetic')
            binary = root / 'bin'; binary.mkdir()
            python = binary / 'python'
            python.write_text('''#!/bin/sh
case "$*" in
  *--check-only*) printf 'check\\n' >> "$TASK_CALLS"; exit "$TASK_CHECK_STATUS" ;;
  *stop_multiclass_training.py*) printf 'stop\\n' >> "$TASK_CALLS"; exit "$TASK_STOP_STATUS" ;;
  *) printf 'train\\n' >> "$TASK_CALLS"; echo simulated-child-exit; exit 3 ;;
esac
''')
            python.chmod(0o755)
            calls = root / 'calls'
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
                CONDA_DEFAULT_ENV='fsdp-med3d', FSDP_FOUNDATION_DIR=str(base), TASK_CALLS=str(calls),
                TASK_CHECK_STATUS='4', TASK_STOP_STATUS='0', FSDP_FOUNDATION_RUN_NAME='fsdp_private_foundation_test', RESUME='0')
            command = ['bash', str(scripts / 'recover_fsdp_foundation_nohup.sh')]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 4)
            self.assertEqual(calls.read_text().splitlines(), ['check'])
            calls.unlink(); env.update(TASK_CHECK_STATUS='0', TASK_STOP_STATUS='5')
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 5)
            self.assertEqual(calls.read_text().splitlines(), ['check', 'stop'])
            calls.unlink(); env['TASK_STOP_STATUS'] = '0'
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(calls.read_text().splitlines(), ['check', 'stop', 'train'])
            self.assertIn('simulated-child-exit', result.stderr)
            self.assertTrue((root / 'logs/fsdp_foundation/latest.log').is_file())
            self.assertEqual((base / 'hybrid_best.pth').read_text(), 'synthetic')


if __name__ == '__main__':
    unittest.main()
