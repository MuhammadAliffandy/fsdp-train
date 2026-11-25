import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fsdp_signal_provenance as provenance


class SignalProvenanceTests(unittest.TestCase):
    def test_records_sender_then_signals_only_itself_to_preserve_termination(self):
        info = argparse.Namespace(si_signo=15, si_pid=1234, si_uid=1001, si_code=0)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            output = Path(folder) / 'tokens_signal.json'
            with patch.object(provenance.signal, 'sigwaitinfo', return_value=info, create=True), \
                 patch.object(provenance.signal, 'pthread_sigmask') as mask, \
                 patch.object(provenance, 'sender_identity', return_value={'comm': 'supervisor'}), \
                 patch.object(provenance.os, 'kill') as terminate, patch.object(provenance.os, '_exit') as fallback:
                provenance.receive_and_terminate(output)
            record = json.loads(output.read_text())
            self.assertEqual(record['sender_pid'], 1234)
            self.assertEqual(record['sender_uid'], 1001)
            self.assertEqual(record['sender_identity']['comm'], 'supervisor')
            terminate.assert_called_once_with(os.getpid(), signal.SIGTERM)
            mask.assert_called_once_with(signal.SIG_UNBLOCK, {signal.SIGTERM})
            fallback.assert_called_once_with(143)

    def test_logging_failure_does_not_prevent_sigterm_exit(self):
        info = argparse.Namespace(si_signo=15, si_pid=1234, si_uid=1001, si_code=0)
        with patch.object(provenance.signal, 'sigwaitinfo', return_value=info, create=True), \
             patch.object(provenance.signal, 'pthread_sigmask'), patch.object(provenance, 'write', side_effect=OSError('disk full')), \
             patch.object(provenance.os, 'kill') as terminate, patch.object(provenance.os, '_exit') as fallback, \
             contextlib.redirect_stderr(io.StringIO()):
            provenance.receive_and_terminate(Path('/synthetic/signal.json'))
        terminate.assert_called_once_with(os.getpid(), signal.SIGTERM)
        fallback.assert_called_once_with(143)

    def test_installs_before_native_threads_and_records_setup(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'tokens_signal.json'
            with patch.object(provenance.signal, 'sigwaitinfo', create=True), \
                 patch.object(provenance.signal, 'getsignal', return_value=signal.SIG_DFL), \
                 patch.object(provenance.signal, 'pthread_sigmask', return_value=set()) as mask, \
                 patch.object(provenance.threading, 'Thread') as thread:
                self.assertTrue(provenance.install(output))
                mask.assert_called_once_with(signal.SIG_BLOCK, {signal.SIGTERM})
                thread.assert_called_once_with(target=provenance.receive_and_terminate, args=(output,),
                                               name='sigterm-provenance', daemon=True)
                thread.return_value.start.assert_called_once()
            self.assertTrue(json.loads(output.with_name('tokens_signal_setup.json').read_text())['enabled'])

    def test_existing_signal_disposition_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'tokens_signal.json'
            with patch.object(provenance.signal, 'sigwaitinfo', create=True), \
                 patch.object(provenance.signal, 'getsignal', return_value=signal.SIG_IGN), \
                 patch.object(provenance.signal, 'pthread_sigmask') as mask:
                self.assertFalse(provenance.install(output))
                mask.assert_not_called()

    def test_inherited_blocked_mask_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'tokens_signal.json'
            with patch.object(provenance.signal, 'sigwaitinfo', create=True), \
                 patch.object(provenance.signal, 'getsignal', return_value=signal.SIG_DFL), \
                 patch.object(provenance.signal, 'pthread_sigmask', return_value={signal.SIGTERM}) as mask, \
                 patch.object(provenance.threading, 'Thread') as thread:
                self.assertFalse(provenance.install(output))
                thread.assert_not_called()
                mask.assert_called_once_with(signal.SIG_BLOCK, {signal.SIGTERM})

    def test_thread_start_failure_restores_the_original_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(provenance.signal, 'sigwaitinfo', create=True), \
                 patch.object(provenance.signal, 'getsignal', return_value=signal.SIG_DFL), \
                 patch.object(provenance.signal, 'pthread_sigmask', return_value=set()) as mask, \
                 patch.object(provenance.threading, 'Thread') as thread:
                thread.return_value.start.side_effect = RuntimeError('cannot start observer')
                with self.assertRaisesRegex(RuntimeError, 'cannot start'):
                    provenance.install(Path(folder) / 'signal.json')
                self.assertEqual(mask.call_args_list[-1].args, (signal.SIG_SETMASK, set()))

    @unittest.skipUnless(sys.platform.startswith('linux') and hasattr(signal, 'sigwaitinfo'), 'Requires Linux sigwaitinfo')
    def test_linux_real_sigterm_retains_sender_and_negative_return_code(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'signal.json'
            code = ('import sys,time; from pathlib import Path; '
                    'sys.path.insert(0,sys.argv[1]); from fsdp_signal_provenance import install; '
                    'assert install(Path(sys.argv[2])); print("ready",flush=True); time.sleep(60)')
            child = subprocess.Popen([sys.executable, '-u', '-c', code, str(Path(provenance.__file__).parent), str(output)],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), 'ready')
                os.kill(child.pid, signal.SIGTERM)
                self.assertEqual(child.wait(timeout=10), -signal.SIGTERM)
                record = json.loads(output.read_text())
                self.assertEqual(record['sender_pid'], os.getpid())
                self.assertEqual(record['sender_uid'], os.getuid())
            finally:
                if child.poll() is None:
                    child.kill(); child.wait(timeout=10)
                child.stdout.close(); child.stderr.close()


if __name__ == '__main__':
    unittest.main()
