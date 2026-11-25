"""Record a Linux SIGTERM sender, then terminate with the same signal."""
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def sender_identity(pid):
    result = {}
    if pid <= 0:
        return result
    for name in ('comm', 'exe'):
        try:
            path = Path('/proc') / str(pid) / name
            result[name] = str(path.readlink()) if name == 'exe' else path.read_text().strip()[:256]
        except OSError as error:
            result[name + '_read_error'] = type(error).__name__
    return result


def receive_and_terminate(output):
    try:
        info = signal.sigwaitinfo({signal.SIGTERM})
    except BaseException as error:
        try:
            write(output, {'observer_error': type(error).__name__, 'pid': os.getpid(),
                           'action': 'terminating diagnostic worker; no training retry'})
        finally:
            os._exit(1)
    try:
        payload = {'pid': os.getpid(), 'signal_number': int(info.si_signo), 'signal_name': 'SIGTERM',
                   'sender_pid': int(info.si_pid), 'sender_uid': int(info.si_uid), 'signal_code': int(info.si_code),
                   'received_at_unix': time.time(), 'sender_identity': sender_identity(info.si_pid),
                   'action': 'record then terminate with SIGTERM; no signal ignored and no automatic retry',
                   'scope': 'Immediate signal sender only; a forwarding supervisor may not be the original cause.'}
        write(output, payload)
        print(f'SIGTERM received from PID {info.si_pid}, UID {info.si_uid}; terminating. Record: {output}', flush=True)
    except OSError as error:
        print(f'Could not save SIGTERM sender: {error}', file=sys.stderr, flush=True)
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
        os.kill(os.getpid(), signal.SIGTERM)
        os._exit(128 + signal.SIGTERM)


def install(output):
    output = Path(output)
    setup = output.with_name(output.stem + '_setup.json')
    available = all(hasattr(signal, name) for name in ('sigwaitinfo', 'pthread_sigmask'))
    if not available or signal.getsignal(signal.SIGTERM) != signal.SIG_DFL:
        write(setup, {'enabled': False, 'pid': os.getpid(),
                      'reason': 'Linux sigwaitinfo unavailable or an existing SIGTERM disposition must be preserved'})
        return False
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    if signal.SIGTERM in previous:
        write(setup, {'enabled': False, 'pid': os.getpid(), 'reason': 'SIGTERM was already blocked; inherited mask preserved'})
        return False
    try:
        write(setup, {'enabled': True, 'pid': os.getpid(), 'method': 'sigwaitinfo before CUDA/native threads',
                      'action': 'Record immediate sender then terminate normally with SIGTERM'})
        observer = threading.Thread(target=receive_and_terminate, args=(output,), name='sigterm-provenance', daemon=True)
        observer.start()
    except BaseException:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        raise
    return True
