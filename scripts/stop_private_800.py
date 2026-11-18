#!/usr/bin/env python3
"""Stop exactly one verified 800-epoch private trainer on the Linux DGX."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time


def process_info(pid, proc_root=Path('/proc')):
    folder = proc_root / str(pid)
    try:
        stat = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        command = (folder / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        return {'start_time': stat[19], 'command': command}
    except (OSError, IndexError, UnicodeError):
        return None


def find_target(project, run_name=None, proc_root=Path('/proc')):
    project = project.resolve()
    candidates = []
    for pid_file in (project / 'logs/training').glob('fsdp_private_v2_*.pid'):
        name = pid_file.stem
        if run_name and run_name != name:
            continue
        try:
            pid = int(pid_file.read_text().strip())
            if pid <= 1:
                continue
            current = process_info(pid, proc_root)
            run = project / 'runs' / name
            config = json.loads((run / 'config.json').read_text())
            if not current or config.get('epochs') != 800:
                continue
            command = current['command']
            script = project / 'scripts/train_private_long.py'
            if not any(Path(arg).resolve() == script for arg in command if not arg.startswith('-')):
                continue
            index = command.index('--run-dir')
            if Path(command[index + 1]).resolve() != run.resolve():
                continue
            candidates.append({'pid': pid, 'run_name': name, 'run_dir': str(run),
                               'start_time': current['start_time']})
        except (OSError, ValueError, IndexError, TypeError):
            continue
    if len(candidates) != 1:
        raise ValueError(f'Expected one verified active 800-epoch trainer; found {len(candidates)}. '
                         'Check its progress/PID or specify LONG_RUN_NAME. No signal was sent.')
    return candidates[0]


def stop_target(target, timeout=30):
    pid = target['pid']
    current = process_info(pid)
    if not current or current['start_time'] != target['start_time']:
        raise ValueError('Process changed before stop; no signal sent')
    print(f"Sending SIGTERM to {target['run_name']} (PID {pid})", flush=True)
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = process_info(pid)
        if not current or current['start_time'] != target['start_time']:
            print('Verified trainer stopped. Existing checkpoints and logs were retained.', flush=True)
            return
        time.sleep(.5)
    raise TimeoutError('Trainer did not stop within 30 seconds; refusing to launch a second run')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, required=True)
    parser.add_argument('--run-name')
    args = parser.parse_args()
    try:
        target = find_target(args.project_dir, args.run_name)
        stop_target(target)
        output = args.project_dir / 'outputs/private/stops'
        output.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        target.update(signal='SIGTERM', stopped_at_utc=stamp,
                      latest_checkpoint=str(Path(target['run_dir']) / 'sam_model_latest.pth'))
        (output / f"{target['run_name']}_{stamp}.json").write_text(json.dumps(target, indent=2))
    except (ValueError, OSError, TimeoutError) as error:
        parser.exit(1, f'Stop aborted: {error}\n')


if __name__ == '__main__':
    main()
