#!/usr/bin/env python3
"""Stop verified multiclass jobs from this project, retaining their saved artifacts."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

from stop_private_800 import process_info


def find_targets(project, proc_root=Path('/proc')):
    project = project.resolve()
    script = project / 'scripts/train_multiclass_turbo.py'
    result = {}
    for channel in ('multiclass', 'multiclass_tokens16'):
        for file in (project / 'logs' / channel).glob('fsdp_multiclass_*.pid'):
            pid = int(file.read_text().strip())
            if pid <= 1:
                raise ValueError(f'Invalid PID in {file}')
            current = process_info(pid, proc_root)
            if current is None:
                continue
            if (proc_root / str(pid)).stat().st_uid != os.getuid():
                raise ValueError(f'PID{pid}does not belong to this user; no signal sent')
            command = current['command']
            index = 2 if len(command) > 2 and command[1] == '-u' else 1
            if len(command) <= index or Path(command[index]).resolve() != script:
                print(f'Ignoring stale PID file for a different process: {file.name}', flush=True)
                continue
            if command.count('--run-dir') != 1:
                raise ValueError(f'Cannot identify run for PID{pid}')
            run = Path(command[command.index('--run-dir') + 1]).resolve()
            if run != (project / 'runs' / file.stem).resolve() or not run.is_relative_to(project / 'runs'):
                raise ValueError(f'PID{pid}run does not match project/run name')
            config = json.loads((run / 'config.json').read_text())
            if (config.get('experiment') != 'multiclass_turbo_local_detail'
                    or Path(config['run_dir']).resolve() != run):
                raise ValueError(f'PID{pid}is not a verified multiclass experiment')
            result[pid] = {'pid': pid, 'run_name': file.stem, 'run_dir': str(run), **current}
    return list(result.values())


def stop_target(target, timeout=30):
    pid = target['pid']
    current = process_info(pid)
    if not current or current != {key: target[key] for key in ('start_time', 'command')}:
        raise ValueError('PID identity/command changed; no signal sent')
    print(f'Sending SIGTERM to verified {target["run_name"]} PID{pid}', flush=True)
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = process_info(pid)
        if current is None or current['start_time'] != target['start_time']:
            print('Multiclass process stopped; existing checkpoints/logs retained', flush=True)
            return
        time.sleep(.5)
    raise TimeoutError('Multiclass process still active; refusing to launch a new job')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/proc/self/cmdline').exists():
        parser.error('Run this on the Linux DGX')
    try:
        targets = find_targets(args.project_dir)
        for target in targets:
            stop_target(target)
        remaining = find_targets(args.project_dir)
        if remaining:
            raise ValueError('A tracked multiclass job is still active; refusing to continue')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = args.project_dir / 'outputs/private/stops'
        output.mkdir(parents=True, exist_ok=True)
        (output / f'multiclass_{stamp}.json').write_text(json.dumps({'stopped': targets,
            'signal': 'SIGTERM', 'stopped_at_utc': stamp}, indent=2))
        if not targets:
            print('No verified active multiclass jobs in tracked PID files', flush=True)
    except (ValueError, OSError, IndexError, KeyError, TimeoutError) as error:
        parser.exit(1, f'Stop aborted: {error}\n')


if __name__ == '__main__':
    main()
