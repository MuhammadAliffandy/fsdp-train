#!/usr/bin/env python3
"""Stop this user's tracked resolution workflow before a sequential research launch."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from stop_private_800 import process_info


def script_path(command):
    index = 2 if len(command) > 2 and command[1] == '-u' else 1
    return Path(command[index]).resolve() if len(command) > index else None


def run_path(command):
    if command.count('--run-dir') != 1:
        raise ValueError('Cannot identify a unique run directory')
    return Path(command[command.index('--run-dir') + 1]).resolve()


def inspect_workflow(project, proc_root=Path('/proc')):
    project = project.resolve()
    channel = project / 'logs/fsdp_resolution'
    tracked = (channel / 'latest.pid').resolve(strict=True)
    if tracked.parent != channel or not tracked.name.startswith('fsdp_resolution_') or tracked.suffix != '.pid':
        raise ValueError('Resolution PID file is outside the tracked channel')
    pid = int(tracked.read_text().strip())
    if pid <= 1:
        raise ValueError('Invalid workflow PID')
    current = process_info(pid, proc_root)
    if current is None:
        raise ValueError('Tracked resolution workflow is no longer active; no signal sent')
    if (proc_root / str(pid)).stat().st_uid != os.getuid():
        raise ValueError('Resolution workflow belongs to another user; no signal sent')
    if script_path(current['command']) != project / 'scripts/run_fsdp_resolution.py':
        raise ValueError('Tracked PID is a different process; no signal sent')
    run = run_path(current['command'])
    if run != project / 'runs' / tracked.stem:
        raise ValueError('Workflow run does not match its PID file')
    config = json.loads((run / 'workflow_config.json').read_text())
    if Path(config['run_dir']).resolve() != run:
        raise ValueError('Workflow configuration belongs to a different run')
    env = (proc_root / str(pid) / 'environ').read_bytes().split(b'\0')
    visible = [value.split(b'=', 1)[1].decode() for value in env if value.startswith(b'CUDA_VISIBLE_DEVICES=')]
    if len(visible) != 1 or not visible[0] or ',' in visible[0]:
        raise ValueError('Cannot identify one physical GPU from the resolution workflow')
    return {'pid': pid, 'run_name': tracked.stem, 'run_dir': str(run), 'gpu_selector': visible[0], **current}


def resolve_gpu(selector):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True, timeout=20)
    rows = [tuple(part.strip() for part in line.split(',')) for line in result.stdout.splitlines()]
    matches = [row for row in rows if len(row) == 2 and row[0].isdigit() and selector in row]
    if len(matches) != 1:
        raise ValueError('Cannot resolve the original workflow GPU to a physical index/UUID')
    return {'index': int(matches[0][0]), 'uuid': matches[0][1]}


def active_children(project, run, proc_root=Path('/proc')):
    children = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            current = process_info(int(entry.name), proc_root)
            if current and script_path(current['command']) == project / 'scripts/train_fsdp_foundation.py':
                child_run = run_path(current['command'])
                if child_run.is_relative_to(run):
                    children.append({'pid': int(entry.name), 'run_dir': str(child_run), **current})
        except FileNotFoundError:
            continue
    return children


def stop_workflow(target, project, timeout=60, proc_root=Path('/proc')):
    expected = {key: target[key] for key in ('start_time', 'command')}
    if process_info(target['pid'], proc_root) != expected:
        raise ValueError('Workflow PID identity changed; no signal sent')
    if (proc_root / str(target['pid'])).stat().st_uid != os.getuid():
        raise ValueError('Workflow owner changed; no signal sent')
    print(f'Sending SIGTERM to verified resolution workflow PID {target["pid"]}', flush=True)
    os.kill(target['pid'], signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while True:
        current = process_info(target['pid'], proc_root)
        same_parent = current is not None and current['start_time'] == target['start_time']
        children = active_children(project, Path(target['run_dir']), proc_root)
        if not same_parent and not children:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('Resolution parent/child is still active; research launch blocked. No SIGKILL sent.')
        time.sleep(.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    project = args.project_dir.resolve()
    if not Path('/proc/self/cmdline').exists():
        parser.error('Run this command on the Linux DGX')
    record = {}
    try:
        target = inspect_workflow(project)
        gpu = resolve_gpu(target['gpu_selector'])
        children = active_children(project, Path(target['run_dir']))
        stamp = datetime.now(timezone.utc).isoformat()
        record = {'complete': False, 'target': target, 'gpu': gpu, 'children_before': children, 'requested_at_utc': stamp}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2))
        stop_workflow(target, project)
        files = sorted(Path(target['run_dir']).glob('train/*/hybrid_*.pth'))
        record.update(complete=True, saved_checkpoints=[{'path': str(p), 'bytes': p.stat().st_size} for p in files],
                      stopped_at_utc=datetime.now(timezone.utc).isoformat(), signal='SIGTERM')
        args.output.write_text(json.dumps(record, indent=2))
        print(f'Resolution parent and children stopped; saved files retained. Released physical GPU {gpu["index"]}.', flush=True)
    except (OSError, ValueError, IndexError, KeyError, TimeoutError, subprocess.SubprocessError) as error:
        if record:
            record.update(error=str(error))
            args.output.write_text(json.dumps(record, indent=2))
        parser.exit(1, f'Switch aborted: {error}\n')


if __name__ == '__main__':
    main()
