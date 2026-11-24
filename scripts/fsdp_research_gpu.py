#!/usr/bin/env python3
"""Inspect GPU capacity and optionally select a shared device with an explicit VRAM budget."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import subprocess
import sys


def inspect_gpu(index, allow_sharing=False, min_free_gib=0.):
    if not str(index).isdigit():
        raise ValueError('GPU_ID must be one physical nvidia-smi index')
    if not math.isfinite(min_free_gib) or min_free_gib < 0 or (allow_sharing and min_free_gib <= 0):
        raise ValueError('Shared GPU mode requires a positive free-VRAM floor')
    result = subprocess.run(['nvidia-smi', '-i', str(index), '--query-gpu=index,uuid,memory.total,memory.used',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=20)
    rows = list(csv.reader(io.StringIO(result.stdout)))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError('Expected exactly one physical GPU')
    number, uuid, total, used = [item.strip() for item in rows[0]]
    if number != str(index) or not uuid.startswith('GPU-'):
        raise ValueError('GPU identity differs from requested physical index')
    processes = subprocess.run(['nvidia-smi', '-i', uuid, '--query-compute-apps=pid,gpu_uuid',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=20)
    pids = [int(row[0].strip()) for row in csv.reader(io.StringIO(processes.stdout))
            if len(row) == 2 and row[0].strip().isdigit() and row[1].strip() == uuid]
    total, used = int(total), int(used)
    if total <= 0 or not 0 <= used <= total:
        raise ValueError('Invalid GPU memory counters')
    free = total - used
    enough = free >= math.ceil(min_free_gib * 1024)
    return {'ready': enough and (allow_sharing or not pids), 'index': int(number), 'uuid': uuid,
            'memory_total_mib': total, 'memory_used_mib': used, 'memory_free_mib': free,
            'minimum_free_gib': min_free_gib, 'sharing_allowed': allow_sharing,
            'reason': 'insufficient_free_vram' if not enough else 'busy' if pids and not allow_sharing else 'eligible',
            'active_compute_pids': pids,
            'scope': 'launch-time availability check, not a system-wide reservation', 'jobs_stopped': False}


def contains_resolution_process(pids, run):
    if run is None or not Path(run).exists():
        return False
    root = Path(run).resolve()
    for pid in pids:
        try:
            argv = (Path('/proc') / str(pid) / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            if not any(Path(arg).name == 'train_fsdp_foundation.py' for arg in argv) or '--run-dir' not in argv:
                continue
            if Path(argv[argv.index('--run-dir') + 1]).resolve().is_relative_to(root):
                return True
        except (OSError, IndexError, ValueError):
            continue
    return False


def select_gpu(request, allow_sharing, min_free_gib, exclude=(), memory_budget_gib=None, reserve_gib=8., resolution_run=None):
    if not math.isfinite(min_free_gib) or min_free_gib < 0:
        raise ValueError('Invalid free-VRAM threshold')
    if memory_budget_gib is None:
        memory_budget_gib = 40. if allow_sharing else 0.
    if (not math.isfinite(memory_budget_gib) or memory_budget_gib < 0
            or not math.isfinite(reserve_gib) or reserve_gib < 0
            or (allow_sharing and (memory_budget_gib <= 0 or reserve_gib < 1))):
        raise ValueError('Shared mode requires a positive allocator budget and at least 1 GiB headroom')
    required = max(min_free_gib, memory_budget_gib + reserve_gib if memory_budget_gib else 0.)
    if request == 'auto':
        result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'],
                                check=True, capture_output=True, text=True, timeout=20)
        indices = [s.strip() for s in result.stdout.splitlines() if s.strip()]
        if not indices or any(not s.isdigit() for s in indices):
            raise ValueError('Cannot enumerate physical GPUs')
    else:
        indices = [str(request)]
    rows = [inspect_gpu(index, allow_sharing, required) for index in indices if str(index) not in set(exclude)]
    for row in rows:
        if contains_resolution_process(row['active_compute_pids'], resolution_run):
            row.update(ready=False, reason='active_resolution_gpu')
    eligible = sorted((r for r in rows if r['ready']),
                      key=lambda r: (bool(r['active_compute_pids']), -r['memory_free_mib'], r['index']))
    selected = eligible[0] if eligible else None
    return {**(selected or {}), 'ready': selected is not None, 'requested_gpu': request,
            'candidates': rows, 'excluded_gpu_ids': list(exclude), 'sharing_allowed': allow_sharing,
            'allocator_budget_gib': memory_budget_gib or None, 'reserved_headroom_gib': reserve_gib,
            'protected_resolution_run': str(Path(resolution_run).resolve()) if resolution_run else None,
            'minimum_free_gib': required, 'jobs_stopped': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-id', default='auto')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--list', action='store_true', help='Read capacity only; never launch a job')
    parser.add_argument('--allow-sharing', action='store_true')
    parser.add_argument('--min-free-gib', type=float, default=48.)
    parser.add_argument('--memory-budget-gib', type=float)
    parser.add_argument('--reserve-gib', type=float, default=8.)
    parser.add_argument('--exclude', default='')
    parser.add_argument('--protect-resolution-run', type=Path)
    args = parser.parse_args()
    exclude = [s.strip() for s in args.exclude.split(',') if s.strip()]
    if any(not s.isdigit() for s in exclude):
        parser.error('Exclude must contain comma-separated physical GPU indices')
    if args.list:
        report = select_gpu('auto', args.allow_sharing, args.min_free_gib, exclude,
                            args.memory_budget_gib, args.reserve_gib, args.protect_resolution_run)
        print(json.dumps(report, indent=2))
        return
    if args.output is None:
        parser.error('GPU selection requires an output record')
    report = select_gpu(args.gpu_id, args.allow_sharing, args.min_free_gib, exclude,
                        args.memory_budget_gib, args.reserve_gib, args.protect_resolution_run)
    args.output.write_text(json.dumps(report, indent=2))
    if not report['ready']:
        details = '\n'.join(f'GPU {r["index"]}: free={r["memory_free_mib"] / 1024:.1f} GiB; '
                            f'PIDs={r["active_compute_pids"]}; reason={r["reason"]}' for r in report['candidates'])
        parser.exit(2, f'No eligible GPU; required free={report["minimum_free_gib"]:.1f} GiB; '
                    f'sharing={args.allow_sharing}; excluded={exclude}.\n{details}\nNo jobs stopped.\n')
    print(f'Selected physical GPU {report["index"]}: free={report["memory_free_mib"] / 1024:.1f} GiB; '
          f'sharing={args.allow_sharing}; allocator cap={report["allocator_budget_gib"]} GiB', file=sys.stderr)
    print(report['uuid'])


if __name__ == '__main__':
    main()
