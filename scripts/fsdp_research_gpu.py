#!/usr/bin/env python3
"""Select an idle physical GPU by nvidia-smi index; never stop existing jobs."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess


def inspect_gpu(index):
    if not str(index).isdigit():
        raise ValueError('GPU_ID must be one physical nvidia-smi index')
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
    return {'ready': not pids, 'index': int(number), 'uuid': uuid, 'memory_total_mib': int(total),
            'memory_used_mib': int(used), 'active_compute_pids': pids,
            'scope': 'launch-time availability check, not a system-wide reservation', 'jobs_stopped': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = inspect_gpu(args.gpu_id)
    args.output.write_text(json.dumps(report, indent=2))
    if not report['ready']:
        parser.exit(2, f'GPU{report["index"]}already has compute PIDs{report["active_compute_pids"]}; choose another GPU_ID. No jobs stopped.\n')
    print(report['uuid'])


if __name__ == '__main__':
    main()
