#!/usr/bin/env python3
"""Read process and cgroup memory diagnostics without elevated permissions."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import time


def read_text(path):
    try:
        return {'value': path.read_text().strip()}
    except OSError as error:
        return {'error': type(error).__name__}


def unescape(value):
    return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), value)


def cgroup_memory(proc):
    membership = read_text(proc / 'cgroup')
    mounts = read_text(proc.parent / 'self/mountinfo')
    groups = []
    for line in membership.get('value', '').splitlines():
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        _, controllers, relative = parts
        for entry in mounts.get('value', '').splitlines():
            left, separator, right = entry.partition(' - ')
            fields, kind = left.split(), right.split()
            if not separator or len(fields) < 5 or len(kind) < 3:
                continue
            v2 = kind[0] == 'cgroup2' and not controllers
            v1 = kind[0] == 'cgroup' and 'memory' in controllers.split(',') and 'memory' in kind[2].split(',')
            if not (v1 or v2):
                continue
            root, mount = PurePosixPath(unescape(fields[3])), Path(unescape(fields[4]))
            member = PurePosixPath(relative)
            if '..' in member.parts or not member.is_absolute():
                continue
            if member.is_relative_to(root):
                offset = member.relative_to(root)
            elif member == PurePosixPath('/'):
                offset = PurePosixPath('.')
            else:
                continue
            folder = mount.joinpath(*offset.parts)
            names = ('memory.current', 'memory.max', 'memory.high', 'memory.peak', 'memory.events', 'memory.events.local') if v2 else (
                'memory.usage_in_bytes', 'memory.limit_in_bytes', 'memory.max_usage_in_bytes', 'memory.failcnt', 'memory.oom_control')
            while folder.is_relative_to(mount):
                groups.append({'directory': str(folder), 'version': 2 if v2 else 1,
                               'files': {name: read_text(folder / name) for name in names}})
                if folder == mount:
                    break
                folder = folder.parent
    return {'membership': membership, 'groups': groups, 'mountinfo_error': mounts.get('error'),
            'scope': 'Cgroup counters can include other processes and descendants; they do not identify an OOM victim.'}


def snapshot(pid=None, proc_root=Path('/proc')):
    pid = os.getpid() if pid is None else pid
    proc = proc_root / str(pid)
    status = read_text(proc / 'status')
    allowed = {'Name', 'State', 'VmRSS', 'VmHWM', 'VmSize', 'VmSwap', 'Threads'}
    values = dict(line.split(':', 1) for line in status.get('value', '').splitlines() if ':' in line)
    memory = read_text(proc_root / 'meminfo')
    host = dict(line.split(':', 1) for line in memory.get('value', '').splitlines() if ':' in line)
    return {'timestamp_unix': time.time(), 'pid': pid,
            'process': {key: value.strip() for key, value in values.items() if key in allowed},
            'process_read_error': status.get('error'), 'limits': read_text(proc / 'limits'),
            'host_memory': {key: value.strip() for key, value in host.items()
                            if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree')},
            'host_memory_read_error': memory.get('error'), 'cgroup': cgroup_memory(proc)}


def event_deltas(before, after):
    def counters(report):
        result = {}
        for group in report['cgroup']['groups']:
            for name in ('memory.events', 'memory.events.local', 'memory.oom_control', 'memory.failcnt'):
                value = group['files'].get(name, {}).get('value', '')
                pairs = [('count', value)] if name == 'memory.failcnt' else [line.split() for line in value.splitlines()]
                for pair in pairs:
                    if len(pair) == 2 and pair[1].isdigit():
                        result[(group['directory'], name, pair[0])] = int(pair[1])
        return result
    initial, final = counters(before), counters(after)
    return [{'directory': key[0], 'file': key[1], 'counter': key[2], 'delta': final[key] - value}
            for key, value in initial.items() if key in final and final[key] != value]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.pid is not None and args.pid <= 0:
        parser.error('PID must be positive')
    report = snapshot(args.pid)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n')
    print(rendered)


if __name__ == '__main__':
    main()
