#!/usr/bin/env python3
"""Finish an interrupted resolution study on a GPU separate from active research."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile

from fsdp_research_gpu import select_gpu
from stop_private_800 import process_info


TRACKED = ('train_sam_refine32.py', 'train_fsdp_foundation.py', 'sam_refine32.py', 'sam_refine_data.py',
           'sam_refine_resolution.py', 'fsdp_resolution.py', 'physical_fsdp.py', 'multiclass_model.py',
           'run_fsdp_resolution.py', 'audit_foundation_sources.py', 'fsdp_verified_cache.py')
VARIANTS = ('baseline', 'tokens', 'detail', 'combined')


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify_source(project, original):
    if original.parent != project / 'runs' or not original.name.startswith('fsdp_resolution_'):
        raise ValueError('Original workflow must be this project\'s resolution run')
    config = read(original / 'workflow_config.json')
    if Path(config['run_dir']).resolve() != original or not config['train']:
        raise ValueError('Original workflow configuration does not identify a training run')
    signatures = read(original / 'workflow_code_sha256.json')
    if set(signatures) != set(TRACKED) or any(digest(project / 'scripts' / name) != signatures[name] for name in TRACKED):
        raise ValueError('Original resolution implementation changed; refusing incompatible resume')
    for variant in VARIANTS:
        folder = original / 'train' / variant
        if not folder.resolve().is_relative_to(original):
            raise ValueError('Resolution child directory escapes the original workflow')
        proof = read(original / 'audit' / variant / 'summary.json')
        if (not proof['complete'] or proof['test_images_read'] or proof['optimizer_updates'] != 0
                or not proof['foundation_reproduced_before_adaptation'] or not proof['gradient_probe']['passed']
                or proof['resolution']['variant'] != variant):
            raise ValueError(f'Original {variant} readiness audit did not pass')
    return config


def active_writers(project, original, proc_root=Path('/proc')):
    result = []
    scripts = {project / 'scripts/train_fsdp_foundation.py', project / 'scripts/run_fsdp_resolution.py'}
    for folder in proc_root.iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            if folder.stat().st_uid != os.getuid():
                continue
            state = process_info(int(folder.name), proc_root)
            if not state:
                continue
            command = state['command']
            if not any(Path(arg).resolve() in scripts for arg in command if not arg.startswith('-')):
                continue
            if command.count('--run-dir') == 1:
                folder_run = Path(command[command.index('--run-dir') + 1]).resolve()
                if folder_run.is_relative_to(original):
                    result.append(int(folder.name))
        except FileNotFoundError:
            continue
    return result


def research_gpus(project, proc_root=Path('/proc')):
    identities = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                                check=True, capture_output=True, text=True, timeout=20)
    mapping = {parts[1].strip(): parts[0].strip() for line in identities.stdout.splitlines()
               if len(parts := line.split(',')) == 2}
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'],
                               check=True, capture_output=True, text=True, timeout=20)
    protected = set()
    tracked = project / 'logs/fsdp_research/latest.pid'
    if tracked.exists():
        pid = int(tracked.read_text().strip())
        folder = proc_root / str(pid)
        current = process_info(pid, proc_root)
        parent_script = project / 'scripts/run_fsdp_research.py'
        if current and folder.stat().st_uid == os.getuid() and any(
                Path(arg).resolve() == parent_script for arg in current['command'] if not arg.startswith('-')):
            env = (folder / 'environ').read_bytes().split(b'\0')
            for value in env:
                if value.startswith(b'CUDA_VISIBLE_DEVICES='):
                    selector = value.split(b'=', 1)[1].decode()
                    if selector in mapping:
                        protected.add(mapping[selector])
                    elif selector in mapping.values():
                        protected.add(selector)
    script = project / 'scripts/train_fsdp_research.py'
    for line in processes.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != 2 or not parts[0].isdigit() or parts[1] not in mapping:
            continue
        folder = proc_root / parts[0]
        try:
            if folder.stat().st_uid != os.getuid():
                continue
            current = process_info(int(parts[0]), proc_root)
            if current and any(Path(arg).resolve() == script for arg in current['command'] if not arg.startswith('-')):
                protected.add(mapping[parts[1]])
        except FileNotFoundError:
            continue
    return sorted(protected)


def validate_child_config(folder, variant, original_config):
    config = read(folder / 'config.json')
    if (config['resolution']['variant'] != variant or config['audit_only']
            or Path(config['run_dir']).resolve() != folder
            or Path(config['foundation_run_dir']).resolve() != Path(original_config['foundation_run_dir']).resolve()
            or config.get('source_root') != original_config.get('source_root')
            or config.get('verified_cache', False) != original_config.get('verified_cache', False)
            or any(config[key] != original_config[key] for key in ('epochs', 'lr_multiplier', 'patience_epochs', 'device'))):
        raise ValueError(f'{variant} configuration differs from the original study')
    return config


def completed(folder, variant, original_config):
    config = validate_child_config(folder, variant, original_config)
    summary = read(folder / 'summary.json')
    if (not summary['complete'] or summary['test_images_read'] or summary['resolution']['variant'] != variant
            or summary['resolution'] != config['resolution'] or summary['foundation'] != config['foundation']
            or summary['best_checkpoint_sha256'] != digest(folder / 'hybrid_best.pth')):
        raise ValueError(f'Completed {variant} result/checkpoint is inconsistent')
    return summary


def action(folder, variant, original_config):
    if (folder / 'summary.json').exists():
        return 'skip', completed(folder, variant, original_config)
    if not folder.exists():
        return 'new', None
    if not (folder / 'hybrid_latest.pth').exists() or not (folder / 'config.json').exists():
        raise ValueError(f'{variant} has partial files but no resumable checkpoint; inspect before continuing')
    config = validate_child_config(folder, variant, original_config)
    import torch
    saved = torch.load(folder / 'hybrid_latest.pth', map_location='cpu', weights_only=False)
    if saved['config'] != config or not 0 <= saved['epoch'] <= config['epochs']:
        raise ValueError(f'{variant} checkpoint metadata differs')
    return 'resume', {'saved_epoch': saved['epoch'], 'checkpoint_sha256': digest(folder / 'hybrid_latest.pth')}


def child_command(project, folder, variant, config, mode, signal_report=None):
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--child']
    if signal_report is not None:
        command.extend(['--signal-report', str(signal_report)])
    command.extend([str(project / 'scripts/train_fsdp_foundation.py'), '--foundation-run-dir', config['foundation_run_dir'],
        '--run-dir', str(folder), '--resolution-variant', variant, '--epochs', str(config['epochs']),
        '--lr-multiplier', str(config['lr_multiplier']), '--patience-epochs', str(config['patience_epochs']), '--device', config['device']])
    if config.get('source_root'):
        command.extend(['--source-root', config['source_root']])
    if config.get('verified_cache'):
        command.append('--verified-cache')
    if mode == 'resume':
        command.append('--resume')
    return command


def child_main(arguments):
    signal_report = None
    if arguments[:1] == ['--signal-report']:
        signal_report = Path(arguments[1]).resolve()
        arguments = arguments[2:]
    expected = Path(__file__).with_name('train_fsdp_foundation.py').resolve()
    if not arguments or Path(arguments[0]).resolve() != expected:
        raise ValueError('Only the unchanged project foundation trainer can be delegated')
    if signal_report is not None:
        from fsdp_signal_provenance import install
        install(signal_report)
    import torch
    budget = float(os.environ.get('FSDP_RESOLUTION_GPU_MEMORY_GIB', '40'))
    total = torch.cuda.get_device_properties(0).total_memory
    if not math.isfinite(budget) or not 0 < budget * 2**30 <= total:
        raise ValueError('Invalid resolution CUDA allocator budget')
    torch.cuda.set_per_process_memory_fraction(budget * 2**30 / total, 0)
    print(f'Resolution allocator budget: {budget:g} GiB on visible cuda:0', flush=True)
    sys.argv = [str(expected), *arguments[1:]]
    runpy.run_path(str(expected), run_name='__main__')


def execute(args):
    from train_fsdp_foundation import inspect_foundation
    from run_fsdp_research import run_child
    original = verify_source(args.project_dir, args.original_run_dir)
    writers = active_writers(args.project_dir, args.original_run_dir)
    if writers:
        raise ValueError(f'Original resolution still has active writers {writers}; no jobs stopped')
    inspect_foundation(Path(original['foundation_run_dir']), Path(original['source_root']) if original.get('source_root') else None,
                       original.get('verified_cache', False))
    frozen = {'original_workflow': original, 'original_code_sha256': read(args.original_run_dir / 'workflow_code_sha256.json'),
              'continuation_code_sha256': digest(Path(__file__)), 'variants': list(VARIANTS),
              'signal_observer_sha256': digest(Path(__file__).with_name('fsdp_signal_provenance.py')),
              'original_run_dir': str(args.original_run_dir), 'test_images_read': False}
    if args.resume:
        if read(args.run_dir / 'workflow_config.json') != frozen:
            raise ValueError('Resolution continuation settings/source changed')
    else:
        save(args.run_dir / 'workflow_config.json', frozen)
    entries = []
    for variant in VARIANTS:
        verify_source(args.project_dir, args.original_run_dir)
        if (digest(Path(__file__)) != frozen['continuation_code_sha256']
                or digest(Path(__file__).with_name('fsdp_signal_provenance.py')) != frozen['signal_observer_sha256']):
            raise ValueError('Continuation source changed between variants')
        if active_writers(args.project_dir, args.original_run_dir):
            raise ValueError('Another resolution writer is active; no child launched')
        folder = args.original_run_dir / 'train' / variant
        mode, previous = action(folder, variant, original)
        print(f'Resolution {variant}: {mode}' + (f' from saved additional epoch {previous["saved_epoch"]}' if mode == 'resume' else ''), flush=True)
        save(args.run_dir / 'progress.json', {'complete': False, 'active_variant': variant, 'action': mode, 'entries': entries})
        if mode != 'skip':
            signal_report = args.run_dir / f'{variant}_signal.json'
            status = run_child(child_command(args.project_dir, folder, variant, original, mode, signal_report), args.run_dir / 'active_child.json')
            save(args.run_dir / f'{variant}_process_exit.json', {'returncode': status, 'variant': variant, 'mode': mode})
            if status != 0:
                sender = read(signal_report) if signal_report.exists() else None
                detail = f'; signal sender PID={sender["sender_pid"]}, UID={sender["sender_uid"]}' if sender and 'sender_pid' in sender else ''
                raise ValueError(f'{variant} exited with returncode {status}{detail}; later variants were not launched. '
                                 f'Signal report, when available: {signal_report}')
        summary = completed(folder, variant, original)
        entries.append({'variant': variant, 'mode': mode, 'directory': str(folder), 'completed_epochs': summary['completed_epochs'],
                        'best_epoch': summary['best_epoch'], 'best_validation_metrics': summary['best_validation_metrics'],
                        'checkpoint_sha256': summary['best_checkpoint_sha256']})
    baseline = entries[0]['best_validation_metrics']['5']['mean_bilateral_patient_dice']
    for entry in entries:
        entry['delta_vs_resolution_baseline'] = entry['best_validation_metrics']['5']['mean_bilateral_patient_dice'] - baseline
    report = {'complete': True, 'entries': entries, 'gpu': read(args.gpu_record) if args.gpu_record else None,
              'original_run_dir': str(args.original_run_dir), 'test_images_read': False,
              'interpretation': 'Original resolution study completed with unchanged recipe; independent from the history continuation.'}
    save(args.run_dir / 'report.json', report)
    with tarfile.open(args.run_dir / 'results.tar.gz', 'w:gz') as archive:
        archive.add(args.run_dir / 'report.json', arcname='results/summary.json')
        for entry in entries:
            archive.add(Path(entry['directory']) / 'results.tar.gz', arcname=f'results/{entry["variant"]}/results.tar.gz')
        for path in args.run_dir.glob('*_signal*.json'):
            archive.add(path, arcname='results/' + path.name)
    save(args.run_dir / 'summary.json', report); save(args.run_dir / 'progress.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--original-run-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--gpu-record', type=Path)
    parser.add_argument('--gpu-check-only', action='store_true')
    parser.add_argument('--gpu-id', default='auto')
    parser.add_argument('--exclude', default='7')
    parser.add_argument('--min-free-gib', type=float, default=48.)
    parser.add_argument('--memory-budget-gib', type=float, default=40.)
    parser.add_argument('--allow-sharing', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    for key in ('project_dir', 'original_run_dir', 'run_dir', 'gpu_record'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.gpu_check_only:
        excluded = {value.strip() for value in args.exclude.split(',') if value.strip()}
        if any(not value.isdigit() for value in excluded):
            parser.error('Excluded GPU indices must be numeric')
        protected = research_gpus(args.project_dir)
        report = select_gpu(args.gpu_id, args.allow_sharing, args.min_free_gib, sorted(excluded | set(protected)),
                            args.memory_budget_gib, 8., args.original_run_dir)
        report['protected_research_gpu_ids'] = protected
        if args.gpu_record is None:
            parser.error('GPU selection needs a record path')
        save(args.gpu_record, report)
        if not report['ready']:
            details = '\n'.join(f'GPU {row["index"]}: free={row["memory_free_mib"] / 1024:.1f} GiB; '
                                f'reason={row["reason"]}' for row in report['candidates'])
            parser.exit(2, f'No eligible separate GPU; required free={report["minimum_free_gib"]:.1f} GiB; '
                        f'excluded={report["excluded_gpu_ids"]}.\n{details}\nNo jobs stopped.\n')
        print(f'Selected physical GPU {report["index"]}; protected research GPUs {protected}; free {report["memory_free_mib"] / 1024:.1f} GiB', file=sys.stderr)
        print(report['uuid'])
        return
    if args.run_dir is None or args.run_dir == args.original_run_dir or args.run_dir.is_relative_to(args.original_run_dir):
        parser.error('Use a separate continuation report directory')
    if args.run_dir.parent != args.project_dir / 'runs' or not args.run_dir.name.startswith('fsdp_resolution_remaining_'):
        parser.error('Continuation reports must use a new project runs/fsdp_resolution_remaining_ name')
    if args.resume:
        if not (args.run_dir / 'workflow_config.json').exists() or (args.run_dir / 'summary.json').exists():
            parser.error('Resume needs an unfinished continuation report directory')
    elif args.run_dir.exists():
        parser.error('Continuation report directory already exists')
    verify_source(args.project_dir, args.original_run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=args.resume)
    with (args.original_run_dir / '.remaining_workflow.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(2, 'Another continuation owns this original workflow; no child launched.\n')
        try:
            execute(args)
        except Exception as error:
            save(args.run_dir / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
            raise


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        child_main(sys.argv[2:])
    else:
        main()
