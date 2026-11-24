#!/usr/bin/env python3
"""Run controlled research arms sequentially on the second GPU."""
import argparse
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

import physical_fsdp as data
from evaluate_private_validation import file_hash
from fsdp_research_model import VARIANTS
from train_fsdp_research import code_signature
from train_fsdp_foundation import inspect_foundation


def run_child(command, record):
    started = time.monotonic()
    child = subprocess.Popen(command)
    data.save(record, {'pid': child.pid, 'command': command})
    def forward(signum, frame):
        if child.poll() is None:
            child.send_signal(signum)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print('Research child has not exited; no subsequent arm will start', flush=True)
        raise SystemExit(128 + signum)
    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        status = child.wait()
        data.save(record, {'pid': child.pid, 'command': command, **exit_details(status),
                           'elapsed_seconds': time.monotonic() - started})
        return status
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def exit_details(status):
    number = -status if status < 0 else None
    try:
        name = signal.Signals(number).name if number else None
    except ValueError:
        name = f'SIGNAL_{number}'
    return {'returncode': status, 'signal_number': number, 'signal_name': name,
            'interpretation': ('Terminated by SIGKILL; OOM is possible but requires kernel/cgroup evidence.'
                               if name == 'SIGKILL' else 'Child process exit status; inspect its traceback or system logs.')}


def execute(args):
    inspect_foundation(args.foundation_run_dir, args.source_root, args.verified_cache)
    config = {'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'},
              'code_sha256': code_signature(), 'variants': list(VARIANTS),
              'scope': 'independent controlled arms on one GPU; runs alongside resolution workflow on another GPU'}
    if args.resume:
        if data.read(args.run_dir / 'workflow_config.json') != config:
            raise ValueError('Research workflow resume settings/source changed')
    else:
        data.save(args.run_dir / 'workflow_config.json', config)
    entries = []
    for variant in VARIANTS:
        if code_signature() != config['code_sha256']:
            raise ValueError('Research source changed between arms')
        child = args.run_dir / variant
        data.save(args.run_dir / 'progress.json', {'complete': False, 'active_variant': variant,
            'entries': entries, 'test_images_read': False})
        command = [sys.executable, '-u', str(Path(__file__).with_name('train_fsdp_research.py')),
            '--foundation-run-dir', str(args.foundation_run_dir), '--run-dir', str(child), '--variant', variant,
            '--epochs', str(args.epochs), '--lr-multiplier', str(args.lr_multiplier),
            '--patience-epochs', str(args.patience_epochs), '--boundary-weight', str(args.boundary_weight),
            '--boundary-clip-mm', str(args.boundary_clip_mm), '--device', args.device]
        if args.source_root:
            command.extend(['--source-root', str(args.source_root)])
        if args.verified_cache:
            command.append('--verified-cache')
        if args.resume and (child / 'summary.json').exists():
            summary = data.read(child / 'summary.json')
            if (not summary['complete'] or summary['variant'] != variant or summary['test_images_read']
                    or file_hash(child / 'research_best.pth') != summary['best_checkpoint_sha256']):
                raise ValueError('Completed research arm/checkpoint changed')
            print(f'Skipping verified completed research arm: {variant}', flush=True)
        else:
            if args.resume and child.exists():
                if not (child / 'research_latest.pth').exists():
                    raise ValueError('Incomplete arm has no latest checkpoint; inspect failure before a new workflow')
                command.append('--resume')
            print(f'Starting research: {variant}; outputs={child}', flush=True)
            status = run_child(command, args.run_dir / 'active_child.json')
            child.mkdir(parents=True, exist_ok=True)
            termination = {**exit_details(status), 'variant': variant,
                           'child_failure_present': (child / 'failure.json').exists()}
            data.save(child / 'process_exit.json', termination)
            if status != 0 or not (child / 'summary.json').exists():
                detail = f'returncode={status}; signal={termination["signal_name"]}'
                raise ValueError(f'Research arm {variant} failed ({detail}); no subsequent arm launched; '
                                 f'inspect {child}/process_exit.json and child failure.json if present')
            summary = data.read(child / 'summary.json')
            if not summary['complete'] or summary['test_images_read'] or summary['variant'] != variant:
                raise ValueError('Research child summary incomplete or inconsistent')
            if file_hash(child / 'research_best.pth') != summary['best_checkpoint_sha256']:
                raise ValueError('Research child checkpoint checksum differs')
        entry = {'variant': variant, 'directory': str(child), 'summary_sha256': file_hash(child / 'summary.json'),
            'best_checkpoint_sha256': summary['best_checkpoint_sha256'], 'best_epoch': summary['best_epoch'],
            'completed_epochs': summary['completed_epochs'], 'best_validation_metrics': summary['best_validation_metrics'],
            'best_frozen_training_metrics': summary['best_frozen_training_metrics'],
            'validation_control_fixed_0_5': summary['validation_control_fixed_0_5'],
            'native_boundary_metrics': summary.get('native_boundary_metrics'),
            'delta_vs_foundation': summary['validation_change_vs_foundation'],
            'delta_vs_candidate_initial': summary['validation_change_vs_candidate_initial']}
        entries.append(entry)
        data.save(args.run_dir / 'progress.json', {'complete': False, 'active_variant': None,
            'entries': entries, 'test_images_read': False})
    control = entries[0]['best_validation_metrics']['5']['mean_bilateral_patient_dice']
    for entry in entries:
        entry['delta_vs_matched_control'] = entry['best_validation_metrics']['5']['mean_bilateral_patient_dice'] - control
    report = {'complete': True, 'entries': entries, 'gpu': data.read(args.gpu_record) if args.gpu_record else None,
        'test_images_read': False, 'validation_images_read': True, 'epochs_per_arm': args.epochs,
        'interpretation': 'Research screen; one controlled change per arm. History changes prompt information; no automatic model promotion.',
        'archive': str(args.run_dir / 'results.tar.gz')}
    data.save(args.run_dir / 'report.json', report)
    with tarfile.open(args.run_dir / 'results.tar.gz', 'w:gz') as archive:
        archive.add(args.run_dir / 'report.json', arcname='research/summary.json')
        archive.add(args.run_dir / 'workflow_config.json', arcname='research/workflow_config.json')
        if args.gpu_record:
            archive.add(args.gpu_record, arcname='research/gpu.json')
        for entry in entries:
            archive.add(Path(entry['directory']) / 'results.tar.gz', arcname=f'research/{entry["variant"]}/results.tar.gz')
    data.save(args.run_dir / 'summary.json', report)
    data.save(args.run_dir / 'progress.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foundation-run-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--verified-cache', action='store_true')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gpu-record', type=Path)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr-multiplier', type=float, default=.2)
    parser.add_argument('--patience-epochs', type=int, default=50)
    parser.add_argument('--boundary-weight', type=float, default=.05)
    parser.add_argument('--boundary-clip-mm', type=float, default=5.)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (args.epochs < 1 or args.patience_epochs < 0 or not math.isfinite(args.lr_multiplier)
            or not 0 < args.lr_multiplier < 1 or not math.isfinite(args.boundary_weight) or args.boundary_weight <= 0
            or not math.isfinite(args.boundary_clip_mm) or args.boundary_clip_mm <= 0):
        parser.error('Invalid research budget/objective')
    for key in ('foundation_run_dir', 'run_dir', 'source_root', 'gpu_record'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).expanduser().resolve())
    old, _ = inspect_foundation(args.foundation_run_dir, args.source_root, args.verified_cache)
    control = data.read(Path(old['cohort_run_dir']) / 'config.json')
    protected = [args.foundation_run_dir, Path(old['sam_dir']).resolve(), Path(old['cohort_run_dir']).resolve(),
                 Path(control['data_dir']).resolve(), Path(control['source_root']).resolve()]
    if args.source_root:
        protected.append(args.source_root)
    if any(args.run_dir.is_relative_to(p) for p in protected):
        parser.error('Research outputs must be outside original data/foundation')
    if args.resume:
        if not (args.run_dir / 'workflow_config.json').exists() or (args.run_dir / 'summary.json').exists():
            parser.error('Resume requires an unfinished research workflow')
    elif args.run_dir.exists():
        parser.error('Choose a new research workflow directory')
    args.run_dir.mkdir(parents=True, exist_ok=args.resume)
    try:
        execute(args)
    except Exception as error:
        data.save(args.run_dir / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
