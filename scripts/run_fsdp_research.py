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
from fsdp_research_resources import snapshot, event_deltas
from train_fsdp_research import code_signature
from train_fsdp_foundation import inspect_foundation


def run_child(command, record):
    started = time.monotonic()
    before = snapshot()
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
    latest = snapshot(child.pid)
    try:
        with record.with_suffix('.resources.jsonl').open('a') as samples:
            while True:
                samples.write(json.dumps(latest) + '\n')
                samples.flush()
                try:
                    status = child.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    latest = snapshot(child.pid)
        data.save(record, {'pid': child.pid, 'command': command, **exit_details(status),
                           'elapsed_seconds': time.monotonic() - started})
        return status
    finally:
        after = snapshot()
        data.save(record.with_suffix('.resources.json'), {'before_parent': before, 'last_child_sample': latest,
            'after_parent': after, 'cgroup_event_deltas': event_deltas(before, after),
            'scope': 'Sampled every 5 seconds; shared cgroup counter changes alone do not identify the killed PID.'})
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
    source = getattr(args, 'research_source_dir', None)
    variants = ('history',) if source else VARIANTS
    config = {'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'},
              'code_sha256': code_signature(), 'variants': list(variants),
              'scope': ('one history continuation from a completed source on the selected GPU' if source
                        else 'independent controlled arms on one GPU; runs alongside resolution workflow on another GPU')}
    if args.resume:
        if data.read(args.run_dir / 'workflow_config.json') != config:
            raise ValueError('Research workflow resume settings/source changed')
    else:
        data.save(args.run_dir / 'workflow_config.json', config)
    entries = []
    for variant in variants:
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
        if source:
            command.extend(['--research-source-dir', str(source)])
        if getattr(args, 'roi_jitter', False):
            command.extend(['--roi-jitter', '--matched-control-dir', str(args.matched_control_dir)])
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
        if source:
            entry['research_source'] = summary['research_source']
            entry['delta_vs_research_source'] = summary['validation_change_vs_research_source']
        if getattr(args, 'roi_jitter', False):
            for key in ('roi_jitter', 'matched_control', 'validation_change_vs_matched_control'):
                entry[key] = summary[key]
        entries.append(entry)
        data.save(args.run_dir / 'progress.json', {'complete': False, 'active_variant': None,
            'entries': entries, 'test_images_read': False})
    control = next((entry['best_validation_metrics']['5']['mean_bilateral_patient_dice']
                    for entry in entries if entry['variant'] == 'control'), None)
    for entry in entries:
        entry['delta_vs_matched_control'] = entry.get('validation_change_vs_matched_control',
            entry['best_validation_metrics']['5']['mean_bilateral_patient_dice'] - control if control is not None else None)
    report = {'complete': True, 'entries': entries, 'gpu': data.read(args.gpu_record) if args.gpu_record else None,
        'test_images_read': False, 'validation_images_read': True, 'epochs_per_arm': args.epochs,
        'interpretation': ('History continuation from a completed best checkpoint; new optimizer/schedule; no matched control for the extended budget.'
                           if source else 'Research screen; one controlled change per arm. History changes prompt information; no automatic model promotion.'),
        'archive': str(args.run_dir / 'results.tar.gz')}
    if getattr(args, 'roi_jitter', False):
        report['interpretation'] = 'Train-only paired ROI translation ablation; completed matched continuation control; same GT-assisted evaluation.'
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
    parser.add_argument('--research-source-dir', type=Path)
    parser.add_argument('--roi-jitter', action='store_true')
    parser.add_argument('--matched-control-dir', type=Path)
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
    if args.roi_jitter != bool(args.matched_control_dir) or (args.roi_jitter and not args.research_source_dir):
        parser.error('ROI jitter requires a trained history source and a matched control directory together')
    if (args.epochs < 1 or args.patience_epochs < 0 or not math.isfinite(args.lr_multiplier)
            or not 0 < args.lr_multiplier < 1 or not math.isfinite(args.boundary_weight) or args.boundary_weight <= 0
            or not math.isfinite(args.boundary_clip_mm) or args.boundary_clip_mm <= 0):
        parser.error('Invalid research budget/objective')
    for key in ('foundation_run_dir', 'run_dir', 'source_root', 'gpu_record', 'research_source_dir', 'matched_control_dir'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).expanduser().resolve())
    old, _ = inspect_foundation(args.foundation_run_dir, args.source_root, args.verified_cache)
    control = data.read(Path(old['cohort_run_dir']) / 'config.json')
    protected = [args.foundation_run_dir, Path(old['sam_dir']).resolve(), Path(old['cohort_run_dir']).resolve(),
                 Path(control['data_dir']).resolve(), Path(control['source_root']).resolve()]
    if args.source_root:
        protected.append(args.source_root)
    if args.research_source_dir:
        protected.append(args.research_source_dir)
    if args.matched_control_dir:
        protected.append(args.matched_control_dir)
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
