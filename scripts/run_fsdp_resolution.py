#!/usr/bin/env python3
"""Audit four inherited FSDP resolution variants; optionally fine-tune after all pass."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import tarfile

import physical_fsdp as data
from evaluate_private_validation import file_hash
from sam_refine_resolution import VARIANTS
from train_fsdp_foundation import inspect_foundation
from audit_foundation_sources import inventory, problem_message, bind_sources


def run_child(command):
    child = subprocess.Popen(command)
    def forward(signum, frame):
        if child.poll() is None:
            child.send_signal(signum)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print(f'Child PID{child.pid}has not exited; no further stage will start', flush=True)
        raise SystemExit(128 + signum)
    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        print(f'Active child PID{child.pid}; stop signals to workflow are forwarded', flush=True)
        return subprocess.CompletedProcess(command, child.wait())
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def execute(args):
    source_root = getattr(args, 'source_root', None)
    sources = inventory(bind_sources(data.read(args.foundation_run_dir / 'source_provenance.json'), source_root))
    sources['source_root_override'] = str(source_root) if source_root else None
    data.save(args.run_dir / 'source_precheck.json', sources)
    if not sources['ready']:
        raise ValueError(problem_message(sources))
    inspect_foundation(args.foundation_run_dir, source_root)
    code = Path(__file__).resolve().parent
    tracked = ('train_sam_refine32.py', 'train_fsdp_foundation.py', 'sam_refine32.py', 'sam_refine_data.py',
               'sam_refine_resolution.py', 'fsdp_resolution.py', 'physical_fsdp.py', 'multiclass_model.py',
               'run_fsdp_resolution.py', 'audit_foundation_sources.py')
    signature = {name: file_hash(code / name) for name in tracked}
    data.save(args.run_dir / 'workflow_code_sha256.json', signature)
    entries = []
    for stage in ('audit', 'train') if args.train else ('audit',):
        for variant in VARIANTS:
            if any(file_hash(code / name) != digest for name, digest in signature.items()):
                raise ValueError('Workflow source changed between variants; refusing to mix implementations')
            child = args.run_dir / stage / variant
            child.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-u', str(code / 'train_fsdp_foundation.py'),
                '--foundation-run-dir', str(args.foundation_run_dir), '--run-dir', str(child),
                '--resolution-variant', variant, '--epochs', str(args.epochs), '--lr-multiplier', str(args.lr_multiplier),
                '--patience-epochs', str(args.patience_epochs), '--device', args.device]
            if source_root:
                command.extend(['--source-root', str(source_root)])
            if stage == 'audit':
                command.append('--audit-only')
            print(f'Starting {stage}: {variant}; outputs={child}', flush=True)
            result = run_child(command)
            entry = {'stage': stage, 'variant': variant, 'directory': str(child), 'returncode': result.returncode}
            summary_path = child / 'summary.json'
            if result.returncode == 0 and summary_path.exists():
                summary = data.read(summary_path)
                entry.update(complete=summary.get('complete', False), summary_sha256=file_hash(summary_path))
                if stage == 'audit':
                    if (not summary['foundation_reproduced_before_adaptation'] or not summary['gradient_probe']['passed']
                            or summary['optimizer_updates'] != 0):
                        entry['complete'] = False
                    entry.update(initial_val=summary['candidate_val'], probe=summary['gradient_probe'],
                        fixed_0_5=summary.get('candidate_validation_fixed_0_5'),
                        delta_before_training=summary['candidate_validation_delta_before_training'])
                else:
                    entry.update(best_epoch=summary['best_epoch'], best_val=summary['best_validation_metrics'],
                        delta_vs_foundation=summary['validation_change_vs_foundation'],
                        checkpoint_sha256=summary['best_checkpoint_sha256'])
            else:
                entry['complete'] = False
                if (child / 'failure.json').exists():
                    entry['failure'] = data.read(child / 'failure.json')
            entries.append(entry)
            data.save(args.run_dir / 'progress.json', {'complete': False, 'entries': entries,
                'current_stage': stage, 'test_images_read': False})
        if any(not r['complete'] for r in entries if r['stage'] == stage):
            raise ValueError(f'{stage}incomplete; no subsequent stage launched; inspect progress.json and child failures')
    if args.train:
        baseline = next(r['best_val']['5']['mean_bilateral_patient_dice'] for r in entries
                        if r['stage'] == 'train' and r['variant'] == 'baseline')
        for entry in entries:
            if entry['stage'] == 'train':
                entry['delta_vs_finetuned_baseline'] = entry['best_val']['5']['mean_bilateral_patient_dice'] - baseline
    report = {'complete': True, 'training_requested': args.train, 'entries': entries,
        'foundation': str(args.foundation_run_dir), 'test_images_read': False, 'validation_images_read': True,
        'interpretation': 'Matched binary interactive FSDP variants from one trained checkpoint; audit has no optimizer updates. '
            'Resolution is not selected automatically. Validation scores are GT-assisted;20test patients reserved.'}
    archive = args.run_dir / 'results.tar.gz'
    report['archive'] = str(archive)
    data.save(args.run_dir / 'report.json', report)
    with tarfile.open(archive, 'w:gz') as package:
        for name in ('report.json', 'workflow_config.json', 'workflow_code_sha256.json'):
            if (args.run_dir / name).exists():
                package.add(args.run_dir / name, arcname='results/' + ('summary.json' if name == 'report.json' else name))
        for entry in entries:
            folder = Path(entry['directory'])
            names = ('summary.json', 'config.json', 'resolution_probe.json', 'foundation_reproduction.json',
                     'weight_preservation.json', 'train_5_sides.csv', 'val_5_sides.csv', 'overlays') if entry['stage'] == 'audit' else ('results.tar.gz',)
            for name in names:
                if (folder / name).exists():
                    package.add(folder / name, arcname=f'results/{entry["stage"]}/{entry["variant"]}/{name}')
    data.save(args.run_dir / 'summary.json', report)
    data.save(args.run_dir / 'progress.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foundation-run-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr-multiplier', type=float, default=.2)
    parser.add_argument('--patience-epochs', type=int, default=50)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.foundation_run_dir = args.foundation_run_dir.expanduser().resolve()
    if args.source_root:
        args.source_root = args.source_root.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    source = data.read(args.foundation_run_dir / 'config.json')
    control = data.read(Path(source['cohort_run_dir']) / 'config.json')
    protected = (args.foundation_run_dir, Path(source['sam_dir']).resolve(), Path(source['cohort_run_dir']).resolve(),
                 Path(control['source_root']).resolve(), Path(control['data_dir']).resolve())
    if args.source_root:
        protected += (args.source_root,)
    if args.run_dir.exists() or any(args.run_dir.is_relative_to(path) for path in protected):
        parser.error('Choose a new output directory outside the successful foundation')
    args.run_dir.mkdir(parents=True)
    data.save(args.run_dir / 'workflow_config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    try:
        execute(args)
    except Exception as error:
        data.save(args.run_dir / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
