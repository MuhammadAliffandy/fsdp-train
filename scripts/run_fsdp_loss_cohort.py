#!/usr/bin/env python3
"""Run matched control/balanced full-cohort arms sequentially, with strict resume."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import physical_fsdp as data
from evaluate_private_validation import file_hash
from fsdp_loss_cohort import diagnostic
from run_fsdp_research import run_child, exit_details
from train_fsdp_foundation import inspect_foundation
from train_fsdp_research import code_signature


def execute(args):
    proof = diagnostic(args.diagnostic_dir, args.research_source_dir)
    old, _ = inspect_foundation(args.foundation_run_dir, args.source_root, verified_cache=True)
    config = {'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'},
              'diagnostic': proof, 'code_sha256': code_signature(), 'arms': ['control', 'balanced'],
              'interpretation': 'Matched new recipe; only detail BCE differs between arms. Both begin at the full-cohort best.'}
    protected = [args.foundation_run_dir, args.research_source_dir, args.diagnostic_dir, Path(old['sam_dir']), Path(old['cohort_run_dir'])]
    cohort = data.read(Path(old['cohort_run_dir']) / 'config.json')
    protected.extend(Path(p).resolve() for p in (cohort['data_dir'], cohort['source_root'], args.source_root) if p)
    if any(args.run_dir.is_relative_to(p.resolve()) for p in protected):
        raise ValueError('Outputs must be outside source/model/data directories')
    ancestor = args.run_dir.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < max(12 * 2**30, 8 * (args.research_source_dir / 'research_best.pth').stat().st_size):
        raise ValueError('Insufficient output filesystem space for full training; keep at least12GiBfree')
    if args.resume:
        if (data.read(args.run_dir / 'workflow_config.json') != config or (args.run_dir / 'summary.json').exists()):
            raise ValueError('Resume requires the unchanged unfinished cohort workflow')
    else:
        args.run_dir.mkdir(parents=True, exist_ok=False)
        data.save(args.run_dir / 'workflow_config.json', config)
    with (args.run_dir / 'workflow.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('A cohort parent is already writing this workflow')
        record = args.run_dir / 'active_child.json'
        if record.exists():
            previous = data.read(record)
            result = subprocess.run(['ps', '-p', str(previous['pid']), '-o', 'args='], capture_output=True, text=True)
            if result.returncode == 0 and str(args.run_dir) in result.stdout:
                raise ValueError('An existing cohort child is still active; refusing duplicate training')
        entries = []
        for arm in config['arms']:
            if code_signature() != config['code_sha256']:
                raise ValueError('Training code changed between cohort arms')
            if diagnostic(args.diagnostic_dir, args.research_source_dir) != proof:
                raise ValueError('Diagnostic/source proof changed between arms')
            child = args.run_dir / arm
            data.save(args.run_dir / 'progress.json', {'complete': False, 'active_arm': arm, 'entries': entries,
                                                     'test_images_read': False})
            command = [sys.executable, '-u', str(Path(__file__).with_name('train_fsdp_loss_cohort.py')),
                '--foundation-run-dir', str(args.foundation_run_dir), '--research-source-dir', str(args.research_source_dir),
                '--diagnostic-dir', str(args.diagnostic_dir), '--run-dir', str(child), '--variant', 'history',
                '--loss-arm', arm, '--epochs', str(args.epochs), '--lr-multiplier', str(args.lr_multiplier),
                '--patience-epochs', str(args.patience_epochs), '--cohort-accumulation', str(args.accumulation),
                '--verified-cache', '--device', args.device]
            if args.source_root:
                command.extend(['--source-root', str(args.source_root)])
            if args.resume and (child / 'summary.json').exists():
                summary = data.read(child / 'summary.json')
            else:
                if args.resume and child.exists():
                    if not (child / 'research_latest.pth').exists():
                        raise ValueError(f'{arm} has no recoverable checkpoint; inspect failure before a new run')
                    command.append('--resume')
                print(f'Starting FULL54train/10val: {arm}; outputs={child}', flush=True)
                status = run_child(command, record)
                data.save(args.run_dir / f'{arm}_exit.json', exit_details(status))
                if status != 0 or not (child / 'summary.json').exists():
                    raise ValueError(f'{arm} failed returncode={status}; later arm not launched; inspect {child}/failure.json')
                summary = data.read(child / 'summary.json')
            if (not summary['complete'] or summary['test_images_read'] or summary['loss_arm'] != arm
                    or summary['train_patients'] != 54 or summary['validation_patients'] != 10
                    or file_hash(child / 'research_best.pth') != summary['best_checkpoint_sha256']):
                raise ValueError('Completed cohort arm or checkpoint is inconsistent')
            entries.append({'arm': arm, 'directory': str(child), 'summary_sha256': file_hash(child / 'summary.json'),
                'checkpoint_sha256': summary['best_checkpoint_sha256'], 'completed_epochs': summary['completed_epochs'],
                'best_epoch': summary['best_epoch'], 'best_validation_metrics': summary['best_validation_metrics'],
                'best_frozen_training_metrics': summary['best_frozen_training_metrics'],
                'validation_change_vs_source': summary['validation_change_vs_research_source']})
        control = entries[0]['best_validation_metrics']['5']['mean_bilateral_patient_dice']
        for entry in entries:
            entry['validation_delta_vs_matched_control'] = entry['best_validation_metrics']['5']['mean_bilateral_patient_dice'] - control
        selected = max(entries, key=lambda entry: entry['best_validation_metrics']['5']['mean_bilateral_patient_dice'])
        report = {'complete': True, 'entries': entries, 'selected_arm': selected['arm'], 'selected_directory': selected['directory'],
                  'test_images_read': False, 'interpretation': 'Exploratory GT-assisted validation selection; inherited epoch0 included; '
                  'not85%generalization, test evaluation or automatic deployment.', 'archive': str(args.run_dir / 'results.tar.gz')}
        data.save(args.run_dir / 'report.json', report)
        with tarfile.open(args.run_dir / 'results.tar.gz', 'w:gz') as archive:
            archive.add(args.run_dir / 'report.json', arcname='cohort/report.json')
            archive.add(args.run_dir / 'workflow_config.json', arcname='cohort/workflow_config.json')
            for entry in entries:
                archive.add(Path(entry['directory']) / 'results.tar.gz', arcname=f'cohort/{entry["arm"]}/results.tar.gz')
        data.save(args.run_dir / 'summary.json', report)
        data.save(args.run_dir / 'progress.json', report)
        print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('foundation-run-dir', 'research-source-dir', 'diagnostic-dir', 'run-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr-multiplier', type=float, default=.3)
    parser.add_argument('--patience-epochs', type=int, default=30)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.patience_epochs < 1 or args.accumulation < 1 or not 0 < args.lr_multiplier < 1:
        parser.error('Invalid cohort budget, accumulation or LR multiplier')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    existed_before = args.run_dir.exists()
    try:
        execute(args)
    except Exception as error:
        if args.run_dir.is_dir() and not existed_before:
            data.save(args.run_dir / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
