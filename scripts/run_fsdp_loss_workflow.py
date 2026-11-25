#!/usr/bin/env python3
"""Snapshot once, audit gradients, then compare three independent two-patient fits."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import physical_fsdp as data
from audit_fsdp_loss_gradients import snapshot
from evaluate_private_validation import file_hash
from run_fsdp_research import run_child


def execute(args):
    source, output = args.run_dir.resolve(), args.output_dir.resolve()
    if not (source / 'research_best.pth').exists() and (source / 'history').is_dir():
        source = (source / 'history').resolve()
    if output.is_relative_to(source):
        raise ValueError('Workflow output must be outside its source')
    config = data.read(source / 'config.json')
    protected = [Path(config['parameters'][key]).resolve() for key in ('foundation_run_dir', 'sam_dir', 'cohort_run_dir')]
    cohort = data.read(Path(config['parameters']['cohort_run_dir']) / 'config.json')
    protected.extend(Path(p).expanduser().resolve() for p in (cohort['data_dir'], cohort['source_root'], args.source_root) if p)
    if any(output.is_relative_to(p) for p in protected):
        raise ValueError('Workflow output must be outside dataset/model/source paths')
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    available = shutil.disk_usage(ancestor).free
    required = max(10 * 2**30, 10 * (source / 'research_best.pth').stat().st_size)
    if available < required:
        raise ValueError(f'Insufficient output filesystem space: free={available / 2**30:.1f}GiB; required={required / 2**30:.1f}GiB')
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / 'frozen_source'; frozen.mkdir()
    data.save(frozen / 'config.json', config)
    checksum = snapshot(source / 'research_best.pth', frozen / 'research_best.pth')
    config_checksum = file_hash(frozen / 'config.json')
    code = {name: file_hash(Path(__file__).with_name(name)) for name in
            ('run_fsdp_loss_workflow.py', 'audit_fsdp_loss_gradients.py', 'fsdp_loss_diagnostics.py',
             'train_fsdp_loss_tiny.py', 'train_fsdp_research.py', 'run_fsdp_research.py', 'fsdp_research_model.py')}
    data.save(output / 'workflow_config.json', {'source': str(source), 'source_sha256': checksum,
        'code_sha256': code, 'source_config_sha256': config_checksum,
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'disk_free_at_start_gib': available / 2**30, 'disk_required_gib': required / 2**30,
        'interpretation': 'Independent identical-source tiny fits; no automatic full-cohort promotion'})
    stages = [('audit', 'audit_fsdp_loss_gradients.py', []),
              *[(f'tiny_{arm}', 'train_fsdp_loss_tiny.py', ['--loss', arm, '--epochs', str(args.epochs), '--target', '.85'])
                for arm in ('control', 'balanced', 'auxiliary')]]
    entries = []
    for name, script, extra in stages:
        if any(file_hash(Path(__file__).with_name(key)) != value for key, value in code.items()):
            raise ValueError('Diagnostic code changed between stages')
        if file_hash(frozen / 'config.json') != config_checksum:
            raise ValueError('Frozen source config changed')
        if file_hash(frozen / 'research_best.pth') != checksum:
            raise ValueError('Frozen source checkpoint changed')
        child = output / name
        command = [sys.executable, '-u', str(Path(__file__).with_name(script)), '--run-dir', str(frozen),
                   '--output-dir', str(child), '--device', args.device, *extra]
        if args.source_root:
            command.extend(['--source-root', str(args.source_root)])
        data.save(output / 'progress.json', {'complete': False, 'stage': name, 'entries': entries})
        status = run_child(command, output / 'active_child.json')
        if status != 0 or not (child / 'summary.json').exists():
            raise ValueError(f'{name} failed with returncode={status}; later stages not launched; inspect {child}/failure.json')
        report = data.read(child / 'summary.json')
        if not report['complete'] or report['test_images_read'] or report['validation_inference_performed']:
            raise ValueError('Diagnostic stage has inconsistent completion/split scope')
        entry = {'stage': name, 'directory': str(child)}
        if name != 'audit':
            entry.update(best_epoch=report['best_epoch'], completed_epochs=report['completed_epochs'],
                         diagnostic_passed=report['diagnostic_passed'], best_training_metrics=report['best_training_metrics'])
        entries.append(entry)
    report = {'complete': True, 'entries': entries, 'source_checkpoint_sha256': checksum,
        'target_training_dice': .85, 'test_images_read': False, 'validation_inference_performed': False,
        'interpretation': 'Gradient audit and three two-patient memorization diagnostics, not validation80–85%. '
                          'Augmentation is off during tiny diagnosis. No cohort run automatically launched.'}
    data.save(output / 'summary.json', report)
    data.save(output / 'progress.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('Epoch cap must be positive')
    existed_before = args.output_dir.exists()
    try:
        execute(args)
    except Exception as error:
        if not existed_before and args.output_dir.is_dir():
            data.save(args.output_dir / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
