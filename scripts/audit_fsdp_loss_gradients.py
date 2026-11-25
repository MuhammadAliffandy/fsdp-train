#!/usr/bin/env python3
"""Audit a frozen snapshot of a history checkpoint on train ROIs, without optimizer updates."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch

import physical_fsdp as data
import train_fsdp_foundation as foundation
import train_fsdp_research as trainer
from evaluate_private_validation import file_hash
from fsdp_loss_diagnostics import RecordedLoss, inspect_forward, virtual_adamw
from fsdp_research_model import ResearchModel, run_sample
from fsdp_verified_cache import validate_targets
from train_private_unet import state_hash


def snapshot(source, output):
    with source.open('rb') as src, output.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    return file_hash(output)


def plan():
    return {'status': 'proposals_only; no training automatically launched',
        'shared': 'same frozen trained source, split, physical ROI, five clicks, seed, LR/schedule, budget and native evaluation',
        'arms': [
            {'name': 'control', 'change': 'none; repeat original recipe at the matched new budget'},
            {'name': 'loss', 'change': 'only moderate capped normalized foreground BCE weighting; original Dice and auxiliary'},
            {'name': 'auxiliary', 'change': 'only context coefficient0.25to0.1; original Dice/BCE'},
            {'name': 'augmentation', 'change': 'only paired physical translation; existing jitter arm available if source/budget match'}],
        'follow_up': 'Confirm promising arms with another seed before combining; no simultaneous LR/loss/architecture changes.',
        'train_target': '80–85% is an objective, not a promised result; tiny memorization must be distinguished from validation.',
        'augmentation': 'On-the-fly variants reuse the same patients; no extra independent patients or new validation evidence. '
                        'No default sharpening, elastic distortion or label changes.',
        'decision': 'Use measurements and overlays to choose the next arm; do not auto-promote based on gradient size.'}


def load_source(args):
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    config = data.read(run / 'config.json')
    if config['variant'] != 'history' or config['test_images_read'] or config['parameters']['auxiliary_weight'] != .25:
        raise ValueError('Expected the original history recipe with auxiliary0.25')
    protected = [run, Path(config['parameters']['foundation_run_dir']).resolve()]
    protected.extend(Path(config['parameters'][key]).resolve() for key in ('sam_dir', 'checkpoint', 'cohort_run_dir'))
    cohort_config = data.read(Path(config['parameters']['cohort_run_dir']) / 'config.json')
    protected.append(Path(cohort_config['data_dir']).resolve())
    for path in (args.source_root, config['parameters'].get('source_root'), cohort_config.get('source_root')):
        if path:
            protected.append(Path(path).expanduser().resolve())
    if any(output.is_relative_to(path) for path in protected):
        raise ValueError('Audit output must be outside source runs/data/model paths')
    output.mkdir(parents=True, exist_ok=False)
    data.save(output / 'source_config.json', config)
    checksum = snapshot(run / 'research_best.pth', output / 'checkpoint_snapshot.pth')
    saved = torch.load(output / 'checkpoint_snapshot.pth', map_location='cpu', weights_only=False)
    if saved['config'] != config:
        raise ValueError('Snapshot checkpoint and source config differ')
    computational = (*trainer.base.CODE_FILES, 'fsdp_research_model.py', 'fsdp_research_metrics.py')
    if any(file_hash(Path(__file__).with_name(name)) != config['code_sha256'][name] for name in computational):
        raise ValueError('Frozen source model/data implementation differs')
    if any(importlib.metadata.version(name) != version for name, version in config['versions'].items()):
        raise ValueError('Source package versions differ')
    frozen = SimpleNamespace(**config['parameters'])
    for key in ('foundation_run_dir', 'sam_dir', 'checkpoint', 'cohort_run_dir'):
        setattr(frozen, key, Path(getattr(frozen, key)).resolve())
    source_root = args.source_root or getattr(frozen, 'source_root', None)
    old, _ = foundation.inspect_foundation(frozen.foundation_run_dir, source_root, verified_cache=True)
    if old['cache_sha256'] != config['cache_sha256'] or old['physical_policy'] != config['physical_policy']:
        raise ValueError('Source cache/policy differs')
    prepared = torch.load(frozen.foundation_run_dir / 'hybrid_data.pt', map_location='cpu', weights_only=False)
    cohort = data.read(frozen.cohort_run_dir / 'config.json')
    validate_targets(prepared, data.read(Path(cohort['data_dir']) / 'manifest.json'))
    selected = [s for s in prepared['samples']['train'] if s['case'] in args.cases]
    if len(selected) != 2 * len(args.cases) or len(set(args.cases)) != len(args.cases):
        raise ValueError('Choose unique cached training aliases, each with both sides')
    device = torch.device(args.device)
    trainer.allocator_limit(device)
    torch.manual_seed(frozen.seed)
    original, _ = trainer.base.build(frozen)
    model = ResearchModel.from_foundation(original, 'history').to(device)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    del original, saved['model_state_dict']
    model.amp_dtype = torch.bfloat16 if frozen.precision == 'bf16' else torch.float16
    trainer.base.set_stage(model, frozen.source_best_epoch, 0)
    return model, prepared, frozen, saved, checksum, selected


def execute(args):
    model, prepared, frozen, saved, checksum, selected = load_source(args)
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    device = torch.device(args.device)
    before = state_hash(model.state_dict())
    rows = []
    for index, sample in enumerate(selected):
        criterion = RecordedLoss()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        loss, outputs = run_sample(model, sample, device, frozen.seed + index, criterion, augment=False)
        report, names, candidates = inspect_forward(model, criterion)
        for key, gradients in candidates.items():
            report['candidate_gradients'][key]['hypothetical_adamw_update'] = virtual_adamw(
                model, names, gradients, saved['optimizer_state_dict'])
        report.update(case=sample['case'], side=sample['side'], name=sample['name'],
                      peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None)
        rows.append(report)
        data.save(output / 'progress.json', {'complete': False, 'completed_rois': len(rows), 'total_rois': len(selected), 'rows': rows})
        print(f'{sample["name"]}: loss/gradient audit complete ({len(rows)}/{len(selected)})', flush=True)
        del outputs, loss, criterion, candidates
        model.zero_grad(set_to_none=True)
    if state_hash(model.state_dict()) != before:
        raise ValueError('Read-only gradient audit changed the model state')
    summary = {'complete': True, 'source_run': str(run), 'checkpoint_snapshot_sha256': checksum,
        'checkpoint_epoch': saved['epoch'], 'checkpoint_best_score': saved['best_score'],
        'optimizer_updates': 0, 'model_state_unchanged': True, 'rows': rows, 'test_images_read': False,
        'cached_validation_tensors_deserialized': True, 'validation_inference_performed': False,
        'raw_validation_files_read': False, 'augmentation_used': False, 'oracle_clicks_per_side': 5,
        'precision': frozen.precision, 'model_mode': 'same partial train/eval stage as cohort training',
        'update_scope': 'Hypothetical one-ROI AdamW step using saved optimizer moments and next LR; '
                        'not the actual accumulated eight-ROI cohort update; no parameters/state mutated.',
        'limitations': 'Selected training ROIs and one checkpoint; gradient size/cosine alone do not establish '
                       'why validation stalls or predict which loss improves it.',
        'code_sha256': {name: file_hash(Path(__file__).with_name(name)) for name in
                        ('audit_fsdp_loss_gradients.py', 'fsdp_loss_diagnostics.py')},
        'experiments': plan()}
    data.save(output / 'experiment_plan.json', plan())
    data.save(output / 'summary.json', summary)
    print(f'Report: {output / "summary.json"}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--cases', nargs='+', default=['train_01', 'train_02', 'train_53'])
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    if not (args.run_dir / 'research_best.pth').exists() and (args.run_dir / 'history').is_dir():
        args.run_dir = (args.run_dir / 'history').resolve()
    if args.output_dir is None:
        args.output_dir = Path('outputs/private/fsdp_loss_audit') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    existed_before = args.output_dir.exists()
    try:
        execute(args)
    except Exception as error:
        if not existed_before and args.output_dir.is_dir():
            data.save(args.output_dir / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
