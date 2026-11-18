#!/usr/bin/env python3
"""Compare three frozen SAM-Med3D checkpoints on the same ten side-ROI validation patients."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from evaluate_private_validation import file_hash, write_csv
from train_private_tiny import evaluate, prepare_samples, select_patients


def checkpoint_sources(args, manifest, manifest_hash):
    tiny = args.tiny_run_dir
    tiny_summary = json.loads((tiny / 'summary.json').read_text())
    if tiny_summary['train_patients'] != 2 or tiny_summary['side_rois'] != 4 or tiny_summary['validation_images_read'] or tiny_summary['test_images_read']:
        raise ValueError('Tiny run is not the completed two-training-patient diagnostic')
    audit = None
    if args.v2_run_dir:
        v2 = args.v2_run_dir
    else:
        audit = json.loads(args.v2_audit_summary.read_text())
        recorded_path = Path(audit['checkpoint']).resolve()
        if recorded_path.name != 'sam_model_val_best.pth':
            raise ValueError('Previous audit does not identify a v2 validation-best checkpoint')
        v2 = recorded_path.parent
    previous = json.loads((v2 / 'summary.json').read_text())
    stages = [('turbo_original', args.turbo_checkpoint), ('v2_best', v2 / 'sam_model_val_best.pth'),
              ('tiny_best', tiny / 'sam_model_tiny_best.pth')]
    if len({path.resolve() for _, path in stages}) != 3:
        raise ValueError('Three distinct checkpoint paths are required')
    turbo_hash = file_hash(args.turbo_checkpoint)
    provenance = json.loads((tiny / 'selected_training_patients.json').read_text())
    if len(provenance) != 2 or len({row['patient_id'] for row in provenance}) != 2:
        raise ValueError('Tiny training provenance must contain exactly two distinct patients')
    records = {row['patient_id']: row for row in manifest['cases']}
    for row in provenance:
        record = records[row['patient_id']]
        if record['split'] != 'train' or any(Path(row[field]).resolve() != Path(record[field]).resolve() for field in ('source_image', 'source_label')):
            raise ValueError('Tiny provenance source paths differ from the frozen training manifest')
    metadata = []
    for name, path in stages:
        digest = file_hash(path)
        state = torch.load(path, map_location='cpu', weights_only=False)
        if name != 'turbo_original':
            config = state['config']
            if config['manifest_sha256'] != manifest_hash or config['initial_checkpoint_sha256'] != turbo_hash:
                raise ValueError(f'{name}: frozen manifest or original turbo initialization differs')
            if config != json.loads((path.parent / 'config.json').read_text()):
                raise ValueError(f'{name}: checkpoint and saved run config differ')
            if name == 'tiny_best':
                if digest != tiny_summary['checkpoint_sha256'] or state['epoch'] != tiny_summary['best_epoch']:
                    raise ValueError('Tiny checkpoint differs from its completed best summary')
                expected_ids = set(select_patients(manifest, config['seed']))
                if {row['patient_id'] for row in provenance} != expected_ids:
                    raise ValueError('Tiny provenance differs from seeded training-only selection')
                if config['shared_code_sha256']['evaluate_private_validation.py'] != file_hash(Path(__file__).with_name('evaluate_private_validation.py')):
                    raise ValueError('Core preprocessing/inference changed since tiny training')
            else:
                if state['epoch'] != previous['best_epoch'] or previous['test_images_read']:
                    raise ValueError('V2 checkpoint differs from its completed run summary')
                if [Path(value).resolve() for value in state['used_datas']] != [args.data_dir / 'train/ct_private']:
                    raise ValueError('V2 training source differs from the frozen prepared training path')
                if audit and digest != audit['checkpoint_sha256']:
                    raise ValueError('V2 checkpoint differs from the earlier independent audit')
        metadata.append({'checkpoint': name, 'path': str(path.resolve()), 'sha256': digest,
                         'saved_epoch': int(state['epoch']) if 'epoch' in state else None})
        del state
    return metadata, tiny_summary


def paired_changes(rows):
    baseline = {(row['case'], row['clicks']): row for row in rows if row['checkpoint'] == 'turbo_original'}
    paired = []
    for row in rows:
        if row['checkpoint'] == 'turbo_original':
            continue
        reference = baseline[(row['case'], row['clicks'])]
        paired.append({'checkpoint': row['checkpoint'], 'case': row['case'], 'clicks_per_side': row['clicks'],
                       'total_bilateral_clicks': 2 * row['clicks'],
                       'dice_turbo': reference['dice_full_target'], 'dice_checkpoint': row['dice_full_target'],
                       'delta_dice_vs_turbo': row['dice_full_target'] - reference['dice_full_target']})
    return paired


def export_representatives(model, samples, patients, device, stage_dir, side_rows, patient_rows):
    ranked = sorted([row for row in patient_rows if row['clicks'] == 5], key=lambda row: (row['dice_full_target'], row['case']))
    aliases = {ranked[0]['case'], ranked[len(ranked) // 2]['case'], ranked[-1]['case']}
    subset = [sample for sample in samples if sample['case'] in aliases]
    _, exported_sides, exported_patients = evaluate(model, subset, {key: patients[key] for key in aliases}, device,
                                                   export=stage_dir / 'representatives', progress_label='export')
    for exported, original, fields in [(exported_sides, side_rows, ('case', 'side', 'clicks')),
                                       (exported_patients, patient_rows, ('case', 'clicks'))]:
        lookup = {tuple(row[key] for key in fields): row for row in original}
        for row in exported:
            expected = lookup[tuple(row[key] for key in fields)]['dice_full_target']
            if not np.isclose(row['dice_full_target'], expected, atol=1e-6, rtol=0):
                raise ValueError('Representative export does not reproduce the original frozen-model score')
    (stage_dir / 'representative_selection.json').write_text(json.dumps({'selection': 'worst, middle, best bilateral validation Dice @5 per side',
        'worst': ranked[0]['case'], 'middle': ranked[len(ranked) // 2]['case'], 'best': ranked[-1]['case']}, indent=2))


def execute(args):
    output, data = args.output_dir, args.data_dir
    manifest_file = data / 'manifest.json'
    manifest = json.loads(manifest_file.read_text())
    select_patients(manifest, args.seed)
    checkpoints, tiny_summary = checkpoint_sources(args, manifest, file_hash(manifest_file))
    protocol = {'manifest_sha256': file_hash(manifest_file), 'checkpoints': checkpoints,
                'script_sha256': file_hash(Path(__file__)),
                'shared_code_sha256': {name: file_hash(Path(__file__).with_name(name)) for name in
                    ('train_private_tiny.py', 'evaluate_private_validation.py', 'train_private_long.py', 'audit_expert_agreement.py')},
                'sam_builder_sha256': file_hash(args.sam_dir / 'segment_anything/build_sam3D.py'),
                'versions': {name: importlib.metadata.version(name) for name in ('torch', 'torchio', 'monai', 'numpy', 'SimpleITK')},
                'seed': args.seed, 'validation_patients': 10, 'side_rois': 20,
                'roi': 'native 128 cube centered separately on final RFSDP/LFSDP GT; require full target retention; no resampling',
                'prompts': 'native coordinates; same fixed per-side seed and oracle error-click algorithm for all checkpoints',
                'threshold': 'logits > 0 (probability > 0.5)', 'clicks_per_side': [1, 5],
                'total_bilateral_clicks': {'1': 2, '5': 10},
                'model_mode': 'eval with inference_mode; frozen parameters; CUDA autocast identical across checkpoints',
                'bilateral_scoring': 'restore both side predictions to original XYZ voxel grid, exclude padding and union once; entire final 1+9 target',
                'training_images_read': False, 'test_images_read': False,
                'interpretation': 'GT-assisted interactive validation; not automatic deployment performance. Compare these checkpoints within this protocol, not directly with old five-total-click validation.'}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    samples, patients = prepare_samples(args.source_root, data, manifest, output, args.seed, split='val')
    if len(samples) != 20 or len(patients) != 10:
        raise ValueError('Expected all ten validation patients and twenty nonempty complete side ROIs')
    sys.path.insert(0, str(args.sam_dir))
    from segment_anything.build_sam3D import sam_model_registry3D
    model = sam_model_registry3D['vit_b_ori'](checkpoint=None).to(device).eval().requires_grad_(False)
    all_sides, all_patients, comparison = [], [], []
    for index, entry in enumerate(checkpoints, 1):
        name, path = entry['checkpoint'], Path(entry['path'])
        if file_hash(path) != entry['sha256']:
            raise ValueError('Checkpoint changed after provenance checks')
        state = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(state.get('model_state_dict', state), strict=True)
        del state
        print(f'{name}: evaluating all ten validation patients', flush=True)
        means, sides, bilateral = evaluate(model, samples, patients, device, progress_label=name)
        stage_dir = output / name
        stage_dir.mkdir()
        write_csv(stage_dir / 'sides.csv', sides)
        write_csv(stage_dir / 'patients.csv', bilateral)
        for clicks in (1, 5):
            selected = [row for row in bilateral if row['clicks'] == clicks]
            side_selected = [row for row in sides if row['clicks'] == clicks]
            comparison.append({'checkpoint': name, 'saved_epoch': entry['saved_epoch'], 'clicks_per_side': clicks,
                'total_bilateral_clicks': 2 * clicks, 'validation_patients': len(selected), 'side_rois': len(side_selected),
                **means[str(clicks)], 'median_bilateral_patient_dice': float(np.median([row['dice_full_target'] for row in selected])),
                'mean_bilateral_precision': float(np.mean([row['precision'] for row in selected])),
                'mean_bilateral_recall': float(np.mean([row['recall_full_target'] for row in selected])),
                'minimum_target_retention': min(row['target_retention'] for row in side_selected)})
        all_sides.extend({'checkpoint': name, **row} for row in sides)
        all_patients.extend({'checkpoint': name, **row} for row in bilateral)
        write_csv(output / 'side_cases.csv', all_sides)
        write_csv(output / 'patient_cases.csv', all_patients)
        write_csv(output / 'comparison.csv', comparison)
        export_representatives(model, samples, patients, device, stage_dir, sides, bilateral)
        (output / 'progress.json').write_text(json.dumps({'checkpoints_completed': index, 'checkpoints_total': 3,
            'last_checkpoint': name, 'last_validation_dice': means}, indent=2))
        print(json.dumps({'checkpoint_completed': name, 'validation_dice': means}), flush=True)
    write_csv(output / 'paired_changes_vs_turbo.csv', paired_changes(all_patients))
    tiny_validation = next(row['mean_bilateral_patient_dice'] for row in comparison if row['checkpoint'] == 'tiny_best' and row['clicks_per_side'] == 5)
    summary = {'complete': True, 'validation_patients': 10, 'side_rois': 20, 'comparison': comparison,
               'checkpoints': checkpoints,
               'tiny_training_reference_bilateral_dice_5_per_side': tiny_summary['best_training_only_dice']['5']['mean_bilateral_patient_dice'],
               'tiny_train_validation_gap_5_per_side': tiny_summary['best_training_only_dice']['5']['mean_bilateral_patient_dice'] - tiny_validation,
               'training_reference_note': 'Training score copied from completed tiny summary; only validation images read in this comparison',
               'training_images_read': False, 'validation_images_read': True, 'test_images_read': False,
               'protocol': 'GT-assisted separate side ROIs and oracle prompts; bilateral @5 uses ten clicks total; original-grid bilateral scoring',
               'interpretation': 'Observed validation comparison only; no training or automatic checkpoint replacement performed',
               'artifacts': str(output)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sam-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--tiny-run-dir', type=Path, required=True)
    parser.add_argument('--turbo-checkpoint', type=Path, required=True)
    parser.add_argument('--v2-run-dir', type=Path)
    parser.add_argument('--v2-audit-summary', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if not args.v2_run_dir and not args.v2_audit_summary:
        parser.error('Supply the previous v2 run or its independent audit summary')
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    if args.output_dir.exists():
        parser.error('Output directory exists; choose a new comparison output')
    args.output_dir.mkdir(parents=True)
    try:
        execute(args)
    except Exception as error:
        failure = {'complete': False, 'error_type': type(error).__name__, 'error': str(error),
                   'interpretation': 'Comparison incomplete; do not interpret partial results as a complete ten-patient comparison'}
        (args.output_dir / 'summary.json').write_text(json.dumps(failure, indent=2))
        (args.output_dir / 'failure.json').write_text(json.dumps(failure, indent=2))
        raise


if __name__ == '__main__':
    main()
