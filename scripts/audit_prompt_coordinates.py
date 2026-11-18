#!/usr/bin/env python3
"""Check positional encoding and compare two prompt orders on validation only."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_private_validation import file_hash, infer_clicks, overlay, preprocess, save_crop, scores, write_csv


def position_contract(encoder):
    dense = encoder.get_dense_pe()
    shape = dense.shape[-3:]
    size = encoder.input_image_size
    cells = [(0, shape[1] - 1, shape[2] // 2),
             (shape[0] // 3, 2 * shape[1] // 3, shape[2] // 3),
             (shape[0] - 1, 0, shape[2] - 1)]
    centers = torch.tensor([[(cell[axis] + .5) * size[axis] / shape[axis] - .5 for axis in range(3)]
                            for cell in cells], device=dense.device)
    expected = torch.stack([dense[0, :, x, y, z] for x, y, z in cells])
    result = {}
    for name, order in [('native', [0, 1, 2]), ('swap_xy', [1, 0, 2])]:
        encoded = encoder.pe_layer.forward_with_coords(centers[None, :, order] + .5, size)[0]
        error = (expected - encoded).abs()
        result[name] = {'mean_absolute_error': float(error.mean()), 'max_absolute_error': float(error.max()),
                        'mean_cosine': float(F.cosine_similarity(expected, encoded, dim=-1).mean()),
                        'aligned_at_cell_centers': bool(error.max() < 1e-5)}
    result['scope'] = 'Positional features at non-diagonal embedding-cell centers; this alone does not establish checkpoint performance'
    result['input_image_size'] = list(size)
    result['embedding_grid'] = list(shape)
    return result


class PromptOrder:
    def __init__(self, encoder, order):
        self.encoder, self.order = encoder, order
        self.trace = []

    def get_dense_pe(self):
        return self.encoder.get_dense_pe()

    def __call__(self, *, points, boxes, masks):
        if points is not None:
            coordinates, labels = points
            points = (coordinates[..., self.order], labels)
            self.trace.append({'voxel_coordinates': coordinates.detach().cpu().tolist(),
                               'encoder_coordinates': points[0].detach().cpu().tolist(),
                               'labels': labels.detach().cpu().tolist()})
        return self.encoder(points=points, boxes=boxes, masks=masks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sam-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    run = args.run_dir.resolve() if args.run_dir else None
    if run is None:
        candidates = list((project / 'runs').glob('fsdp_private_v2_*/summary.json'))
        if not candidates:
            parser.error('No completed v2 run found')
        run = max(candidates, key=lambda path: path.stat().st_mtime).parent
    data = args.data_dir.resolve()
    manifest_file = data / 'manifest.json'
    manifest = json.loads(manifest_file.read_text())
    groups = manifest['splits']
    ids = sorted(groups['val'])
    flat = sum([groups[key] for key in ('train', 'val', 'test')], [])
    if len(ids) != 10 or len(groups['test']) != 20 or len(set(flat)) != len(flat):
        parser.error('Expected a disjoint frozen split with 10 val and 20 test patients')
    for folder in ('imagesTr', 'labelsTr'):
        if {p.name for p in (data / 'val/ct_private' / folder).glob('*.nii.gz')} != {key + '.nii.gz' for key in ids}:
            parser.error('Validation files differ from the frozen manifest')
    if args.output_dir.exists():
        parser.error('Output exists')
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    checkpoint = run / 'sam_model_val_best.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['config']['manifest_sha256'] != file_hash(manifest_file):
        parser.error('Checkpoint and frozen dataset differ')
    if [Path(path).resolve() for path in state['used_datas']] != [data / 'train/ct_private']:
        parser.error('Checkpoint training source differs')
    seed, epoch = state['config']['seed'], state['epoch']
    prior = json.loads((run / 'summary.json').read_text())
    if epoch != prior['best_epoch']:
        parser.error('Best checkpoint and summary disagree')
    sys.path.insert(0, str(args.sam_dir.resolve()))
    from segment_anything.build_sam3D import sam_model_registry3D
    model = sam_model_registry3D['vit_b_ori'](checkpoint=None).to(device).eval()
    model.load_state_dict(state['model_state_dict'], strict=True)
    del state
    with torch.inference_mode():
        contract = position_contract(model.prompt_encoder)
    rows, paired, traces = [], [], {}
    for index, key in enumerate(ids, 1):
        alias = f'val_{index:02d}'
        image, target, raw, metadata, valid = preprocess(data / 'val/ct_private/imagesTr' / f'{key}.nii.gz',
                                                        data / 'val/ct_private/labelsTr' / f'{key}.nii.gz')
        gt = target[0, 0].numpy().astype(bool)
        traces[alias] = {}
        for name, order in [('native', [0, 1, 2]), ('swap_xy', [1, 0, 2])]:
            proxy = SimpleNamespace(image_encoder=model.image_encoder, mask_decoder=model.mask_decoder,
                                    prompt_encoder=PromptOrder(model.prompt_encoder, order))
            predictions = infer_clicks(proxy, image.to(device), target.to(device), [1, 5],
                                       seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
            traces[alias][name] = proxy.prompt_encoder.trace
            for clicks, logits in predictions.items():
                rows.append({'case': alias, 'order': name, 'clicks': clicks,
                             **scores(logits > 0, gt, metadata['full_target_voxels'], valid)})
            folder = output / name
            folder.mkdir(exist_ok=True)
            prediction = predictions[5] > 0
            overlay(folder / f'{name}_{alias}_overlay.png', raw, gt, prediction, metadata['target_retention'])
            save_crop(folder / f'{alias}_prediction.nii.gz', prediction.astype(np.uint8), metadata)
        for clicks in (1, 5):
            selected = [row for row in rows if row['case'] == alias and row['clicks'] == clicks]
            native = next(row['dice_full_target'] for row in selected if row['order'] == 'native')
            swapped = next(row['dice_full_target'] for row in selected if row['order'] == 'swap_xy')
            paired.append({'case': alias, 'clicks': clicks, 'native_dice': native, 'swap_xy_dice': swapped, 'difference': swapped - native})
        write_csv(output / 'cases.csv', rows)
        write_csv(output / 'paired_changes.csv', paired)
        (output / 'prompt_traces.json').write_text(json.dumps(traces, indent=2))
        print(f'{alias}: both prompt orders complete', flush=True)
    comparison = []
    for order in ('native', 'swap_xy'):
        for clicks in (1, 5):
            selected = [row for row in rows if row['order'] == order and row['clicks'] == clicks]
            comparison.append({'order': order, 'clicks': clicks, 'cases': len(selected),
                               'mean_dice': float(np.mean([row['dice_full_target'] for row in selected])),
                               'mean_precision': float(np.mean([row['precision'] for row in selected])),
                               'mean_recall': float(np.mean([row['recall_full_target'] for row in selected]))})
    reproduced = all(abs(row['mean_dice'] - prior['best_validation_dice'][str(row['clicks'])]) < 1e-4
                     for row in comparison if row['order'] == 'native')
    summary = {'run': run.name, 'checkpoint_epoch': epoch, 'checkpoint_sha256': file_hash(checkpoint),
               'audit_script_sha256': file_hash(Path(__file__)),
               'manifest_sha256': file_hash(manifest_file), 'position_encoding_contract': contract,
               'comparison': comparison, 'native_reproduces_saved_validation': reproduced, 'test_images_read': False,
               'protocol': 'same checkpoint, CT, GT crop, threshold and oracle-click budget; only coordinates entering prompt encoder are reordered',
               'limitation': 'An encoding inconsistency is not by itself proof of the cause; pretrained weights may have learned with the upstream convention'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2), flush=True)
    if not reproduced:
        raise SystemExit('Native control did not reproduce the prior result; do not attribute changes to coordinate order yet')


if __name__ == '__main__':
    main()
