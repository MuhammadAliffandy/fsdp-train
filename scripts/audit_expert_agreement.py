#!/usr/bin/env python3
"""Measure FSDP annotation agreement on the frozen training patients only."""

import argparse
from collections import defaultdict
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
import SimpleITK as sitk


SOURCES = ('labelsTr', 'labels_exp1', 'labels_exp2', 'labels_exp3')
SIDES = ('RFSDP', 'LFSDP', 'bilateral')


def index_files(folder):
    index = defaultdict(list)
    unrecognized = 0
    for path in sorted(folder.iterdir()):
        lower = path.name.lower()
        if not path.is_file() or not lower.endswith(('.nii', '.nii.gz')):
            continue
        stem = path.name[:-7] if lower.endswith('.nii.gz') else path.name[:-4]
        match = re.fullmatch(r'(\d+)(?:[-_].*)?', stem)
        if match:
            index[match.group(1)].append(path)
        else:
            unrecognized += 1
    return index, unrecognized


def header(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    if reader.GetDimension() != 3 or reader.GetNumberOfComponents() != 1:
        raise ValueError('Expected a scalar 3D mask')
    return reader


def same_geometry(a, b):
    return a.GetSize() == b.GetSize() and all(np.allclose(x, y, atol=tol, rtol=0) for x, y, tol in [
        (a.GetSpacing(), b.GetSpacing(), 1e-5), (a.GetOrigin(), b.GetOrigin(), 1e-4),
        (a.GetDirection(), b.GetDirection(), 1e-5)])


def agreement(a, b, spacing, direction):
    na, nb = int(a.sum()), int(b.sum())
    overlap = int(np.count_nonzero(a & b))
    distance = None
    if na and nb:
        ca, cb = np.argwhere(a).mean(axis=0)[::-1], np.argwhere(b).mean(axis=0)[::-1]
        distance = float(np.linalg.norm(np.asarray(direction).reshape(3, 3) @ ((cb - ca) * spacing)))
    return {'dice': 2 * overlap / (na + nb) if na + nb else None,
            'both_empty': not (na or nb), 'one_empty': bool(na) != bool(nb),
            'a_voxels': na, 'b_voxels': nb, 'overlap_voxels': overlap,
            'a_volume_mm3': na * float(np.prod(spacing)), 'b_volume_mm3': nb * float(np.prod(spacing)),
            'centroid_distance_mm': distance}


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def heatmap(path, rows, patients):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
    labels = ['Final ref', 'Expert 1', 'Expert 2', 'Expert 3']
    for axis, side in zip(axes, SIDES):
        matrix = np.full((4, 4), np.nan)
        for row in rows:
            if row['target'] != side:
                continue
            a, b = SOURCES.index(row['source_a']), SOURCES.index(row['source_b'])
            value = row['mean_dice_nonempty']
            matrix[a, b] = matrix[b, a] = value if value is not None else np.nan
            annotation = f"{value:.2f}\nn={row['nonempty_pairs']}" if value is not None else 'n=0'
            for x, y in ((b, a), (a, b)):
                axis.text(x, y, annotation, ha='center', va='center', fontsize=8,
                          color='white' if value is not None and value < .5 else 'black')
        plot = axis.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap='viridis')
        axis.set(title=side, xticks=range(4), yticks=range(4), xticklabels=labels, yticklabels=labels)
        axis.tick_params(axis='x', rotation=25)
    fig.colorbar(plot, ax=list(axes), label='Mean Dice; both-empty pairs excluded', shrink=.8)
    fig.suptitle(f'Annotation agreement: {patients} training patients; final reference shown separately')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def audit(root, manifest_path, output, confirmed=False, mapping_path=None):
    root, output = root.expanduser().resolve(), output.resolve()
    manifest = json.loads(manifest_path.read_text())
    groups = manifest['splits']
    ids = sorted(groups['train'])
    flat = sum([groups[name] for name in ('train', 'val', 'test')], [])
    if len(set(flat)) != len(flat) or len(ids) != 54 or len(groups['val']) != 10 or len(groups['test']) != 20:
        raise ValueError('Expected the frozen disjoint 54/10/20 patient split')
    if output.exists():
        raise ValueError('Output already exists')
    records = {row['patient_id']: row for row in manifest['cases']}
    mapping = {'labelsTr': {'RFSDP': 1, 'LFSDP': 9}}
    if mapping_path:
        supplied = json.loads(mapping_path.read_text())
        for source in SOURCES[1:]:
            values = supplied[source]
            right, left = values['RFSDP'], values['LFSDP']
            if any(type(value) is not int or value < 1 for value in (right, left)) or right == left:
                raise ValueError('Each expert needs distinct positive integer RFSDP/LFSDP IDs')
            mapping[source] = {'RFSDP': right, 'LFSDP': left}
    elif confirmed:
        mapping.update({source: {'RFSDP': 1, 'LFSDP': 9} for source in SOURCES[1:]})
    verified_mapping = len(mapping) == 4
    output.mkdir(parents=True)
    indices, unknown = {}, {}
    for source in SOURCES:
        if not (root / source).is_dir():
            raise ValueError(f'Missing annotation folder: {source}')
        indices[source], unknown[source] = index_files(root / source)
    rows, issues, inventory = [], [], []
    all_values = {source: set() for source in SOURCES}
    complete_cases = 0
    for index, key in enumerate(ids, 1):
        alias = f'train_{index:02d}'
        masks = {}
        try:
            reference_path = Path(records[key]['source_label']).resolve()
            image_path = Path(records[key]['source_image']).resolve()
            if not reference_path.is_relative_to(root / 'labelsTr') or not image_path.is_relative_to(root / 'imagesTr'):
                raise ValueError('Manifest source paths are outside the requested private root')
            expected = header(image_path)
            with (root / 'ColorTable' / f'{key}_ColorTable.csv').open(encoding='utf-8-sig', newline='') as stream:
                colors = {int(row['LabelValue']): row['Name'].strip().upper() for row in csv.DictReader(stream)}
            if colors.get(1) != 'RFSDP' or colors.get(9) != 'LFSDP':
                raise ValueError('Final-reference ColorTable does not confirm RFSDP=1/LFSDP=9')
        except Exception as error:
            issues.append({'case': alias, 'source': 'reference', 'reason': str(error)})
            print(f'{alias}: reference metadata failed', flush=True)
            continue
        for source in SOURCES:
            try:
                candidates = indices[source].get(key, [])
                if len(candidates) != 1:
                    raise ValueError(f'Expected one matching mask; found {len(candidates)}')
                path = candidates[0].resolve()
                if source == 'labelsTr' and path != reference_path:
                    raise ValueError('Reference file differs from frozen manifest')
                info = header(path)
                if not same_geometry(expected, info):
                    raise ValueError('Mask grid differs from the source CT; no resampling performed')
                image = sitk.ReadImage(str(path))
                array = sitk.GetArrayViewFromImage(image)
                integer_values = np.issubdtype(array.dtype, np.integer) or (np.isfinite(array).all() and np.equal(array, np.floor(array)).all())
                if not integer_values or (array < 0).any():
                    raise ValueError('Invalid segmentation label values')
                unique = sorted(int(value) for value in np.unique(array))
                all_values[source].update(unique)
                inventory.append({'case': alias, 'source': source, 'values': '|'.join(map(str, unique)),
                                  'shape_xyz': str(info.GetSize()), 'spacing_mm': str(info.GetSpacing())})
                if verified_mapping:
                    right = array == mapping[source]['RFSDP']
                    left = array == mapping[source]['LFSDP']
                    masks[source] = {'RFSDP': right, 'LFSDP': left, 'bilateral': right | left}
            except Exception as error:
                issues.append({'case': alias, 'source': source, 'reason': str(error)})
        if verified_mapping and len(masks) == 4:
            complete_cases += 1
            for a, b in combinations(SOURCES, 2):
                for side in SIDES:
                    rows.append({'case': alias, 'source_a': a, 'source_b': b, 'target': side,
                                 **agreement(masks[a][side], masks[b][side], np.asarray(expected.GetSpacing()), expected.GetDirection())})
        write_csv(output / 'inventory.csv', inventory)
        write_csv(output / 'case_agreement.csv', rows)
        (output / 'issues.json').write_text(json.dumps(issues, indent=2))
        print(f'{alias}: checked {index}/{len(ids)}', flush=True)
    aggregate = []
    for a, b in combinations(SOURCES, 2):
        for side in SIDES:
            selected = [row for row in rows if row['source_a'] == a and row['source_b'] == b and row['target'] == side]
            nonempty = [row['dice'] for row in selected if row['dice'] is not None]
            distances = [row['centroid_distance_mm'] for row in selected if row['centroid_distance_mm'] is not None]
            aggregate.append({'source_a': a, 'source_b': b, 'target': side, 'complete_cases': len(selected),
                              'nonempty_pairs': len(nonempty), 'both_empty_pairs': sum(row['both_empty'] for row in selected),
                              'one_empty_pairs': sum(row['one_empty'] for row in selected),
                              'mean_dice_nonempty': float(np.mean(nonempty)) if nonempty else None,
                              'median_dice_nonempty': float(np.median(nonempty)) if nonempty else None,
                              'minimum_dice_nonempty': min(nonempty) if nonempty else None,
                              'maximum_dice_nonempty': max(nonempty) if nonempty else None,
                              'mean_centroid_distance_mm': float(np.mean(distances)) if distances else None})
    complete = verified_mapping and complete_cases == len(ids) and not issues
    summary = {'mode': 'agreement' if verified_mapping else 'inventory_only_mapping_pending', 'complete': complete,
               'train_patients': len(ids), 'fully_comparable_patients': complete_cases,
               'label_values': {source: sorted(values) for source, values in all_values.items()},
               'unrecognized_nifti_names': unknown, 'issues': len(issues),
               'pairwise_agreement': aggregate if verified_mapping else [],
               'validation_images_read': False, 'test_images_read': False,
               'empty_pair_policy': 'Both-empty targets omitted from Dice averages and counted separately; one-empty Dice is 0',
               'interpretation': 'Annotation consistency evidence; not a mathematical model-performance ceiling',
               'artifacts': str(output)}
    protocol = {'root': str(root), 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'label_mapping': mapping, 'mapping_source': str(mapping_path) if mapping_path else 'user-confirmed shared IDs' if confirmed else 'not yet confirmed',
                'mapping_sha256': hashlib.sha256(mapping_path.read_bytes()).hexdigest() if mapping_path else None,
                'patients': ids, 'reference': 'labelsTr remains final reference; no consensus or training data written',
                'geometry_policy': 'Exact-size and physical-geometry tolerance checks; mismatches excluded from all-rater comparisons'}
    write_csv(output / 'pairwise_summary.csv', aggregate if verified_mapping else [])
    if complete:
        heatmap(output / 'agreement_heatmap.png', aggregate, len(ids))
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2), flush=True)
    if issues:
        raise ValueError('Audit incomplete; inspect issues.json before interpreting partial agreement')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--confirm-shared-label-ids', action='store_true', help='Use only after confirming all three experts use 1=RFSDP, 9=LFSDP')
    mode.add_argument('--mapping-json', type=Path, help='Per-expert RFSDP/LFSDP integer mapping')
    args = parser.parse_args()
    try:
        audit(args.root, args.manifest, args.output_dir, args.confirm_shared_label_ids, args.mapping_json)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f'Expert audit stopped: {error}\n')


if __name__ == '__main__':
    main()
