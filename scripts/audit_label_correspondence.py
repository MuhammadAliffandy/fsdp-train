#!/usr/bin/env python3
"""Find raw-label overlap candidates on frozen training patients; never infer anatomy."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from audit_expert_agreement import SOURCES, header, index_files, same_geometry, write_csv


TARGETS = tuple(range(1, 18)) + ('1+9',)


def counts(reference, expert):
    """Return compact contingency counts, including background, without assuming expert IDs."""
    reference_values, reference_counts = np.unique(reference, return_counts=True)
    raw_values, raw_counts = np.unique(expert, return_counts=True)
    ref = dict(zip(map(int, reference_values), map(int, reference_counts)))
    raw = dict(zip(map(int, raw_values), map(int, raw_counts)))
    intersections = {}
    for value in raw:
        values, sizes = np.unique(reference[expert == value], return_counts=True)
        intersections[value] = dict(zip(map(int, values), map(int, sizes)))
    return ref, raw, intersections


def summarize(cases, raw_values, names):
    rows = []
    for raw_value in sorted(set(raw_values) - {0}):
        for target in TARGETS:
            target_ids = (1, 9) if target == '1+9' else (target,)
            scores, a_total, b_total, overlap_total = [], 0, 0, 0
            both_empty = one_empty = raw_present = reference_present = 0
            for case in cases:
                a = case['raw'].get(raw_value, 0)
                b = sum(case['ref'].get(value, 0) for value in target_ids)
                intersection = sum(case['overlap'].get(raw_value, {}).get(value, 0) for value in target_ids)
                raw_present += a > 0
                reference_present += b > 0
                if a + b:
                    scores.append(2 * intersection / (a + b))
                else:
                    both_empty += 1
                one_empty += bool(a) != bool(b)
                a_total += a
                b_total += b
                overlap_total += intersection
            rows.append({'expert_raw_value': raw_value, 'reference_value': target,
                         'reference_name': 'RFSDP+LFSDP' if target == '1+9' else names.get(target, f'label_{target}'),
                         'compared_patients': len(cases), 'raw_present_patients': raw_present,
                         'reference_present_patients': reference_present, 'nonempty_pairs': len(scores),
                         'both_empty_pairs': both_empty, 'one_empty_pairs': one_empty,
                         'mean_patient_dice_nonempty': float(np.mean(scores)) if scores else None,
                         'pooled_voxel_dice': 2 * overlap_total / (a_total + b_total) if a_total + b_total else None,
                         'expert_voxels': a_total, 'reference_voxels': b_total, 'overlap_voxels': overlap_total,
                         'fraction_expert_overlapping_reference': overlap_total / a_total if a_total else None,
                         'fraction_reference_covered': overlap_total / b_total if b_total else None})
    return rows


def audit(root, manifest_path, output):
    root, output = root.expanduser().resolve(), output.resolve()
    manifest = json.loads(manifest_path.read_text())
    groups = manifest['splits']
    ids = sorted(groups['train'])
    flat = [key for split in ('train', 'val', 'test') for key in groups[split]]
    if len(ids) != 54 or len(groups['val']) != 10 or len(groups['test']) != 20 or len(set(flat)) != len(flat):
        raise ValueError('Expected frozen disjoint 54/10/20 split')
    if output.exists():
        raise ValueError('Output already exists')
    output.mkdir(parents=True)
    records = {row['patient_id']: row for row in manifest['cases']}
    indices, unknown = {}, {}
    for source in SOURCES:
        indices[source], unknown[source] = index_files(root / source)
    compact = {source: [] for source in SOURCES[1:]}
    inventory, issues, names = [], [], {1: 'RFSDP', 9: 'LFSDP'}
    all_values = {source: set() for source in SOURCES}
    contingency = {source: Counter() for source in SOURCES[1:]}
    for index, key in enumerate(ids, 1):
        alias = f'train_{index:02d}'
        try:
            record = records[key]
            ref_path = Path(record['source_label']).resolve()
            ct_path = Path(record['source_image']).resolve()
            if not ref_path.is_relative_to(root / 'labelsTr') or not ct_path.is_relative_to(root / 'imagesTr'):
                raise ValueError('Manifest source outside private root')
            with (root / 'ColorTable' / f'{key}_ColorTable.csv').open(encoding='utf-8-sig', newline='') as stream:
                colors = {int(row['LabelValue']): row['Name'].strip().upper() for row in csv.DictReader(stream)}
            if colors.get(1) != 'RFSDP' or colors.get(9) != 'LFSDP':
                raise ValueError('Reference ColorTable does not confirm final FSDP IDs')
            for value, name in colors.items():
                if value in names and names[value] != name:
                    raise ValueError(f'Inconsistent reference name for value {value}')
                names[value] = name
            ct = header(ct_path)
            loaded = {}
            for source in SOURCES:
                candidates = indices[source].get(key, [])
                if len(candidates) != 1:
                    raise ValueError(f'{source}: expected one mask; found {len(candidates)}')
                path = candidates[0].resolve()
                if not path.is_relative_to(root / source):
                    raise ValueError(f'{source}: mask resolves outside annotation folder')
                if source == 'labelsTr' and path != ref_path:
                    raise ValueError('Final reference differs from manifest')
                if not same_geometry(ct, header(path)):
                    raise ValueError(f'{source}: geometry mismatch; no resampling performed')
                image = sitk.ReadImage(str(path))
                array = sitk.GetArrayFromImage(image)
                integer = np.issubdtype(array.dtype, np.integer) or (np.isfinite(array).all() and np.equal(array, np.floor(array)).all())
                if not integer or (array < 0).any() or (array > 65535).any():
                    raise ValueError(f'{source}: invalid or unsupported raw values')
                values, sizes = np.unique(array, return_counts=True)
                if source == 'labelsTr' and (values > 17).any():
                    raise ValueError('Reference value outside expected 0..17')
                all_values[source].update(map(int, values))
                for value, size in zip(values, sizes):
                    inventory.append({'case': alias, 'source': source, 'raw_value': int(value),
                                      'voxels': int(size), 'volume_mm3': int(size) * float(np.prod(ct.GetSpacing()))})
                loaded[source] = array
            for source in SOURCES[1:]:
                ref, raw, overlap = counts(loaded['labelsTr'], loaded[source])
                compact[source].append({'case': alias, 'ref': ref, 'raw': raw, 'overlap': overlap})
                for a, reference_counts in overlap.items():
                    for b, size in reference_counts.items():
                        contingency[source][a, b] += size
        except (ValueError, KeyError, OSError, RuntimeError) as error:
            issues.append({'case': alias, 'reason': str(error)})
        print(f'{alias}: correspondence checked {index}/{len(ids)}', flush=True)
    rows, candidates = [], []
    for source in SOURCES[1:]:
        selected = summarize(compact[source], all_values[source], names)
        rows.extend({'source': source, **row} for row in selected)
        for value in sorted(all_values[source] - {0}):
            ranked = sorted([row for row in selected if row['expert_raw_value'] == value and row['reference_value'] != '1+9'],
                            key=lambda row: row['mean_patient_dice_nonempty'] if row['mean_patient_dice_nonempty'] is not None else -1, reverse=True)
            candidates.append({'source': source, 'expert_raw_value': value,
                               'top_reference_overlap_candidates': [row for row in ranked if row['overlap_voxels'] > 0][:3],
                               'fsdp_comparisons': [row for row in selected if row['expert_raw_value'] == value and row['reference_value'] in (1, 9, '1+9')]})
    complete = not issues and all(len(value) == len(ids) for value in compact.values())
    summary = {'mode': 'raw_label_correspondence', 'complete': complete, 'train_patients': len(ids),
               'compared_patients': {source: len(cases) for source, cases in compact.items()},
               'label_values': {source: sorted(values) for source, values in all_values.items()},
               'issues': len(issues), 'unrecognized_nifti_names': unknown,
               'candidates': candidates, 'semantic_mapping_verified': False,
               'validation_images_read': False, 'test_images_read': False,
               'interpretation': 'Overlap ranks are candidates only; anatomical names require expert-specific label definitions. No remapping or training performed.',
               'empty_pair_policy': 'Both-empty omitted from patient mean; one-empty scores zero. Pooled Dice weights voxel counts, not patients.',
               'artifacts': str(output)}
    write_csv(output / 'inventory.csv', inventory)
    write_csv(output / 'reference_overlap.csv', rows)
    write_csv(output / 'pooled_contingency.csv', [{'source': source, 'expert_raw_value': a, 'reference_value': b, 'voxels': size}
              for source, counter in contingency.items() for (a, b), size in sorted(counter.items())])
    (output / 'issues.json').write_text(json.dumps(issues, indent=2))
    (output / 'protocol.json').write_text(json.dumps({'root': str(root), 'train_patient_ids': ids,
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'shared_helpers_sha256': hashlib.sha256(Path(__file__).with_name('audit_expert_agreement.py').read_bytes()).hexdigest(),
        'geometry_policy': 'CT and all four masks must match; exclude entire case on any mismatch; no resampling',
        'reference': 'Original labelsTr and patient splits unchanged'}, indent=2))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2), flush=True)
    if not complete:
        raise ValueError('Correspondence incomplete; inspect issues.json before interpreting partial candidates')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        audit(args.root, args.manifest, args.output_dir)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f'Label correspondence stopped: {error}\n')


if __name__ == '__main__':
    main()
