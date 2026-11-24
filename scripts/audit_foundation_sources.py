#!/usr/bin/env python3
"""Inspect missing/renamed foundation inputs without loading images or changing data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


FOLDERS = {'source_image': 'imagesTr', 'source_label': 'labelsTr', 'color_table': 'ColorTable'}


def bind_sources(provenance, source_root=None):
    if source_root is None:
        return provenance
    if any(split not in ('train', 'val') for split in provenance):
        raise ValueError('Only train/val source provenance is supported')
    root = Path(source_root).expanduser().resolve()
    return {split: [{**record, **{key: str((root / folder / Path(record[key]).name).resolve())
                    for key, folder in FOLDERS.items()}} for record in records]
            for split, records in provenance.items()}


def bind_record(record, source_root):
    root = Path(source_root).expanduser().resolve()
    return {**record, **{key: str((root / FOLDERS[key] / Path(record[key]).name).resolve())
                        for key in ('source_image', 'source_label')}}


def load_bound_case(record, source_root, expected, alias):
    """Read relocated byte-identical inputs, including symlinked role directories."""
    import csv
    import numpy as np
    import physical_fsdp as physical
    root = Path(source_root).resolve()
    paths = {key: Path(expected[key]).resolve() for key in FOLDERS}
    for key, folder in FOLDERS.items():
        if not paths[key].is_relative_to((root / folder).resolve()):
            raise ValueError('Relocated source is outside the specified role directory')
        if key != 'color_table' and Path(record[key]).resolve() != paths[key]:
            raise ValueError('Relocated manifest/provenance path differs')
        if digest(paths[key]) != expected[key + '_sha256']:
            raise ValueError('Relocated source content differs from the verified foundation')
    with paths['color_table'].open(encoding='utf-8-sig', newline='') as stream:
        colors = {int(row['LabelValue']): row['Name'].strip().upper() for row in csv.DictReader(stream)}
    if colors.get(1) != 'RFSDP' or colors.get(9) != 'LFSDP':
        raise ValueError('Relocated ColorTable does not confirm FSDP1/9')
    image, label = [physical.sitk.ReadImage(str(paths[key])) for key in ('source_image', 'source_label')]
    if not physical.same_geometry(image, label):
        raise ValueError('Relocated CT/label physical geometry differs')
    ct, labels = physical.xyz(image), physical.xyz(label)
    if (not np.isfinite(ct).all() or not np.isin(labels, range(18)).all()
            or int((labels == 1).sum()) != record['right_voxels']
            or int((labels == 9).sum()) != record['left_voxels']):
        raise ValueError('Relocated CT/final labels/counts differ')
    return image, label, ct, labels, {**expected, 'case': alias}


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def inventory(provenance, search_roots=(), verify=False):
    """Only examine patient IDs in train/val provenance; never scan expert/test masks."""
    if any(split not in ('train', 'val') for split in provenance):
        raise ValueError('Only train/val source provenance is supported')
    rows = []
    hashed = {}
    def hash_once(path):
        resolved = path.resolve()
        if resolved not in hashed:
            hashed[resolved] = digest(resolved)
        return hashed[resolved]
    for split, records in provenance.items():
        for record in records:
            patient = str(record['patient_id'])
            for key, folder in FOLDERS.items():
                path = Path(record[key])
                expected = record[key + '_sha256']
                row = {'split': split, 'case': record['case'], 'role': key, 'path': str(path),
                       'expected_sha256': expected, 'directory_exists': path.parent.is_dir(),
                       'status': 'present_unchecked' if path.is_file() else 'missing'}
                if path.is_file() and verify:
                    try:
                        actual = hash_once(path)
                        row.update(actual_sha256=actual, status='ok' if actual == expected else 'changed')
                    except OSError as error:
                        row.update(status='unreadable', error=str(error))
                if row['status'] in ('missing', 'changed', 'unreadable'):
                    alternatives = []
                    seen = {path.resolve()}
                    for root in dict.fromkeys([path.parent.parent, *(Path(p) for p in search_roots)]):
                        directory = root / folder
                        for candidate in sorted(directory.glob(patient + '*')):
                            suffix = candidate.name[len(patient):]
                            if (not suffix or suffix[0] not in '_.- ' or not candidate.is_file()
                                    or candidate.resolve() in seen):
                                continue
                            seen.add(candidate.resolve())
                            item = {'path': str(candidate), 'sha256_matches': False}
                            try:
                                item['actual_sha256'] = hash_once(candidate)
                                item['sha256_matches'] = item['actual_sha256'] == expected
                            except OSError as error:
                                item['error'] = str(error)
                            alternatives.append(item)
                    row['alternatives'] = alternatives
                rows.append(row)
    counts = dict(Counter(row['status'] for row in rows))
    return {'ready': all(row['status'] in ('ok', 'present_unchecked') for row in rows),
            'status_counts': counts, 'content_verified': verify,
            'problems': [r for r in rows if r['status'] not in ('ok', 'present_unchecked')],
            'files_checked': len(rows), 'rows': rows, 'image_tensors_loaded': False, 'test_images_read': False,
            'data_modified': False, 'interpretation': 'Alternatives are diagnostic only; even a checksum match is not applied automatically.'}


def check_folder(folder, source_root=None):
    path = Path(folder) / 'source_provenance.json'
    report = inventory(bind_sources(json.loads(path.read_text()), source_root))
    report['source_root_override'] = str(Path(source_root).resolve()) if source_root else None
    if not report['ready']:
        raise ValueError(problem_message(report))
    return report


def problem_message(report):
    lines = [f'Foundation inputs unavailable: {report["status_counts"]}. No model training started.']
    for row in report['problems'][:8]:
        matches = [a['path'] for a in row.get('alternatives', []) if a['sha256_matches']]
        lines.append(f'{row["case"]} {row["role"]}: {row["status"]}: {row["path"]}')
        if matches:
            lines.append('Byte-identical alternative found: ' + ', '.join(matches))
    lines.append('Run scripts/audit_foundation_sources.py with --foundation-run-dir and --verify; '
                 'restore the original files/paths from verified data. Different masks must not replace the foundation labels.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foundation-run-dir', type=Path, required=True)
    parser.add_argument('--search-root', type=Path, action='append', default=[])
    parser.add_argument('--source-root', type=Path, help='Explicit relocated dataset root; original filenames/checksums remain required')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    base = args.foundation_run_dir.expanduser().resolve()
    original = json.loads((base / 'source_provenance.json').read_text())
    provenance = bind_sources(original, args.source_root)
    report = inventory(provenance, [p.expanduser().resolve() for p in args.search_root], args.verify)
    report['source_root_override'] = str(args.source_root.expanduser().resolve()) if args.source_root else None
    if args.output:
        output = args.output.expanduser().resolve()
        roots = {Path(row[key]).resolve().parent.parent for mapping in (original, provenance)
                 for records in mapping.values() for row in records for key in FOLDERS}
        roots.update(p.expanduser().resolve() for p in args.search_root)
        if output.is_relative_to(base) or any(output.is_relative_to(root) for root in roots):
            parser.error('Report output must be outside foundation and original dataset folders')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
    brief = {key: value for key, value in report.items() if key != 'rows'}
    print(json.dumps(brief, indent=2))
    if not report['ready']:
        parser.exit(2, 'Source audit incomplete; no data was changed.\n')


if __name__ == '__main__':
    main()
