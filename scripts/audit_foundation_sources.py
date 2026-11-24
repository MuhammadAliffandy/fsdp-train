#!/usr/bin/env python3
"""Inspect missing/renamed foundation inputs without loading images or changing data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


FOLDERS = {'source_image': 'imagesTr', 'source_label': 'labelsTr', 'color_table': 'ColorTable'}


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


def check_folder(folder):
    path = Path(folder) / 'source_provenance.json'
    report = inventory(json.loads(path.read_text()))
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
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    base = args.foundation_run_dir.expanduser().resolve()
    provenance = json.loads((base / 'source_provenance.json').read_text())
    report = inventory(provenance, [p.expanduser().resolve() for p in args.search_root], args.verify)
    if args.output:
        output = args.output.expanduser().resolve()
        roots = {Path(row[key]).resolve().parent.parent for records in provenance.values() for row in records for key in FOLDERS}
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
