import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fsdp_verified_cache as recovery
import fsdp_resolution as resolution
import physical_fsdp as physical
import train_fsdp_foundation as foundation
from audit_foundation_sources import digest
from test_fsdp_foundation import fixture as foundation_fixture


def fixture(root):
    source = root / 'actual data'
    for name in ('imagesTr', 'labelsTr', 'ColorTable'):
        (source / name).mkdir(parents=True)
    split_ids = {split: [str(start + i) for i in range(count)] for split, start, count in
                 (('train', 1000001, 54), ('val', 2000001, 10), ('test', 3000001, 20))}
    manifest = {'splits': split_ids, 'target': {'1': 'RFSDP', '9': 'LFSDP'}, 'cases': []}
    meta = {'size': [12] * 3, 'spacing': [.5] * 3, 'origin': [-4., 7., -10.],
            'direction': [0., -1., 0., 1., 0., 0., 0., 0., 1.]}
    ct = np.arange(12 ** 3, dtype=np.float32).reshape((12,) * 3)
    image = physical.image_xyz(ct, meta)
    labels = np.zeros(ct.shape, np.uint8)
    labels[3, 5, 5], labels[8, 5, 5] = 1, 9
    plan = {'shape': [6] * 3, 'spacing': [.5] * 3}
    norm = {'clip_lower': -1000., 'clip_upper': 2000., 'mean': 0., 'std': 100.}
    templates = []
    for side, value in (('RFSDP', 1), ('LFSDP', 9)):
        sample = physical.make_sample(image, ct, labels == value, 'train_01', side, plan, norm)
        sample.update(sam_image=sample['image'].clone(), sam_target=sample['target'].clone(),
            sam_valid=sample['valid'].clone(), sam_grid=copy.deepcopy(sample['grid']))
        templates.append(sample)
    prepared = {'samples': {'train': [], 'val': []}, 'patients': {'train': {}, 'val': {}},
        'source_provenance': {'train': [], 'val': []}, 'policy': {'normalization': norm, 'selected': plan}, 'coverage': {}}
    for split in ('train', 'val'):
        for i, patient in enumerate(split_ids[split], 1):
            alias = f'{split}_{i:02d}'
            path = source / 'imagesTr' / f'{patient}.nii.gz'
            physical.sitk.WriteImage(image, str(path))
            record = {'patient_id': patient, 'case': alias, 'split': split,
                'source_image': str(path), 'source_image_sha256': digest(path),
                'source_label': str(source / 'labelsTr' / f'{patient}-label.nii.gz'), 'source_label_sha256': 'missing_original',
                'color_table': str(source / 'ColorTable' / f'{patient}_ColorTable.csv'), 'color_table_sha256': 'missing_original',
                'right_voxels': 1, 'left_voxels': 1}
            manifest['cases'].append(record)
            prepared['source_provenance'][split].append(record)
            prepared['patients'][split][alias] = {'shape': list(ct.shape), 'target_indices': np.flatnonzero(labels)}
            for template in templates:
                sample = copy.deepcopy(template)
                sample.update(case=alias, name=f'{alias}_{sample["side"]}')
                prepared['samples'][split].append(sample)
    return source, manifest, prepared, labels


class VerifiedCacheTests(unittest.TestCase):
    def test_nohup_launchers_forward_explicit_cache_mode_and_spaced_source_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'project with spaces'
            scripts = root / 'scripts'; scripts.mkdir(parents=True)
            binary = root / 'bin'; binary.mkdir()
            fake = binary / 'python'
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\nexit 3\n')
            fake.chmod(0o755)
            base = root / 'base'; base.mkdir()
            for name in ('config.json', 'summary.json', 'hybrid_best.pth', 'hybrid_data.pt'):
                (base / name).write_text('synthetic')
            for channel in ('resolution', 'foundation'):
                name = f'launch_fsdp_{channel}_nohup.sh'
                shutil.copyfile(Path(__file__).resolve().parents[1] / 'scripts' / name, scripts / name)
                env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
                    CONDA_DEFAULT_ENV='fsdp-med3d', FSDP_FOUNDATION_DIR=str(base),
                    FSDP_SOURCE_ROOT=str(root / 'actual data'), FSDP_VERIFIED_CACHE='1', RUN_TRAINING='1', RESUME='0')
                result = subprocess.run(['bash', str(scripts / name)], env=env, capture_output=True, text=True, timeout=15)
                self.assertNotEqual(result.returncode, 0)
                args = (root / 'logs' / f'fsdp_{channel}' / 'latest.log').read_text().splitlines()
                self.assertIn('--verified-cache', args)
                self.assertEqual(args[args.index('--source-root') + 1], str(root / 'actual data'))
                if channel == 'resolution':
                    self.assertIn('--train', args)

    def test_all_128_targets_match_frozen_bilateral_targets_without_annotation_files(self):
        with tempfile.TemporaryDirectory() as folder:
            source, manifest, prepared, _ = fixture(Path(folder))
            before = copy.deepcopy(prepared['source_provenance'])
            report = recovery.validate_targets(prepared, manifest)
            self.assertEqual(report['splits']['train']['native_targets_intact'], 108)
            self.assertEqual(report['splits']['val']['native_targets_intact'], 20)
            sources = recovery.image_inventory(prepared['source_provenance'], source, verify=True)
            recovery.require_images(sources)
            self.assertTrue(sources['ct_content_verified'])
            self.assertFalse(sources['annotation_files_verified'])
            self.assertEqual(prepared['source_provenance'], before)
            self.assertEqual(list((source / 'labelsTr').iterdir()), [])
            self.assertEqual(list((source / 'ColorTable').iterdir()), [])

    def test_changed_ct_is_rejected_and_test_paths_are_never_examined(self):
        with tempfile.TemporaryDirectory() as folder:
            source, manifest, prepared, _ = fixture(Path(folder))
            row = prepared['source_provenance']['train'][0]
            Path(row['source_image']).write_bytes(b'changed_CT')
            report = recovery.image_inventory(prepared['source_provenance'], source, verify=True)
            with self.assertRaisesRegex(ValueError, 'CT inputs unavailable or changed'):
                recovery.require_images(report)
            self.assertEqual(sum(r['status'] == 'changed' for r in report['rows']), 1)
            with self.assertRaisesRegex(ValueError, 'train/val only'):
                recovery.image_inventory({**prepared['source_provenance'], 'test': []}, source)
            prepared['patients']['test'] = {}
            with self.assertRaisesRegex(ValueError, 'unsupported split'):
                recovery.validate_targets(prepared, manifest)

    def test_native_target_corruption_and_patient_mix_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            _, manifest, prepared, _ = fixture(Path(folder))
            sample = prepared['samples']['train'][0]
            target = sample['native_target'].copy()
            sample['native_target'] = np.roll(target, 1, axis=1)
            with self.assertRaisesRegex(ValueError, 'bilateral target'):
                recovery.validate_targets(prepared, manifest)
            sample['native_target'] = target
            prepared['source_provenance']['train'][0]['patient_id'] = manifest['splits']['test'][0]
            with self.assertRaisesRegex(ValueError, 'patient identity'):
                recovery.validate_targets(prepared, manifest)

    def test_token_context_uses_direct_original_ct_and_exact_cached_binary_on_original_grid(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            source, manifest, prepared, labels = fixture(root)
            recovery.validate_targets(prepared, manifest)
            samples = prepared['samples']['train'][:2]
            prior = prepared['source_provenance']['train'][0]
            record = manifest['cases'][0]
            image = recovery.read_image(record, source, prior, samples)
            for sample in samples:
                binary = recovery.source_label(sample, image)
                original = labels == (1 if sample['side'] == 'RFSDP' else 9)
                np.testing.assert_array_equal(physical.xyz(binary), original)
                rebuilt = physical.xyz(physical.resample(binary, sample['sam_grid'], physical.sitk.sitkNearestNeighbor,
                                                        pixel=physical.sitk.sitkUInt8))
                np.testing.assert_array_equal(rebuilt, sample['sam_target'][0, 0].numpy())
            prepared['samples'] = {'train': samples, 'val': []}
            cohort = root / 'cohort'; cohort.mkdir()
            physical.save(root / 'config.json', {'cohort_run_dir': str(cohort)})
            physical.save(cohort / 'config.json', {'data_dir': str(root), 'source_root': '/unavailable_old_root'})
            physical.save(root / 'manifest.json', manifest)
            pointers = [s['image'].data_ptr() for s in samples]
            def records(unused, split):
                return [record] if split == 'train' else []
            with patch.object(physical, 'records_for', records):
                resolution.replace_contexts(prepared, argparse.Namespace(foundation_run_dir=root,
                    source_root=source, verified_cache=True))
            self.assertEqual(pointers, [s['image'].data_ptr() for s in samples])
            for sample in samples:
                original = labels == (1 if sample['side'] == 'RFSDP' else 9)
                wanted = physical.xyz(physical.resample(physical.image_xyz(original.astype(np.uint8), physical.geometry(image)),
                    sample['sam_grid'], physical.sitk.sitkNearestNeighbor, pixel=physical.sitk.sitkUInt8))
                self.assertEqual(sample['sam_target'].shape[-3:], (256,) * 3)
                np.testing.assert_array_equal(sample['sam_target'][0, 0].numpy(), wanted)
            samples[0]['native']['origin'][0] += 1
            with self.assertRaisesRegex(ValueError, 'CT geometry differs'):
                recovery.read_image(record, source, prior, samples)

    def test_explicit_mode_preserves_original_cache_and_default_still_requires_labels(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            base, old, summary, _, _ = foundation_fixture(root)
            source, manifest, prepared, _ = fixture(root)
            physical.save(Path(old['cohort_run_dir']) / 'config.json', {'data_dir': str(root / 'data'), 'source_root': str(source)})
            physical.save(root / 'data/manifest.json', manifest)
            old['manifest_sha256'] = digest(root / 'data/manifest.json')
            torch.save(prepared, base / 'hybrid_data.pt')
            old['cache_sha256'] = digest(base / 'hybrid_data.pt')
            old['physical_policy'] = summary['physical_policy'] = prepared['policy']
            physical.save(base / 'config.json', old)
            physical.save(base / 'summary.json', summary)
            physical.save(base / 'source_provenance.json', prepared['source_provenance'])
            args = argparse.Namespace(foundation_run_dir=base, run_dir=root / 'new', source_root=source, verified_cache=True,
                resume=False, sam_dir=Path(old['sam_dir']), checkpoint=Path(old['checkpoint']), **foundation.recipe_values(old))
            args.run_dir.mkdir()
            with patch.object(foundation, 'FOUNDATION_SHA256', summary['best_checkpoint_sha256']):
                with self.assertRaisesRegex(ValueError, 'Foundation inputs unavailable'):
                    foundation.inspect_foundation(base, source)
                actual, config = foundation.prepare_foundation(args)
                self.assertEqual(digest(base / 'hybrid_data.pt'), old['cache_sha256'])
                self.assertEqual(actual['source_provenance'], prepared['source_provenance'])
                self.assertFalse(config['target_recovery']['original_annotation_files_verified'])
                self.assertTrue(json.loads((args.run_dir / 'cache_target_verification.json').read_text())['passed'])
                (base / 'hybrid_data.pt').write_bytes(b'corrupt cache')
                with self.assertRaisesRegex(ValueError, 'checksum differs'):
                    foundation.inspect_foundation(base, source, verified_cache=True)


if __name__ == '__main__':
    unittest.main()
