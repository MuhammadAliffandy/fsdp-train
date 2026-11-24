import argparse
import contextlib
import copy
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_fsdp_research as research
import train_fsdp_foundation as foundation
import train_sam_refine32 as base
import physical_fsdp as physical
from sam_refine32 import SamRefine32
from test_fsdp_foundation import fixture as foundation_fixture
from test_fsdp_verified_cache import fixture as cache_fixture
from test_sam_refine32 import FakeSAM
from evaluate_private_validation import file_hash


class ResearchTrainerTests(unittest.TestCase):
    def test_verified_prepare_keeps_cache_and_source_immutable_and_resume_settings_are_strict(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            source, old, summary, _, _ = foundation_fixture(root)
            dataset, manifest, prepared, _ = cache_fixture(root)
            physical.save(root / 'data/manifest.json', manifest)
            physical.save(Path(old['cohort_run_dir']) / 'config.json',
                {'source_root': str(dataset), 'data_dir': str(root / 'data')})
            torch.save(prepared, source / 'hybrid_data.pt')
            old.update(cache_sha256=file_hash(source / 'hybrid_data.pt'), physical_policy=prepared['policy'],
                       manifest_sha256=file_hash(root / 'data/manifest.json'))
            summary['physical_policy'] = prepared['policy']
            for name, value in (('config.json', old), ('summary.json', summary),
                                ('source_provenance.json', prepared['source_provenance'])):
                physical.save(source / name, value)
            args = argparse.Namespace(foundation_run_dir=source, run_dir=root / 'research', source_root=dataset,
                verified_cache=True, epochs=20, lr_multiplier=.2, patience_epochs=50, boundary_weight=.05,
                boundary_clip_mm=5., device='cpu', resume=False, variant='control')
            args.run_dir.mkdir()
            with patch.object(foundation, 'FOUNDATION_SHA256', summary['best_checkpoint_sha256']):
                actual, config = research.prepare(args)
                self.assertEqual(len(actual['samples']['train']), 108)
                self.assertEqual(file_hash(source / 'hybrid_data.pt'), old['cache_sha256'])
                self.assertEqual(actual['source_provenance'], prepared['source_provenance'])
                self.assertFalse(config['target_recovery']['original_annotation_files_verified'])
                self.assertTrue((args.run_dir / 'foundation_inputs/cache_target_verification.json').is_file())
                args.resume = True
                research.prepare(args)
                args.boundary_weight = .1
                with self.assertRaisesRegex(ValueError, 'changed'):
                    research.prepare(args)

    def test_interrupted_training_resumes_exactly_and_exports_native_masks_with_no_test_reads(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            image = physical.sitk.GetImageFromArray(np.zeros((16,) * 3, np.int16))
            image.SetSpacing((.35,) * 3)
            ct = np.zeros((16,) * 3, np.float32)
            norm = {'clip_lower': -1000., 'clip_upper': 1000., 'mean': 0., 'std': 1000.}
            plan = {'shape': [64] * 3, 'spacing': [.35] * 3}
            prepared = {'samples': {}, 'patients': {}, 'policy': {}, 'coverage': {}}
            for split, count in (('train', 2), ('val', 1)):
                prepared['samples'][split], prepared['patients'][split] = [], {}
                for index in range(1, count + 1):
                    alias = f'{split}_{index:02d}'
                    mask = np.zeros_like(ct, dtype=bool); mask[6:10, 6:10, 6:10] = True
                    prepared['patients'][split][alias] = {'shape': [16] * 3, 'target_indices': np.flatnonzero(mask)}
                    for side in ('RFSDP', 'LFSDP'):
                        prepared['samples'][split].append(physical.make_sample(image, ct, mask, alias, side, plan, norm))
            metric = {'mean_bilateral_patient_dice': .456}
            config = {'foundation': {'source_best_epoch': 30, 'checkpoint_sha256': 'synthetic',
                                    'expected_validation': {'5': metric}},
                      'cache_sha256': 'synthetic', 'target_recovery': None, 'evaluation_protocol': 'synthetic CPU proxy'}
            def prepare(args):
                for key, value in dict(seed=2026, precision='bf16', source_best_epoch=30, augmentation='none',
                    accumulation=3, sam_lr=1e-5, head_lr=1e-4, auxiliary_weight=.25, val_every=1, min_delta=.001).items():
                    setattr(args, key, value)
                physical.save(args.run_dir / 'config.json', config)
                return prepared, config
            def build(args):
                sam = FakeSAM()
                sam.image_encoder.blocks = nn.ModuleList([nn.Conv3d(8, 8, 1) for _ in range(12)])
                return SamRefine32(sam), 'synthetic'
            def run_sample(model, sample, device, seed, criterion=None, augment=False, auxiliary_weight=.25):
                value = sample['image'].to(device)
                features = model.full[0](value)
                logits = model.residual(torch.cat((features, features), 1))
                logits = logits + model.sam.mask_decoder.project(model.sam.image_encoder.project(value))
                loss = physical.masked_loss(criterion, logits, sample['target'], sample['valid']) if criterion is not None else None
                return loss, {k: {'refined': logits, 'coarse_on_detail': logits} for k in (1, 5)}
            args = argparse.Namespace(run_dir=root / 'interrupted', variant='control', device='cpu', resume=False,
                                      epochs=2, patience_epochs=50, boundary_weight=.05)
            args.run_dir.mkdir()
            with patch.object(research, 'prepare', prepare), patch.object(base, 'build', build), \
                 patch.object(foundation, 'load_foundation_weights'), patch.object(foundation, 'verify_reproduction'), \
                 patch.object(research, 'gradient_probe', return_value={'passed': True}), \
                 patch.object(base, 'run_sample', run_sample), patch.object(research, 'run_sample', run_sample):
                original = research.train_epoch
                def interrupted(*values):
                    if values[6] == 1:
                        raise RuntimeError('simulated interruption')
                    return original(*values)
                with patch.object(research, 'train_epoch', interrupted), self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    research.execute(args)
                saved = torch.load(args.run_dir / 'research_latest.pth', weights_only=False)
                self.assertEqual(saved['epoch'], 1)
                args.resume = True
                research.execute(args)
                resumed = torch.load(args.run_dir / 'research_latest.pth', weights_only=False)
                reference_args = copy.copy(args); reference_args.run_dir = root / 'reference'; reference_args.resume = False
                reference_args.run_dir.mkdir()
                research.execute(reference_args)
                reference = torch.load(reference_args.run_dir / 'research_latest.pth', weights_only=False)
            self.assertEqual(resumed['history'], reference['history'])
            for key, value in reference['model_state_dict'].items():
                torch.testing.assert_close(resumed['model_state_dict'][key], value, rtol=0, atol=0)
            self.assertTrue(all(r['optimizer_updates'] == 2 and r['expected_updates'] == 2 for r in resumed['history']))
            summary = physical.read(args.run_dir / 'summary.json')
            self.assertTrue(summary['complete'])
            self.assertFalse(summary['test_images_read'])
            self.assertEqual(len(list((args.run_dir / 'best_validation').glob(f'*_t{summary["threshold"]:g}_prediction.nii.gz'))), 2)
            with tarfile.open(args.run_dir / 'results.tar.gz') as archive:
                self.assertFalse(any(n.endswith('.pth') or n.endswith('.pt') for n in archive.getnames()))


if __name__ == '__main__':
    unittest.main()
