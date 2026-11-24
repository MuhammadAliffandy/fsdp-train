import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_fsdp_foundation as foundation
import train_sam_refine32 as trainer
from evaluate_private_validation import file_hash
from train_private_unet import state_hash


def fixture(root):
    base = root / 'base'; base.mkdir()
    sam = root / 'sam/segment_anything'; sam.mkdir(parents=True)
    (sam / 'build_sam3D.py').write_text('synthetic_source')
    turbo = root / 'turbo.pth'; turbo.write_bytes(b'synthetic_turbo')
    cohort = root / 'cohort'; cohort.mkdir()
    dataset = root / 'data'; dataset.mkdir()
    manifest = {'splits': {'train': [str(i) for i in range(54)], 'val': [str(i) for i in range(54, 64)],
                          'test': [str(i) for i in range(64, 84)]}, 'target': {'1': 'RFSDP', '9': 'LFSDP'}}
    (dataset / 'manifest.json').write_text(json.dumps(manifest))
    (cohort / 'config.json').write_text(json.dumps({'data_dir': str(dataset), 'source_root': str(root / 'source')}))
    prepared = {'policy': {'sentinel_original_ROI': [96, 160, 208]},
                'patients': {'train': {str(i): {} for i in range(54)}, 'val': {str(i): {} for i in range(10)}},
                'source_provenance': {'train': [], 'val': []}}
    torch.save(prepared, base / 'hybrid_data.pt')
    old = {'versions': {}, 'cache_sha256': file_hash(base / 'hybrid_data.pt'), 'physical_policy': prepared['policy'],
           'sam_dir': str(sam.parent), 'checkpoint': str(turbo), 'turbo_sha256': file_hash(turbo),
           'sam_source_sha256': foundation.views.source_hashes(sam.parent), 'cohort_run_dir': str(cohort),
           'manifest_sha256': file_hash(dataset / 'manifest.json'),
           'code_sha256': {name: file_hash(Path(trainer.__file__).with_name(name)) for name in trainer.CODE_FILES},
           'augmentation': 'shared physical LR reflection and shared gain0.9-1.1/shift+-0.1 only',
           'accumulation': 8, 'auxiliary_weight': .25, 'seed': 2026, 'precision': 'bf16', 'val_every': 5, 'min_delta': .001,
           'architecture': 'synthetic', 'protocol': 'synthetic', 'loss': 'synthetic', 'prompts': 'synthetic',
           'threshold_protocol': 'train_only', 'selection': 'FSDP_val_5', 'sam_lr': 1e-5, 'head_lr': 1e-4}
    (base / 'config.json').write_text(json.dumps(old))
    model = nn.Linear(2, 1)
    torch.save({'config': old, 'epoch': 30, 'best_score': foundation.FOUNDATION_DICE, 'best_threshold': .35,
                'model_state_dict': model.state_dict()}, base / 'hybrid_best.pth')
    metric = lambda value: {'mean_bilateral_patient_dice': value}
    summary = {'complete': True, 'experiment': 'sam_turbo_cnn_refine32', 'train_patients': 54,
               'validation_patients': 10, 'reserved_test_patients': 20, 'test_images_read': False,
               'best_validation_metrics': {'1': metric(.37), '5': metric(foundation.FOUNDATION_DICE)},
               'best_frozen_training_metrics': {'1': metric(.4), '5': metric(.5)},
               'validation_control_fixed_0_5': metric(.43), 'best_epoch': 30, 'threshold': .35,
               'best_checkpoint_sha256': file_hash(base / 'hybrid_best.pth'),
               'detail_feature_grid': [32] * 3, 'sam_token_grid': [8] * 3, 'oracle_clicks_per_side': [1, 5],
               'physical_policy': prepared['policy']}
    (base / 'summary.json').write_text(json.dumps(summary))
    return base, old, summary, prepared, model


class FoundationTests(unittest.TestCase):
    def test_cli_inherits_recipe_and_only_reduces_learning_rates_in_new_run(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            base, old, summary, _, _ = fixture(root)
            captured = []
            def execute(args):
                prepared, config = foundation.prepare_foundation(args)
                captured.append((args, config))
                self.assertEqual(prepared['policy'], old['physical_policy'])
            argv = ['train_fsdp_foundation.py', '--foundation-run-dir', str(base),
                    '--run-dir', str(root / 'continuation'), '--device', 'cpu', '--epochs', '1']
            with patch.object(foundation, 'FOUNDATION_SHA256', summary['best_checkpoint_sha256']), \
                 patch.object(sys, 'argv', argv), patch.object(trainer, 'execute', execute):
                foundation.main()
            args, config = captured[0]
            self.assertAlmostEqual(args.sam_lr, old['sam_lr'] * .2)
            self.assertAlmostEqual(args.head_lr, old['head_lr'] * .2)
            self.assertEqual(args.augmentation, 'light')
            self.assertEqual(args.accumulation, old['accumulation'])
            self.assertEqual(args.head_warmup_epochs, 0)
            self.assertEqual(file_hash(base / 'hybrid_data.pt'), config['cache_sha256'])

    def test_hashes_frozen_split_cached_roi_and_full_weights_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            base, old, summary, prepared, model = fixture(root)
            args = argparse.Namespace(foundation_run_dir=base, run_dir=root / 'new', resume=False,
                sam_dir=Path(old['sam_dir']), checkpoint=Path(old['checkpoint']), **foundation.recipe_values(old))
            args.run_dir.mkdir()
            with patch.object(foundation, 'FOUNDATION_SHA256', summary['best_checkpoint_sha256']):
                foundation.inspect_foundation(base)
                actual, config = foundation.prepare_foundation(args)
                self.assertEqual(actual['policy'], prepared['policy'])
                self.assertEqual(config['cache_sha256'], old['cache_sha256'])
                original = copy.deepcopy(model.state_dict())
                with torch.no_grad():
                    model.weight.add_(20)
                foundation.load_foundation_weights(model, args, config)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, original[key], rtol=0, atol=0)
                (base / 'hybrid_data.pt').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    foundation.inspect_foundation(base)

    def test_baseline_reproduction_includes_train_val_clicks_threshold_and_fixed_control(self):
        train = {'refined': {}}
        val = {'refined': {}}
        expected = {'source_threshold': .35, 'expected_train': {}, 'expected_validation': {},
                    'expected_validation_fixed_0_5': {'mean_bilateral_patient_dice': .43}}
        for clicks in (1, 5):
            for bundle, key, score in ((train, 'expected_train', .5), (val, 'expected_validation', foundation.FOUNDATION_DICE)):
                expected[key][str(clicks)] = {'mean_bilateral_patient_dice': score}
                bundle['refined'][clicks] = {.35: {'metrics': expected[key][str(clicks)].copy()}}
        val['refined'][5][.5] = {'metrics': {'mean_bilateral_patient_dice': .43}}
        with contextlib.redirect_stdout(io.StringIO()):
            foundation.verify_reproduction(.35, train, val, expected)
        val['refined'][1][.35]['metrics']['mean_bilateral_patient_dice'] = .2
        with self.assertRaisesRegex(ValueError, 'does not reproduce'):
            foundation.verify_reproduction(.35, train, val, expected)

    def test_regression_keeps_foundation_checkpoint_as_best(self):
        from test_sam_refine32 import FakeSAM
        from sam_refine32 import SamRefine32
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            metrics = lambda score: {'mean_bilateral_patient_dice': score}
            meta = {'source_best_epoch': 30, 'source_threshold': .35,
                    'expected_train': {str(k): metrics(.5) for k in (1, 5)},
                    'expected_validation': {str(k): metrics(foundation.FOUNDATION_DICE) for k in (1, 5)},
                    'expected_validation_fixed_0_5': metrics(.43)}
            config = {'experiment': 'synthetic_continuation', 'protocol': 'synthetic', 'foundation': meta}
            (root / 'config.json').write_text(json.dumps(config))
            model = SamRefine32(FakeSAM())
            initial = copy.deepcopy(model.state_dict())
            def bundle(score):
                return {'metrics': metrics(score), 'sides': [], 'patients': [{'case': 'val_01', 'dice_full_target': score}]}
            def evaluate(*unused, **kwargs):
                equal = all(torch.equal(v, initial[k]) for k, v in model.state_dict().items())
                score = foundation.FOUNDATION_DICE if equal else .2
                refined = {k: {.35: bundle(score), .5: bundle(.43)} for k in (1, 5)}
                return {'refined': refined, 'coarse_on_detail': refined}
            def calibrate(*unused):
                validation = evaluate()
                return .35, {'refined': {k: {.35: bundle(.5)} for k in (1, 5)}}, validation
            def epoch(*values):
                self.assertEqual(values[6], 30)
                with torch.no_grad():
                    model.residual.bias.add_(1)
                values[2].step()
                return {'optimizer_updates': 1, 'amp_skipped_updates': 0, 'expected_updates': 1, 'loss': 1.}
            args = argparse.Namespace(run_dir=root, resume=False, seed=2026, device='cpu', precision='bf16',
                epochs=1, val_every=1, accumulation=8, sam_lr=2e-6, head_lr=2e-5, head_warmup_epochs=0,
                auxiliary_weight=.25, augmentation='light', patience_epochs=50, min_delta=.001)
            with patch.object(trainer, 'prepare', return_value=({'policy': {}, 'coverage': {}}, config)), \
                 patch.object(trainer, 'build', return_value=(model, state_hash(model.sam.state_dict()))), \
                 patch.object(foundation, 'load_foundation_weights'), patch.object(trainer, 'calibrate_validate', calibrate), \
                 patch.object(trainer, 'train_epoch', epoch), patch.object(trainer, 'plot_results'), \
                 patch.object(trainer, 'evaluate', evaluate):
                trainer.execute(args)
            best = torch.load(root / 'hybrid_best.pth', weights_only=False)
            self.assertEqual(best['epoch'], 0)
            for key, value in best['model_state_dict'].items():
                torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
            summary = json.loads((root / 'summary.json').read_text())
            self.assertEqual(summary['validation_change_vs_foundation'], 0.)
            self.assertTrue(summary['selection_includes_unchanged_foundation'])


if __name__ == '__main__':
    unittest.main()
