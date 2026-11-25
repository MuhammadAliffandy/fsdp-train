import argparse
import contextlib
import copy
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import physical_fsdp as data
import train_fsdp_research as trainer
import run_fsdp_research as workflow
from evaluate_private_validation import file_hash
from test_fsdp_research_workflow import arguments, completed_child


def fixture(root):
    source = root / 'source'; source.mkdir()
    metrics = {'1': {'mean_bilateral_patient_dice': .4}, '5': {'mean_bilateral_patient_dice': .4918}}
    foundation = {'source_best_epoch': 30}
    control = {'foundation': foundation, 'cache_sha256': 'cache', 'manifest_sha256': 'manifest',
               'physical_policy': {'shape': [96, 160, 208]}, 'versions': {},
               'seed': 2026, 'precision': 'bf16', 'augmentation': 'light', 'accumulation': 8,
               'auxiliary_weight': .25, 'val_every': 5, 'min_delta': .001}
    config = {**control, 'experiment': 'fsdp_research_screening', 'variant': 'history',
              'code_sha256': trainer.code_signature(),
              'parameters': {key: control[key] for key in ('seed', 'precision', 'augmentation', 'accumulation',
                                                          'auxiliary_weight', 'val_every', 'min_delta')},
              'versions': {'scipy': trainer.importlib.metadata.version('scipy')}}
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(.75); model.bias.fill_(.125)
    torch.save({'config': config, 'epoch': 5, 'best_epoch': 5, 'best_score': .4918, 'best_threshold': .35,
                'model_state_dict': model.state_dict(), 'optimizer_state_dict': {'unused': 'fresh optimizer required'}},
               source / 'research_best.pth')
    summary = {'complete': True, 'variant': 'history', 'test_images_read': False,
        'train_patients': 54, 'validation_patients': 10, 'detail_grid': [32] * 3, 'token_grid': [8] * 3,
        'oracle_clicks_per_side': [1, 5], 'foundation': foundation, 'best_epoch': 5, 'completed_epochs': 20,
        'threshold': .35, 'best_checkpoint_sha256': file_hash(source / 'research_best.pth'),
        'best_frozen_training_metrics': metrics, 'best_validation_metrics': metrics,
        'validation_control_fixed_0_5': metrics['5']}
    data.save(source / 'config.json', config); data.save(source / 'summary.json', summary)
    return source, control, config, summary, model


class HistoryContinuationTests(unittest.TestCase):
    def test_history_launcher_defaults_to_single_source_100_epochs_half_screen_lr_and_25_patience(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); scripts = root / 'scripts'; scripts.mkdir()
            shutil.copy(Path(trainer.__file__).with_name('launch_fsdp_history_nohup.sh'), scripts)
            capture = root / 'settings'
            (scripts / 'launch_fsdp_research_nohup.sh').write_text('''#!/bin/sh
printf '%s\\n' "$FSDP_RESEARCH_SOURCE_DIR" "$ADDITIONAL_EPOCHS" "$LR_MULTIPLIER" "$PATIENCE_EPOCHS" "$FSDP_RESEARCH_RUN_NAME" > "$TASK_CAPTURE"
''')
            env = dict(os.environ, TASK_CAPTURE=str(capture))
            for key in ('FSDP_RESEARCH_SOURCE_DIR', 'ADDITIONAL_EPOCHS', 'LR_MULTIPLIER', 'PATIENCE_EPOCHS', 'FSDP_RESEARCH_RUN_NAME'):
                env.pop(key, None)
            result = subprocess.run(['bash', str(scripts / 'launch_fsdp_history_nohup.sh')], env=env,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            settings = capture.read_text().splitlines()
            self.assertEqual(settings[0], str(root / 'runs/fsdp_research_gpu7_20261124T132503Z_3250370/history'))
            self.assertEqual(settings[1:4], ['100', '0.1', '25'])
            self.assertTrue(settings[4].startswith('fsdp_research_history_'))

    def test_best_trained_history_weights_and_cumulative_epoch_are_inherited_without_source_writes(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            source, control, _, summary, original = fixture(Path(folder))
            metadata = trainer.inspect_research_source(source, control)
            self.assertEqual(metadata['source_plus_additional_epoch'], 35)
            model = nn.Linear(2, 1)
            trainer.load_research_source(model, metadata)
            for key, tensor in original.state_dict().items():
                torch.testing.assert_close(model.state_dict()[key], tensor, rtol=0, atol=0)
            self.assertEqual(file_hash(source / 'research_best.pth'), summary['best_checkpoint_sha256'])

    def test_other_variant_cache_changed_implementation_or_modified_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source, control, config, summary, _ = fixture(Path(folder))
            wrong = copy.deepcopy(control); wrong['cache_sha256'] = 'different'
            with self.assertRaisesRegex(ValueError, 'same frozen'):
                trainer.inspect_research_source(source, wrong)
            summary['variant'] = 'refiner'; data.save(source / 'summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'completed history'):
                trainer.inspect_research_source(source, control)
            summary['variant'] = 'history'; data.save(source / 'summary.json', summary)
            config['code_sha256']['fsdp_research_model.py'] = 'changed'; data.save(source / 'config.json', config)
            with self.assertRaisesRegex(ValueError, 'implementation changed'):
                trainer.inspect_research_source(source, control)
            config['code_sha256'] = trainer.code_signature(); data.save(source / 'config.json', config)
            metadata = trainer.inspect_research_source(source, control)
            (source / 'research_best.pth').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum differs'):
                trainer.inspect_research_source(source, control)
            with self.assertRaisesRegex(ValueError, 'changed before load'):
                trainer.load_research_source(nn.Linear(2, 1), metadata)

    def test_source_reproduction_checks_both_click_budgets_and_fixed_threshold(self):
        metrics = {'1': {'mean_bilateral_patient_dice': .4}, '5': {'mean_bilateral_patient_dice': .4918}}
        bundle = {'refined': {k: {t: {'metrics': dict(metrics[str(k)])} for t in (.35, .5)} for k in (1, 5)}}
        source = {'threshold': .35, 'expected_train': metrics, 'expected_validation': metrics,
                  'expected_validation_fixed_0_5': metrics['5']}
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.verify_research_source(.35, bundle, bundle, source)
        wrong = copy.deepcopy(bundle); wrong['refined'][5][.5]['metrics']['mean_bilateral_patient_dice'] = .3
        with self.assertRaisesRegex(ValueError, 'fixed-threshold'):
            trainer.verify_research_source(.35, bundle, wrong, source)
        with self.assertRaisesRegex(ValueError, 'threshold does not reproduce'):
            trainer.verify_research_source(.5, bundle, bundle, source)

    def test_parent_runs_history_only_and_does_not_invent_a_matched_control(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder); args = arguments(root); args.run_dir.mkdir()
            args.research_source_dir = root / 'source'; args.epochs = 100
            def child(command, record):
                self.assertEqual(command[command.index('--variant') + 1], 'history')
                self.assertEqual(command[command.index('--research-source-dir') + 1], str(args.research_source_dir))
                completed_child(command, record)
                summary = data.read(args.run_dir / 'history/summary.json')
                summary.update(research_source={'directory': str(args.research_source_dir)}, validation_change_vs_research_source=.01)
                data.save(args.run_dir / 'history/summary.json', summary)
                return 0
            with patch.object(workflow, 'inspect_foundation'), patch.object(workflow, 'run_child', side_effect=child) as call:
                workflow.execute(args)
            call.assert_called_once()
            report = data.read(args.run_dir / 'summary.json')
            self.assertEqual(report['entries'][0]['variant'], 'history')
            self.assertIsNone(report['entries'][0]['delta_vs_matched_control'])
            self.assertEqual(report['entries'][0]['delta_vs_research_source'], .01)


if __name__ == '__main__':
    unittest.main()
