import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import compare_side_validation as comparison
import train_private_tiny as tiny
from evaluate_private_validation import file_hash
from test_tiny_training import fixture, ToySAM


def checkpoints(root, data, manifest):
    turbo = root / 'turbo.pth'
    torch.save({'model_state_dict': ToySAM().state_dict(), 'epoch': 0}, turbo)
    runs = {}
    common = {'manifest_sha256': file_hash(data / 'manifest.json'), 'initial_checkpoint_sha256': file_hash(turbo), 'seed': 2026}
    for name, epoch, bias in [('tiny', 190, -.2), ('v2', 20, -.3)]:
        run = root / name
        run.mkdir()
        runs[name] = run
        config = dict(common)
        if name == 'tiny':
            config['shared_code_sha256'] = {'evaluate_private_validation.py': file_hash(Path(tiny.__file__).with_name('evaluate_private_validation.py'))}
        (run / 'config.json').write_text(json.dumps(config))
        model = ToySAM()
        with torch.no_grad():
            model.bias.fill_(bias)
        state = {'model_state_dict': model.state_dict(), 'epoch': epoch, 'config': config}
        if name == 'v2':
            state['used_datas'] = [str(data / 'train/ct_private')]
        path = run / ('sam_model_tiny_best.pth' if name == 'tiny' else 'sam_model_val_best.pth')
        torch.save(state, path)
        summary = {'best_epoch': epoch, 'test_images_read': False}
        if name == 'tiny':
            summary.update(train_patients=2, side_rois=4, validation_images_read=False,
                           checkpoint_sha256=file_hash(path), best_training_only_dice={'5': {'mean_bilateral_patient_dice': .79}})
            ids = set(tiny.select_patients(manifest, 2026))
            (run / 'selected_training_patients.json').write_text(json.dumps([row for row in manifest['cases'] if row['patient_id'] in ids]))
        (run / 'summary.json').write_text(json.dumps(summary))
    audit = root / 'v2_audit.json'
    audit.write_text(json.dumps({'checkpoint': str(runs['v2'] / 'sam_model_val_best.pth'),
                                'checkpoint_sha256': file_hash(runs['v2'] / 'sam_model_val_best.pth')}))
    sam = root / 'sam'
    (sam / 'segment_anything').mkdir(parents=True)
    (sam / 'segment_anything/build_sam3D.py').write_text('# proxy SAM for CPU flow verification\n')
    return argparse.Namespace(data_dir=data, source_root=root / 'normal case_86', tiny_run_dir=runs['tiny'],
                              v2_run_dir=None, v2_audit_summary=audit, turbo_checkpoint=turbo, sam_dir=sam,
                              output_dir=root / 'comparison', seed=2026, device='cpu')


class SideValidationTests(unittest.TestCase):
    def test_pairwise_changes_are_patient_matched_and_disclose_total_clicks(self):
        rows = [{'checkpoint': name, 'case': alias, 'clicks': 5, 'dice_full_target': score}
                for name, alias, score in [('turbo_original', 'val_01', .2), ('turbo_original', 'val_02', .6),
                                            ('tiny_best', 'val_02', .5), ('v2_best', 'val_01', .4)]]
        paired = comparison.paired_changes(rows)
        self.assertAlmostEqual(paired[0]['delta_dice_vs_turbo'], -.1)
        self.assertAlmostEqual(paired[1]['delta_dice_vs_turbo'], .2)
        self.assertTrue(all(row['total_bilateral_clicks'] == 10 for row in paired))

    def test_checkpoint_provenance_rejects_hash_and_training_split_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, data, manifest = fixture(root, image_split='val')
            args = checkpoints(root, data, manifest)
            checked, _ = comparison.checkpoint_sources(args, manifest, file_hash(data / 'manifest.json'))
            self.assertEqual(len(checked), 3)
            path = args.tiny_run_dir / 'summary.json'
            summary = json.loads(path.read_text())
            summary['checkpoint_sha256'] = 'incorrect'
            path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'completed best summary'):
                comparison.checkpoint_sources(args, manifest, file_hash(data / 'manifest.json'))
            command = ['compare_side_validation.py', '--sam-dir', str(args.sam_dir), '--source-root', str(args.source_root),
                       '--data-dir', str(data), '--tiny-run-dir', str(args.tiny_run_dir), '--turbo-checkpoint', str(args.turbo_checkpoint),
                       '--v2-audit-summary', str(args.v2_audit_summary), '--output-dir', str(args.output_dir), '--device', 'cpu']
            with patch.object(sys, 'argv', command), self.assertRaisesRegex(ValueError, 'completed best summary'):
                comparison.main()
            self.assertFalse(json.loads((args.output_dir / 'summary.json').read_text())['complete'])
            self.assertFalse((args.output_dir / 'inputs').exists())
            summary['checkpoint_sha256'] = file_hash(args.tiny_run_dir / 'sam_model_tiny_best.pth')
            path.write_text(json.dumps(summary))
            selected = args.tiny_run_dir / 'selected_training_patients.json'
            provenance = json.loads(selected.read_text())
            provenance[0] = next(row for row in manifest['cases'] if row['split'] == 'val')
            selected.write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, 'training manifest'):
                comparison.checkpoint_sources(args, manifest, file_hash(data / 'manifest.json'))

    def test_validation_only_three_checkpoints_complete_with_no_training_or_test_files(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory).resolve()
            _, data, manifest = fixture(root, image_split='val')
            args = checkpoints(root, data, manifest)
            args.output_dir.mkdir()
            registry = types.ModuleType('segment_anything.build_sam3D')
            registry.sam_model_registry3D = {'vit_b_ori': lambda checkpoint: ToySAM()}
            modules = {'segment_anything': types.ModuleType('segment_anything'), 'segment_anything.build_sam3D': registry}
            previous = {name: sys.modules.get(name) for name in modules}
            sys.modules.update(modules)
            def stub_overlay(path, *args):
                path.write_text('renderer invocation checked')
            try:
                with patch.object(tiny, 'overlay', stub_overlay):
                    comparison.execute(args)
            finally:
                for name, value in previous.items():
                    if value is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = value
            summary = json.loads((args.output_dir / 'summary.json').read_text())
            self.assertTrue(summary['complete'])
            self.assertFalse(summary['training_images_read'])
            self.assertFalse(summary['test_images_read'])
            self.assertEqual(len(summary['comparison']), 6)
            self.assertTrue(all(row['validation_patients'] == 10 and row['side_rois'] == 20 for row in summary['comparison']))
            self.assertTrue(all(row['minimum_target_retention'] == 1 for row in summary['comparison']))
            provenance = json.loads((args.output_dir / 'selected_validation_patients.json').read_text())
            self.assertEqual({row['patient_id'] for row in provenance}, set(manifest['splits']['val']))
            self.assertFalse((data / 'train').exists())
            self.assertFalse((data / 'test').exists())
            self.assertTrue((args.output_dir / 'paired_changes_vs_turbo.csv').is_file())
            for name in ('turbo_original', 'v2_best', 'tiny_best'):
                self.assertEqual(len(list((args.output_dir / name / 'representatives').glob('*prediction_crop.nii.gz'))), 12)
            with self.assertRaisesRegex(ValueError, 'Only training diagnostic'):
                tiny.prepare_samples(args.source_root, data, manifest, args.output_dir, 2026, split='test')


if __name__ == '__main__':
    unittest.main()
