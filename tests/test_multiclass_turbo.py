import argparse
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import multiclass_data as data
import multiclass_model as architecture
import train_multiclass_turbo as trainer
import physical_fsdp as physical


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv3d(1, 8, 1)
        self.blocks = nn.ModuleList([nn.Conv3d(8, 8, 1) for _ in range(12)])

    def forward(self, image):
        result = self.first(F.avg_pool3d(image, 16))
        for layer in self.blocks:
            result = result + .1 * layer(result)
        return result


def synthetic_case():
    labels = np.zeros((32, 36, 40), np.uint8)
    for value in range(1, 18):
        x, y, z = 4 + (value % 4) * 5, 4 + (value // 4) * 5, 8 + value % 3
        labels[x:x+2, y:y+2, z:z+3] = value
    ct = np.where(labels, -700, 0).astype(np.float32)
    meta = {'size': list(ct.shape), 'spacing': [.25] * 3, 'origin': [-10., -20., -30.], 'direction': np.eye(3).ravel().tolist()}
    image = physical.image_xyz(ct, meta)
    norm = {'clip_lower': -1000., 'clip_upper': 200., 'mean': -700., 'std': 300.}
    plan = {'shape': [128] * 3, 'spacing': [.25] * 3, 'anchor': 'source_origin', 'origin_shift_fraction_xyz': [0.] * 3}
    return data.make_case(image, ct, labels, 'train_01', plan, norm), image, labels, norm


class ArchitectureTests(unittest.TestCase):
    def test_64_is_learned_detail_and_identical_weights_for_comparison(self):
        torch.manual_seed(2026)
        model = architecture.MulticlassTurbo(Encoder(), 64, semantic_channels=8).eval()
        original = trainer.state_hash(model.state_dict())
        image = torch.randn((1, 1, 128, 128, 128))
        context = torch.randn_like(image)
        with torch.inference_mode():
            logits, shapes = model(image, context, (0, 0, 0), (128,) * 3, True)
        self.assertEqual(tuple(logits.shape), (1, 18, 128, 128, 128))
        self.assertEqual(shapes['local_detail'][-3:], [64] * 3)
        self.assertEqual(shapes['full_skip'][-3:], [128] * 3)
        model.detail_grid = 32
        with torch.inference_mode():
            _, shapes = model(image, context, (0, 0, 0), (128,) * 3, True)
        self.assertEqual(shapes['local_detail'][-3:], [32] * 3)
        self.assertEqual(trainer.state_hash(model.state_dict()), original)

    def test_global_to_local_spatial_coordinates(self):
        axis = 2 * (torch.arange(8).float() + .5) / 8 - 1
        feature = axis[None, None, :, None, None].expand(1, 1, 8, 8, 8)
        result = architecture.spatial_crop(feature, (64, 0, 0), (256, 128, 128), 64)
        expected = 2 * (64 + (torch.arange(64).float() + .5) * 2) / 256 - 1
        torch.testing.assert_close(result[0, 0, :, 32, 32], expected)

    def test_valid_loss_excludes_padding_and_reaches_all_present_classes(self):
        logits = torch.randn((1, 18, 6, 6, 6), requires_grad=True)
        target = torch.arange(216).reshape(1, 6, 6, 6) % 18
        valid = torch.ones_like(target, dtype=torch.bool)
        valid[:, 0] = False
        loss = architecture.segmentation_loss(logits, target, valid)
        loss.backward()
        self.assertTrue((logits.grad[:, :, 0] == 0).all())
        self.assertTrue((logits.grad.abs().sum((0, 2, 3, 4)) > 0).all())
        changed = logits.detach().clone(); changed[:, :, 0] = 10000
        torch.testing.assert_close(architecture.segmentation_loss(changed, target, valid), loss)

    def test_stage_and_gradient_through_frozen_neck(self):
        encoder = Encoder()
        model = architecture.MulticlassTurbo(encoder, 64, semantic_channels=8)
        architecture.set_stage(model, 0, 5)
        self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
        architecture.set_stage(model, 5, 5)
        encoder(torch.randn(1, 1, 128, 128, 128)).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in encoder.blocks[9].parameters()))
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.blocks[10].parameters()))
        self.assertFalse(encoder.training)
        self.assertTrue(encoder.blocks[11].training)


class DataTests(unittest.TestCase):
    def test_17_labels_native_restore_padding_and_patch_pairing(self):
        case, image, original, _ = synthetic_case()
        self.assertEqual(set(np.unique(case['target'])), set(range(18)))
        self.assertTrue(all(r['roundtrip_dice'] >= .95 for r in case['coverage']))
        self.assertTrue(all(r['retention'] for r in case['coverage']))
        start = data.choose_start(case, 2026)
        np.testing.assert_array_equal(start, data.choose_start(case, 2026))
        patch, target, valid = data.local_patch(case, start)
        self.assertEqual(tuple(patch.shape), (1, 1, 128, 128, 128))
        self.assertTrue((patch[:, 0][~valid] == 0).all())
        self.assertFalse(target[~valid].any())
        np.testing.assert_array_equal(physical.xyz(image), np.where(original, -700, 0))
        self.assertEqual(len(data.window_starts((160, 192, 224))), 8)

    def test_full_color_mapping_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / 'ColorTable').mkdir()
            file = root / 'ColorTable/123_ColorTable.csv'
            file.write_text('LabelValue,Name\n1,RFSDP\n9,LFSDP\n')
            with self.assertRaisesRegex(ValueError, 'every class'):
                data.color_mapping(root, {'patient_id': '123'})
            file.write_text('LabelValue,Name\n' + '\n'.join(f'{i},LABEL{i}' for i in range(1, 18)))
            self.assertEqual(len(data.color_mapping(root, {'patient_id': '123'})), 18)

    def test_confusion_does_not_inflate_empty_classes_and_union_handles_side_swap(self):
        c = np.zeros((18, 18), np.int64)
        c[0, 0] = 100; c[1, 9] = 5; c[9, 1] = 7; c[2, 2] = 10; c[0, 3] = 2
        rows, score = trainer.confusion_metrics(c, {str(i): str(i) for i in range(18)}, 'val_01')
        self.assertEqual(score['fsdp_bilateral_dice'], 1.)
        self.assertEqual(rows[0]['dice'], 0.)
        self.assertIsNone(rows[3]['dice'])
        self.assertEqual(score['evaluable_classes'], 4)
        self.assertEqual(score['macro_dice'], .25)

class ProxySemantic(nn.Module):
    def __init__(self, encoder, grid):
        super().__init__()
        self.image_encoder, self.detail_grid = encoder, grid
        self.detail = nn.Conv3d(1, 1, 1)
        self.classifier = nn.Conv3d(1, 18, 1)

    def encode(self, context):
        return self.image_encoder(context)

    def decode(self, image, embedding, offset, shape):
        return self.classifier(self.detail(image) + embedding.mean())

    def forward(self, image, context, offset, shape, return_shapes=False):
        logits = self.decode(image, self.encode(context), offset, shape)
        return (logits, {'logits': list(logits.shape)}) if return_shapes else logits


class WorkflowTests(unittest.TestCase):
    def test_whole_field_native_inference_reconstructs_all17_classes(self):
        class Oracle(nn.Module):
            amp_dtype = torch.bfloat16
            def encode(self, context):
                return None
            def decode(self, patch, embedding, offset, shape):
                logits = torch.zeros((1, 18, 128, 128, 128))
                return logits.scatter_(1, patch.long(), 20.)
        case, _, _, _ = synthetic_case()
        case['image'] = case['target'][None, None].float()
        prediction = trainer.infer_case(Oracle(), case, torch.device('cpu'))
        np.testing.assert_array_equal(prediction, case['native_target'])

    def test_global_plan_is_selected_on_train_before_any_validation(self):
        import copy
        records = {'train': [{'patient_id': '1'}, {'patient_id': '2'}], 'val': [{'patient_id': '3'}]}
        inspected = [{'case': 'train_01'}, {'case': 'train_02'}]
        sources = {'train': [{'patient_id': '1'}, {'patient_id': '2'}], 'val': [{'patient_id': '3'}]}
        policy = {'required_field_mm': [32.] * 3, 'normalization': {}, 'mapping': {str(i): str(i) for i in range(18)},
                  'train_voxel_counts_by_class': {i: 1 for i in range(1, 18)}}
        calls = []
        def load(record, root, expected, alias, mapping):
            return None, None, None, {'case': alias}, mapping
        def make(image, ct, labels, alias, plan, norm):
            calls.append((alias, plan['anchor'], plan['origin_shift_fraction_xyz']))
            good = plan['anchor'] == 'bbox_center' and plan['origin_shift_fraction_xyz'] == [.5] * 3
            return {'minimum_margin_mm': 6., 'coverage': [{'case': alias, 'label': 1, 'target_voxels': 1,
                'retention': True, 'roundtrip_dice': .99 if good else .9}]}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(data, 'audit_train', return_value=(copy.deepcopy(policy), inspected)), \
             patch.object(data.physical, 'records_for', side_effect=lambda manifest, split: records[split]), \
             patch.object(data, 'load', load), patch.object(data, 'make_case', make):
            prepared = data.prepare({}, Path(directory), sources, Path(directory) / 'prepared')
        self.assertEqual(len(calls), 7)
        self.assertEqual(calls[-1], ('val_01', 'bbox_center', [.5] * 3))
        self.assertEqual(len(prepared['paths']['train']), 2)
        self.assertFalse(prepared['test_images_read'])
        self.assertTrue(prepared['policy']['attempts'][-1]['eligible'])

    def test_unattended_two_variants_checkpoint_resume_and_archive(self):
        import tarfile
        case, _, _, _ = synthetic_case()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            (root / 'preprocessing').mkdir()
            paths = {}
            coverage = {}
            for split in ('train', 'val'):
                path = root / f'{split}_01.pt'
                item = {**case, 'case': f'{split}_01'}
                torch.save(item, path)
                paths[split] = [str(path)]
                coverage[split] = case['coverage']
            prepared = {'paths': paths, 'policy': {'mapping': {str(i): str(i) for i in range(18)},
                        'selected': {'spacing': [.25] * 3}}, 'coverage': coverage}
            config = {'experiment': 'synthetic_test_only', 'protocol': 'synthetic GT-field test only'}
            for name in ('config.json', 'preprocessing/policy.json', 'preprocessing/audit_train.json'):
                physical.save(root / name, config)
            for split in ('train', 'val'):
                (root / f'preprocessing/{split}_coverage.csv').write_text('synthetic_test\n')
            args = argparse.Namespace(run_dir=root, resume=False, seed=2026, device='cpu', precision='bf16',
                sam_dir=root, checkpoint=root / 'unused', epochs=2, val_every=1, patches_per_patient=1,
                accumulation=4, encoder_lr=1e-5, head_lr=1e-4, head_warmup_epochs=1,
                patience_epochs=50, min_delta=.001, stride=96, detail_grids=[64, 32])
            def encoder(*unused):
                result = Encoder()
                return result, trainer.state_hash(result.state_dict())
            with patch.object(trainer, 'control', return_value=(prepared, config)), \
                 patch.object(trainer, 'load_encoder', encoder), patch.object(trainer, 'MulticlassTurbo', ProxySemantic):
                original = trainer.train_epoch
                def interrupt(*values):
                    if values[0].detail_grid == 64 and values[5] == 1:
                        raise RuntimeError('simulated interruption')
                    return original(*values)
                with patch.object(trainer, 'train_epoch', interrupt), self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    trainer.execute(args)
                saved = torch.load(root / 'detail64/model_latest.pth', weights_only=False)
                self.assertEqual(saved['epoch'], 1)
                args.resume = True
                trainer.execute(args)
            summary = physical.read(root / 'summary.json')
            self.assertTrue(summary['complete'])
            self.assertFalse(summary['test_images_read'])
            self.assertEqual([r['detail_grid'] for r in summary['comparison']], [64, 32])
            self.assertEqual(len({r['initial_state_sha256'] for r in summary['comparison']}), 1)
            latest = torch.load(root / 'detail64/model_latest.pth', weights_only=False)
            self.assertEqual([r['stage'] for r in latest['history']], ['decoder_warmup', 'partial_encoder_and_decoder'])
            self.assertTrue(all(r['optimizer_updates'] == 1 for r in latest['history']))
            with tarfile.open(root / 'results.tar.gz') as package:
                names = package.getnames()
            self.assertEqual(len([n for n in names if n.endswith('.png')]), 6)
            self.assertFalse(any(n.endswith('.pth') for n in names))
            trainer.cached.cache_clear()


class LauncherTests(unittest.TestCase):
    def test_child_failure_logs_gpu_arguments_and_spaced_paths(self):
        import os
        import shutil
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'project with spaces'
            scripts = root / 'scripts'; scripts.mkdir(parents=True)
            launcher = scripts / 'launch_multiclass_nohup.sh'
            shutil.copyfile(Path(__file__).resolve().parents[1] / 'scripts/launch_multiclass_nohup.sh', launcher)
            sam = root / 'SAM source'; (sam / 'segment_anything').mkdir(parents=True)
            (sam / 'segment_anything/build_sam3D.py').write_text('synthetic')
            checkpoint = sam / 'turbo.pth'; checkpoint.write_text('synthetic')
            control = root / 'old cohort'; control.mkdir()
            for name in ('config.json', 'summary.json', 'source_provenance.json'):
                (control / name).write_text('{}')
            tools = root / 'bin'; tools.mkdir()
            child = tools / 'python'
            child.write_text('#!/bin/sh\nprintf "%s\\n" "$CUDA_VISIBLE_DEVICES" > "$TASK_GPU_RECORD"\nprintf "%s\\n" "$@" > "$TASK_ARGS_RECORD"\necho simulated-child-failure\nexit 3\n')
            child.chmod(0o755)
            env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'], CONDA_DEFAULT_ENV='fsdp-med3d',
                SAM_MED3D_DIR=str(sam), TURBO_CHECKPOINT=str(checkpoint), UNET_COHORT_RUN_DIR=str(control),
                MULTICLASS_RUN_NAME='fsdp_multiclass_test', GPU_ID='2', TASK_GPU_RECORD=str(root / 'gpu'),
                TASK_ARGS_RECORD=str(root / 'args'), RESUME='0')
            result = subprocess.run(['bash', str(launcher)], env=env, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('simulated-child-failure', result.stderr)
            self.assertEqual((root / 'gpu').read_text().strip(), '2')
            self.assertIn('--detail-grids\n64\n32\n', (root / 'args').read_text())
            self.assertTrue((root / 'logs/multiclass/latest.log').is_file())
            self.assertEqual(checkpoint.read_text(), 'synthetic')
            wrapper = scripts / 'launch_multiclass_tokens16_nohup.sh'
            shutil.copyfile(Path(__file__).resolve().parents[1] / 'scripts' / wrapper.name, wrapper)
            env['MULTICLASS_RUN_NAME'] = 'fsdp_multiclass_tokens16_test'
            result = subprocess.run(['bash', str(wrapper)], env=env, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('simulated-child-failure', result.stderr)
            self.assertIn('--encoder-grid\n16\n--detail-grids\n64\n--epochs', (root / 'args').read_text())
            self.assertTrue((root / 'logs/multiclass_tokens16/latest.log').is_file())
            self.assertTrue((root / 'logs/multiclass/latest.log').is_file())


if __name__ == '__main__':
    unittest.main()
