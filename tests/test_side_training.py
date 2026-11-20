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

import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_private_sides as trainer
import train_private_tiny as tiny
from evaluate_private_validation import file_hash, write_csv
from test_tiny_training import fixture, ToyPrompt


class SmallSAM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(.1))
        self.prompt_encoder = ToyPrompt()

    def image_encoder(self, image):
        return image

    def mask_decoder(self, image_embeddings, **kwargs):
        return (self.bias + .001 * image_embeddings.mean()).expand(1, 1, 32, 32, 32), torch.zeros(1)


def small_preprocess(image_path, label_path):
    image, label = sitk.ReadImage(str(image_path)), sitk.ReadImage(str(label_path))
    raw = sitk.GetArrayFromImage(image).transpose(2, 1, 0).copy().astype(np.float32)
    target = sitk.GetArrayFromImage(label).transpose(2, 1, 0).copy()
    count = int(target.sum())
    metadata = {'full_target_voxels': count, 'crop_target_voxels': count, 'target_retention': 1.,
                'crop_origin_mm': list(image.GetOrigin()), 'spacing_mm': list(image.GetSpacing()),
                'direction': list(image.GetDirection())}
    return torch.from_numpy(raw / 1000)[None, None], torch.from_numpy(target)[None, None].long(), raw, metadata, np.ones(target.shape, bool)


@contextlib.contextmanager
def proxy_environment():
    registry = types.ModuleType('segment_anything.build_sam3D')
    registry.sam_model_registry3D = {'vit_b_ori': lambda checkpoint: SmallSAM()}
    modules = {'segment_anything': types.ModuleType('segment_anything'), 'segment_anything.build_sam3D': registry}
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        with patch.object(tiny, 'preprocess', small_preprocess), patch.object(tiny, 'overlay', lambda path, *args: path.write_text('renderer invocation checked')):
            yield
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def setup_run(root):
    source, data, manifest = fixture(root, all_training_patients=True, include_validation=True)
    checkpoint = root / 'turbo.pth'
    torch.save({'model_state_dict': SmallSAM().state_dict()}, checkpoint)
    sam = root / 'sam'
    (sam / 'segment_anything').mkdir(parents=True)
    (sam / 'segment_anything/build_sam3D.py').write_text('# small CPU proxy; actual SAM uses 128 cubes\n')
    baseline = root / 'baseline'
    baseline.mkdir()
    preparation = root / 'baseline_preparation'
    preparation.mkdir()
    samples, patients = tiny.prepare_samples(source, data, manifest, preparation, 2026, split='val')
    _, sides, bilateral = tiny.evaluate(SmallSAM().eval(), samples, patients, torch.device('cpu'))
    (baseline / 'turbo_original').mkdir()
    write_csv(baseline / 'turbo_original/sides.csv', sides)
    write_csv(baseline / 'turbo_original/patients.csv', bilateral)
    (baseline / 'summary.json').write_text(json.dumps({'complete': True, 'validation_patients': 10, 'side_rois': 20,
        'test_images_read': False, 'checkpoints': [{'checkpoint': 'turbo_original', 'sha256': file_hash(checkpoint)}]}))
    (baseline / 'protocol.json').write_text(json.dumps({'manifest_sha256': file_hash(data / 'manifest.json'), 'seed': 2026,
        'sam_builder_sha256': file_hash(sam / 'segment_anything/build_sam3D.py'),
        'shared_code_sha256': {'evaluate_private_validation.py': file_hash(Path(tiny.__file__).with_name('evaluate_private_validation.py'))}}))
    args = argparse.Namespace(sam_dir=sam, source_root=source, data_dir=data, checkpoint=checkpoint, baseline_dir=baseline,
                              run_dir=root / 'run', epochs=3, val_every=1, accumulation=8, lr=.01, seed=2026,
                              patience_epochs=50, min_delta=.001, resume=False, device='cpu')
    return args, sides, bilateral


class SideTrainingTests(unittest.TestCase):
    def test_108_rois_use_correct_partial_accumulation_window(self):
        class Scalar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.2))
        model, reference = Scalar(), Scalar()
        optimizer, expected_optimizer = [torch.optim.SGD(item.parameters(), lr=.05) for item in (model, reference)]
        samples = [{'image': torch.full((1, 1, 2, 2, 2), (i + 1) / 108), 'target': torch.ones((1, 1, 2, 2, 2))} for i in range(108)]
        criterion = torch.nn.MSELoss()
        seen = []
        def loss(model, image, target, criterion, clicks, seed):
            seen.append(float(image.flatten()[0]))
            logits = image * model.weight
            return criterion(logits, target.float()), logits.detach()
        with patch.object(trainer, 'interactive_loss', loss), contextlib.redirect_stdout(io.StringIO()):
            result = trainer.train_epoch(model, samples, optimizer, torch.amp.GradScaler('cuda', enabled=False), criterion, torch.device('cpu'), 0, 2026, 8)
        order = torch.randperm(108, generator=torch.Generator().manual_seed(2026)).tolist()
        for start in range(0, 108, 8):
            indices = order[start:start + 8]
            expected_optimizer.zero_grad()
            value = sum(criterion(samples[i]['image'] * reference.weight, samples[i]['target']) for i in indices) / len(indices)
            value.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
            expected_optimizer.step()
        self.assertEqual(len(set(seen)), 108)
        self.assertEqual(result['optimizer_updates'], 14)
        self.assertEqual(result['amp_skipped_updates'], 0)
        torch.testing.assert_close(model.weight, reference.weight, rtol=1e-6, atol=1e-7)

    def test_full_cohort_preflight_resume_and_selected_checkpoint_scores(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), proxy_environment():
            root = Path(directory).resolve()
            args, sides, bilateral = setup_run(root)
            self.assertEqual(trainer.reproduce_baseline(args.baseline_dir, sides, bilateral), 0)
            wrong = [dict(row) for row in bilateral]
            wrong[0]['dice_full_target'] += .1
            with self.assertRaisesRegex(ValueError, 'fails to reproduce'):
                trainer.reproduce_baseline(args.baseline_dir, sides, wrong)
            args.run_dir.mkdir()
            trainer.execute(args)
            reference = torch.load(args.run_dir / 'sam_model_side_latest.pth', weights_only=False)
            summary = json.loads((args.run_dir / 'summary.json').read_text())
            self.assertEqual(summary['completed_epochs'], 3)
            self.assertEqual(summary['train_side_rois'], 108)
            self.assertEqual(summary['validation_side_rois'], 20)
            self.assertFalse(summary['test_images_read'])
            self.assertFalse((args.data_dir / 'test').exists())
            self.assertTrue(all(row['optimizer_updates'] == 14 for row in reference['history']))
            source_records = json.loads((args.run_dir / 'source_provenance.json').read_text())
            self.assertEqual(len(source_records['train']), 54)
            self.assertEqual(len(source_records['val']), 10)
            self.assertTrue(json.loads((args.run_dir / 'preflight.json').read_text())['passed'])
            self.assertEqual(len(reference['history']), 3)
            args.run_dir = root / 'resumed'
            args.run_dir.mkdir()
            original = trainer.train_epoch
            def interrupted(*values, **kwargs):
                if values[6] == 1:
                    raise RuntimeError('simulated epoch-two interruption')
                return original(*values, **kwargs)
            with patch.object(trainer, 'train_epoch', interrupted), self.assertRaisesRegex(RuntimeError, 'simulated'):
                trainer.execute(args)
            first = torch.load(args.run_dir / 'sam_model_side_latest.pth', weights_only=False)
            self.assertEqual(first['epoch'], 1)
            args.resume = True
            trainer.execute(args)
            resumed = torch.load(args.run_dir / 'sam_model_side_latest.pth', weights_only=False)
            torch.testing.assert_close(reference['model_state_dict']['bias'], resumed['model_state_dict']['bias'], rtol=0, atol=0)
            self.assertEqual(reference['history'], resumed['history'])
            args.lr = .02
            with self.assertRaisesRegex(ValueError, 'Resume code/data/config differs'):
                trainer.execute(args)
            with self.assertRaisesRegex(ValueError, 'Full training selection'):
                tiny.prepare_samples(args.source_root, args.data_dir, json.loads((args.data_dir / 'manifest.json').read_text()), root, 2026,
                                     split='val', all_training_patients=True)


if __name__ == '__main__':
    unittest.main()
