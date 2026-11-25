import argparse
import contextlib
import copy
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import physical_fsdp as data
import physical_resunet as architecture
import train_physical_resunet as trainer
from evaluate_private_validation import file_hash


class PhysicalResUNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_multiscale_outputs_gradients_and_padding_gradient(self):
        model = architecture.PhysicalResUNet((8, 8, 16, 16, 24))
        image = torch.randn(1, 1, 32, 40, 48)
        target = torch.zeros_like(image).byte(); target[..., 14:17, 18:22, 23:26] = 1
        valid = torch.ones_like(target).bool(); valid[..., :3, :, :] = False
        outputs = model(image)
        self.assertEqual(outputs[0].shape, image.shape)
        self.assertEqual(outputs[1].shape[-3:], (16, 20, 24))
        for output in outputs:
            output.retain_grad()
        loss = architecture.supervised_loss(outputs, target, valid); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(outputs[0].grad[~valid].any())
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
        self.assertGreater(model.encoder[0].layers[0].weight.grad.abs().sum(), 0)
        self.assertTrue(all(head.weight.grad.abs().sum() > 0 for head in model.heads))

    def test_augmentations_are_deterministic_aligned_and_do_not_modify_cache(self):
        image = torch.zeros(1, 1, 32, 32, 32)
        target = image.byte(); target[..., 15:18, 15:18, 15:18] = 1
        image.copy_(target)
        valid = torch.ones_like(target).bool(); valid[..., :3, :, :] = False
        sample = {'image': image, 'target': target, 'valid': valid}
        before = {k: v.clone() for k, v in sample.items()}
        counts = set()
        for seed in range(30):
            first, record = architecture.training_input(sample, torch.device('cpu'), seed, patch=(16, 16, 16))
            second, again = architecture.training_input(sample, torch.device('cpu'), seed, patch=(16, 16, 16))
            self.assertEqual(record, again); counts.add(record['view'])
            for a, b in zip(first, second):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertFalse(first[0][~first[2]].any())
            self.assertFalse(first[1][~first[2]].any())
            if record['foreground_centered']:
                self.assertTrue(first[1].any())
        self.assertEqual(counts, {'full_roi', 'patch'})
        for key in sample:
            torch.testing.assert_close(sample[key], before[key], rtol=0, atol=0)

    def test_empty_foreground_loss_is_finite(self):
        logits = torch.zeros(1, 1, 16, 16, 16, requires_grad=True)
        loss = architecture.supervised_loss((logits, logits, logits), torch.zeros_like(logits), torch.ones_like(logits).bool())
        loss.backward()
        self.assertTrue(loss.isfinite()); self.assertGreater(logits.grad.sum(), 0)

    def test_prepare_validates_same_cache_benchmark_and_strict_resume(self):
        from test_fsdp_verified_cache import fixture
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); dataset, manifest, prepared, _ = fixture(root)
            source = root / 'foundation'; source.mkdir()
            torch.save(prepared, source / 'hybrid_data.pt')
            cohort = root / 'cohort'; cohort.mkdir()
            data_root = root / 'data'; data_root.mkdir()
            data.save(data_root / 'manifest.json', manifest)
            data.save(cohort / 'config.json', {'data_dir': str(data_root), 'source_root': str(dataset)})
            old = {'cohort_run_dir': str(cohort), 'sam_dir': str(root / 'sam'),
                   'cache_sha256': file_hash(source / 'hybrid_data.pt'), 'manifest_sha256': file_hash(data_root / 'manifest.json'),
                   'physical_policy': prepared['policy']}
            compare = root / 'benchmark'; compare.mkdir()
            (compare / 'research_best.pth').write_bytes(b'original SAM benchmark')
            summary = {'complete': True, 'test_images_read': False, 'train_patients': 54, 'validation_patients': 10,
                'best_checkpoint_sha256': file_hash(compare / 'research_best.pth'),
                'best_validation_metrics': {'5': {'mean_bilateral_patient_dice': .4947}}}
            data.save(compare / 'summary.json', summary); data.save(compare / 'config.json', old)
            args = argparse.Namespace(foundation_run_dir=source, sam_comparison_dir=compare, source_root=dataset,
                run_dir=root / 'new', resume=False, seed=2026)
            with patch.object(trainer, 'inspect_foundation', return_value=(old, {})), \
                 patch.object(trainer.shutil, 'disk_usage', return_value=argparse.Namespace(free=100 * 2**30)):
                _, config = trainer.prepare(args)
                self.assertEqual(config['sam_comparison']['dice_5_per_side'], .4947)
                args.resume = True
                self.assertEqual(trainer.prepare(args)[1], config)
                (compare / 'research_best.pth').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'SAM benchmark'):
                    trainer.prepare(args)

    def fixture(self):
        ct = np.arange(16 ** 3, dtype=np.float32).reshape((16,) * 3) / 1000
        image = data.sitk.GetImageFromArray(ct.transpose(2, 1, 0)); image.SetSpacing((.25,) * 3)
        prepared = {'samples': {'train': [], 'val': []}, 'patients': {'train': {}, 'val': {}}}
        for split, count in (('train', 2), ('val', 1)):
            for index in range(1, count + 1):
                case = f'{split}_{index:02d}'; union = np.zeros_like(ct).astype(bool)
                for side, start in (('RFSDP', 4), ('LFSDP', 10)):
                    mask = np.zeros_like(union); mask[start:start+2, 7:9, 7:9] = True; union |= mask
                    prepared['samples'][split].append(data.make_sample(image, ct, mask, case, side,
                        {'shape': [16] * 3, 'spacing': [.25] * 3},
                        {'clip_lower': -1000., 'clip_upper': 1000., 'mean': 0., 'std': 1.}))
                prepared['patients'][split][case] = {'shape': list(ct.shape), 'target_indices': np.flatnonzero(union)}
        return prepared

    def test_real_native_export_and_exact_interrupted_resume(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder); prepared = self.fixture()
            config = {'experiment': 'synthetic', 'sam_comparison': {'dice_5_per_side': .4947}, 'protocol': 'GT ROI0clicks', 'comparison_note': 'synthetic'}
            def prepare(args):
                args.run_dir.mkdir(exist_ok=True)
                data.save(args.run_dir / 'config.json', config)
                return prepared, config
            args = argparse.Namespace(run_dir=root / 'interrupted', device='cpu', seed=2026, precision='bf16',
                                      lr=.001, epochs=2, patience=60, val_every=1, accumulation=3, resume=False)
            model = architecture.PhysicalResUNet
            train = trainer.train_epoch
            def interrupt(*values):
                if values[6] == 1:
                    raise RuntimeError('interrupted')
                return train(*values)
            with patch.object(trainer, 'prepare', side_effect=prepare), \
                 patch.object(trainer, 'PhysicalResUNet', side_effect=lambda: model((8, 8, 16, 16, 24))), \
                 patch.object(trainer, 'overlay'):
                with patch.object(trainer, 'train_epoch', side_effect=interrupt), self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    trainer.execute(args)
                self.assertEqual(torch.load(args.run_dir / 'resunet_latest.pth', weights_only=False)['epoch'], 1)
                args.resume = True; trainer.execute(args)
                resumed = torch.load(args.run_dir / 'resunet_latest.pth', weights_only=False)
                other = copy.copy(args); other.run_dir = root / 'reference'; other.resume = False
                trainer.execute(other)
                reference = torch.load(other.run_dir / 'resunet_latest.pth', weights_only=False)
                self.assertEqual(resumed['history'], reference['history'])
                for key, value in reference['model'].items():
                    torch.testing.assert_close(resumed['model'][key], value, rtol=0, atol=0)
            report = data.read(args.run_dir / 'summary.json')
            self.assertEqual(report['oracle_clicks'], 0); self.assertFalse(report['test_images_read'])
            self.assertTrue(list((args.run_dir / 'best_exports').glob('*prediction.nii.gz')))
            with tarfile.open(args.run_dir / 'results.tar.gz') as archive:
                self.assertFalse(any(n.endswith('.pth') for n in archive.getnames()))

    def test_launcher_gpu_refusal_and_default_full_cohort_command(self):
        for success in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); scripts = root / 'scripts'; scripts.mkdir()
                shutil.copy(Path(trainer.__file__).with_name('launch_physical_resunet_nohup.sh'), scripts)
                (scripts / 'fsdp_research_gpu.py').write_text('print("GPU-test")\n' if success else 'raise SystemExit(3)\n')
                (scripts / 'train_physical_resunet.py').write_text('import os, pathlib, sys\npathlib.Path(os.environ["TASK_CAPTURE"]).write_text(os.environ["CUDA_VISIBLE_DEVICES"] + "\\n" + "\\n".join(sys.argv))\n')
                capture = root / 'capture'
                env = dict(os.environ, TASK_CAPTURE=str(capture), CONDA_DEFAULT_ENV='fsdp-med3d', RESUME='0')
                for key in ('RESUNET_EPOCHS', 'RESUNET_LR', 'PATIENCE_EPOCHS', 'ACCUMULATION_STEPS', 'RESUNET_RUN_NAME'):
                    env.pop(key, None)
                result = subprocess.run(['bash', str(scripts / 'launch_physical_resunet_nohup.sh')], env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0 if success else 3, result.stderr)
                if success:
                    self.assertTrue(capture.read_text().startswith('GPU-test\n'))
                    self.assertIn('--epochs\n300', capture.read_text())
                else:
                    self.assertFalse(capture.exists())


if __name__ == '__main__':
    unittest.main()
