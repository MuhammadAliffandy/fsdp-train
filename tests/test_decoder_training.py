import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from decoder_experiment import DecoderExperiment, encoder_hash
from evaluate_private_validation import write_csv
from train_private_sides import train_epoch


class Prompt(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))
        self.register_buffer('position_matrix', torch.tensor([.4]))

    def forward(self, points, boxes, masks):
        if self.training:
            raise AssertionError('Prompt encoder must remain in eval')
        return self.weight, masks.mean() * self.weight

    def get_dense_pe(self):
        return self.position_matrix


class Mask(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.3))

    def forward(self, image_embeddings, dense_prompt_embeddings, **kwargs):
        logits = self.weight * image_embeddings.mean() + dense_prompt_embeddings
        return logits.expand(1, 1, 32, 32, 32), torch.zeros(1)


class MiniSAM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = torch.nn.Sequential(torch.nn.Conv3d(1, 1, 1), torch.nn.BatchNorm3d(1), torch.nn.Dropout(.5))
        self.prompt_encoder = Prompt()
        self.mask_decoder = Mask()


class DecoderTests(unittest.TestCase):
    def test_nohup_launcher_uses_isolated_logs_gpu_and_paths_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix='fsdp decoder ') as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            launcher = root / 'scripts/launch_private_decoder_nohup.sh'
            shutil.copyfile(Path(__file__).resolve().parents[1] / 'scripts/launch_private_decoder_nohup.sh', launcher)
            (root / 'bin').mkdir()
            python = root / 'bin/python'
            python.write_text('#!/bin/bash\nprintf "GPU=%s\\n" "$CUDA_VISIBLE_DEVICES"\nprintf "%s\\n" "$@"\n')
            python.chmod(0o755)
            for name in ('sam/ckpt/sam_med3d_turbo.pth', 'data/manifest.json',
                         'baseline/summary.json', 'baseline/protocol.json',
                         'v3/summary.json', 'v3/sam_model_side_val_best.pth',
                         'v4/summary.json', 'v4/sam_model_side_val_best.pth'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            env = {**os.environ, 'PATH': str(root / 'bin') + os.pathsep + os.environ['PATH'],
                   'CONDA_DEFAULT_ENV': 'fsdp-med3d', 'SAM_MED3D_DIR': str(root / 'sam'),
                   'PRIVATE_DATA_DIR': str(root / 'data'), 'SIDE_BASELINE_DIR': str(root / 'baseline'),
                   'PRIVATE_V3_RUN_DIR': str(root / 'v3'), 'PRIVATE_V4_RUN_DIR': str(root / 'v4'),
                   'PRIVATE_SOURCE_ROOT': str(root / 'normal case_86'),
                   'DECODER_RUN_NAME': 'fsdp_private_decoder_v5_check', 'GPU_ID': '3'}
            result = subprocess.run(['bash', str(launcher)], env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            log = (root / 'logs/decoder_training/latest.log').read_text()
            self.assertIn('GPU=3', log)
            self.assertIn('--v4-run-dir\n' + str(root / 'v4'), log)
            self.assertIn('--epochs\n200', log)
            self.assertIn('--source-root\n' + str(root / 'normal case_86'), log)
            self.assertTrue((root / 'outputs/private/decoder_training/latest').is_symlink())
            self.assertFalse((root / 'logs/zoom_training').exists())
            (root / 'runs/fsdp_private_decoder_v5_check').mkdir(parents=True)
            refused = subprocess.run(['bash', str(launcher)], env=env, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn('Run exists', refused.stderr)

    def test_real_interactive_training_updates_only_decoder_and_preserves_encoder_buffers(self):
        from monai.losses import DiceCELoss
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            model = MiniSAM()
            torch.save({'model_state_dict': model.state_dict()}, root / 'turbo.pth')
            args = types.SimpleNamespace(checkpoint=root / 'turbo.pth')
            experiment = DecoderExperiment()
            original = {key: value.clone() for key, value in model.state_dict().items()}
            parameters = experiment.configure_model(model, root)
            optimizer = torch.optim.AdamW(parameters, lr=.02, weight_decay=.1)
            self.assertEqual({id(p) for p in parameters}, {id(p) for p in model.mask_decoder.parameters()})
            samples = [{'image': torch.full((1, 1, 4, 4, 4), .4 + i * .1),
                        'target': torch.ones((1, 1, 4, 4, 4), dtype=torch.uint8)} for i in range(3)]
            result = train_epoch(model, samples, optimizer, torch.amp.GradScaler('cuda', enabled=False),
                                 DiceCELoss(sigmoid=True, squared_pred=True), torch.device('cpu'),
                                 0, 2026, 2, train_mode=experiment.train_mode)
            self.assertEqual(result['optimizer_updates'], 2)
            self.assertFalse(model.image_encoder.training)
            self.assertFalse(model.prompt_encoder.training)
            self.assertTrue(model.mask_decoder.training)
            for key, value in model.state_dict().items():
                if key.startswith(('image_encoder.', 'prompt_encoder.')):
                    torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            self.assertFalse(torch.equal(model.mask_decoder.weight, original['mask_decoder.weight']))
            experiment.verify_frozen(model, args, root)
            self.assertTrue(json.loads((root / 'frozen_encoder_check.json').read_text())['matches_original_turbo'])
            with torch.no_grad():
                model.image_encoder[1].running_mean.add_(1)
            with self.assertRaisesRegex(ValueError, 'buffers differ'):
                experiment.verify_frozen(model, args, root)

    def test_initial_control_reproduces_rows_and_restores_turbo_including_frozen_encoders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v4, run, export = root / 'v4', root / 'run', root / 'v4/best_validation'
            run.mkdir()
            export.mkdir(parents=True)
            model = MiniSAM()
            turbo = root / 'turbo.pth'
            torch.save({'model_state_dict': model.state_dict()}, turbo)
            initial_hash = encoder_hash(model.state_dict())
            with torch.no_grad():
                model.image_encoder[0].weight.add_(1)
            checkpoint = v4 / 'best.pth'
            torch.save({'epoch': 15, 'model_state_dict': model.state_dict()}, checkpoint)
            empty_sources = {'train': [], 'val': []}
            for folder in (run, v4):
                (folder / 'source_provenance.json').write_text(json.dumps(empty_sources))
            def metrics(score):
                means = {'5': {'mean_bilateral_patient_dice': score}}
                sides = [{'case': 'val_01', 'side': side, 'clicks': clicks, 'dice_full_target': score}
                         for side in ('RFSDP', 'LFSDP') for clicks in (1, 5)]
                patients = [{'case': 'val_01', 'clicks': clicks, 'dice_full_target': score} for clicks in (1, 5)]
                return means, sides, patients
            control, sides, patients = metrics(.22)
            for name, rows in [('sides', sides), ('patients', patients)]:
                write_csv(export / f'{name}.csv', rows)
            initial, sides, patients = metrics(.04)
            for name, rows in [('sides', sides), ('patients', patients)]:
                write_csv(v4 / f'validation_initial_{name}.csv', rows)
            experiment = DecoderExperiment()
            experiment.v4, experiment.v4_path, experiment.v4_export = v4, checkpoint, export
            experiment.v4_summary = {'best_epoch': 15, 'best_validation_dice': control}
            with patch('decoder_experiment.evaluate_zoom', side_effect=[metrics(.22), metrics(.04)]):
                result = experiment.initial_validation(model, [], {}, torch.device('cpu'),
                                                       types.SimpleNamespace(checkpoint=turbo), run)
            self.assertEqual(result[3], 0.)
            self.assertEqual(encoder_hash(model.state_dict()), initial_hash)
            self.assertTrue((run / 'decoder_controls.json').is_file())
            with patch('decoder_experiment.evaluate_zoom', side_effect=[metrics(.15)]), self.assertRaisesRegex(ValueError, 'fails to reproduce'):
                experiment.initial_validation(model, [], {}, torch.device('cpu'), types.SimpleNamespace(checkpoint=turbo), run)


if __name__ == '__main__':
    unittest.main()
