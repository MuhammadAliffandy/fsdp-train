"""Decoder-only adaptation with the frozen v4 ROI and inference protocol."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_private_validation import file_hash, write_csv
from roi_zoom import ZoomExperiment, compare_saved, evaluate_zoom


def encoder_hash(state):
    digest = hashlib.sha256()
    keys = sorted(key for key in state if key.startswith(('image_encoder.', 'prompt_encoder.')))
    if not keys:
        raise ValueError('Checkpoint contains no encoder state')
    for key in keys:
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class DecoderExperiment(ZoomExperiment):
    def configuration(self, args, manifest, config):
        result = super().configuration(args, manifest, config)
        self.v4 = args.v4_run_dir
        self.v4_summary = json.loads((self.v4 / 'summary.json').read_text())
        old = json.loads((self.v4 / 'config.json').read_text())
        summary = self.v4_summary
        if (not summary['complete'] or summary['test_images_read'] or summary['train_patients'] != 54
                or summary['validation_patients'] != 10 or old['experiment'] != 'v4_anisotropic_roi_zoom'):
            raise ValueError('Decoder-only comparison requires a completed v4 anisotropic run')
        for name in ('epochs', 'val_every', 'accumulation', 'lr', 'seed', 'patience_epochs', 'min_delta',
                     'manifest_sha256', 'initial_checkpoint_sha256', 'versions', 'sam_source_sha256', 'baseline_sha256'):
            if old[name] != config[name]:
                raise ValueError(f'V5 must keep v4 setting/source {name} unchanged')
        for name in ('zoom_module_sha256', 'zoom_entrypoint_sha256', 'native_roi_shape_selection',
                     'normalization', 'protocol'):
            if old[name] != result[name]:
                raise ValueError(f'V4 ROI/preprocessing implementation differs: {name}')
        if old['shared_code_sha256'] != config['shared_code_sha256']:
            raise ValueError('V4 inference/loss/source helpers differ')
        self.v4_path = self.v4 / 'sam_model_side_val_best.pth'
        if file_hash(self.v4_path) != summary['best_checkpoint_sha256']:
            raise ValueError('V4 checkpoint checksum differs from its summary')
        self.v4_export = Path(summary['export_directory']).resolve()
        if not self.v4_export.is_relative_to(self.v4):
            raise ValueError('V4 exports must belong to its run')
        return {**result, 'experiment': 'v5_decoder_only',
                'optimizer': 'AdamW; mask_decoder parameters only; weight_decay=0.1',
                'encoder_mode': 'image_encoder and prompt_encoder frozen, eval during optimization',
                'decoder_experiment_sha256': file_hash(Path(__file__)),
                'decoder_entrypoint_sha256': file_hash(Path(__file__).with_name('train_private_decoder.py')),
                'v4_control_sha256': {name: file_hash(self.v4 / name) for name in
                    ('summary.json', 'config.json', 'source_provenance.json', 'roi_policy.json', 'sam_model_side_val_best.pth')},
                'v4_export_sha256': {name: file_hash(self.v4_export / name) for name in ('sides.csv', 'patients.csv')}}

    def prepare_split(self, split, samples, patients, directory, run, resume):
        result = super().prepare_split(split, samples, patients, directory, run, resume)
        if json.loads((run / 'roi_policy.json').read_text()) != json.loads((self.v4 / 'roi_policy.json').read_text()):
            raise ValueError('Training-selected ROI differs from completed v4')
        return result

    def configure_model(self, model, run):
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith('mask_decoder.'))
            parameter.grad = None
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError('No mask decoder parameters found')
        self.parameter_counts = {'total': sum(parameter.numel() for parameter in model.parameters()),
                                 'trainable': sum(parameter.numel() for parameter in parameters)}
        self.parameter_counts['frozen'] = self.parameter_counts['total'] - self.parameter_counts['trainable']
        (run / 'trainable_parameters.json').write_text(json.dumps(self.parameter_counts, indent=2))
        print(json.dumps({'decoder_only_parameter_counts': self.parameter_counts}), flush=True)
        return parameters

    @staticmethod
    def train_mode(model):
        model.image_encoder.eval()
        model.prompt_encoder.eval()
        model.mask_decoder.train()
        if any(parameter.requires_grad for module in (model.image_encoder, model.prompt_encoder) for parameter in module.parameters()):
            raise ValueError('An encoder parameter was unexpectedly unfrozen')

    def verify_frozen(self, model, args, run):
        if not hasattr(self, 'reference_encoder_hash'):
            source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            self.reference_encoder_hash = encoder_hash(source.get('model_state_dict', source))
            del source
        observed = encoder_hash(model.state_dict())
        if observed != self.reference_encoder_hash:
            raise ValueError('Frozen encoder parameters or buffers differ from original Turbo')
        if any(parameter.grad is not None for module in (model.image_encoder, model.prompt_encoder) for parameter in module.parameters()):
            raise ValueError('Unexpected gradients on frozen encoders')
        (run / 'frozen_encoder_check.json').write_text(json.dumps(
            {'matches_original_turbo': True, 'encoder_state_sha256': observed,
             'scope': 'encoder parameters and persistent buffers; checked at initialization/resume, validation epochs, final training state and selected-checkpoint completion'}, indent=2))

    def initial_validation(self, model, samples, patients, device, args, run):
        current = json.loads((run / 'source_provenance.json').read_text())
        if current != json.loads((self.v4 / 'source_provenance.json').read_text()):
            raise ValueError('Original sources differ from completed v4')
        source = torch.load(self.v4_path, map_location='cpu', weights_only=False)
        if source['epoch'] != self.v4_summary['best_epoch']:
            raise ValueError('V4 checkpoint epoch differs from completed summary')
        model.load_state_dict(source['model_state_dict'], strict=True)
        del source
        control, sides, bilateral = evaluate_zoom(model, samples, patients, device, progress_label='v4_zoom_control')
        error = compare_saved(self.v4_export, sides, bilateral)
        if not np.isclose(control['5']['mean_bilateral_patient_dice'], self.v4_summary['best_validation_dice']['5']['mean_bilateral_patient_dice'], atol=1e-4, rtol=0):
            raise ValueError('V4 aggregate score differs from completed summary')
        folder = run / 'v4_control'
        folder.mkdir()
        write_csv(folder / 'sides.csv', sides)
        write_csv(folder / 'patients.csv', bilateral)
        source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(source.get('model_state_dict', source), strict=True)
        del source
        initial, sides, bilateral = evaluate_zoom(model, samples, patients, device, progress_label='turbo_zoom_initial')
        initial_error = 0.
        # V4 stored its initial rows at the run root, rather than an export directory.
        for name, rows, keys in [('sides', sides, ('case', 'side', 'clicks')), ('patients', bilateral, ('case', 'clicks'))]:
            with (self.v4 / f'validation_initial_{name}.csv').open(newline='') as stream:
                saved = list(csv.DictReader(stream))
            key = lambda row: tuple(str(row[field]) for field in keys)
            lookup = {key(row): float(row['dice_full_target']) for row in saved}
            if len(lookup) != len(rows) or len(saved) != len(rows) or set(lookup) != {key(row) for row in rows}:
                raise ValueError('Initial Turbo coverage differs from v4')
            for row in rows:
                difference = abs(lookup[key(row)] - row['dice_full_target'])
                initial_error = max(initial_error, difference)
                if difference > 1e-4:
                    raise ValueError('Initial Turbo fails to reproduce v4 zoom baseline')
        (run / 'decoder_controls.json').write_text(json.dumps({'v4_zoom': control, 'turbo_zoom': initial,
            'v4_maximum_reproduction_error': error, 'turbo_maximum_reproduction_error': initial_error}, indent=2))
        return initial, sides, bilateral, max(error, initial_error)

    def finalize(self, model, patients_by_split, device, means, args, run, provenance):
        self.verify_frozen(model, args, run)
        controls = json.loads((run / 'decoder_controls.json').read_text())
        reference = controls['v4_zoom']['5']['mean_bilateral_patient_dice']
        return {'experiment': 'v5_decoder_only', 'native_roi_size_xyz': self.native_size_xyz,
                'roi_coverage': self.retention, 'parameter_counts': self.parameter_counts,
                'frozen_encoders_match_original_turbo': True,
                'v4_validation_same_protocol': controls['v4_zoom'],
                'validation_dice_change_vs_v4': means['5']['mean_bilateral_patient_dice'] - reference,
                'comparison_note': 'Same v4 ROI/loss/click/split recipe, original Turbo initialization; only mask decoder trainable and frozen encoders kept in eval. GT-assisted validation.'}
