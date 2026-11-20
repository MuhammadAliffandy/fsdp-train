"""Adapt only image-encoder blocks 10/11 and the mask decoder from original Turbo."""

import hashlib
import json
from pathlib import Path

import torch

from decoder_experiment import DecoderExperiment
from evaluate_private_validation import file_hash


TRAINABLE_PREFIXES = ('image_encoder.blocks.10.', 'image_encoder.blocks.11.', 'mask_decoder.')


def frozen_state_hash(state):
    digest = hashlib.sha256()
    keys = sorted(key for key in state if not key.startswith(TRAINABLE_PREFIXES))
    if not keys or not any(key.startswith('prompt_encoder.') for key in keys):
        raise ValueError('Expected frozen SAM state including prompt encoder')
    for key in keys:
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class PartialExperiment(DecoderExperiment):
    def configuration(self, args, manifest, config):
        result = super().configuration(args, manifest, config)
        return {**result, 'experiment': 'v6_partial_encoder',
                'optimizer': 'AdamW; image_encoder.blocks.10/11 and mask_decoder only; weight_decay=0.1',
                'encoder_mode': '12-block encoder expected; only blocks10/11 train; early blocks, patch/position embedding, neck and prompt encoder frozen/eval',
                'trainable_prefixes': list(TRAINABLE_PREFIXES),
                'partial_experiment_sha256': file_hash(Path(__file__)),
                'partial_entrypoint_sha256': file_hash(Path(__file__).with_name('train_private_partial.py'))}

    def configure_model(self, model, run):
        blocks = getattr(model.image_encoder, 'blocks', None)
        if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) != 12:
            raise ValueError('V6 requires vit_b_ori image_encoder.blocks with exactly12 blocks')
        names = []
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(TRAINABLE_PREFIXES))
            parameter.grad = None
            if parameter.requires_grad:
                names.append(name)
        if any(not any(name.startswith(prefix) for name in names) for prefix in TRAINABLE_PREFIXES):
            raise ValueError('Missing trainable block or decoder parameters')
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.parameter_counts = {'total': sum(p.numel() for p in model.parameters()),
                                 'trainable': sum(p.numel() for p in parameters)}
        self.parameter_counts['frozen'] = self.parameter_counts['total'] - self.parameter_counts['trainable']
        self.parameter_counts['by_trainable_region'] = {
            prefix: sum(p.numel() for name, p in model.named_parameters() if name.startswith(prefix))
            for prefix in TRAINABLE_PREFIXES}
        (run / 'trainable_parameters.json').write_text(json.dumps(
            {**self.parameter_counts, 'trainable_names': names}, indent=2))
        print(json.dumps({'partial_encoder_parameter_counts': self.parameter_counts}), flush=True)
        return parameters

    @staticmethod
    def train_mode(model):
        model.image_encoder.eval()
        model.prompt_encoder.eval()
        for index in (10, 11):
            model.image_encoder.blocks[index].train()
        model.mask_decoder.train()
        if any(p.requires_grad != name.startswith(TRAINABLE_PREFIXES) for name, p in model.named_parameters()):
            raise ValueError('V6 parameter trainability differs from its frozen policy')

    def verify_frozen(self, model, args, run):
        if not hasattr(self, 'reference_frozen_hash'):
            source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            self.reference_frozen_hash = frozen_state_hash(source.get('model_state_dict', source))
            del source
        observed = frozen_state_hash(model.state_dict())
        if observed != self.reference_frozen_hash:
            raise ValueError('Frozen parameters or persistent buffers differ from original Turbo')
        if any(p.grad is not None for name, p in model.named_parameters() if not name.startswith(TRAINABLE_PREFIXES)):
            raise ValueError('Unexpected gradient outside the two final blocks and decoder')
        (run / 'frozen_regions_check.json').write_text(json.dumps(
            {'matches_original_turbo': True, 'frozen_state_sha256': observed,
             'excluded_trainable_prefixes': list(TRAINABLE_PREFIXES),
             'scope': 'all parameters/persistent buffers except blocks10/11 and decoder; initialization/resume, validation epochs and completion'}, indent=2))

    def finalize(self, model, patients_by_split, device, means, args, run, provenance):
        result = super().finalize(model, patients_by_split, device, means, args, run, provenance)
        result.pop('frozen_encoders_match_original_turbo')
        return {**result, 'experiment': 'v6_partial_encoder',
                'frozen_regions_match_original_turbo': True, 'trainable_prefixes': list(TRAINABLE_PREFIXES),
                'comparison_note': 'Original Turbo initialization and v4 ROI/loss/click/split/schedule; train only image encoder blocks10/11 and mask decoder. Frozen neck remains differentiable for gradients to selected blocks. GT-assisted validation.'}
