#!/usr/bin/env python3
"""Audit completed tiny U-Net inference and gradients without optimizer updates."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import train_private_unet as baseline
from evaluate_private_validation import file_hash
from roi_zoom import restore_logits, zoom_samples
from train_private_tiny import prepare_samples


def hard_dice(logits, target, full_count=None, valid=None):
    prediction, target = np.asarray(logits) > 0, np.asarray(target).astype(bool)
    if valid is not None:
        prediction = prediction & valid
    predicted, target_count = int(prediction.sum()), int(target.sum())
    full_count = target_count if full_count is None else full_count
    intersection = int((prediction & target).sum())
    return {'dice': 2 * intersection / (predicted + full_count) if predicted + full_count else 1.,
            'predicted_voxels': predicted, 'target_voxels_in_crop': target_count,
            'full_target_voxels': full_count, 'true_positives': intersection}


def gradient_probe(model, image, target):
    from monai.losses import DiceCELoss
    model.eval()
    model.zero_grad(set_to_none=True)
    logits = baseline.checked_logits(model, image.float())
    criterion = DiceCELoss(sigmoid=True, squared_pred=True)
    dice, bce = criterion.dice(logits, target.float()), criterion.bce(logits, target.float())
    component_gradients = {}
    for name, loss in [('dice', dice), ('bce', bce)]:
        gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0].detach()
        component_gradients[name] = {
            'foreground_absolute_gradient_sum': finite_scalar(gradient[target.bool()].abs().sum()),
            'background_absolute_gradient_sum': finite_scalar(gradient[~target.bool()].abs().sum())}
    (dice + bce).backward()
    present = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    nonfinite = sum(int((~torch.isfinite(value)).sum()) for value in present)
    squared = sum(float(value.double().square().sum()) for value in present)
    norm = squared ** .5 if np.isfinite(squared) else None
    result = {'precision': 'FP32; no autocast', 'dice_loss': finite_scalar(dice.detach()), 'bce_loss': finite_scalar(bce.detach()),
              'combined_loss': finite_scalar((dice + bce).detach()), 'component_logit_gradients': component_gradients,
              'parameters_with_gradient': len(present), 'nonfinite_gradient_elements': nonfinite,
              'global_gradient_l2_norm': norm, 'optimizer_updates': 0}
    model.zero_grad(set_to_none=True)
    return result


def finite_scalar(value):
    value = float(value)
    return value if np.isfinite(value) else None


def history_report(run):
    with (run / 'history.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    measured = [{key: float(row[key]) if row[key] else None for key in
                ('epoch', 'loss', 'moving_model_grid_train_dice', 'selection_score', 'mean_bilateral_patient_dice')}
                for row in rows if row['selection_score']]
    return {'epochs': len(rows), 'successful_optimizer_updates': sum(int(row['optimizer_updates']) for row in rows),
            'amp_skipped_updates': sum(int(row['amp_skipped_updates']) for row in rows),
            'expected_updates': sum(int(row['expected_updates']) for row in rows),
            'measured_epochs': measured}


def execute(args):
    run, output = args.run_dir, args.output_dir
    summary = json.loads((run / 'summary.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    if (not summary['complete'] or summary['stage'] != 'tiny' or summary['train_patients'] != 2
            or summary['validation_images_read'] or summary['test_images_read']):
        raise ValueError('Audit requires a completed two-patient train-only U-Net diagnostic')
    if config['script_sha256'] != file_hash(Path(baseline.__file__)):
        raise ValueError('U-Net training implementation differs from the frozen run')
    for name, checksum in config['shared_code_sha256'].items():
        if checksum != file_hash(Path(baseline.__file__).with_name(name)):
            raise ValueError(f'Frozen helper differs: {name}')
    from monai.networks.nets import UNet
    import inspect
    if file_hash(Path(inspect.getfile(UNet))) != config['monai_source_sha256']:
        raise ValueError('Installed U-Net implementation differs from the diagnostic')
    data = Path(config['data_dir']).resolve()
    frozen_args = SimpleNamespace(v4_run_dir=Path(config['v4_run_dir']).resolve(), seed=config['seed'])
    _, policy, _ = baseline.v4_control(frozen_args, data / 'manifest.json')
    if file_hash(data / 'manifest.json') != config['manifest_sha256']:
        raise ValueError('Frozen manifest differs')
    preparation = output / 'preparation'
    preparation.mkdir()
    samples, patients = prepare_samples(Path(config['source_root']), data,
        json.loads((data / 'manifest.json').read_text()), preparation, config['seed'])
    records = json.loads((preparation / 'selected_training_patients.json').read_text())
    if records != json.loads((run / 'source_provenance.json').read_text())['train']:
        raise ValueError('Two training sources differ from the completed run')
    for sample in samples:
        sample['target'] = sample['target'].to(torch.uint8)
    samples = zoom_samples(samples, policy['selected_native_size_xyz'])
    device = torch.device(args.device)
    results, audit_rows = {}, []
    for checkpoint_name in ('unet_best', 'unet_latest'):
        path = run / f'{checkpoint_name}.pth'
        checksum = file_hash(path)
        if checkpoint_name == 'unet_best' and checksum != summary['best_checkpoint_sha256']:
            raise ValueError('Best checkpoint hash differs from completed summary')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint['config'] != config:
            raise ValueError('Checkpoint/config metadata differ')
        if checkpoint_name == 'unet_best' and (checkpoint['epoch'] != summary['best_epoch']
                or not np.isclose(checkpoint['best_score'], summary['best_selection_score'], atol=1e-6, rtol=0)):
            raise ValueError('Best-checkpoint epoch/selection metadata differ from the summary')
        model = baseline.build_model().to(device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        epoch = checkpoint['epoch']
        del checkpoint
        frozen_means, _, _ = baseline.evaluate(model, samples, patients, device, progress_label=checkpoint_name)
        if checkpoint_name == 'unet_best':
            for key, expected in summary['best_selected_metrics'].items():
                if not np.isclose(frozen_means[key], expected, atol=1e-4, rtol=0):
                    raise ValueError('Best checkpoint does not reproduce the supplied summary')
        mode_difference = None
        for index, sample in enumerate(samples):
            image, target = sample['image'].to(device), sample['target'].to(device).float()
            with torch.no_grad():
                model.eval()
                fp32 = baseline.checked_logits(model, image.float()).float()
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    amp = baseline.checked_logits(model, image).float()
                probability = fp32.sigmoid()
                if index == 0:
                    saved = {key: value.detach().clone() for key, value in model.state_dict().items()}
                    model.train()
                    train_logits = baseline.checked_logits(model, image.float()).float()
                    mode_difference = float((train_logits - fp32).abs().max())
                    model.load_state_dict(saved)
                    model.eval()
                    del saved, train_logits
                gt = sample['target'][0, 0].numpy().astype(bool)
                native_fp32 = restore_logits(fp32[0, 0].cpu().numpy(), sample['native_target'].shape)
                native_amp = restore_logits(amp[0, 0].cpu().numpy(), sample['native_target'].shape)
                synthetic_logits = gt.astype(np.float32) * 2 - 1
                oracle_roundtrip = restore_logits(synthetic_logits, sample['native_target'].shape)
                metrics = {
                    'model_grid_fp32': hard_dice(fp32[0, 0].cpu().numpy(), gt),
                    'model_grid_amp': hard_dice(amp[0, 0].cpu().numpy(), gt),
                    'native_fp32': hard_dice(native_fp32, sample['native_target'], sample['metadata']['full_target_voxels'], sample['valid']),
                    'native_amp': hard_dice(native_amp, sample['native_target'], sample['metadata']['full_target_voxels'], sample['valid']),
                    'label_roundtrip': hard_dice(oracle_roundtrip, sample['native_target'], sample['metadata']['full_target_voxels'], sample['valid'])}
                fg, bg = probability[target.bool()], probability[~target.bool()]
                foreground = int(target.sum())
                row = {'checkpoint': checkpoint_name, 'epoch': epoch, 'case': sample['case'], 'side': sample['side'],
                       'input_min': float(image.min()), 'input_max': float(image.max()),
                       'input_mean': float(image.mean()), 'input_std': float(image.std()),
                       'model_foreground_voxels': foreground, 'model_total_voxels': target.numel(),
                       'foreground_fraction': foreground / target.numel(),
                       'foreground_probability_mean': float(fg.mean()), 'background_probability_mean': float(bg.mean()),
                       'foreground_probability_median': float(fg.median()), 'background_probability_median': float(bg.median()),
                       'amp_fp32_maximum_logit_difference': float((amp - fp32).abs().max()), 'metrics': metrics}
                audit_rows.append(row)
            print(f'{checkpoint_name}: {sample["name"]} audit complete', flush=True)
        probe = gradient_probe(model, samples[0]['image'].to(device), samples[0]['target'].to(device).float())
        results[checkpoint_name] = {'epoch': epoch, 'sha256': checksum, 'reproduced_metrics': frozen_means,
            'train_eval_maximum_logit_difference_first_roi_fp32': mode_difference,
            'first_roi_gradient_probe': probe}
        del model
    report = {'complete': True, 'run': run.name, 'training_history': history_report(run),
              'audit_script_sha256': file_hash(Path(__file__)),
              'checkpoints': results, 'per_side': audit_rows, 'train_patients': 2, 'side_rois': 4,
              'validation_images_read': False, 'test_images_read': False, 'optimizer_updates': 0,
              'training_artifacts_modified': False,
              'limitations': 'Gradient probe is FP32 on the first ROI of each checkpoint; no causal conclusion from loss scale alone. Label roundtrip is a transform check, not an achievable Dice ceiling. No threshold tuning or retraining performed.',
              'artifacts': str(output)}
    baseline.save_json(output / 'summary.json', report)
    overview = {'complete': True, 'run': run.name,
                'update_counts': {key: value for key, value in report['training_history'].items() if key != 'measured_epochs'},
                'curve_first_best_last': [row for row in report['training_history']['measured_epochs']
                                         if row['epoch'] in (1, summary['best_epoch'], summary['completed_epochs'])],
                'checkpoints': {}, 'validation_images_read': False, 'test_images_read': False,
                'optimizer_updates_in_audit': 0, 'artifacts': str(output)}
    for name, result in results.items():
        rows = [row for row in audit_rows if row['checkpoint'] == name]
        probe = result['first_roi_gradient_probe']
        overview['checkpoints'][name] = {'epoch': result['epoch'], 'native_bilateral_dice': result['reproduced_metrics']['mean_bilateral_patient_dice'],
            'mean_side_dice_model_grid_amp': float(np.mean([row['metrics']['model_grid_amp']['dice'] for row in rows])),
            'mean_side_dice_native_amp': float(np.mean([row['metrics']['native_amp']['dice'] for row in rows])),
            'mean_side_dice_native_fp32': float(np.mean([row['metrics']['native_fp32']['dice'] for row in rows])),
            'minimum_label_roundtrip_dice': min(row['metrics']['label_roundtrip']['dice'] for row in rows),
            'mean_foreground_fraction_model_grid': float(np.mean([row['foreground_fraction'] for row in rows])),
            'mean_foreground_probability_fp32': float(np.mean([row['foreground_probability_mean'] for row in rows])),
            'mean_background_probability_fp32': float(np.mean([row['background_probability_mean'] for row in rows])),
            'first_roi_gradient_probe': probe,
            'train_eval_maximum_logit_difference_fp32': result['train_eval_maximum_logit_difference_first_roi_fp32']}
    baseline.save_json(output / 'overview.json', overview)
    print(json.dumps(overview, indent=2, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.run_dir, args.output_dir = args.run_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if args.output_dir.exists() or args.output_dir.is_relative_to(args.run_dir):
        parser.error('Use a new audit output directory outside the training run')
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    args.output_dir.mkdir(parents=True)
    try:
        execute(args)
    except Exception as error:
        baseline.save_json(args.output_dir / 'failure.json', {'error_type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
