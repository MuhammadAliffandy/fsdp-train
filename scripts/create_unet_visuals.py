#!/usr/bin/env python3
"""Export all tiny U-Net segmentation cases as PNG figures without training."""

import argparse
import csv
import inspect
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import torch

import train_private_unet as baseline
from evaluate_private_validation import file_hash, write_csv
from roi_zoom import restore_logits, zoom_samples
from train_private_tiny import native_prediction_indices, prepare_samples, sparse_scores
import unet_visuals as plots


def prepare_verified(args):
    from monai.networks.nets import UNet
    run = args.run_dir
    config = json.loads((run / 'config.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    audit = json.loads((args.audit_dir / 'summary.json').read_text())
    thresholds = json.loads((args.audit_dir / 'thresholds.json').read_text())
    overview = json.loads((args.audit_dir / 'threshold_overview.json').read_text())
    for result in (summary, audit, thresholds, overview):
        if not result['complete'] or result.get('test_images_read') or result.get('validation_images_read'):
            raise ValueError('Expected completed train-only artifacts with no validation/test image reads')
    if summary['stage'] != 'tiny' or summary['train_patients'] != 2 or thresholds['training_patients'] != 2:
        raise ValueError('Visual export is limited to the two tiny training patients')
    if audit['run'] != run.name or thresholds['run'] != run.name or overview['run'] != run.name:
        raise ValueError('Training and audit run identifiers differ')
    for name in ('unet_best', 'unet_latest'):
        for precision in ('amp', 'fp32'):
            for key in ('at_threshold_0_5', 'best_train_only_candidate'):
                item = overview['checkpoints'][name][precision][key]
                row = next(row for row in thresholds['checkpoints'][name][precision]['curve'] if row['threshold'] == item['threshold'])
                if any(not np.isclose(value, row[field], atol=1e-8, rtol=0) for field, value in item.items()):
                    raise ValueError('Compact threshold metrics differ from the full audit')
    if config['script_sha256'] != file_hash(Path(baseline.__file__)):
        raise ValueError('Frozen training implementation differs')
    for name, checksum in config['shared_code_sha256'].items():
        if file_hash(Path(baseline.__file__).with_name(name)) != checksum:
            raise ValueError('Frozen data/scoring helper differs')
    if file_hash(Path(inspect.getfile(UNet))) != config['monai_source_sha256']:
        raise ValueError('Installed MONAI U-Net differs')
    data = Path(config['data_dir']).resolve()
    _, policy, _ = baseline.v4_control(SimpleNamespace(v4_run_dir=Path(config['v4_run_dir']), seed=config['seed']), data / 'manifest.json')
    if file_hash(data / 'manifest.json') != config['manifest_sha256']:
        raise ValueError('Source manifest differs')
    preparation = args.output_dir / 'preparation'
    preparation.mkdir()
    samples, patients = prepare_samples(Path(config['source_root']), data,
        json.loads((data / 'manifest.json').read_text()), preparation, config['seed'])
    if json.loads((preparation / 'selected_training_patients.json').read_text()) != json.loads((run / 'source_provenance.json').read_text())['train']:
        raise ValueError('Source image/final label hashes differ from the tiny run')
    for sample in samples:
        sample['target'] = sample['target'].to(torch.uint8)
    return config, summary, audit, thresholds, overview, zoom_samples(samples, policy['selected_native_size_xyz']), patients


def numeric_history(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    return [{key: float(value) if value else None for key, value in row.items()} for row in rows]


def confusion_row(alias, checkpoint, epoch, threshold, predicted, target, universe):
    predicted, target, universe = [np.unique(value) for value in (predicted, target, universe)]
    if not np.isin(predicted, universe).all() or not np.isin(target, universe).all():
        raise ValueError('Confusion domain must contain predictions and the full target')
    metrics = sparse_scores(predicted, target)
    tp = metrics['true_positive_voxels']
    fp, fn = len(predicted) - tp, len(target) - tp
    return {'case': alias, 'checkpoint': checkpoint, 'epoch': epoch, 'threshold': threshold,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': len(universe) - tp - fp - fn,
            'domain_voxels': len(universe), **metrics}


@torch.inference_mode()
def execute(args):
    config, summary, audit, thresholds, overview, samples, patients = prepare_verified(args)
    output, figures = args.output_dir, args.output_dir / 'figures'
    figures.mkdir()
    chosen = overview['checkpoints']['unet_latest']['amp']['best_train_only_candidate']['threshold']
    comparisons = sorted(set((.5, chosen)), reverse=True)
    plots.metric_comparison(overview, figures / '01_metrics_latest.png')
    plots.dice_matrix(thresholds, 'side', figures / '02_dice_per_side.png')
    plots.dice_matrix(thresholds, 'patient', figures / '03_dice_per_patient.png')
    plots.threshold_curves(thresholds, figures / '04_threshold_curves.png')
    plots.training_curves(numeric_history(args.run_dir / 'history.csv'), figures / '05_training_curves.png')
    side_csv, patient_csv, checkpoint_hashes = [], [], {}
    domains = {}
    for alias, patient in patients.items():
        parts = [native_prediction_indices(sample['valid'], sample['valid'], sample['metadata'], patient['shape'])
                 for sample in samples if sample['case'] == alias]
        domains[alias] = np.unique(np.concatenate(parts + [patient['target_indices']]))
    device = torch.device(args.device)
    for name in ('unet_best', 'unet_latest'):
        path = args.run_dir / f'{name}.pth'
        checkpoint_hashes[name] = file_hash(path)
        if checkpoint_hashes[name] != audit['checkpoints'][name]['sha256']:
            raise ValueError('Checkpoint differs from the supplied threshold audit')
        if name == 'unet_best' and checkpoint_hashes[name] != summary['best_checkpoint_sha256']:
            raise ValueError('Best checkpoint differs from the training summary')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint['config'] != config:
            raise ValueError('Checkpoint configuration differs from source run')
        model = baseline.build_model().to(device).eval()
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        epoch = checkpoint['epoch']
        del checkpoint
        restored = {}
        for sample in samples:
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = baseline.checked_logits(model, sample['image'].to(device))
            native_logits = restore_logits(logits[0, 0].float().cpu().numpy(), sample['native_target'].shape)
            predictions = {}
            for threshold in comparisons:
                prediction = (native_logits > np.log(threshold / (1 - threshold))) & sample['valid']
                predictions[threshold] = prediction
                metrics = baseline.scores(prediction, sample['native_target'], sample['metadata']['full_target_voxels'], sample['valid'])
                reference = next(row for row in thresholds['checkpoints'][name]['amp']['curve'] if row['threshold'] == threshold)
                expected = next(row['dice'] for row in reference['per_side'] if row['case'] == sample['case'] and row['side'] == sample['side'])
                if not np.isclose(metrics['dice_full_target'], expected, atol=1e-4, rtol=0):
                    raise ValueError('Visual prediction does not reproduce threshold-audit side Dice')
                side_csv.append({'checkpoint': name, 'epoch': epoch, 'threshold': threshold,
                                 'case': sample['case'], 'side': sample['side'], **metrics})
                restored.setdefault((sample['case'], threshold), []).append(native_prediction_indices(
                    prediction, sample['valid'], sample['metadata'], patients[sample['case']]['shape']))
            plots.overlay_grid(sample, predictions, chosen, figures / f'{name}_{sample["name"]}_three_planes.png', name, epoch)
            if name == 'unet_latest':
                plots.slice_montages(sample, predictions[chosen], chosen, figures)
            print(f'{name}: {sample["name"]} exported and score verified', flush=True)
        for (alias, threshold), parts in sorted(restored.items()):
            row = confusion_row(alias, name, epoch, threshold, np.concatenate(parts), patients[alias]['target_indices'], domains[alias])
            reference = next(item for item in thresholds['checkpoints'][name]['amp']['curve'] if item['threshold'] == threshold)
            expected = next(item['dice_full_target'] for item in reference['per_patient'] if item['case'] == alias)
            if not np.isclose(row['dice_full_target'], expected, atol=1e-4, rtol=0):
                raise ValueError('Visual bilateral union does not reproduce threshold audit')
            patient_csv.append(row)
        del model
    write_csv(output / 'side_metrics.csv', side_csv)
    write_csv(output / 'patient_confusion_metrics.csv', patient_csv)
    plots.confusion_matrices(patient_csv, figures / '06_confusion_matrices_native.png')
    rows = [row for row in patient_csv if row['checkpoint'] == 'unet_latest' and row['threshold'] == chosen]
    expected = overview['checkpoints']['unet_latest']['amp']['best_train_only_candidate']
    for field, average in [('mean_bilateral_patient_dice', 'dice_full_target'), ('mean_precision', 'precision'), ('mean_recall', 'recall_full_target')]:
        if not np.isclose(np.mean([row[average] for row in rows]), expected[field], atol=1e-4, rtol=0):
            raise ValueError('Plotted aggregate metrics do not reproduce exported predictions')
    figure_files = sorted(path.name for path in figures.glob('*.png'))
    text = [
        'VISUALISASI SEGMENTASI FSDP - U-NET TINY', '',
        'Scope: dua pasien train / empat ROI sisi. Semua gambar adalah held-in training.',
        f'Threshold kandidat {chosen:g} dipilih menggunakan dua pasien train yang sama.',
        'Validation dan test belum dievaluasi; original gate threshold0.5 tidak diubah.',
        'GT berasal dari anotasi final labelsTr, RFSDP1 dan LFSDP9. ROI masih dibantu GT.', '',
        '01-06: metrik, matriks Dice, kurva threshold/training dan confusion matrix.',
        '*three_planes.png: semua sisi, kedua checkpoint; cyan GT, magenta prediksi.',
        '*slices_z*.png: seluruh slice Z yang memuat GT atau prediksi pada checkpoint terakhir.',
        'Error overlay: hijau TP, magenta FP, kuning FN. Slice memakai spacing native.',
        'Native X/Y/Z tidak berarti orientasi anatomi standar tanpa pemeriksaan direction.',
        'Confusion matrix memakai union ROI valid dan seluruh target, bukan background seluruh CT.',
        'Skor gambar diverifikasi terhadap audit, toleransi Dice1e-4.', '',
        'Source run: ' + args.run_dir.name, 'Source audit: ' + args.audit_dir.name, '', 'PNG files:', *figure_files]
    (output / 'README.txt').write_text('\n'.join(text) + '\n')
    manifest = {'complete': True, 'run': args.run_dir.name, 'audit': args.audit_dir.name,
                'train_patients': 2, 'side_rois': 4, 'visualized_thresholds': comparisons,
                'checkpoint_sha256': checkpoint_hashes, 'generator_sha256': file_hash(Path(__file__)),
                'plots_module_sha256': file_hash(Path(plots.__file__)),
                'input_artifact_sha256': {name: file_hash(args.audit_dir / name) for name in
                                        ('summary.json', 'thresholds.json', 'threshold_overview.json')},
                'figures': {name: file_hash(figures / name) for name in figure_files},
                'source_summary': summary['best_selected_metrics'], 'threshold_candidate_metrics': overview['checkpoints']['unet_latest']['amp']['best_train_only_candidate'],
                'validation_images_read': False, 'test_images_read': False, 'optimizer_updates': 0,
                'training_artifacts_modified': False, 'gate_modified': False,
                'artifacts': str(output)}
    baseline.save_json(output / 'manifest.json', manifest)
    archive = output.with_suffix('.tar.gz')
    if archive.exists():
        raise ValueError('Archive path already exists')
    with tarfile.open(archive, 'w:gz') as bundle:
        for name in ('figures', 'side_metrics.csv', 'patient_confusion_metrics.csv', 'README.txt', 'manifest.json'):
            bundle.add(output / name, arcname=f'{output.name}/{name}')
    print(json.dumps({'complete': True, 'png_count': len(figure_files), 'figures': str(figures),
                      'archive': str(archive), 'validation_images_read': False, 'test_images_read': False}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'audit-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    for name in ('run_dir', 'audit_dir', 'output_dir'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if (args.output_dir.exists() or args.output_dir.is_relative_to(args.run_dir)
            or args.output_dir.is_relative_to(args.audit_dir)):
        parser.error('Use a new output directory outside the training/audit artifacts')
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
