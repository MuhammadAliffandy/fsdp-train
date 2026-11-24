#!/usr/bin/env python3
"""Continue the verified FSDP hybrid, optionally auditing compatible resolution variants."""
import argparse
import importlib.metadata
import math
from pathlib import Path

import torch

import physical_fsdp as data
import sam_refine_data as views
import train_sam_refine32 as trainer
from evaluate_private_validation import file_hash
from train_private_tiny import select_patients
from audit_foundation_sources import check_folder, inventory, problem_message, bind_sources


FOUNDATION_SHA256 = '81b3e436c25062d25544e6a0b89d1c7887e7935cbf087ce3676fea08f1144a39'
FOUNDATION_DICE = .4563322489
RECIPE = ('accumulation', 'auxiliary_weight', 'augmentation', 'seed', 'precision', 'val_every', 'min_delta')


def recipe_values(config):
    result = {key: config[key] for key in RECIPE}
    modes = {'shared physical LR reflection and shared gain0.9-1.1/shift+-0.1 only': 'light', 'none': 'none'}
    if config['augmentation'] not in modes:
        raise ValueError('Unknown foundation augmentation protocol')
    result['augmentation'] = modes[config['augmentation']]
    return result


def inspect_foundation(folder, source_root=None, verified_cache=False):
    folder = folder.resolve()
    config, summary = [data.read(folder / name) for name in ('config.json', 'summary.json')]
    score = summary['best_validation_metrics']['5']['mean_bilateral_patient_dice']
    if (not summary['complete'] or summary['experiment'] != 'sam_turbo_cnn_refine32'
            or summary['train_patients'] != 54 or summary['validation_patients'] != 10
            or summary['reserved_test_patients'] != 20 or summary['test_images_read']
            or not math.isfinite(score) or abs(score - FOUNDATION_DICE) > 1e-8 or summary['best_checkpoint_sha256'] != FOUNDATION_SHA256
            or summary['detail_feature_grid'] != [32] * 3 or summary['sam_token_grid'] != [8] * 3
            or summary['oracle_clicks_per_side'] != [1, 5]):
        raise ValueError('Expected the completed45.63%FSDP foundation; refusing another run')
    if verified_cache:
        from fsdp_verified_cache import image_inventory, require_images
        require_images(image_inventory(data.read(folder / 'source_provenance.json'), source_root))
    else:
        check_folder(folder, source_root)
    for name, expected in (('hybrid_best.pth', FOUNDATION_SHA256), ('hybrid_data.pt', config['cache_sha256'])):
        print(f'Checking unchanged foundation {name}', flush=True)
        if file_hash(folder / name) != expected:
            raise ValueError(f'Foundation {name} checksum differs')
    if summary['physical_policy'] != config['physical_policy']:
        raise ValueError('Foundation physical policy differs')
    versions = {k: importlib.metadata.version(k) for k in config['versions']}
    if versions != config['versions']:
        raise ValueError('Use the original foundation package versions')
    if (views.source_hashes(Path(config['sam_dir'])) != config['sam_source_sha256']
            or file_hash(Path(config['checkpoint'])) != config['turbo_sha256']):
        raise ValueError('Foundation Turbo source/checkpoint changed')
    recipe_values(config)
    # The trainer has explicit continuation hooks; all underlying model/data/metric
    # implementations must still be byte-identical to the successful experiment.
    for name in trainer.CODE_FILES:
        if name != 'train_sam_refine32.py' and file_hash(Path(__file__).with_name(name)) != config['code_sha256'][name]:
            raise ValueError(f'Foundation implementation changed: {name}')
    manifest_path = Path(config['cohort_run_dir']) / 'config.json'
    control = data.read(manifest_path)
    manifest = Path(control['data_dir']) / 'manifest.json'
    if file_hash(manifest) != config['manifest_sha256']:
        raise ValueError('Frozen patient manifest changed')
    select_patients(data.read(manifest), config['seed'])
    return config, summary


def prepare_foundation(args):
    source_root = getattr(args, 'source_root', None)
    verified_cache = getattr(args, 'verified_cache', False)
    old, summary = inspect_foundation(args.foundation_run_dir, source_root, verified_cache)
    if (views.source_hashes(args.sam_dir) != old['sam_source_sha256']
            or file_hash(args.checkpoint) != old['turbo_sha256']):
        raise ValueError('Original Turbo source/checkpoint differs from foundation')
    expected_recipe = recipe_values(old)
    if any(getattr(args, key) != value for key, value in expected_recipe.items()):
        raise ValueError('FSDP recipe changed unexpectedly')
    prepared = torch.load(args.foundation_run_dir / 'hybrid_data.pt', map_location='cpu', weights_only=False)
    if prepared['policy'] != old['physical_policy']:
        raise ValueError('Foundation cache physical policy differs')
    original_sources = prepared['source_provenance']
    if verified_cache:
        from fsdp_verified_cache import image_inventory, require_images, validate_targets
        control = data.read(Path(old['cohort_run_dir']) / 'config.json')
        manifest = data.read(Path(control['data_dir']) / 'manifest.json')
        cache_check = validate_targets(prepared, manifest)
        data.save(args.run_dir / 'cache_target_verification.json', cache_check)
        source_check = image_inventory(original_sources, source_root, verify=True)
        bound_sources = original_sources
    else:
        bound_sources = bind_sources(original_sources, source_root)
        source_check = inventory(bound_sources, verify=True)
    source_check['source_root_override'] = str(source_root) if source_root else None
    source_check['authority'] = ('CT hashes and binary targets from original verified cache; original annotation files not verified'
        if verified_cache else 'checksums from the original verified cache; only paths may be rebound')
    data.save(args.run_dir / 'source_verification.json', source_check)
    if verified_cache:
        require_images(source_check)
    elif not source_check['ready']:
        raise ValueError(problem_message(source_check))
    prepared['source_provenance'] = bound_sources
    if len(prepared['patients']['train']) != 54 or len(prepared['patients']['val']) != 10:
        raise ValueError('Expected exact54/10foundation cohort')
    config = {key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items() if key != 'resume'}
    config.update(experiment='sam_turbo_cnn_refine32_foundation', versions=old['versions'],
        architecture=old['architecture'], protocol=old['protocol'], loss=old['loss'], prompts=old['prompts'],
        augmentation_protocol=old['augmentation'], threshold_protocol=old['threshold_protocol'],
        selection=old['selection'], physical_policy=prepared['policy'], cache_sha256=old['cache_sha256'],
        manifest_sha256=old['manifest_sha256'], test_images_read=False,
        code_sha256={name: file_hash(Path(__file__).with_name(name)) for name in (*trainer.CODE_FILES,
            'train_fsdp_foundation.py', 'audit_foundation_sources.py')},
        sam_source_sha256=views.source_hashes(args.sam_dir),
        foundation={'run': str(args.foundation_run_dir), 'checkpoint_sha256': FOUNDATION_SHA256,
            'config_sha256': file_hash(args.foundation_run_dir / 'config.json'),
            'summary_sha256': file_hash(args.foundation_run_dir / 'summary.json'),
            'source_best_epoch': summary['best_epoch'], 'source_threshold': summary['threshold'],
            'expected_train': summary['best_frozen_training_metrics'], 'expected_validation': summary['best_validation_metrics'],
            'expected_validation_fixed_0_5': summary['validation_control_fixed_0_5'],
            'change': 'inherit full best hybrid weights and exact cached ROI/normalization; fresh optimizer at LR multiplier; same active groups as source best epoch'})
    if getattr(args, 'resolution_variant', None):
        from fsdp_resolution import configuration
        config['resolution'] = configuration(args, old)
        config['experiment'] = 'sam_turbo_fsdp_resolution'
        config['foundation_architecture'] = config['architecture']
        spec = config['resolution']
        config['architecture'] = (f'Trained binary SAM with{spec["token_grid"]}cube tokens, '
            f'{spec["detail_feature_shape"]}CNNfusion, original prompt/mask decoder and residual/full/half skips')
        for name in ('fsdp_resolution.py', 'sam_refine_resolution.py', 'multiclass_model.py'):
            config['code_sha256'][name] = file_hash(Path(__file__).with_name(name))
    if verified_cache:
        config['target_recovery'] = {'mode': 'verified_cache', 'target_cache_sha256': old['cache_sha256'],
            'original_annotation_files_verified': False, 'cache_targets_verified': cache_check,
            'source_image_hashes_verified': True,
            'limitation': 'Uses original cached binary FSDP targets; does not restore original17-class labels or ColorTables.'}
        config['code_sha256']['fsdp_verified_cache.py'] = file_hash(Path(__file__).with_name('fsdp_verified_cache.py'))
    if args.resume:
        if data.read(args.run_dir / 'config.json') != config:
            raise ValueError('Continuation configuration/code/foundation changed')
    else:
        data.save(args.run_dir / 'config.json', config)
        data.save(args.run_dir / 'source_provenance.json', prepared['source_provenance'])
        if source_root:
            data.save(args.run_dir / 'foundation_source_provenance.json', original_sources)
    return prepared, config


def load_foundation_weights(model, args, config):
    path = args.foundation_run_dir / 'hybrid_best.pth'
    if file_hash(path) != config['foundation']['checkpoint_sha256']:
        raise ValueError('Foundation checkpoint changed before load')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if (saved['config'] != data.read(args.foundation_run_dir / 'config.json')
            or saved['epoch'] != config['foundation']['source_best_epoch']
            or abs(saved['best_score'] - FOUNDATION_DICE) > 1e-8
            or saved['best_threshold'] != config['foundation']['source_threshold']):
        raise ValueError('Foundation checkpoint metadata differs')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    print('Loaded full successful hybrid: encoder, prompt encoder, mask decoder and trained CNN refiner', flush=True)


def verify_reproduction(threshold, training, validation, foundation):
    if threshold != foundation['source_threshold']:
        raise ValueError('Foundation train-calibrated threshold does not reproduce; no training started')
    for bundle, expected, split in ((training, foundation['expected_train'], 'train'),
                                     (validation, foundation['expected_validation'], 'val')):
        for clicks in (1, 5):
            actual = bundle['refined'][clicks][threshold]['metrics']['mean_bilateral_patient_dice']
            wanted = expected[str(clicks)]['mean_bilateral_patient_dice']
            if not math.isfinite(actual) or abs(actual - wanted) > 1e-4:
                raise ValueError(f'{split}@{clicks}foundation Dice does not reproduce; no training started')
    fixed = validation['refined'][5][.5]['metrics']['mean_bilateral_patient_dice']
    if not math.isfinite(fixed) or abs(fixed - foundation['expected_validation_fixed_0_5']['mean_bilateral_patient_dice']) > 1e-4:
        raise ValueError('Foundation fixed-threshold validation does not reproduce')
    print('Foundation reproduced before optimizer updates: validation Dice@5=45.63%; exact same cached ROI and prompts', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foundation-run-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, help='Relocated dataset root; all source hashes must match the foundation')
    parser.add_argument('--verified-cache', action='store_true', help='Use checksum-verified cached targets; original CT hashes remain required')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--epochs', type=int, default=100, help='Additional epochs')
    parser.add_argument('--lr-multiplier', type=float, default=.2)
    parser.add_argument('--patience-epochs', type=int, default=50)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--resolution-variant', choices=('baseline', 'detail', 'tokens', 'combined'))
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    args.foundation_run_dir = args.foundation_run_dir.expanduser().resolve()
    if args.source_root:
        args.source_root = args.source_root.expanduser().resolve()
    old, _ = inspect_foundation(args.foundation_run_dir, args.source_root, args.verified_cache)
    if args.check_only:
        print('Foundation weights/cache/source-presence/split/package checks passed; target/content verification follows in preparation; no tensors loaded', flush=True)
        return
    if (args.run_dir is None or args.epochs < 1 or args.patience_epochs < 0
            or not math.isfinite(args.lr_multiplier) or not 0 < args.lr_multiplier < 1):
        parser.error('Need a new run directory, positive additional epochs and LR multiplier in(0,1)')
    if args.audit_only and (args.resolution_variant is None or args.resume):
        parser.error('Audit-only requires a resolution variant and a new run')
    args.run_dir = args.run_dir.expanduser().resolve()
    args.sam_dir, args.checkpoint, args.cohort_run_dir = [Path(old[k]).resolve() for k in ('sam_dir', 'checkpoint', 'cohort_run_dir')]
    for key, value in recipe_values(old).items():
        setattr(args, key, value)
    args.sam_lr, args.head_lr = old['sam_lr'] * args.lr_multiplier, old['head_lr'] * args.lr_multiplier
    args.head_warmup_epochs = 0
    del args.check_only
    control = data.read(args.cohort_run_dir / 'config.json')
    protected = (args.foundation_run_dir, args.sam_dir, args.cohort_run_dir,
                 Path(control['source_root']).resolve(), Path(control['data_dir']).resolve())
    if args.source_root:
        protected += (args.source_root,)
    for path in protected:
        if args.run_dir.is_relative_to(path):
            parser.error('New run must be outside foundation/source/control')
    if args.resume:
        if not (args.run_dir / 'hybrid_latest.pth').exists() or (args.run_dir / 'summary.json').exists():
            parser.error('Resume requires an unfinished continuation with latest checkpoint')
    elif args.run_dir.exists():
        parser.error('Run exists; use a new timestamped name')
    else:
        args.run_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == 'cuda' and (not torch.cuda.is_available() or (args.precision == 'bf16' and not torch.cuda.is_bf16_supported())):
        parser.error('Requested CUDA/BF16 device unavailable')
    try:
        data.save(args.run_dir / 'progress.json', {'complete': False, 'phase': 'loading_foundation_and_reproducing_baseline',
            'foundation': str(args.foundation_run_dir), 'optimizer_updates': 0})
        trainer.execute(args)
    except Exception as error:
        data.save(args.run_dir / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
