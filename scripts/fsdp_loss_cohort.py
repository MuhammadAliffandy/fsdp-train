"""Full-cohort loss ablation: identical history weights, prompts and paired augmentation."""
from pathlib import Path

import torch

import physical_fsdp as data
from evaluate_private_validation import file_hash
from fsdp_history_jitter import POLICY, translate_pair
from fsdp_research_model import ResearchModel
from train_fsdp_loss_tiny import CandidateLoss


def diagnostic(folder, source):
    folder, source = Path(folder).resolve(), Path(source).resolve()
    report = data.read(folder / 'summary.json')
    checksum = file_hash(source / 'research_best.pth')
    if (not report['complete'] or report['test_images_read'] or report['validation_inference_performed']
            or report['source_checkpoint_sha256'] != checksum):
        raise ValueError('Completed train-only diagnostics must match this full-cohort source checkpoint')
    frozen = folder / 'frozen_source'
    if file_hash(frozen / 'research_best.pth') != checksum or data.read(frozen / 'config.json') != data.read(source / 'config.json'):
        raise ValueError('Diagnostic frozen checkpoint/config differs from the cohort source')
    proofs = {}
    for arm in ('control', 'balanced'):
        path = folder / f'tiny_{arm}' / 'summary.json'
        child = data.read(path)
        metric = child['best_training_metrics']
        if (not child['complete'] or not child['diagnostic_passed'] or child['test_images_read']
                or child['validation_inference_performed'] or child['patients'] != ['train_01', 'train_02']
                or child['source_checkpoint_snapshot_sha256'] != checksum
                or min(metric['minimum_side_dice'], metric['minimum_bilateral_patient_dice']) < .85):
            raise ValueError(f'{arm} tiny-fit proof is incomplete or incompatible')
        proofs[arm] = {'summary_sha256': file_hash(path), 'best_epoch': child['best_epoch'], 'metrics': metric}
    return {'directory': str(folder), 'summary_sha256': file_hash(folder / 'summary.json'),
            'source_checkpoint_sha256': checksum, 'tiny_proofs': proofs,
            'interpretation': 'Both losses can fit two train patients; this does not choose the validation winner.'}


class LossCohortModel(ResearchModel):
    @staticmethod
    def interaction_function(model, image, context, ratio, spacing, target, valid, milestones, seed,
                             criterion=None, context_target=None, context_valid=None, auxiliary_weight=.25, field=None):
        if criterion is not None:
            (image, context, target, valid, context_target, context_valid), record = translate_pair(
                image, context, ratio, spacing, target, valid, context_target, context_valid, seed)
            model.jitter_counts[record['reason']] = model.jitter_counts.get(record['reason'], 0) + 1
            criterion = CandidateLoss(model.loss_arm == 'balanced')
        return ResearchModel.interaction_function(model, image, context, ratio, spacing, target, valid,
            milestones, seed, criterion, context_target, context_valid, auxiliary_weight, field)


def configure(args, config):
    if args.variant != 'history' or args.research_source_dir is None or args.loss_arm not in ('control', 'balanced'):
        raise ValueError('Full loss study requires a trained history source and control/balanced arm')
    if args.cohort_accumulation < 1:
        raise ValueError('Cohort accumulation must be positive')
    proof = diagnostic(args.diagnostic_dir, args.research_source_dir)
    if proof['source_checkpoint_sha256'] != config['research_source']['checkpoint_sha256']:
        raise ValueError('Source proof and training initialization differ')
    args.accumulation = args.cohort_accumulation
    args.record_jitter_counts = True
    args.record_gradient_stats = True
    config['parameters'] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'}
    config.update(training_experiment='full_cohort_loss_ablation', loss_arm=args.loss_arm,
        diagnostic=proof, cohort_augmentation=POLICY,
        loss_protocol='original squared Dice + detail BCE (control) or capped normalized positive BCE (balanced); context DiceCE coefficient0.25',
        cohort_recipe='Same150epoch cap, LR multiplier0.3, accumulation4, warmup5/cosine, patience30, light+jitter in both arms by default. '
                      'These LR/update/augmentation settings differ jointly from the old continuation; only loss differs BETWEEN new arms.',
        evaluation_protocol='54train/10validation frozen patient split; same GT-side physical ROI; 1/5oracle clicks per side; '
                            'threshold calibrated on train only; full-target native patient-macro Dice; test unread')


def adapt_model(original, args, config):
    model = LossCohortModel.from_foundation(original, 'history')
    model.loss_arm, model.jitter_counts = args.loss_arm, {}
    return model


def finish_summary(summary, config):
    summary.update(experiment='full_cohort_loss_ablation', loss_arm=config['loss_arm'],
        diagnostic=config['diagnostic'], cohort_augmentation=config['cohort_augmentation'],
        loss_protocol=config['loss_protocol'],
        interpretation='Full54train/10validation controlled loss experiment from cohort best weights, not tiny weights. '
                       'GT-assisted repeatedly inspected validation; no80%guarantee and no automatic test/deployment claim.')


def extend_parser(parser):
    parser.set_defaults(record_jitter_counts=True, record_gradient_stats=True)
    parser.add_argument('--loss-arm', choices=('control', 'balanced'), required=True)
    parser.add_argument('--diagnostic-dir', type=Path, required=True)
    parser.add_argument('--cohort-accumulation', type=int, default=4)
