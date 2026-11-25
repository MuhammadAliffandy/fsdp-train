#!/usr/bin/env python3
"""Two-patient training-only fit check; source weights and validation inference stay untouched."""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from monai.losses import DiceCELoss, DiceLoss
from torch.nn import functional as F

import physical_fsdp as data
import train_fsdp_research as trainer
from audit_fsdp_loss_gradients import load_source
from fsdp_research_model import run_sample
from train_private_long import atomic_save


class CandidateLoss(torch.nn.Module):
    def __init__(self, balanced=False):
        super().__init__()
        self.original = DiceCELoss(sigmoid=True, squared_pred=True)
        self.dice = DiceLoss(sigmoid=True, squared_pred=True)
        self.balanced, self.calls = balanced, 0

    def forward(self, logits, target):
        detail = self.calls % 2 == 0
        self.calls += 1
        if not self.balanced or not detail:
            return self.original(logits, target)
        fraction = target.mean().detach()
        weight = torch.sqrt((1 - fraction) / fraction.clamp_min(1e-8)).clamp(1., 8.)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
        return self.dice(logits, target) + bce / (1 + (weight - 1) * fraction)


def execute(args):
    if len(args.cases) != 2 or len(set(args.cases)) != 2:
        raise ValueError('Tiny diagnosis requires exactly two unique training patients')
    model, prepared, frozen, source, checksum, samples = load_source(args)
    device = torch.device(args.device)
    subset = copy.copy(prepared)
    subset['samples'] = {'train': samples}
    subset['patients'] = {'train': {key: prepared['patients']['train'][key] for key in args.cases}}
    threshold = source['best_threshold']
    optimizer = torch.optim.AdamW([
        {'params': model.sam.parameters(), 'lr': args.sam_lr, 'weight_decay': .01},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('sam.')],
         'lr': args.head_lr, 'weight_decay': 1e-4}])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and frozen.precision == 'fp16')
    def measure():
        result = trainer.base.evaluate(model, subset, 'train', device, [threshold], frozen.seed)
        return result['refined'][5][threshold]['metrics']
    initial = measure()
    best, best_epoch, history = initial, 0, []
    def selection(metrics):
        return min(metrics['minimum_side_dice'], metrics['minimum_bilateral_patient_dice'])
    atomic_save(args.output_dir / 'tiny_best.pth', {'model_state_dict': model.state_dict(), 'epoch': 0,
        'metrics': initial, 'threshold': threshold, 'source_sha256': checksum})
    passed = selection(initial) >= args.target
    for epoch in range(args.epochs) if not passed else []:
        trainer.base.set_stage(model, frozen.source_best_epoch, 0)
        order = np.random.default_rng(frozen.seed + epoch).permutation(len(samples))
        losses, updates, skips, norms = [], 0, 0, []
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            criterion = CandidateLoss(args.loss == 'balanced')
            loss, outputs = run_sample(model, samples[index], device, frozen.seed + epoch * 1000 + int(index),
                criterion, augment=False, auxiliary_weight=.1 if args.loss == 'auxiliary' else .25)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite tiny loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            length = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(length) and not scaler.is_enabled():
                raise ValueError('Nonfinite tiny gradient')
            previous = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            success = scaler.get_scale() >= previous
            updates += int(success); skips += int(not success)
            norms.append(float(length)); losses.append(float(loss.detach()))
            del loss, outputs, criterion
        row = {'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'optimizer_updates': updates,
               'amp_skipped_updates': skips, 'gradient_norm_before_clipping_mean': float(np.mean(norms)),
               'metrics': None}
        if epoch == 0 or (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            row['metrics'] = measure()
            if selection(row['metrics']) > selection(best):
                best, best_epoch = row['metrics'], epoch + 1
                atomic_save(args.output_dir / 'tiny_best.pth', {'model_state_dict': model.state_dict(),
                    'epoch': best_epoch, 'metrics': best, 'threshold': threshold, 'source_sha256': checksum})
            passed = selection(best) >= args.target
        history.append(row)
        progress = {'complete': False, 'mode': 'two_patient_train_only', 'loss_variant': args.loss,
                    'completed_epochs': epoch + 1, 'target': args.target, 'best_epoch': best_epoch,
                    'best_training_metrics': best, 'diagnostic_passed': passed, 'latest': row}
        data.save(args.output_dir / 'progress.json', progress)
        atomic_save(args.output_dir / 'tiny_latest.pth', {'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(),
            'epoch': epoch + 1, 'history': history, 'arguments': vars(args), 'source_sha256': checksum,
            'recovery_note': 'State retained for inspection; this diagnostic has no automatic resume command.'})
        print(json.dumps(progress), flush=True)
        if passed:
            break
    summary = {'complete': True, 'mode': 'two_patient_train_only', 'loss_variant': args.loss,
        'completed_epochs': len(history), 'planned_epochs': args.epochs, 'initial_training_metrics': initial,
        'best_training_metrics': best, 'best_epoch': best_epoch, 'diagnostic_passed': passed,
        'target_minimum_side_and_patient_dice': args.target, 'threshold_fixed_from_source': threshold,
        'sam_lr': args.sam_lr, 'head_lr': args.head_lr, 'augmentation': 'none', 'patients': args.cases,
        'source_checkpoint_snapshot_sha256': checksum, 'oracle_clicks_per_side': 5,
        'cached_validation_tensors_deserialized': True, 'validation_inference_performed': False,
        'test_images_read': False, 'interpretation': 'Memorization diagnostic on two training patients only; '
        '85%training Dice does not establish validation performance. Four updates per epoch, no accumulation; '
        'both tiny arms use this same explicit diagnostic recipe; cohort LR/budget comparisons are separate.'}
    data.save(args.output_dir / 'history.json', history)
    data.save(args.output_dir / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--cases', nargs=2, default=['train_01', 'train_02'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--loss', choices=('control', 'balanced', 'auxiliary'), default='control')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--target', type=float, default=.85)
    parser.add_argument('--sam-lr', type=float, default=1e-5)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    args = parser.parse_args()
    if (args.epochs < 1 or args.eval_every < 1 or not 0 < args.target <= 1
            or any(not math.isfinite(v) or v <= 0 for v in (args.sam_lr, args.head_lr))):
        parser.error('Invalid tiny diagnostic budget/target/LR')
    args.run_dir = args.run_dir.expanduser().resolve()
    if not (args.run_dir / 'research_best.pth').exists() and (args.run_dir / 'history').is_dir():
        args.run_dir = (args.run_dir / 'history').resolve()
    existed_before = args.output_dir.exists()
    try:
        execute(args)
    except Exception as error:
        if not existed_before and args.output_dir.is_dir():
            data.save(args.output_dir / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
