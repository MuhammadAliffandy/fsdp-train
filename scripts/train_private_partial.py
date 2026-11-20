#!/usr/bin/env python3
"""Run the v6 partial-encoder experiment with the frozen v4 ROI protocol."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from partial_experiment import PartialExperiment
from train_private_sides import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sam-dir', 'source-root', 'data-dir', 'checkpoint', 'baseline-dir', 'v3-run-dir', 'v4-run-dir', 'run-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--patience-epochs', type=int, default=50)
    parser.add_argument('--min-delta', type=float, default=.001)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if min(args.epochs, args.val_every, args.accumulation) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('Training counts and LR must be positive')
    if args.patience_epochs < 0 or not np.isfinite(args.min_delta) or args.min_delta < 0:
        parser.error('Invalid early-stopping settings')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    if args.resume:
        if not (args.run_dir / 'sam_model_side_latest.pth').is_file():
            parser.error('Resume requires an existing v6 run and its latest checkpoint')
    elif args.run_dir.exists():
        parser.error('Run directory exists; choose a new v6 name or --resume')
    else:
        args.run_dir.mkdir(parents=True)
    try:
        execute(args, experiment=PartialExperiment())
    except Exception as error:
        (args.run_dir / 'failure.json').write_text(json.dumps({'error_type': type(error).__name__, 'error': str(error)}, indent=2))
        raise


if __name__ == '__main__':
    main()
