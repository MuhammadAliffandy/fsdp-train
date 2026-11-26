#!/usr/bin/env python3
"""Run nnU-Net train or predict with a per-process PyTorch CUDA allocator cap."""
import argparse
from importlib.metadata import entry_points
import math
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('train', 'predict'))
    parser.add_argument('--memory-budget-gib', type=float, required=True)
    args, command_args = parser.parse_known_args()
    if not math.isfinite(args.memory_budget_gib) or args.memory_budget_gib <= 0:
        parser.error('memory budget must be a positive finite number')

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error('expected exactly one CUDA device visible to the nnU-Net process')
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    fraction = args.memory_budget_gib * 1024**3 / total_bytes
    if not 0 < fraction < 1:
        parser.error('memory budget must be below the visible GPU total capacity')
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(f'PyTorch CUDA allocator cap: {args.memory_budget_gib:g} GiB '
          f'({fraction:.3f} of visible GPU)', flush=True)

    command = f'nnUNetv2_{args.action}'
    matches = [entry for entry in entry_points(group='console_scripts') if entry.name == command]
    if len(matches) != 1:
        parser.error(f'expected one installed {command} console entry point')
    sys.argv = [command, *command_args]
    matches[0].load()()


if __name__ == '__main__':
    main()
