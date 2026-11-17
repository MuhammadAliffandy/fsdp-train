#!/usr/bin/env python3
"""Print geometry and label diagnostics for NIfTI volumes."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def inspect(path: Path, *, count_values: bool) -> None:
    image = nib.load(str(path))
    header = image.header
    print(f"File: {path}")
    print(f"  shape: {image.shape}")
    print(f"  dtype: {header.get_data_dtype()}")
    print(f"  voxel spacing (mm): {tuple(round(float(x), 5) for x in header.get_zooms()[:3])}")
    print(f"  orientation: {''.join(nib.aff2axcodes(image.affine))}")
    print(f"  affine:\n{np.array2string(image.affine, precision=4, suppress_small=True)}")

    if count_values:
        values, counts = np.unique(np.asanyarray(image.dataobj), return_counts=True)
        print("  values and voxel counts:")
        for value, count in zip(values, counts):
            print(f"    {value}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="NIfTI files (.nii or .nii.gz)")
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Print unique voxel values and counts; use for masks, not large CT images.",
    )
    args = parser.parse_args()

    missing = [path for path in args.paths if not path.is_file()]
    if missing:
        parser.error("Files not found: " + ", ".join(str(path) for path in missing))

    for index, path in enumerate(args.paths):
        if index:
            print()
        inspect(path, count_values=args.labels)


if __name__ == "__main__":
    main()
