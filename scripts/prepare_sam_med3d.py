#!/usr/bin/env python3
"""Package TotalSegmentator frontal-sinus pseudo-labels for SAM-Med3D."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def same_geometry(image: sitk.Image, mask: sitk.Image) -> bool:
    return (
        image.GetSize() == mask.GetSize()
        and np.allclose(image.GetSpacing(), mask.GetSpacing(), atol=1e-5)
        and np.allclose(image.GetOrigin(), mask.GetOrigin(), atol=1e-4)
        and np.allclose(image.GetDirection(), mask.GetDirection(), atol=1e-5)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True, help="Folder with <case>_img.nii.gz")
    parser.add_argument("--pseudo-dir", type=Path, required=True, help="TotalSegmentator output folder with <case>/sinus_frontal.nii.gz")
    parser.add_argument("--manifest", type=Path, required=True, help="Pseudo-label QC CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("data/medical_preprocessed/fsdp/ct_nifti"))
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output case files")
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as csv_file:
        rows = {row["case_id"]: row for row in csv.DictReader(csv_file)}
    if not rows:
        parser.error(f"No cases found in manifest: {args.manifest}")

    image_out = args.output_dir / "imagesTr"
    label_out = args.output_dir / "labelsTr"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    low_voxel_cases: list[tuple[str, int]] = []
    prepared = 0
    for case_id, row in sorted(rows.items()):
        if row.get("status") != "ok":
            failures.append(f"{case_id}: manifest status is {row.get('status')!r}")
            continue
        image_path = args.image_dir / f"{case_id}_img.nii.gz"
        mask_path = args.pseudo_dir / case_id / "sinus_frontal.nii.gz"
        target_image = image_out / f"{case_id}.nii.gz"
        target_label = label_out / f"{case_id}.nii.gz"
        if not image_path.is_file() or not mask_path.is_file():
            failures.append(f"{case_id}: image or pseudo-mask file is missing")
            continue
        if not args.overwrite and (target_image.exists() or target_label.exists()):
            failures.append(f"{case_id}: output already exists (use --overwrite to replace)")
            continue

        image = sitk.ReadImage(str(image_path))
        mask = sitk.ReadImage(str(mask_path))
        if not same_geometry(image, mask):
            failures.append(f"{case_id}: image and mask geometry differ")
            continue
        binary = sitk.Cast(mask > 0, sitk.sitkUInt8)
        foreground = int(np.count_nonzero(sitk.GetArrayViewFromImage(binary)))
        if foreground == 0:
            failures.append(f"{case_id}: mask is empty")
            continue
        if foreground <= 1000:
            low_voxel_cases.append((case_id, foreground))

        if target_image.exists():
            target_image.unlink()
        try:
            target_image.hardlink_to(image_path)
        except OSError:
            shutil.copy2(image_path, target_image)
        sitk.WriteImage(binary, str(target_label), useCompression=True)
        prepared += 1

    if failures:
        print("Preparation stopped with issues:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print(f"Prepared {prepared} cases at {args.output_dir}")
    print("SAM-Med3D layout: imagesTr/<case>.nii.gz + labelsTr/<case>.nii.gz")
    print("Masks are binary uint8 (foreground=1) and retain source geometry.")
    print(f"Cases with <=1000 foreground voxels: {len(low_voxel_cases)}")
    if low_voxel_cases:
        print("The official trainer filters masks at this threshold; set its threshold to -1 to include every non-empty mask.")
        for case_id, count in low_voxel_cases:
            print(f"  {case_id}: {count}")
    print("Labels are TotalSegmentator pseudo-labels, not expert ground truth.")


if __name__ == "__main__":
    main()
