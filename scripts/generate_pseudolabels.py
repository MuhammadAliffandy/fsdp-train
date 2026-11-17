#!/usr/bin/env python3
"""Generate frontal-sinus pseudo-labels for NRRD CT scans with TotalSegmentator."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def geometry_matches(image: sitk.Image, mask: sitk.Image) -> bool:
    return (
        image.GetSize() == mask.GetSize()
        and np.allclose(image.GetSpacing(), mask.GetSpacing(), atol=1e-5)
        and np.allclose(image.GetOrigin(), mask.GetOrigin(), atol=1e-4)
        and np.allclose(image.GetDirection(), mask.GetDirection(), atol=1e-5)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing *_img.nrrd CT files")
    parser.add_argument("--nifti-dir", type=Path, required=True, help="Folder for temporary NIfTI conversions")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder for per-case TotalSegmentator masks")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV progress and mask audit file")
    parser.add_argument("--device", default="gpu", help="TotalSegmentator device: gpu, cpu, or gpu:N")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of scans, useful for a pilot")
    args = parser.parse_args()

    images = sorted(args.input_dir.glob("*_img.nrrd"))
    if not images:
        parser.error(f"No *_img.nrrd files found in {args.input_dir}")
    if args.limit is not None:
        images = images[: args.limit]

    args.nifti_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    columns = [
        "case_id", "status", "image_nrrd", "image_nifti", "pseudo_mask",
        "shape_xyz", "spacing_mm", "foreground_voxels", "foreground_volume_mm3",
        "geometry_matches", "return_code", "log_file",
    ]
    rows: list[dict[str, object]] = []

    for index, image_path in enumerate(images, start=1):
        case_id = image_path.name.removesuffix("_img.nrrd")
        nifti_path = args.nifti_dir / f"{case_id}_img.nii.gz"
        case_output = args.output_dir / case_id
        mask_path = case_output / "sinus_frontal.nii.gz"
        log_path = log_dir / f"{case_id}.log"
        return_code: int | str = ""

        try:
            source_image = sitk.ReadImage(str(image_path))
            if not nifti_path.exists():
                sitk.WriteImage(source_image, str(nifti_path), useCompression=True)

            if not mask_path.exists():
                case_output.mkdir(parents=True, exist_ok=True)
                command = [
                    "TotalSegmentator",
                    "-i", str(nifti_path),
                    "-o", str(case_output),
                    "-ta", "craniofacial_structures",
                    "--device", args.device,
                ]
                with log_path.open("w", encoding="utf-8") as log_file:
                    result = subprocess.run(
                        command,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                return_code = result.returncode
                if result.returncode != 0:
                    rows.append({
                        "case_id": case_id,
                        "status": "failed_inference",
                        "image_nrrd": str(image_path),
                        "image_nifti": str(nifti_path),
                        "pseudo_mask": str(mask_path),
                        "return_code": return_code,
                        "log_file": str(log_path),
                    })
                    print(f"[{index}/{len(images)}] {case_id}: inference failed; see {log_path}", flush=True)
                    write_manifest(args.manifest, columns, rows)
                    continue

            if not mask_path.exists():
                rows.append({
                    "case_id": case_id,
                    "status": "missing_output",
                    "image_nrrd": str(image_path),
                    "image_nifti": str(nifti_path),
                    "pseudo_mask": str(mask_path),
                    "return_code": return_code,
                    "log_file": str(log_path),
                })
                print(f"[{index}/{len(images)}] {case_id}: expected mask missing", flush=True)
                write_manifest(args.manifest, columns, rows)
                continue

            mask = sitk.ReadImage(str(mask_path))
            aligned = geometry_matches(source_image, mask)
            foreground_voxels = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask)))
            voxel_volume = float(np.prod(mask.GetSpacing()))
            status = "ok" if aligned and foreground_voxels > 0 else (
                "geometry_mismatch" if not aligned else "empty_mask"
            )
            rows.append({
                "case_id": case_id,
                "status": status,
                "image_nrrd": str(image_path),
                "image_nifti": str(nifti_path),
                "pseudo_mask": str(mask_path),
                "shape_xyz": "x".join(str(value) for value in mask.GetSize()),
                "spacing_mm": "x".join(f"{value:.6g}" for value in mask.GetSpacing()),
                "foreground_voxels": foreground_voxels,
                "foreground_volume_mm3": round(foreground_voxels * voxel_volume, 3),
                "geometry_matches": aligned,
                "return_code": return_code,
                "log_file": str(log_path) if log_path.exists() else "",
            })
            print(
                f"[{index}/{len(images)}] {case_id}: {status}, "
                f"foreground={foreground_voxels} voxels",
                flush=True,
            )
        except Exception as error:  # keep a per-case failure and continue the batch
            rows.append({
                "case_id": case_id,
                "status": f"error:{type(error).__name__}",
                "image_nrrd": str(image_path),
                "image_nifti": str(nifti_path),
                "pseudo_mask": str(mask_path),
                "log_file": str(log_path) if log_path.exists() else "",
            })
            print(f"[{index}/{len(images)}] {case_id}: {error}", flush=True)

        write_manifest(args.manifest, columns, rows)

    summary: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        summary[status] = summary.get(status, 0) + 1
    print(f"\nCompleted {len(rows)}/{len(images)} cases. Status counts: {summary}")
    print(f"Manifest: {args.manifest}")
    print("These model predictions are pseudo-labels, not expert ground truth.")


def write_manifest(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
