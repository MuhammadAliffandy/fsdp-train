#!/usr/bin/env python3
"""Freeze a patient split and extract bilateral FSDP masks from private labels."""

import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from audit_private_dataset import nifti_id


def prepare(root, audit_path, output, test_count=20, val_count=10, seed=2026, exclude_geometry_mismatch=False):
    root, output = root.expanduser().resolve(), output.resolve()
    if output.exists():
        raise ValueError("Output already exists; use the frozen manifest or choose a new output directory")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    all_rows = audit["cases"]
    if not all_rows or len({row['case_id'] for row in all_rows}) != len(all_rows):
        raise ValueError("Audit must contain distinct case IDs")
    allowed = {"ok", "geometry_mismatch"} if exclude_geometry_mismatch else {"ok"}
    if any(row["status"] not in allowed for row in all_rows):
        from collections import Counter
        counts = dict(Counter(row["status"] for row in all_rows))
        raise ValueError(f"Audit contains unaccepted statuses: {counts}; only explicit geometry-mismatch exclusion is supported")
    excluded = [{"patient_id": row["case_id"], "reason": row["status"]}
                for row in all_rows if row["status"] != "ok"]
    rows = [row for row in all_rows if row["status"] == "ok"]
    ids = sorted(row["case_id"] for row in rows)
    if len(ids) != len(set(ids)) or any(not key.isdigit() for key in ids):
        raise ValueError("Expected unique numeric patient IDs, one image per patient; otherwise supply a patient mapping first")
    if len({int(key) for key in ids}) != len(ids):
        raise ValueError("Patient IDs collide after removing leading zeros")
    if test_count != 20 or val_count <= 0 or len(ids) <= test_count + val_count:
        raise ValueError("This experiment requires 20 test patients and nonempty train/validation sets")
    split_file = root / "held_out_test_set_30percent.json"
    previous = json.loads(split_file.read_text(encoding="utf-8"))["test_set"]
    if not isinstance(previous, list) or not all(isinstance(item, str) for item in previous):
        raise ValueError("Expected test_set to be a list of case IDs or filenames")
    candidates = [nifti_id(Path(item)).removesuffix("-label").removesuffix("_0000") for item in previous]
    if len(set(candidates)) != len(candidates) or not set(candidates) <= {row["case_id"] for row in all_rows}:
        raise ValueError("Old test set contains duplicate or unknown patient IDs")
    original_test_count = len(candidates)
    candidates = sorted(set(candidates) & set(ids))
    if len(candidates) < test_count:
        raise ValueError("Old test pool contains fewer than 20 patients")
    rng = random.Random(seed)
    test_ids = set(rng.sample(sorted(candidates), test_count))
    val_ids = set(rng.sample(sorted(set(ids) - test_ids), val_count))
    split_by_id = {key: "test" if key in test_ids else "val" if key in val_ids else "train" for key in ids}
    records = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fsdp-prepare-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        for split in ("train", "val", "test"):
            for folder in ("imagesTr", "labelsTr"):
                (staging / split / "ct_private" / folder).mkdir(parents=True)
        for row in sorted(rows, key=lambda item: item["case_id"]):
            key = row["case_id"]
            if len(row["images"]) != 1 or len(row["labels"]) != 1:
                raise ValueError("Expected one CT and one label per patient")
            image_path, label_path = (Path(row[field][0]).resolve() for field in ("images", "labels"))
            if not image_path.is_relative_to(root / "imagesTr") or not label_path.is_relative_to(root / "labelsTr"):
                raise ValueError("Audit paths do not belong to the requested dataset")
            if nifti_id(image_path).removesuffix("_0000") != key or nifti_id(label_path).removesuffix("-label") != key:
                raise ValueError("Audit patient ID does not match source filenames")
            with (root / "ColorTable" / f"{key}_ColorTable.csv").open(encoding="utf-8-sig", newline="") as stream:
                color_rows = list(csv.DictReader(stream))
            mapping = {int(item["LabelValue"]): item["Name"].strip().upper() for item in color_rows}
            if len(mapping) != len(color_rows) or mapping.get(1) != "RFSDP" or mapping.get(9) != "LFSDP":
                raise ValueError("Per-patient ColorTable does not confirm label 1=RFSDP and 9=LFSDP")
            image = sitk.ReadImage(str(image_path))
            label = sitk.ReadImage(str(label_path))
            if image.GetDimension() != 3 or label.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1 or label.GetNumberOfComponentsPerPixel() != 1:
                raise ValueError("Expected scalar 3D CT and label volumes")
            geometry = image.GetSize() == label.GetSize() and all(
                np.allclose(a, b, atol=tol, rtol=0) for a, b, tol in [
                    (image.GetSpacing(), label.GetSpacing(), 1e-5),
                    (image.GetOrigin(), label.GetOrigin(), 1e-4),
                    (image.GetDirection(), label.GetDirection(), 1e-5)])
            array = sitk.GetArrayViewFromImage(label)
            if not geometry or not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all() or (array < 0).any():
                raise ValueError("Source geometry or label values changed since the audit")
            binary_array = np.isin(array, [1, 9]).astype(np.uint8)
            if not binary_array.any():
                raise ValueError("A patient has no RFSDP/LFSDP voxels; review the source before splitting")
            binary = sitk.GetImageFromArray(binary_array)
            binary.CopyInformation(label)
            target = staging / split_by_id[key] / "ct_private"
            destination = target / "imagesTr" / f"{key}.nii.gz"
            # Recompress uncompressed NIfTI instead of giving it a .nii.gz suffix.
            if image_path.name.endswith(".nii.gz"):
                try:
                    destination.hardlink_to(image_path)
                except OSError:
                    shutil.copy2(image_path, destination)
            else:
                sitk.WriteImage(image, str(destination), useCompression=True)
            sitk.WriteImage(binary, str(target / "labelsTr" / f"{key}.nii.gz"), useCompression=True)
            points = np.argwhere(binary_array)
            records.append({"patient_id": key, "split": split_by_id[key],
                            "source_image": str(image_path), "source_label": str(label_path),
                            "right_voxels": int(np.count_nonzero(array == 1)),
                            "left_voxels": int(np.count_nonzero(array == 9)),
                            "foreground_voxels": int(binary_array.sum()),
                            "bbox_xyz": (points.max(axis=0) - points.min(axis=0) + 1)[::-1].tolist()})
        counts = {split: sum(value == split for value in split_by_id.values()) for split in ("train", "val", "test")}
        manifest = {"seed": seed, "patient_unit": "numeric filename ID, one CT per patient",
                    "excluded_patients": excluded,
                    "old_test_candidates": original_test_count, "eligible_old_test_candidates": len(candidates),
                    "target": {"1": "RFSDP", "9": "LFSDP"}, "binary_target": "union of labels 1 and 9",
                    "preprocessing": "native spacing and orientation; no resampling or intensity changes",
                    "counts": counts, "previous_test_file": str(split_file),
                    "previous_test_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
                    "test_selection": "seeded subset of old test_set; remaining patients eligible for train/val",
                    "splits": {split: sorted(key for key, value in split_by_id.items() if value == split) for split in counts},
                    "cases": records}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        staging.rename(output)
    print("Private FSDP data prepared:", counts)
    print("Excluded geometry-mismatch patients:", len(excluded))
    print("Target: binary RFSDP (1) + LFSDP (9)")
    print("Frozen manifest:", output / "manifest.json")
    print("Patients with a target bounding box exceeding 128 voxels on any axis:",
          sum(any(size > 128 for size in record["bbox_xyz"]) for record in records))
    print("Train loader uses only train/ct_private. Validation and test are reserved separately.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=Path("outputs/private/audit.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/private/fsdp"))
    parser.add_argument("--val-patients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--exclude-geometry-mismatch", action="store_true",
                        help="Explicitly omit audited geometry mismatches and record them in the frozen manifest")
    args = parser.parse_args()
    try:
        prepare(args.root, args.audit, args.output_dir, val_count=args.val_patients, seed=args.seed,
                exclude_geometry_mismatch=args.exclude_geometry_mismatch)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Private preparation stopped: {error}\n")


if __name__ == "__main__":
    main()
