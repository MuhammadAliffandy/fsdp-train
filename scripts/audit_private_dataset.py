#!/usr/bin/env python3
"""Audit private NIfTI pairs and summarize an existing split without printing IDs."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def nifti_id(path):
    return path.name.removesuffix(".gz").removesuffix(".nii")


def schema(value, depth=0):
    if isinstance(value, list):
        return {"type": "list", "length": len(value), "item": schema(value[0], depth + 1) if value else None}
    if isinstance(value, dict):
        if depth > 3:
            return {"type": "dict", "length": len(value)}
        known = {"train", "val", "validation", "test", "test_set", "held_out", "held_out_test", "test_cases", "held_out_test_set", "case_ids", "patients", "cases", "splits", "seed", "test_size", "train_cases", "test_patients", "train_patients"}
        return {"type": "dict", "length": len(value),
                "fields": {key: schema(item, depth + 1) for key, item in value.items() if key in known},
                "other_key_count": sum(key not in known for key in value)}
    return {"type": type(value).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/private/audit.json"))
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "imagesTr").is_dir() or not (root / "labelsTr").is_dir():
        parser.error("Expected imagesTr and labelsTr directories under --root")
    images, labels = defaultdict(list), defaultdict(list)
    for folder, index in [("imagesTr", images), ("labelsTr", labels)]:
        for path in sorted((root / folder).iterdir()):
            if path.is_file() and path.name.endswith((".nii", ".nii.gz")):
                key = nifti_id(path)
                if folder == "imagesTr":
                    key = key.removesuffix("_0000")
                else:
                    key = key.removesuffix("-label")
                index[key].append(path)

    rows, values = [], set()
    for case in sorted(set(images) | set(labels)):
        row = {"case_id": case, "images": [str(p) for p in images[case]], "labels": [str(p) for p in labels[case]]}
        if len(images[case]) != 1 or len(labels[case]) != 1:
            row["status"] = "missing_or_ambiguous_pair"
            rows.append(row)
            continue
        try:
            image = sitk.ReadImage(str(images[case][0]))
            label = sitk.ReadImage(str(labels[case][0]))
            if image.GetDimension() != 3 or label.GetDimension() != 3:
                raise ValueError("Expected scalar 3D NIfTI volumes")
            array = sitk.GetArrayFromImage(label)
            unique, counts = np.unique(array, return_counts=True)
            finite = bool(np.isfinite(array).all())
            integral = finite and bool(np.equal(array, np.floor(array)).all())
            values.update(float(v) for v in unique if np.isfinite(v))
            geometry = image.GetSize() == label.GetSize() and all(
                np.allclose(a, b, atol=tolerance, rtol=0)
                for a, b, tolerance in [(image.GetSpacing(), label.GetSpacing(), 1e-5),
                                         (image.GetOrigin(), label.GetOrigin(), 1e-4),
                                         (image.GetDirection(), label.GetDirection(), 1e-5)])
            row.update(shape_xyz=list(image.GetSize()), spacing_mm=list(image.GetSpacing()),
                       geometry_matches=geometry, foreground_voxels=int(np.count_nonzero(array)),
                       label_counts={str(v): int(n) for v, n in zip(unique, counts)})
            row["status"] = ("invalid_label_values" if not integral or (array < 0).any() else
                             "geometry_mismatch" if not geometry else
                             "empty_mask" if row["foreground_voxels"] == 0 else "ok")
        except Exception as error:
            row.update(status="read_error", error=str(error))
        rows.append(row)

    split = root / "held_out_test_set_30percent.json"
    split_schema = None
    heldout_membership = None
    if split.is_file():
        try:
            split_data = json.loads(split.read_text(encoding="utf-8"))
            split_schema = schema(split_data)
            entries = split_data.get("test_set") if isinstance(split_data, dict) else None
            if isinstance(entries, list) and all(isinstance(entry, str) for entry in entries):
                test_ids = [nifti_id(Path(entry)).removesuffix("-label").removesuffix("_0000") for entry in entries]
                paired_ids = {row["case_id"] for row in rows if len(row["images"]) == len(row["labels"]) == 1}
                heldout_membership = {"entries": len(entries), "unique_ids": len(set(test_ids)),
                                      "matched_pairs": len(set(test_ids) & paired_ids),
                                      "unmatched_ids": len(set(test_ids) - paired_ids),
                                      "duplicate_entries": len(test_ids) - len(set(test_ids))}
        except (ValueError, OSError) as error:
            split_schema = {"error_type": type(error).__name__}
    summary = {"image_files": sum(map(len, images.values())), "label_files": sum(map(len, labels.values())),
               "case_keys": len(rows), "status_counts": dict(Counter(row["status"] for row in rows)),
               "label_values": sorted(values), "heldout_json_found": split.is_file(), "heldout_schema": split_schema,
               "heldout_membership": heldout_membership,
               "label_candidates": {folder.name: sum(p.is_file() and p.name.endswith((".nii", ".nii.gz")) for p in folder.iterdir())
                                    for folder in root.glob("labels*") if folder.is_dir()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "cases": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Detailed local report: {args.output}")
    print("Split membership has not been interpreted or applied. No training data was created.")


if __name__ == "__main__":
    main()
