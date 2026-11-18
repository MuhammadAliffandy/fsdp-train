#!/usr/bin/env python3
"""Independently verify saved validation masks and train/validation source pairing."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import SimpleITK as sitk

from evaluate_private_validation import preprocess, write_csv


def geometry(a, b):
    return a.GetSize() == b.GetSize() and all(np.allclose(x, y, atol=t, rtol=0) for x, y, t in [
        (a.GetSpacing(), b.GetSpacing(), 1e-5), (a.GetOrigin(), b.GetOrigin(), 1e-4),
        (a.GetDirection(), b.GetDirection(), 1e-5)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data, diagnostics = args.data_dir.resolve(), args.diagnostics.resolve()
    manifest = json.loads((data / "manifest.json").read_text())
    protocol = json.loads((diagnostics / "protocol.json").read_text())
    if protocol.get("manifest_sha256") != hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest():
        parser.error("Saved diagnostics used a different frozen manifest")
    splits = manifest["splits"]
    flattened = sum([splits[name] for name in ("train", "val", "test")], [])
    if len(set(flattened)) != len(flattened) or len(splits["val"]) != 10 or len(splits["test"]) != 20:
        parser.error("Expected disjoint frozen patient splits with 10 validation and 20 test cases")
    if args.output_dir.exists():
        parser.error("Preflight output already exists; choose a new path")
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    failures, verified, crop_rows = [], [], []
    allowed = set(splits["train"] + splits["val"])
    records = {row["patient_id"]: row for row in manifest["cases"]}
    for index, key in enumerate(sorted(allowed), 1):
        alias = f"source_{index:02d}"
        try:
            row = records[key]
            split = "train" if key in splits["train"] else "val"
            source_image = sitk.ReadImage(row["source_image"])
            source_label = sitk.ReadImage(row["source_label"])
            image = sitk.ReadImage(str(data / split / "ct_private/imagesTr" / f"{key}.nii.gz"))
            label = sitk.ReadImage(str(data / split / "ct_private/labelsTr" / f"{key}.nii.gz"))
            if not all(geometry(image, other) for other in (source_image, source_label, label)):
                raise ValueError("Source/prepared geometry differs")
            if not np.array_equal(sitk.GetArrayViewFromImage(source_image), sitk.GetArrayViewFromImage(image)):
                raise ValueError("Prepared CT pixels differ from the source")
            expected = np.isin(sitk.GetArrayViewFromImage(source_label), [1, 9])
            if not np.array_equal(expected, sitk.GetArrayViewFromImage(label)):
                raise ValueError("Prepared mask differs from original RFSDP/LFSDP labels")
            normalized, cropped_target, raw, metadata, _ = preprocess(
                data / split / 'ct_private/imagesTr' / f'{key}.nii.gz',
                data / split / 'ct_private/labelsTr' / f'{key}.nii.gz')
            gt = cropped_target[0, 0].numpy().astype(bool)
            if not gt.any():
                raise ValueError("Training-style crop loses the entire FSDP target")
            crop_rows.append({'case': alias, 'split': split, 'target_retention': metadata['target_retention'],
                              'target_hu_mean': float(raw[gt].mean()),
                              'target_normalized_mean': float(normalized[0, 0].numpy()[gt].mean()),
                              'normalization_positive_fraction': float((raw > 0).mean())})
            verified.append(alias)
        except Exception as error:
            failures.append({"case": alias, "error": str(error)})
        print(f"Source verification {index}/{len(allowed)}", flush=True)
    with (diagnostics / "validation_cases.csv").open(newline="") as stream:
        metric_rows = list(csv.DictReader(stream))
    stage = "private_finetuned" if any(row["checkpoint"] == "private_finetuned" for row in metric_rows) else "sam_model_latest"
    private_rows = [row for row in metric_rows if row["checkpoint"] == stage]
    budget = max(int(row["clicks"]) for row in private_rows)
    selected = {row["case"]: row for row in private_rows if int(row["clicks"]) == budget}
    aliases = [f"val_{index:02d}" for index in range(1, 11)]
    if set(selected) != set(aliases):
        failures.append({"case": "validation", "error": "Expected ten saved private validation cases"})
    dice_rows = []
    for alias, key in zip(aliases, sorted(splits["val"])):
        try:
            folder = diagnostics / stage
            prediction = sitk.ReadImage(str(folder / f"{alias}_prediction_crop.nii.gz"))
            target = sitk.ReadImage(str(folder / f"{alias}_target_crop.nii.gz"))
            if not geometry(prediction, target):
                raise ValueError("Saved prediction and GT crop geometry differs")
            pred_values, gt_values = sitk.GetArrayFromImage(prediction), sitk.GetArrayFromImage(target)
            if not np.isin(pred_values, [0, 1]).all() or not np.isin(gt_values, [0, 1]).all():
                raise ValueError("Saved masks must be binary")
            pred, gt = pred_values.astype(bool), gt_values.astype(bool)
            denominator = int(pred.sum()) + int(gt.sum())
            dice = 2 * int(np.count_nonzero(pred & gt)) / denominator if denominator else 1.0
            difference = abs(dice - float(selected[alias]["dice_crop"]))
            if difference > 1e-6:
                raise ValueError("Independent Dice disagrees with the reported crop Dice")
            full_target = sitk.ReadImage(str(data / "val/ct_private/labelsTr" / f"{key}.nii.gz"))
            physical_crop = sitk.Resample(full_target, target, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            if not np.array_equal(sitk.GetArrayViewFromImage(physical_crop), gt_values):
                raise ValueError("Physical crop geometry does not reproduce the prepared full target")
            dice_rows.append({"case": alias, "dice_independent_crop": dice, "absolute_difference": difference})
        except Exception as error:
            failures.append({"case": alias, "error": str(error)})
    if len(dice_rows) == 10:
        ranking = sorted(dice_rows, key=lambda row: row["dice_independent_crop"])
        for name, row in [("worst", ranking[0]), ("middle", ranking[len(ranking) // 2]), ("best", ranking[-1])]:
            source = diagnostics / stage / f"{row['case']}_overlay.png"
            if source.is_file():
                shutil.copy2(source, output / f"{name}_{row['case']}_overlay.png")
    result = {"passed": not failures and len(verified) == len(allowed) and len(dice_rows) == 10,
              "source_pairs_verified": len(verified), "saved_validation_masks_verified": len(dice_rows),
              "mean_independent_crop_dice": float(np.mean([row["dice_independent_crop"] for row in dice_rows])) if dice_rows else None,
              "test_images_read": False, "cases": dice_rows, "failures": failures,
              "minimum_train_val_target_retention": min(row['target_retention'] for row in crop_rows) if crop_rows else None,
              "partial_target_crops": sum(row['target_retention'] < 1 for row in crop_rows),
              "limitation": "Checks numeric consistency and physical metadata; anatomical annotation correctness still requires overlay/expert review"}
    (output / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    write_csv(output / 'pipeline_cases.csv', crop_rows)
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit("Preflight failed; long training was not started")


if __name__ == "__main__":
    main()
