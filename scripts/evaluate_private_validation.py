#!/usr/bin/env python3
"""Diagnose a private SAM-Med3D run on its frozen validation patients only."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def scores(prediction, target, full_target_count, valid=None):
    predicted = int(prediction.sum())
    valid_predicted = int(np.count_nonzero(prediction & valid)) if valid is not None else predicted
    cropped = int(target.sum())
    intersection = int(np.count_nonzero(prediction & target))
    return {"dice_crop": 2 * intersection / (predicted + cropped) if predicted + cropped else 1.0,
            "dice_full_target": 2 * intersection / (valid_predicted + full_target_count) if valid_predicted + full_target_count else 1.0,
            "recall_full_target": intersection / full_target_count if full_target_count else 1.0,
            "precision": intersection / valid_predicted if valid_predicted else (1.0 if not full_target_count else 0.0),
            "predicted_voxels_crop": predicted, "predicted_voxels_valid": valid_predicted}


def preprocess(image_path, label_path):
    import torchio as tio
    from torchio.data.io import sitk_to_nib

    image = sitk.ReadImage(str(image_path))
    label = sitk.ReadImage(str(label_path))
    if image.GetSize() != label.GetSize() or not all(np.allclose(a, b, atol=tol, rtol=0) for a, b, tol in [
        (image.GetSpacing(), label.GetSpacing(), 1e-5), (image.GetOrigin(), label.GetOrigin(), 1e-4),
        (image.GetDirection(), label.GetDirection(), 1e-5)]):
        raise ValueError("Validation image/label geometry differs")
    image_array, _ = sitk_to_nib(image)
    label_array, _ = sitk_to_nib(label)
    if not np.isin(label_array, [0, 1]).all() or not np.any(label_array):
        raise ValueError("Expected a nonempty binary FSDP label")
    full_count = int(np.count_nonzero(label_array))
    # Match upstream: it discards the physical affine before ToCanonical/CropOrPad.
    subject = tio.Subject(image=tio.ScalarImage(tensor=image_array), label=tio.LabelMap(tensor=label_array),
                          valid=tio.LabelMap(tensor=np.ones_like(label_array, dtype=np.uint8)))
    subject = tio.Clamp(-1000, 1000)(subject)
    subject = tio.ToCanonical()(subject)
    subject = tio.CropOrPad((128, 128, 128), mask_name="label")(subject)
    raw = subject.image.data.clone()
    normalizer = tio.ZNormalization(masking_method=lambda x: x > 0)
    normalized = normalizer(raw).unsqueeze(0).float()
    target = subject.label.data.unsqueeze(0).long()
    if not torch.isfinite(normalized).all():
        raise ValueError("Non-finite normalized CT")
    offset = subject.image.affine[:3, 3]
    crop_origin = image.TransformContinuousIndexToPhysicalPoint(tuple(float(x) for x in offset))
    metadata = {"full_target_voxels": full_count, "crop_target_voxels": int(target.sum()),
                "target_retention": int(target.sum()) / full_count,
                "crop_origin_mm": list(crop_origin), "spacing_mm": list(image.GetSpacing()),
                "direction": list(image.GetDirection()), "normalization_positive_voxels": int((raw > 0).sum())}
    valid = subject.valid.data[0].numpy().astype(bool)
    metadata["crop_padding_voxels"] = int((~valid).sum())
    return normalized, target, raw[0].numpy(), metadata, valid


def next_prompt(logits, target, generator):
    errors = (logits > 0) != target.bool()
    points = torch.nonzero(errors[0, 0], as_tuple=False)
    if not len(points):
        points = torch.nonzero(target[0, 0] > 0, as_tuple=False)
    if len(points):
        coordinate = points[int(generator.integers(len(points)))]
    else:
        coordinate = torch.tensor([int(generator.integers(size)) for size in target.shape[-3:]], device=target.device)
    label = target[0, 0, coordinate[0], coordinate[1], coordinate[2]].reshape(1, 1)
    return coordinate.reshape(1, 1, 3).float(), label.long()


@torch.inference_mode()
def infer_clicks(model, image, target, milestones, seed):
    generator = np.random.default_rng(seed)
    logits = torch.zeros_like(target, dtype=torch.float32)
    # Zero logits mean background for first-click generation (strict > 0).
    hints = torch.zeros((1, 1, 32, 32, 32), device=image.device)
    results = {}
    with torch.autocast(device_type=image.device.type, enabled=image.device.type == "cuda"):
        embedding = model.image_encoder(image)
        for count in range(1, max(milestones) + 1):
            coordinates, labels = next_prompt(logits, target, generator)
            sparse, dense = model.prompt_encoder(points=(coordinates, labels), boxes=None, masks=hints)
            hints, _ = model.mask_decoder(image_embeddings=embedding, image_pe=model.prompt_encoder.get_dense_pe(),
                                           sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                           multimask_output=False)
            logits = F.interpolate(hints, size=target.shape[-3:], mode="trilinear", align_corners=False)
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite prediction logits")
            if count in milestones:
                results[count] = logits[0, 0].float().cpu().numpy().copy()
    return results


def save_crop(path, array, metadata):
    image = sitk.GetImageFromArray(array.transpose(2, 1, 0))
    image.SetSpacing(metadata["spacing_mm"])
    image.SetDirection(metadata["direction"])
    image.SetOrigin(metadata["crop_origin_mm"])
    sitk.WriteImage(image, str(path), useCompression=True)


def overlay(path, image, target, prediction, retention):
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    for row, axis in enumerate((2, 1, 0)):
        reductions = tuple(i for i in range(3) if i != axis)
        index = int(np.argmax(target.sum(axis=reductions))) if target.any() else target.shape[axis] // 2
        ct, gt, pred = [np.take(value, index, axis=axis).T for value in (image, target, prediction)]
        for column in range(3):
            ax = axes[row, column]
            ax.imshow(ct, cmap="gray", origin="lower", vmin=-1000, vmax=1000)
            for mask, color, active in [(gt, "cyan", column != 1), (pred, "magenta", column != 0)]:
                if active and mask.any() and not mask.all():
                    ax.contour(mask, levels=[0.5], colors=[color], linewidths=0.8)
            ax.set_title(f"native axis {axis}, slice {index}: " + ["GT (cyan)", "prediction (magenta)", "GT + prediction"][column], fontsize=8)
            ax.axis("off")
    fig.suptitle(f"GT-guided 128-cube; target retained {retention:.1%}; probability threshold 0.5")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def training_history(checkpoint, output):
    losses, dices = checkpoint.get("losses", []), checkpoint.get("dices", [])
    if len(losses) != len(dices) or not losses:
        raise ValueError("Latest checkpoint lacks aligned training loss/Dice history")
    rows = [{"epoch": index, "loss": float(loss), "dice_training_legacy": float(dice)}
            for index, (loss, dice) in enumerate(zip(losses, dices))]
    write_csv(output / "training_history.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ("loss", "dice_training_legacy")):
        ax.plot([row["epoch"] for row in rows], [row[key] for row in rows])
        ax.set(xlabel="Epoch (zero-based)", ylabel=key, title="Training only")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)
    return {"completed_epochs": checkpoint.get("epoch"), "last": rows[-1],
            "minimum_loss": min(rows, key=lambda row: row["loss"]),
            "maximum_training_dice": max(rows, key=lambda row: row["dice_training_legacy"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/private/fsdp"))
    parser.add_argument("--run-dir", type=Path, help="Default: newest private run with a latest checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clicks", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if any(count < 1 for count in args.clicks):
        parser.error("Click counts must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable in this environment")
    data = args.data_dir.resolve()
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("target") != {"1": "RFSDP", "9": "LFSDP"}:
        parser.error("Unexpected target mapping")
    parts = [manifest["splits"][split] for split in ("train", "val", "test")]
    all_ids = sum(parts, [])
    val_ids = sorted(manifest["splits"]["val"])
    if (len(set(all_ids)) != len(all_ids) or len(val_ids) != 10 or manifest["counts"]["test"] != 20
            or any(len(manifest["splits"][split]) != manifest["counts"][split] for split in ("train", "val", "test"))):
        parser.error("Expected disjoint frozen splits with 10 validation and 20 test patients")
    for folder in ("imagesTr", "labelsTr"):
        if {p.name for p in (data / "val/ct_private" / folder).glob("*.nii.gz")} != {key + ".nii.gz" for key in val_ids}:
            parser.error("Validation files differ from the frozen manifest")
    sam_dir = args.sam_dir.expanduser().resolve()
    if args.run_dir:
        run = args.run_dir.resolve()
    else:
        candidates = list((sam_dir / "work_dir").glob("fsdp_private_binary_*/sam_model_latest.pth"))
        if not candidates:
            parser.error("No private run checkpoint found")
        run = max(candidates, key=lambda p: p.stat().st_mtime).parent
    if not run.name.startswith("fsdp_private_binary_"):
        parser.error("Select a private run")
    if args.output_dir.exists():
        parser.error("Output directory exists; choose a new diagnostic run directory")
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    latest = torch.load(run / "sam_model_latest.pth", map_location="cpu", weights_only=False)
    if [Path(path).resolve() for path in latest.get("used_datas", [])] != [data / "train/ct_private"]:
        parser.error("Checkpoint was not trained on this private training folder")
    config = latest.get("args")
    if getattr(config, "model_type", "vit_b_ori") != "vit_b_ori" or getattr(config, "img_size", 128) != 128:
        parser.error("This diagnostic expects vit_b_ori with 128-cube inputs")
    if latest.get("epoch", 0) < 50:
        parser.error("Latest private checkpoint has not completed the planned 50 epochs; select the completed run")
    history = training_history(latest, output)
    del latest
    protocol = {"run": str(run), "split": "val", "validation_patients": len(val_ids),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "clicks": sorted(set(args.clicks)), "seed": args.seed, "crop": "GT-bounding-box-centered 128 cube; native voxel grid, matching training loader",
                "normalization": "clamp [-1000,1000], then training ZNormalization(x>0)",
                "prompts": "oracle error clicks; latest point only plus previous low-resolution mask; no prompt dropout",
                "metrics": "probability>0.5 (logit>0), with legacy logit>0.5 crop Dice for comparison",
                "full_target_score": "crop prediction on valid original voxels, background elsewhere; denominator includes all GT voxels; padded voxels excluded",
                "empty_pair_dice": 1.0, "test_images_read": False, "training": history,
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchio", "SimpleITK", "numpy")},
                "evaluation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(output / "protocol.json", protocol)
    sys.path.insert(0, str(sam_dir))
    from segment_anything.build_sam3D import sam_model_registry3D
    model = sam_model_registry3D["vit_b_ori"](checkpoint=None).to(device).eval()
    metrics, failures, comparison = [], [], []
    for filename in ("sam_model_latest.pth", "sam_model_loss_best.pth", "sam_model_dice_best.pth"):
        checkpoint_path = run / filename
        checkpoint_name = filename.removesuffix(".pth")
        destination = output / checkpoint_name
        destination.mkdir()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        epoch = checkpoint["epoch"]
        del checkpoint
        checkpoint_rows = []
        for index, key in enumerate(val_ids, 1):
            alias = f"val_{index:02d}"
            try:
                image, target, raw, metadata, valid = preprocess(data / "val/ct_private/imagesTr" / f"{key}.nii.gz", data / "val/ct_private/labelsTr" / f"{key}.nii.gz")
                case_seed = args.seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                predictions = infer_clicks(model, image.to(device), target.to(device), args.clicks, case_seed)
                gt = target[0, 0].numpy().astype(bool)
                for clicks, logits in predictions.items():
                    prediction = logits > 0
                    row = {"checkpoint": checkpoint_name, "saved_epoch": epoch, "case": alias, "clicks": clicks,
                           "full_target_voxels": metadata["full_target_voxels"], "crop_target_voxels": metadata["crop_target_voxels"],
                           "target_retention": metadata["target_retention"], **scores(prediction, gt, metadata["full_target_voxels"], valid),
                           "dice_crop_legacy_threshold": scores(logits > .5, gt, metadata["full_target_voxels"])["dice_crop"]}
                    metrics.append(row)
                    checkpoint_rows.append(row)
                    if clicks == max(args.clicks):
                        save_crop(destination / f"{alias}_prediction_crop.nii.gz", prediction.astype(np.uint8), metadata)
                        overlay(destination / f"{alias}_overlay.png", raw, gt, prediction, metadata["target_retention"])
                if checkpoint_name == "sam_model_latest":
                    save_crop(destination / f"{alias}_ct_crop.nii.gz", raw.astype(np.float32), metadata)
                    save_crop(destination / f"{alias}_target_crop.nii.gz", gt.astype(np.uint8), metadata)
                    save_json(destination / f"{alias}_preprocessing.json", metadata)
                print(f"{checkpoint_name}: {alias} complete; target retained {metadata['target_retention']:.1%}", flush=True)
            except Exception as error:
                failures.append({"checkpoint": checkpoint_name, "case": alias, "error": str(error)})
                print(f"{checkpoint_name}: {alias} failed: {error}", flush=True)
            write_csv(output / "validation_cases.csv", metrics)
            save_json(output / "failures.json", failures)
        for clicks in sorted(set(args.clicks)):
            selected = [row for row in checkpoint_rows if row["clicks"] == clicks]
            complete = len(selected) == len(val_ids)
            comparison.append({"checkpoint": checkpoint_name, "saved_epoch": epoch, "clicks": clicks,
                               "cases_completed": len(selected), "complete": complete,
                               "mean_dice_full_target": float(np.mean([row["dice_full_target"] for row in selected])) if complete else None,
                               "mean_dice_crop": float(np.mean([row["dice_crop"] for row in selected])) if complete else None,
                               "mean_target_retention": float(np.mean([row["target_retention"] for row in selected])) if complete else None,
                               "empty_crops": sum(row["crop_target_voxels"] == 0 for row in selected)})
        write_csv(output / "checkpoint_comparison.csv", comparison)
    eligible = [row for row in comparison if row["complete"] and row["clicks"] == max(args.clicks)]
    best = max(eligible, key=lambda row: row["mean_dice_full_target"]) if eligible and not failures else None
    summary = {"run": run.name, "protocol": "GT-assisted ROI and oracle prompts; validation only",
               "training": history, "comparison": comparison, "selected_by_validation": best,
               "failures": len(failures), "artifacts": str(output)}
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit("Validation incomplete; inspect failures.json before comparing scores")


if __name__ == "__main__":
    main()
