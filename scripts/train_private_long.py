#!/usr/bin/env python3
"""Private SAM-Med3D fine-tuning with complete accumulation and periodic validation."""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from evaluate_private_validation import file_hash, infer_clicks, next_prompt, overlay, preprocess, save_crop, scores, write_csv


class Crops(Dataset):
    def __init__(self, root, ids, augment=False):
        self.root, self.ids, self.augment = root, ids, augment
        self.cache = {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        key = self.ids[index]
        if key not in self.cache:
            numpy_state, python_state = np.random.get_state(), random.getstate()
            try:
                with torch.random.fork_rng(devices=[]):
                    image, target, _, metadata, valid = preprocess(self.root / "imagesTr" / f"{key}.nii.gz", self.root / "labelsTr" / f"{key}.nii.gz")
            finally:
                np.random.set_state(numpy_state)
                random.setstate(python_state)
            self.cache[key] = (image[0], target[0].to(torch.uint8), metadata, torch.from_numpy(valid))
        image, target, metadata, valid = self.cache[key]
        if self.augment:
            axes = [axis + 1 for axis in range(3) if torch.rand(()).item() < .5]
            if axes:
                image, target = torch.flip(image, axes), torch.flip(target, axes)
        return image, target, index


def accumulation_window(step, batches, accumulation):
    start = (step // accumulation) * accumulation
    size = min(accumulation, batches - start)
    return size, (step + 1) % accumulation == 0 or step + 1 == batches


def interactive_loss(model, image, target, criterion, clicks, seed):
    generator = np.random.default_rng(seed)
    embedding = model.image_encoder(image)
    logits = torch.zeros_like(target, dtype=torch.float32)
    hints = torch.zeros((1, 1, 32, 32, 32), device=image.device)
    loss = 0
    for _ in range(clicks):
        coordinates, labels = next_prompt(logits.detach(), target, generator)
        sparse, dense = model.prompt_encoder(points=(coordinates, labels), boxes=None, masks=hints.detach())
        hints, _ = model.mask_decoder(image_embeddings=embedding, image_pe=model.prompt_encoder.get_dense_pe(),
                                       sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
        logits = F.interpolate(hints, size=target.shape[-3:], mode="trilinear", align_corners=False)
        loss = loss + criterion(logits.float(), target.float())
    return loss / clicks, logits.detach()


def train_epoch(model, dataset, optimizer, scaler, criterion, device, epoch, seed, accumulation, clicks):
    model.train()
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed + epoch))
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_dice, updates, skipped = 0., 0., 0, 0
    for step, (image, target, _) in enumerate(loader):
        image, target = image.to(device), target.to(device).long()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss, logits = interactive_loss(model, image, target, criterion, clicks, seed + epoch * len(loader) + step)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        size, should_step = accumulation_window(step, len(loader), accumulation)
        scaler.scale(loss / size).backward()
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            success = scaler.get_scale() >= old_scale
            updates += int(success)
            skipped += int(not success)
            optimizer.zero_grad(set_to_none=True)
        prediction, gt = (logits[0, 0] > 0).cpu().numpy(), target[0, 0].bool().cpu().numpy()
        total_loss += loss.item()
        total_dice += scores(prediction, gt, int(gt.sum()))["dice_crop"]
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            print(f"epoch={epoch + 1} batch={step + 1}/{len(loader)} loss={loss.item():.6f} optimizer_updates={updates}", flush=True)
    return {"loss": total_loss / len(loader), "dice_crop": total_dice / len(loader),
            "optimizer_updates": updates, "amp_skipped_updates": skipped,
            "expected_updates": math.ceil(len(loader) / accumulation)}


@torch.inference_mode()
def validate(model, dataset, device, seed, export=None):
    model.eval()
    rows = []
    if export is not None:
        export.mkdir(parents=True, exist_ok=True)
    for index, key in enumerate(dataset.ids):
        image, target, _ = dataset[index]
        metadata, valid = dataset.cache[key][2:]
        predictions = infer_clicks(model, image.unsqueeze(0).to(device), target.unsqueeze(0).to(device).long(), [1, 5],
                                   seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
        gt = target[0].numpy().astype(bool)
        for clicks, logits in predictions.items():
            rows.append({"case": f"val_{index + 1:02d}", "clicks": clicks,
                         "target_retention": metadata["target_retention"],
                         **scores(logits > 0, gt, metadata["full_target_voxels"], valid.numpy())})
        if export is not None:
            alias = f"val_{index + 1:02d}"
            _, _, raw, _, _ = preprocess(dataset.root / 'imagesTr' / f'{key}.nii.gz', dataset.root / 'labelsTr' / f'{key}.nii.gz')
            prediction = predictions[5] > 0
            overlay(export / f'{alias}_overlay.png', raw, gt, prediction, metadata['target_retention'])
            save_crop(export / f'{alias}_prediction_crop.nii.gz', prediction.astype(np.uint8), metadata)
            save_crop(export / f'{alias}_target_crop.nii.gz', gt.astype(np.uint8), metadata)
    means = {str(clicks): float(np.mean([row["dice_full_target"] for row in rows if row["clicks"] == clicks])) for clicks in (1, 5)}
    return means, rows


def atomic_save(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--clicks", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.val_every, args.accumulation, args.clicks) < 1 or args.lr <= 0:
        parser.error("Training counts and LR must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    data, run = args.data_dir.resolve(), args.run_dir.resolve()
    manifest_file = data / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    groups = [manifest["splits"][name] for name in ("train", "val", "test")]
    if len(set(sum(groups, []))) != sum(map(len, groups)) or len(groups[1]) != 10 or len(groups[2]) != 20 or manifest["target"] != {"1": "RFSDP", "9": "LFSDP"}:
        parser.error("Frozen private split or target mapping invalid")
    for split in ("train", "val"):
        for folder in ("imagesTr", "labelsTr"):
            if {path.name for path in (data / split / "ct_private" / folder).glob('*.nii.gz')} != {key + '.nii.gz' for key in manifest['splits'][split]}:
                parser.error("Prepared train/validation files differ from the manifest")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "resume"}
    config.update(manifest_sha256=file_hash(manifest_file), script_sha256=file_hash(Path(__file__)),
                  versions={name: importlib.metadata.version(name) for name in ('torch', 'torchio', 'monai', 'numpy', 'SimpleITK')},
                  preprocessing_and_inference_sha256=file_hash(Path(__file__).with_name('evaluate_private_validation.py')),
                  initial_checkpoint_sha256=file_hash(args.checkpoint), model_type="vit_b_ori", img_size=128,
                  optimizer="AdamW; all trainable modules use the same LR; weight_decay=0.1",
                  protocol=f"native GT-centered 128 cube; training normalization x>0; {args.clicks} oracle single-click interactions; detached mask feedback; average click loss",
                  augmentation="independent random flips on three voxel axes", selection="mean validation full-target Dice at 5 clicks",
                  test_images_read=False)
    if args.resume:
        if json.loads((run / "config.json").read_text()) != config:
            parser.error("Resume configuration/code/data differs from the frozen run")
    else:
        if run.exists():
            parser.error("Run directory exists; use --resume or a new run name")
        run.mkdir(parents=True)
        (run / "config.json").write_text(json.dumps(config, indent=2))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    sys.path.insert(0, str(args.sam_dir.resolve()))
    from segment_anything.build_sam3D import sam_model_registry3D
    from monai.losses import DiceCELoss
    model = sam_model_registry3D["vit_b_ori"](checkpoint=None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.1)
    warmup = min(5, args.epochs)
    def schedule(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        return .05 + .95 * .5 * (1 + math.cos(math.pi * min(1., (epoch - warmup) / max(1, args.epochs - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    source = torch.load(run / "sam_model_latest.pth" if args.resume else args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(source.get("model_state_dict", source), strict=True)
    history, start, best = [], 0, -1.
    if args.resume:
        optimizer.load_state_dict(source["optimizer_state_dict"])
        scheduler.load_state_dict(source["lr_scheduler_state_dict"])
        scaler.load_state_dict(source["scaler_state_dict"])
        history, start, best = source["history"], source["epoch"], source["best_validation_dice"]
        torch.set_rng_state(source["torch_rng_state"])
        np.random.set_state(source["numpy_rng_state"])
        random.setstate(source["python_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(source["cuda_rng_state"])
    del source
    training = Crops(data / "train/ct_private", sorted(groups[0]), augment=True)
    validation = Crops(data / "val/ct_private", sorted(groups[1]))
    def state(epoch):
        return {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                "history": history, "best_validation_dice": best, "args": args,
                "used_datas": [str(data / "train/ct_private")], "config": config,
                "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(), "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
    if not args.resume:
        means, rows = validate(model, validation, device, args.seed)
        best = means['5']
        write_csv(run / 'validation_initial.csv', rows)
        write_csv(run / 'validation_best.csv', rows)
        atomic_save(run / 'sam_model_val_best.pth', state(0))
        atomic_save(run / 'sam_model_latest.pth', state(0))
        print(f"Initial validation Dice: {means}", flush=True)
    for epoch in range(start, args.epochs):
        lr = optimizer.param_groups[0]['lr']
        metrics = train_epoch(model, training, optimizer, scaler, criterion, device, epoch, args.seed, args.accumulation, args.clicks)
        scheduler.step()
        row = {"epoch": epoch + 1, "lr": lr, **metrics, "val_dice_1_click": None, "val_dice_5_clicks": None}
        improved = False
        if (epoch + 1) % args.val_every == 0 or epoch == 0 or epoch + 1 == args.epochs:
            means, rows = validate(model, validation, device, args.seed)
            row.update(val_dice_1_click=means['1'], val_dice_5_clicks=means['5'])
            write_csv(run / 'validation_latest.csv', rows)
            improved = means['5'] > best
            if improved:
                best = means['5']
                write_csv(run / 'validation_best.csv', rows)
        history.append(row)
        write_csv(run / 'history.csv', history)
        (run / 'progress.json').write_text(json.dumps({"epoch": epoch + 1, "planned_epochs": args.epochs,
                                                      "best_validation_dice_5_clicks": best, "latest": row}, indent=2))
        if improved:
            atomic_save(run / 'sam_model_val_best.pth', state(epoch + 1))
        if improved or (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            atomic_save(run / 'sam_model_latest.pth', state(epoch + 1))
        print(json.dumps({"epoch_result": row, "best_validation_dice_5_clicks": best}), flush=True)
    selected = torch.load(run / 'sam_model_val_best.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(selected['model_state_dict'], strict=True)
    best_epoch = selected['epoch']
    del selected
    means, rows = validate(model, validation, device, args.seed, export=run / 'best_validation')
    write_csv(run / 'best_validation/cases.csv', rows)
    (run / 'summary.json').write_text(json.dumps({'completed_epochs': args.epochs, 'best_epoch': best_epoch,
                                                 'best_validation_dice': means, 'test_images_read': False,
                                                 'protocol': 'GT-assisted ROI and oracle clicks'}, indent=2))
    print(f"Training complete: {args.epochs} epochs. Best validation Dice@5={best:.6f}. Run: {run}", flush=True)


if __name__ == "__main__":
    main()
