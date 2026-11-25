# FSDP system diagnosis and next dataset set

**Status:** evidence review plus a read-only DGX audit workflow. No training was started, no source labels were changed, and the test split remains reserved.

## What the experiments establish

The strongest currently measured result is the Turbo SAM + CNN32 hybrid: **49.47% validation bilateral-patient Dice with five oracle clicks per side**, using a ground-truth-centered physical ROI. It is a useful foundation and a clear experimental benchmark. It is not automatic full-CT performance.

| Experiment | Result in the available report | What it tells us | What it does not tell us |
| --- | --- | --- | --- |
| Earlier SAM side/zoom/decoder runs | Approximately 19–26% validation in earlier protocols | Early preprocessing and training recipes were weaker | These scores are not directly comparable with the later physical ROI/cache protocol |
| Physical Turbo SAM + CNN32 foundation | 49.47% validation, five oracle clicks/side | Transfer initialization, local CNN detail and interactive prompts together produce the best measured cohort result | Which single component caused the gain; automatic localization performance |
| Controlled SAM research screen | Control 47.75%, history 49.18%, boundary 47.51%, refiner 47.20% after 20 epochs | History prompt information was the only screened change above that screen's control | A completed 100-epoch improvement; the screen was short and not a final selection |
| History continuation / ROI jitter | History continuation did not beat its source; jitter reached 49.38%, only about +0.20 percentage points over its matched control | Light ROI translation is safe to try, but its measured benefit was small | A reliable large improvement or evidence that augmentation alone will solve generalization |
| Full-cohort loss comparison | Control best 49.47%; balanced-loss arm about 49.38%; final measured validation at epoch 30 was 42.74% vs 45.19% | Loss changes modestly affect training and validation behavior; later selected checkpoints can be worse than earlier best checkpoints | That a larger loss weight or sharper gradient will add 30–40 points |
| Tiny loss fitting | About 86.7–87.4% on two training patients | The model/loss can fit a tiny training subset | Cohort validation quality, robust generalization, or an 80% result |
| Physical ResUNet, zero clicks | Best validation 33.27% at epoch 120; frozen train 65.51%; fixed-0.5 validation 31.43%; early stopped at 180/300 | A zero-click random-initialized 3D model learns, but generalizes poorly under this experiment | A fair architecture-only comparison with SAM, which used pretrained weights and five oracle clicks |
| Larger-detail / multiclass runs | Incomplete or a different task/protocol; no confirmed FSDP validation win | These runs identify resource, label and task-design questions | That increasing a feature grid or token count will itself improve Dice |

The displayed ResUNet cases show mixed failure modes at the selected threshold: `train_28` predicts zero voxels at threshold 0.95; `train_53` predicts 3,618 voxels for a 564-voxel target (about 6.4×); several others predict about 2.1–2.8× their target volume. The `train_28` score at threshold 0.95 does not show whether its probabilities are uniformly low; the probability map and threshold 0.5 are needed. `train_53` has an apparent false-positive burden, but the voxel totals alone do not identify its cause.

## Diagnosis by pipeline stage

### 1. Source data and labels

- The fixed patient split is 54 train / 10 validation / 20 test. The two sides of a patient stay together. There are 108 train ROIs, not 108 independent patients; the validation sample is only 10 patients.
- The original `labelsTr` and `ColorTable` folders reported on DGX were empty. The best foundation instead uses the checksum-verified binary 1/9 cache. That cache is the present target authority, but it cannot recreate missing 17-class labels or independently validate label semantics.
- Do not regenerate masks from another `labels_exp*` folder, remove a low-scoring patient, or change a label because the model misses it. A new target set requires verified source annotation provenance and anatomical review.
- Current evaluation uses GT-derived side centers and oracle points. This is suitable for a controlled interactive ROI benchmark, but it supplies localization information unavailable in automatic full-scan use.

### 2. Geometry and preprocessing

- The current physical pipeline retains every train and validation target in the cached ROI; the selected field is 96×160×208 at 0.25 mm (24×40×52 mm). Label roundtrip is at least 0.95 by design. Those checks reduce cropping and representation concerns; they do not verify anatomy, label correctness, or localization.
- CT is linearly resampled and labels use nearest-neighbor sampling. Predictions are restored to native space before thresholding and full-target scoring. Keep this order fixed in comparisons.
- Existing normalization was estimated from training FSDP voxels. That is not automatically wrong, but it is a risk to measure: a tiny foreground distribution may scale surrounding CT context poorly. CT-unit/scaling metadata and clipping fractions must be reviewed before trying a fixed HU window.
- Isotropic 0.25 mm resampling does not create acquired detail that was absent in the original scan. Keep original geometry and native scoring targets in every derivative.

### 3. Sampling, augmentation and loss

- A small target occupies a small part of a large ROI. Report target occupancy and the number of target-containing training patches per patient/epoch. A patch may train on only part of a full target; validation must continue scoring the complete native target.
- Previous loss arms changed weighting but did not produce a consistent cohort gain. The available tiny gradient probe proves finite, nonzero gradients for a few examples only. It does not establish that gradients are too weak on the cohort or that gradient magnitude is the bottleneck.
- ResUNet's class-balanced BCE can favor foreground recall at the cost of false positives. The observed `train_53` volume is consistent with that possibility, not proof. Compare TP/FP/FN and probability calibration before changing weights.
- Prefer paired online geometric transforms applied identically to CT, target and support. Keep augmentation in training only; do not create duplicate “new patients” by saving transformed copies. Use conservative millimeter-scale shifts/rotations, reject transforms that erase or truncate the target, and add intensity/noise changes only after the current input statistics are known.

### 4. Architecture and optimization

- The 49.47% foundation is not “a 100M-parameter model trained end to end.” Most Turbo weights are frozen; the active SAM blocks, mask decoder and CNN refiner are adapted. Its 8³ Turbo embedding supplies context, while CNN full/half-resolution skips retain local detail around the 32³ fusion grid.
- The 4.76M-parameter ResUNet is a different random-initialized, zero-click model. Its decoder returns full-resolution logits and multiscale supervision; the 32³ SAM fusion grid is not an equivalent downsampling bottleneck. The result cannot isolate which model family is better.
- The controlled resolution experiment did not complete its planned variants, so token16/native-half/combined changes remain unproven. Do not increase embedding resolution solely because “more tokens means more learning”; check pretrained position interpolation, feature-cell spacing in mm, activation gradients and memory, then compare on the same initialization and protocol.
- Select the checkpoint by the predeclared validation metric and retain the best checkpoint. Training Dice/loss improvement is not a validation improvement. The 80% goal is a target to measure, not a threshold to manufacture by selecting cases or tuning against validation.

Layer view from the implemented models:

| Path | Tensor flow at the selected physical input | Learning role / current caveat |
| --- | --- | --- |
| Turbo context encoder | 128³ context → 3D patch embedding with stride 16 → 8³ = 512 tokens → transformer blocks | Pretrained anatomical context; most encoder weights are frozen in the successful hybrid. A larger token grid is a separate expensive adaptation, not a free resolution switch. |
| Prompt + SAM mask path | Sparse/dense 3D click embeddings + image tokens → SAM mask decoder → coarse 32³ mask/logits | Provides interactive localization/refinement. Five oracle clicks per side are part of the 49.47% evaluation result. |
| Learned local detail path | 96×160×208 CT → full-resolution CNN features and 48×80×104 half-resolution features → 32³ fusion with mapped Turbo context and prompt/mask channels → residual refiner → full-resolution correction | CNN32 is the fusion lattice, not the sole output lattice: skip features and the full-resolution logit path remain. |
| Physical ResUNet | 96×160×208×16 → 48×80×104×32 → 24×40×52×64 → 12×20×26×128 → 6×10×13×192, then residual upsampling/skip decoder to full logits | Four stride-2 downsamples; three supervised decoder outputs. All 4.76M parameters train from random initialization. Small bottleneck context and limited cohort are plausible concerns, but the low validation score does not isolate them. |
| Multiclass CNN/Turbo candidate | Turbo image encoder plus randomly initialized 18-class detail head at detail32/64; separate 8/16 token experiment | Different objective and label set; incomplete runs and absent original 17-class files mean no completed FSDP comparison is available. |

The ResUNet path's tensor shapes are spatial samples, not physical resolution alone. At 0.25 mm the bottleneck feature cell spans about 4 mm per axis, before receptive-field effects. The reported 32³ SAM fusion cells are anisotropic in mm because the input field is rectangular. Both should be reported in physical mm per axis when comparing a feature-grid change.

## Most likely limitations, ranked by evidence

1. **Evaluation and task mismatch:** the best score is GT-ROI + five oracle clicks, while zero-click ResUNet is a different task and protocol. This is certain and prevents interpreting the gap as architecture alone.
2. **Patient-level generalization and cohort variability:** ResUNet train–validation gap is about 32 points; SAM loss controls also show later degradation. This is measured, though train and validation protocols must be checked per run before attributing it solely to overfitting.
3. **Inconsistent prediction volume/calibration across cases:** directly supported by the supplied ResUNet case counts. The probabilities and spatial overlays are still needed to separate threshold calibration, anatomy/context and labeling effects.
4. **Small, uncertain validation estimate:** 10 patients make the mean sensitive to a few difficult cases. Report patient-level spread and paired changes; use patient-level folds for development if compute permits. Do not use test patients to stabilize the estimate.
5. **Target semantics and quality:** unresolved because original annotation files are absent. Cache integrity establishes bytes and counts, not expert agreement or anatomical truth.
6. **Loss/gradient scaling:** plausible but not yet established as the primary limiter. Current evidence does not justify increasing loss weights before reviewing the existing gradient report and FP/FN breakdown.
7. **Token/detail resolution:** plausible secondary design question, not a confirmed cause. Existing full/half skips mean the SAM path already preserves some high-resolution detail.

## New set plan: immutable control plus one-change candidates

Do not make one large “improved” set that simultaneously changes labels, normalization, crop size, augmentation, loss and architecture. Create versioned derivatives with target/source hashes and a manifest:

1. **S0 — verified control:** original 54/10/20 split, binary cached targets, original grids, masks, support and SAM foundation inputs. No target edits.
2. **S1 — input-normalization candidate:** keep every target, grid and support tensor bit-identical; only regenerate CT channels from checksum-verified source CT using a predeclared train-only normalization candidate. Confirm source intensity units first. Validation uses train-derived statistics; test remains unread. Store separately and label it *candidate*, not a corrected dataset.
3. **S2 — online augmentation recipe:** build on the selected input policy; paired spatial transforms and mild intensity changes occur at training time only. Preserve native targets and deterministic validation.
4. **S3 — sampling/loss candidate:** change foreground patch sampling or a loss component one at a time, with identical initialization, optimizer budget, seed and evaluation. Choose the change only after gradients and TP/FP/FN measurements support it.
5. **S4 — architecture candidate:** start from the best SAM hybrid weights and change one feature path at a time. Preserve 8³ pretrained context initially; compare detail32 with the already-computed native-half path. Test tokens16 only as a separate controlled arm after the resolution audit completes.

For each set/run save: source CT hashes, authoritative target-cache SHA256, patient manifest SHA256, preprocessing policy, code/package hashes, label counts, ROI retention/roundtrip, target voxels/mm³, input percentiles/clipping, patch inclusion counts, model tensor shapes and mm per feature cell, optimizer update counts, per-patient train/validation Dice, precision/recall and TP/FP/FN, selected threshold (train only), and overlays for difficult plus representative patients. Never put validation/test statistics into train normalization or threshold selection.

### Required review gates before long training

- The DGX diagnostic must complete with `train_28`, `train_53`, and representative `train_01` probabilities/overlays; check the selected threshold and 0.5 without deleting hard cases.
- Review intensity units, header geometry, clipping and roundtrip flags for all 54/10 patients. Any geometry or target integrity failure blocks set creation.
- Confirm with an annotator what binary 1/9 represents and whether the intended endpoint is interactive FSDP or automatic localization. If label source is missing or semantics are uncertain, report that explicitly; do not invent additional structure labels.
- Build S1 as a separate candidate only if disk space allows. Validate its tensor shapes, target hashes and split membership against S0 before any training.
- Use the existing SAM foundation as the matched control. First verify its checkpoint/validation reproduction; then train a single candidate arm with a fixed budget and early stopping. Run additional seeds/folds only if the result is promising.
- Do not start a full “80%” run until these gates pass. No experiment so far establishes that validation Dice 80% is attainable on this label definition and split.

## DGX audit command

The new `diagnose_fsdp_system.py` is a CPU-only read/analysis workflow: it inventories available summaries/configs/history/CSV, checks all cached train/validation targets, profiles the 128 ROIs and source CT headers, and does not read test images or perform optimizer updates. Optional ResUNet probes inspect train_01/28/53 only. Optional normalization candidate construction writes a separate cache; neither operation changes a source cache or checkpoint.

From the DGX project directory after pulling the code:

```bash
conda activate fsdp-med3d
cd ~/raid/Clara/fsdp-train
FSDP_SOURCE_ROOT="$HOME/raid/normal case_86" \
FSDP_FOUNDATION_DIR="runs/fsdp_private_sam_refine32_20261123T144159Z_3745292" \
RESUNET_AUDIT_DIR="runs/physical_resunet_20261125T130044Z_641541" \
BUILD_CANDIDATE=1 INCLUDE_PROBE=1 \
bash scripts/launch_fsdp_system_diagnosis_nohup.sh
```

Then follow `tail -f logs/system_diagnosis/latest.log`. The script uses CPU/RAM and disk, not a GPU; candidate creation requires free space of at least twice the source-cache size plus 2 GiB. If space is tight, set `BUILD_CANDIDATE=0` first; the report/probe can run without making S1. Inspect `outputs/private/system_diagnosis/latest/summary.json`, `roi_diagnostics.csv`, `new_set_plan.json`, and the train-only probe overlays before training. Copy the report back for the final evidence review.

**Important limit:** the local workspace does not contain the DGX private runs or CT source tree. This review therefore uses prior DGX outputs pasted in this chat and repository documentation. The command above is required to fill in the patient-wide intensity, layer-probe and artifact-inventory evidence; no such full audit result is claimed yet.
