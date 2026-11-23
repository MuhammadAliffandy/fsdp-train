# FSDP / Frontal Sinus Segmentation Experiments

Research workspace for the public frontal-sinus pseudo-label experiment and later supervised fine-tuning with the available private NIfTI labels. See [PROJECT.md](PROJECT.md) for the research plan and open decisions.

## DGX setup order

Clone the project, create the Conda environment, install Python requirements, verify GPU visibility, then download the dataset.

```bash
git clone https://github.com/MuhammadAliffandy/fsdp-train.git
cd fsdp-train
conda env create -f environment.yml
conda activate fsdp-med3d
pip install -r requirements.txt
```

The environment follows the official SAM-Med3D Python 3.10 and PyTorch 2.6.0 setup, with NIfTI and dataset-audit dependencies added. Confirm that PyTorch sees the assigned GPU before downloading data or launching an experiment:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpus:', torch.cuda.device_count())"
```

## Data audit

Keep raw data under `data/raw/` and do not commit it. Audit NIfTI geometry before resampling or cropping:

```bash
python scripts/audit_nifti.py /path/to/public_ct.nii.gz
python scripts/audit_nifti.py --labels /path/to/private_frontal_sinus_mask.nii.gz
```

The first pass checks image coverage, shape, voxel spacing, orientation, and mask label values. This is needed before generating TotalSegmentator pseudo-labels or converting data to the SAM-Med3D training layout. If an archive contains NRRD or another format, convert it while preserving physical geometry before using this NIfTI audit script.

## Download the public CT source on DGX

NasalSeg v2 is the current public CT source candidate. Its annotations do not include frontal sinus; the scans are used as public images, and frontal-sinus pseudo-labels will be generated separately with TotalSegmentator after checking scan coverage.

```bash
set -euo pipefail
mkdir -p data/raw/public/nasalseg
cd data/raw/public/nasalseg
curl --location --fail --show-error --retry 5 --retry-delay 2 -C - \
  --user-agent 'fsdp-train/0.1 (research dataset download)' \
  -o NasalSeg.zip \
  'https://zenodo.org/api/records/13893419/files/NasalSeg.zip/content'
echo 'ba83df47974d907798542101fa43ac7d  NasalSeg.zip' | md5sum -c -
python - <<'PY'
from pathlib import Path
import tarfile
import zipfile

source = Path("NasalSeg.zip")
target = Path("NasalSeg.tar.gz")
with zipfile.ZipFile(source) as archive:
    bad_member = archive.testzip()
    if bad_member:
        raise SystemExit(f"Corrupt member in source archive: {bad_member}")
    with tarfile.open(target, "w:gz") as output:
        for item in archive.infolist():
            if item.is_dir():
                continue
            member = tarfile.TarInfo(f"NasalSeg/{item.filename}")
            member.size = item.file_size
            with archive.open(item) as content:
                output.addfile(member, content)
print(f"Created {target}")
PY
rm NasalSeg.zip
tar -tzf NasalSeg.tar.gz | head -40
mkdir -p extracted
tar -xzf NasalSeg.tar.gz -C extracted
```

The Zenodo v2 record publishes a 224 MB ZIP, not a TAR.GZ. The download uses Zenodo's file-content API and a descriptive User-Agent. The commands stop on an HTTP/download failure, validate the ZIP checksum, repack its contents as TAR.GZ with Python, remove the source ZIP, and extract the TAR.GZ. Review the extracted layout before writing dataset conversion code; do not assume image or mask extensions from the archive name.

## Generate frontal-sinus pseudo-labels on DGX

NasalSeg v2 contains 130 paired NRRD CT/mask files. Its masks label maxillary sinuses, nasal cavities, and nasopharynx, not the frontal sinus. For the public experiment, run TotalSegmentator's `craniofacial_structures` task on the CT images and use its `sinus_frontal` output as pseudo-labels. This task emits its full class set; TotalSegmentator does not support `--roi_subset` for this task, so the script runs it without that option and retains the frontal-sinus mask.

```bash
cd ~/raid/Clara/fsdp-train
git pull
conda activate fsdp-med3d
python -m pip install TotalSegmentator
python scripts/generate_pseudolabels.py \
  --input-dir data/raw/public/nasalseg/extracted/NasalSeg/images \
  --nifti-dir data/interim/public/nifti \
  --output-dir data/pseudo_labels/public/totalsegmentator \
  --manifest data/manifests/nasalseg_totalsegmentator_pseudo_labels.csv \
  --device gpu
```

The script converts each NRRD image to NIfTI, runs inference per scan, resumes cases with an existing mask, and writes per-case logs plus a manifest. The manifest records inference failures, empty masks, and image/mask geometry mismatches. These checks do not establish anatomical correctness; all predictions remain pseudo-labels, not expert ground truth.

## Prepare the SAM-Med3D training data

After the pseudo-label manifest reports 130 `ok` cases, package the CT volumes and binary frontal-sinus masks in the directory structure expected by the official SAM-Med3D loader:

```bash
cd ~/raid/Clara/fsdp-train
python scripts/prepare_sam_med3d.py \
  --image-dir data/interim/public/nifti \
  --pseudo-dir data/pseudo_labels/public/totalsegmentator \
  --manifest data/manifests/nasalseg_totalsegmentator_pseudo_labels.csv \
  --output-dir data/medical_preprocessed/fsdp/ct_nifti
```

It writes matching `imagesTr/<case>.nii.gz` and `labelsTr/<case>.nii.gz`; masks are binarized to uint8 values 0/1 while preserving the original geometry. Image files are hard-linked when the filesystem permits, otherwise copied. The script stops on missing files, non-`ok` manifest entries, empty masks, geometry mismatch, or existing output files. Use `--overwrite` only to intentionally rebuild this derived dataset.

## Launch and monitor SAM-Med3D training

The setup script clones the official implementation if needed, downloads its recommended `sam_med3d_turbo.pth` checkpoint, configures the prepared data path, and lowers the loader foreground threshold to `-1`. This is necessary because the public masks include many cases with fewer than 1000 foreground voxels. The launcher always runs training under `nohup`, saves a timestamped log and PID under ignored `logs/training/`, and refuses to start a second copy with the same run name.

The setup also converts binary targets to float32 at the `DiceCELoss` call, since recent MONAI Dice computations require floating targets. The integer labels remain available for prompt generation. Re-running setup applies these edits without duplicating them. Training uses unbuffered Python output (`python -u`) so log messages appear promptly when following the log.

If an earlier run stopped with `linalg.vector_norm: Expected a floating point or complex tensor as input. Got Long`, pull the latest project code and run setup again before launching a new run. No dataset conversion or checkpoint download is needed when the existing files are complete. The failed run's log is retained under its original timestamp.

Setup also handles an empty error region in `get_next_click3D_torch_2`. The upstream sampler raises `ValueError: high <= 0` when prediction and target have no differing voxels. The patched sampler keeps sampling errors when available; otherwise it selects a positive target voxel, or a negative voxel when the target is empty. Prompt labels always follow the target at the selected coordinate. For this error, pull the latest code, re-run setup, and launch again using the commands below.

The training Dice metric scores an empty prediction paired with an empty target as `1.0` and returns a tensor consistently. This prevents the upstream float `.item()` crash and NaN epoch averages. Empty pairs therefore contribute to this training metric; it is a crop-level metric and does not establish foreground segmentation quality on full scans.

```bash
cd ~/raid/Clara/fsdp-train
git pull origin main
conda activate fsdp-med3d
bash scripts/setup_sam_med3d.sh
bash scripts/launch_training_nohup.sh
```

To continue after a crash when a completed-epoch checkpoint exists, run `RESUME=latest bash scripts/launch_training_nohup.sh`. It selects the most recently modified `sam_model_latest.pth` among the default public run directories, restores model/optimizer/scheduler state through the upstream `--resume` option, and writes a separate resume log. It restarts at the next epoch after the saved epoch; partial-epoch work is repeated. For a specific run use `TASK_NAME=<run_name> RESUME=1 bash scripts/launch_training_nohup.sh`. The upstream checkpoint does not save RNG or AMP scaler state, so this is epoch-level recovery rather than an exact continuation of the interrupted batch.

Monitor the process and follow its log from another terminal:

```bash
bash scripts/monitor_training.sh status
bash scripts/monitor_training.sh tail
```

Defaults are one GPU (`GPU_IDS=0`), batch size 1, accumulation 8, 128-cube input, and 50 epochs. You can set `TASK_NAME` to choose a run name or override values such as `GPU_IDS`, `BATCH_SIZE`, `NUM_WORKERS`, `ACCUMULATION_STEPS`, `IMG_SIZE`, `NUM_EPOCHS`, and `LEARNING_RATE` when launching. Each run's log is `logs/training/<task_name>.log`; `latest.log` follows the newest launch. This launcher is configured for single-GPU training.

This official training script trains on `labelsTr` and reports training-set metrics; it does not provide a held-out validation split in its standard `train.py` path. Treat those metrics as agreement with TotalSegmentator pseudo-labels, not anatomical accuracy. The first run is a pipeline/adaptation baseline. Do not use it to claim FSDP clinical performance; evaluation against the private expert-labeled holdout is a later, separate stage.

## Audit the external private FSDP dataset

The final private annotations are in `labelsTr` (confirmed by the user). Keep the external source dataset intact and inspect the existing held-out split before preparing fine-tuning data:

```bash
cd ~/raid/Clara/fsdp-train
git pull origin main
conda activate fsdp-med3d
mkdir -p outputs/private
python -u scripts/audit_private_dataset.py \
  --root "$HOME/raid/normal case_86" \
  --output outputs/private/audit.json \
  | tee outputs/private/audit_summary.log
```

The script matches NIfTI basenames, allowing an image-only `_0000` suffix and the confirmed label suffix `-label` (for example `subject.nii.gz` pairs with `subject-label.nii.gz`). It checks all pairs for 3D geometry, label values, and empty masks. Ambiguous pairs are reported rather than guessed. The console summary includes a redacted structural description of `held_out_test_set_30percent.json` and counts how many `test_set` entries match paired case IDs; it does not apply a split. Detailed case IDs and paths remain in the ignored `outputs/private/audit.json` on DGX. Use the audit results to define the target IDs and interpret the existing split before creating private training/validation folders. Private supervised fine-tuning should use a new run/checkpoint initialization, not `RESUME=latest` of the public experiment.

## Prepare and fine-tune private bilateral FSDP

The reviewed ColorTable samples define `1=RFSDP` and `9=LFSDP`. The first private target is their binary union; other structures are background. Preparation checks every included patient's ColorTable before extracting these two labels. Included patients must have an `ok` audit, matching scalar 3D geometry, valid integer labels, and a nonempty FSDP target. No label IDs are inferred from their numeric order.

The actual audit found 84 `ok` pairs and 2 geometry mismatches with substantially different image/mask grids. Use `--exclude-geometry-mismatch` for the first private run: it omits only those audited failures and records the excluded IDs/reasons in the local manifest. It does not modify the original scans/masks, resample mismatched masks, or accept other failure types. This produces **54 train / 10 validation / 20 test** from 84 eligible patients. The 20 test patients are selected only from eligible IDs in the old test pool. Revisit the two excluded cases separately to establish whether they belong to the same scan/coordinate frame before alignment.

```bash
cd ~/raid/Clara/fsdp-train
git pull origin main
conda activate fsdp-med3d
python -u scripts/prepare_private_fsdp.py \
  --root "$HOME/raid/normal case_86" \
  --audit outputs/private/audit.json \
  --output-dir data/processed/private/fsdp \
  --exclude-geometry-mismatch
```

With 86 valid patient IDs, the frozen split is **56 train / 10 validation / 20 test**, seed 2026. The 20 test patients are a deterministic subset of the 26 IDs in the original `test_set`; remaining patients, including the other six old test IDs, are eligible for train/validation. The original JSON is preserved. This assumes each numeric filename ID identifies one patient and one CT; provide a patient grouping map before preparation if there are multiple IDs/scans per patient. The script stops on ambiguous IDs or mismatched inputs, writes through a temporary directory, and refuses to overwrite a frozen dataset. The local `manifest.json` records patient membership, source paths, target definitions, original split checksum, and per-side voxel counts. It is ignored by Git.

Images retain native geometry and intensities, with hard links when possible or copies otherwise; masks are separate binary uint8 NIfTI files. Each split has its own `ct_private/imagesTr` and `ct_private/labelsTr` directories. Only `train/ct_private` is passed to the trainer; directory separation prevents its loader from including validation/test. The prep summary flags target bounding boxes wider than the 128-voxel training crop for further inspection.

Once preparation succeeds, apply the trainer fixes and start a new private run under nohup:

```bash
bash scripts/setup_sam_med3d.sh && bash scripts/launch_private_training_nohup.sh
bash scripts/monitor_training.sh tail
```

The private launcher picks the most recently modified public `sam_model_latest.pth` under the default public run prefix, verifies it records at least 50 completed epochs, and uses its model weights to initialize a fresh private run. Set `PUBLIC_CHECKPOINT=/absolute/path/to/sam_model_latest.pth` to select a public run explicitly. Private runs start with `fsdp_private_binary_`, reset optimizer/scheduler and epoch count, and default to learning rate `2e-5`, batch size 1, accumulation 8, 50 epochs, and GPU 0. The common launcher verifies the runtime data path and paired filenames before starting the background process. Logs/PIDs and checkpoint directories are separate from public runs.

The upstream trainer currently reports **training** metrics only; the 10 validation patients are reserved for a separate evaluation stage and are not used for automatic checkpoint selection. Its `dice_best`/`loss_best` filenames reflect training metrics. Do not evaluate the 20 test patients during model development. This model remains prompt-based, using label-derived prompts and target-centered crops during training; these metrics do not measure automatic full-volume segmentation.

To recover an interrupted private run, use the exact private run name printed at launch:

```bash
TASK_NAME=fsdp_private_binary_<timestamp> RESUME=1 \
  TRAIN_DATA_DIR="$PWD/data/processed/private/fsdp/train/ct_private" \
  LEARNING_RATE=2e-5 bash scripts/launch_training_nohup.sh
```

Do not use `RESUME=latest` for private recovery: that shortcut deliberately searches public run names. Resume restores the saved optimizer learning rate; a new private experiment instead uses the private launcher.

## Diagnose the completed private run on validation

The private training screenshot reached epoch 49/49 with final training loss `7.16802140518471` and Dice `0.2425045617338684`. Prepare diagnostics before changing training settings. This stage reads **only the 10 frozen validation patients** and compares the private run's `latest`, training-`loss_best`, and training-`dice_best` checkpoints. It exports their stored training history, runs deterministic oracle-prompt evaluation at 1 and 5 clicks, and creates overlays and crop NIfTI files. It does not train a model.

```bash
cd ~/raid/Clara/fsdp-train
git pull origin main
conda activate fsdp-med3d
bash scripts/launch_validation_nohup.sh
tail -n 80 -F logs/validation/latest.log
```

The launcher uses nohup, saves a timestamped log/PID under `logs/validation/`, and prints the output directory under `outputs/private/diagnostics/`. Default selection is the newest private run with a latest checkpoint; the evaluator checks that checkpoint's training path and requires 50 completed epochs. To choose a run or physical GPU explicitly:

```bash
PRIVATE_RUN_DIR="$HOME/raid/Clara/SAM-Med3D/work_dir/<private_run_name>" \
  GPU_ID=0 bash scripts/launch_validation_nohup.sh
```

After completion, read `outputs/private/diagnostics/latest/summary.json` to get the latest validation summary. `latest` points to the output directory corresponding to `logs/validation/latest.log`.

Artifacts in the printed diagnostics directory:

- `summary.json`: training extrema, validation comparison, and best of the three candidates by mean full-target Dice at 5 clicks. Any failed case prevents selection.
- `training_history.csv` and `training_curves.png`: stored loss and Dice for every training epoch; their extrema are training extrema, not validation results.
- `validation_cases.csv` and `checkpoint_comparison.csv`: per-case and aggregate validation metrics at 1/5 clicks, plus crop retention.
- `protocol.json`: checkpoint run, frozen manifest checksum, preprocessing, prompt and threshold definitions, seed, and dependency versions.
- `failures.json`: per-case errors, if any. A partial validation result is not used to select a checkpoint.
- Each checkpoint subfolder contains 10 `val_XX_overlay.png` files and 5-click crop predictions. The latest-checkpoint folder also contains CT/target crops and preprocessing metadata. Aliases follow the sorted validation IDs in the frozen manifest; artifacts and detailed IDs stay local in ignored directories.

The diagnostic matches the upstream loader's clamp and native-voxel, GT-centered 128-cube preprocessing (including its discarded affine before cropping), and applies training normalization `ZNormalization(x > 0)`. Physical geometry is restored when saving crops. `target_retention` measures the fraction of full-volume GT inside the crop. `dice_crop` scores the crop, while `dice_full_target` includes all GT voxels and treats voxels outside the crop as predicted background; padded voxels outside the original CT are excluded from this latter score. Both-empty crop pairs score 1 and their count is reported separately.

Prompt clicks are generated from GT/prediction disagreement; the latest click plus previous mask is supplied to the decoder. No random flips or prompt dropout are applied. This is a **GT-assisted ROI and interactive-prompt diagnostic**, not a measurement of automatic full-scan FSDP detection. The main metric uses `logit > 0` (equivalent to probability >0.5). The CSV also reports crop Dice with the trainer's original `logit > 0.5` threshold for context. Existing training logs/checkpoints are unchanged, and their Dice is not directly equivalent to this evaluation protocol.

After the log reports completion, share `summary.json` and selected overlays to decide whether cropping, normalization, prompting, or training settings need revision. Keep the 20 test patients untouched during these decisions.

## Compare original, public, and private weights

The first validation completed all 10 cases with no failures. All three private checkpoint files gave the same scores at saved epoch 50: mean Dice 0.1111009692 at 1 click and 0.1067041250 at 5 clicks. Target retention was 100% with no empty crops, so lost target voxels from cropping do not explain these validation results. Identical scores alone do not prove that checkpoint tensors are identical. The next comparison changes only the weights while retaining the same validation split and preprocessing/prompt protocol.

```bash
cd ~/raid/Clara/fsdp-train
git pull origin main
conda activate fsdp-med3d
bash scripts/compare_pretraining_nohup.sh
tail -n 80 -F logs/validation/latest.log
```

After completion:

```bash
cat outputs/private/diagnostics/latest/summary.json
```

The stages are `turbo_original` (`ckpt/sam_med3d_turbo.pth`), `public_finetuned` (the actual initialization path recorded in the private checkpoint's args), and `private_finetuned` (that private run's latest checkpoint). The script does not guess a different public run when the recorded source is missing or ambiguous; use `PUBLIC_CHECKPOINT=/absolute/path/to/the/actual/public/checkpoint.pth` to supply it explicitly. `TURBO_CHECKPOINT` and `PRIVATE_RUN_DIR` can also override their respective sources. Original turbo checkpoints may contain a model state dictionary directly or under `model_state_dict`; both formats are supported. File paths, SHA256 checksums, and whether the public source was overridden are recorded in `protocol.json`.

Each stage evaluates all 10 validation cases at 1 and 5 clicks, with the same case seeds and the same first positive click. Subsequent oracle clicks adapt to each model's errors, so their coordinates can differ across models. No retraining or test-image reading takes place. Each stage gets its own overlays and crop predictions; the private stage also exports CT/target crops. New runs preserve earlier diagnostic folders and update the same latest log/output links.

`summary.json` and `checkpoint_comparison.csv` now include macro precision/recall, prediction-to-target voxel ratio, zero-prediction counts, zero-overlap counts, median Dice, and legacy-threshold Dice. Ratios above 1 indicate more predicted voxels than target voxels; they do not by themselves establish where the errors occur. `validation_cases.csv` contains TP/FP/FN counts and all case metrics. `paired_changes_vs_turbo.csv` records per-patient Dice differences between each fine-tuned stage and turbo. `validation_comparison.png` plots stage Dice/recall. Overlays use cyan GT and magenta prediction with translucent fill, including for predictions covering an entire slice.

The highest 5-click validation score among these three candidates is reported using the existing `selected_by_validation` field; this is a comparison at a fixed click budget, not a search across all possible checkpoints or settings. Inspect stage scores, paired changes, and overlays before deciding whether to change initialization, preprocessing, or fine-tuning. Findings about transfer are limited to this assisted ROI/prompt protocol.

## Verify the pipeline, then run up to 200 private epochs

The [SAM-Med3D paper, section 4.2](https://arxiv.org/html/2310.15161v3#S4.SS2), reports 800 epochs for its first pretraining stage using 131K masks. This is evidence for that large-scale pretraining schedule, not a guarantee that private FSDP fine-tuning needs 800 epochs or will reach Dice 0.8. The [official training script](https://github.com/uni-medical/SAM-Med3D/blob/main/train.py) defaults to 200 epochs. For this 54-patient training set, the revised initial budget is **up to 200 epochs**, chosen as an engineering starting point rather than a paper-derived optimum. With batch size 1 and accumulation 8, 54 batches give 7 update attempts per epoch: 700 at 100 epochs, 1,400 at 200, and 5,600 at 800. AMP can skip individual updates. Data size alone cannot determine the optimal epoch count.

Early stopping checks validation at the configured intervals. Default patience is 50 epochs without a cumulative Dice improvement exceeding 0.001 (0.1 percentage point) relative to the patience anchor; default validation interval is 10 epochs. The best checkpoint still tracks every positive validation improvement, independently of the patience threshold. Thus the run can end before 200 epochs and exports the selected checkpoint, not simply the final weights.

The user confirmed that an older 800-epoch run is active on DGX and explicitly requested stopping it and starting a new 200-epoch run. New defaults and early stopping do not change that running process. On DGX use:

```bash
cd ~/raid/Clara/fsdp-train
git pull origin main
conda activate fsdp-med3d
bash scripts/restart_private_200.sh
tail -n 80 -F logs/training/latest.log
```

The replacement script identifies exactly one active 800-epoch trainer from this project's PID files, saved configuration, Linux `/proc` command line, and run path. It rechecks process start time to reject a reused PID, sends SIGTERM, waits up to 30 seconds for exit, and then launches a fresh 200-epoch run under nohup. If identification or stopping fails, no new run is launched. If several 800-epoch runs exist, specify `LONG_RUN_NAME=<exact_run_name>`; other ambiguity is not guessed. A process still in the preflight phase is not killed as though it were the trainer. The stop is recorded under `outputs/private/stops/`. Existing checkpoints/logs are not removed; work since the last saved checkpoint may be lost. GPU/process stopping is executed by this command on DGX, not remotely from the local workspace.

The new run uses a fresh name, default original-turbo initialization, and a 200-epoch schedule with patience 50. It does not resume the old optimizer/schedule. Changing the epoch budget on resume would violate the old frozen configuration; do not edit a saved configuration to bypass that check. Older checkpoints must be recovered with their recorded code/config, since they predate the new early-stop state.

Code inspection of the [upstream training loop](https://github.com/uni-medical/SAM-Med3D/blob/main/train.py) found an accumulation defect: it steps at zero-based batches 8,16,...,48 with 54 batches and accumulation=8. The first window contains 9 samples and the final 5 samples' accumulated gradients are discarded at the next epoch's zero_grad. The corrected loop uses 8-sample windows plus a final 6-sample window: 7 updates per epoch rather than 6 when AMP does not skip updates. This is a real implementation issue; its contribution to low Dice has not yet been isolated.

```bash
cd ~/raid/Clara/fsdp-train
git pull origin main
conda activate fsdp-med3d
bash scripts/launch_private_long_nohup.sh
tail -n 80 -F logs/training/latest.log
```

The entire pipeline runs under nohup. First `verify_private_pipeline.py`:

- checks the saved diagnostics' frozen manifest checksum;
- checks original CT pixels, scalar geometry, and original labels 1/9 against all 54 prepared train and 10 validation pairs;
- records training-style crop retention and CT/normalized target intensities in `pipeline_cases.csv`;
- independently recalculates Dice from the 10 saved prediction/GT crop files, checks it against the CSV, and reconstructs GT crops using physical-space resampling to verify their saved geometry;
- copies the best, middle, and worst validation overlays into the preflight folder for manual review.

The 20 test image/label files are not read. A failed numerical consistency check stops the worker before training. Original data and the frozen split are not edited. Preflight artifacts are at `outputs/private/preflight/latest/verification.json` and `outputs/private/preflight/latest/*overlay.png`. These checks cannot establish anatomical annotation correctness or guarantee the absence of every coordinate/model bug; actual overlays still need review.

When preflight passes, the worker starts `train_private_long.py` in a new ignored `runs/fsdp_private_v2_<timestamp>_<pid>/` directory. Defaults:

- original turbo initialization (direct turbo-to-private), all modules trainable;
- maximum 200 epochs, patience 50 epochs, minimum significant validation gain 0.001; batch size 1, accumulation 8, gradient norm clip 1;
- AdamW, LR `2e-5` for all parameters, weight decay 0.1; five-epoch warmup and cosine decay to 5% of base LR;
- training-matched native GT-centered 128-cube crop and `ZNormalization(x>0)`, random flips on three voxel axes;
- 5 oracle point interactions with the latest point and detached previous mask; per-click DiceCE losses are averaged, with no random prompt dropout;
- probability threshold 0.5 for both train and validation metrics;
- initial validation, then validation at epoch 1, every 10 epochs, and the final epoch; checkpoint selection by mean validation full-target Dice at 5 clicks.

This is a revised experiment, not a pure epoch-count ablation: initialization, optimizer parameter-group LR, accumulation, prompt/dropout handling, loss averaging, clipping, and LR schedule differ from the first private run. Old training loss/Dice must not be directly compared as if their protocols were identical. In particular the first trainer summed click losses and used smaller prompt/decoder learning rates. Results remain GT-assisted segmentation measurements, not automatic full-volume detection.

The run stores `config.json` with code/data/initial-weight hashes, `history.csv`, `progress.json`, `validation_initial.csv`, `validation_latest.csv`, and `validation_best.csv`. `sam_model_val_best.pth` is selected by validation, while `sam_model_latest.pth` supports recovery. Latest is saved at validation intervals, on validation improvement, and at completion; the initial state is also saved. A crash can lose up to the unsaved interval of work. Saves use a temporary file and atomic rename. Resume restores model, optimizer, scheduler, AMP scaler, and RNG states and refuses incompatible code/config/data. Preprocessed CPU crops are cached; use one worker to avoid duplicate caches and preserve the tested resume behavior.

When the run completes, `summary.json` reports the selected validation epoch and `best_validation/` contains its per-case metrics and overlays. An initial checkpoint can remain best if training never improves validation. No Dice target is promised.

Monitor the printed run name:

```bash
cat outputs/private/preflight/latest/verification.json
cat runs/<printed_run_name>/progress.json
```

To resume, repeat the original settings and provide its name:

```bash
LONG_RUN_NAME=fsdp_private_v2_<timestamp>_<pid> RESUME=1 \
  bash scripts/launch_private_long_nohup.sh
```

Optional launch overrides include `NUM_EPOCHS`, `VAL_EVERY`, `PATIENCE_EPOCHS`, `MIN_DELTA`, `LEARNING_RATE`, `ACCUMULATION_STEPS`, `TRAIN_CLICKS`, `SEED`, `GPU_ID`, `INITIAL_CHECKPOINT`, `PRIVATE_DATA_DIR`, and `DIAGNOSTICS_DIR`. Set `PATIENCE_EPOCHS=0` only to disable early stopping intentionally. `DIAGNOSTICS_DIR` should point to a completed diagnostic folder with saved private CT/target/prediction crops. Resume requires the same values as the original run. For a standalone preflight before allocating a long GPU run:

```bash
python -u scripts/verify_private_pipeline.py \
  --data-dir data/processed/private/fsdp \
  --diagnostics outputs/private/diagnostics/latest \
  --output-dir outputs/private/preflight/manual_check_01
```

Regression tests: `python -m unittest discover -s tests -v`. The 54-sample accumulation test compares learned parameters with an independent explicit-window reference, including the partial final window.

## Audit the epoch-20 validation-best checkpoint

The v2 run stopped at epoch 70 with the selected checkpoint at epoch 20 (validation Dice about 0.12696). The last epoch's training Dice was about 0.96439. Before assigning this gap entirely to overfitting, use the same frozen model, no augmentation, and the identical inference/prompt protocol on train and validation:

```bash
cd ~/raid/Clara/fsdp-train
git pull origin main
conda activate fsdp-med3d
bash scripts/launch_best_audit_nohup.sh
tail -n 80 -F logs/audit/latest.log
```

After completion:

```bash
cat outputs/private/best_checkpoint_audit/latest/summary.json
```

Default selection is the most recently completed v2 run with a `summary.json`. Set `PRIVATE_V2_RUN_DIR=/absolute/path/to/runs/<run>` to select it explicitly. The script verifies the frozen manifest/checkpoint training path and uses only train/validation image files. It compares 1/5-click full-target Dice, crop Dice, precision, recall, and crop retention; both splits use eval mode, deterministic per-case oracle clicks, and no flips. It records the logged training metric at the selected epoch separately because that statistic comes from the original training process. Validation must reproduce the saved best-checkpoint summary within 1e-4 or the audit exits with an error.

`train_ranked_cases.csv`, `val_ranked_cases.csv`, and `cases.csv` contain all case metrics. Six overlay PNGs show the best/middle/worst train and validation cases at 5 clicks, with matching crop predictions. The 20 test patients remain untouched. A high frozen-model train score with low validation score supports a generalization gap under this protocol; if frozen train scores are also low, examine protocol/measurement differences and actual overlays before further training.

## Check prompt coordinates after reviewing overlays

The supplied overlays show partial coverage and predicted regions separated from GT on the selected planes. They support the low-overlap findings, but three displayed slices cannot establish a global transform error or an empty 3D prediction. A code-level candidate is the [upstream prompt encoder](https://github.com/uni-medical/SAM-Med3D/blob/main/segment_anything/modeling/prompt_encoder3D.py): dense positional features use the second tensor axis as their first coordinate, while sparse point features use the incoming coordinate order. A CPU check on the official module found that swapping the first two point coordinates reconciles those positional features at non-diagonal cell centers of the 128/8 cubic grid. Because the released weights may have learned with that convention, inference quality must be checked independently before changing training.

```bash
cd ~/raid/Clara/fsdp-train
git pull origin main
conda activate fsdp-med3d
bash scripts/launch_prompt_audit_nohup.sh
tail -n 80 -F logs/prompt_audit/latest.log
```

After completion:

```bash
cat outputs/private/prompt_audit/latest/summary.json
```

The A/B audit uses the completed v2 run's validation-best checkpoint and the same 10 validation patients. Set `PRIVATE_V2_RUN_DIR` to select the run explicitly. `native` forwards points unchanged; `swap_xy` exchanges only the two coordinates entering the prompt encoder. Dense image features, CT/GT arrays, prior-mask feedback, threshold, and click budgets are unchanged. First sampled voxel coordinates are identical; subsequent oracle error clicks can differ as predictions differ. `prompt_traces.json` records the original voxel coordinates, encoder input coordinates, and labels at every click. No model parameters are updated, and train/test images are not read.

`position_encoding_contract` in `summary.json` tests the actual loaded encoder, not an assumed repository version. It compares sparse and dense features at non-diagonal embedding-cell centers, reporting errors and cosine similarities for both coordinate orders. `native_reproduces_saved_validation` must be true before attributing any performance changes to the coordinate setting. The summary compares Dice/precision/recall; `cases.csv` and `paired_changes.csv` retain per-patient results. Each mode has 10 distinctly named overlays and crop predictions in its own folder. A positional-feature mismatch alone is not sufficient evidence to declare the swap a model fix; use the validation A/B scores and images.

## Measure the three experts' FSDP annotation agreement

The folders `labels_exp1`, `labels_exp2`, and `labels_exp3` are three experts' annotations. The first DGX inventory contradicted the initially supplied shared 1/9 mapping: expert raw values were 0–2, 0–4, and 0–5, with no value 9 in any of the 54 training patients. The previous `complete: true` checked computational completeness but did not detect this unsupported target mapping. Those scores are raw-ID comparisons, not verified anatomical FSDP agreement. The shared mapping config has been removed; the launcher now defaults to inventory only. `labelsTr` remains the final reference, with its 1/9 mapping supported by per-patient ColorTables. Run raw-label correspondence first:

```bash
cd ~/raid/Clara/fsdp-train
git pull origin main
conda activate fsdp-med3d
bash scripts/launch_label_correspondence_nohup.sh
tail -n 80 -F logs/label_correspondence/latest.log
```

After completion:

```bash
cat outputs/private/label_correspondence/latest/summary.json
```

The first DGX correspondence result (54 training patients, issues=0) found zero overlap between every expert nonzero class and final FSDP 1/9. In both exp2 and exp3, raw class 1 exactly equals the union of final 2+3+4, and raw class 2 equals 10+11+12; exp3 raw class 5 equals final 17 (FSC). These voxel identities suggest class regrouping but do not establish annotation provenance. The reported ~0.82 Dice compares raw group masks with individual non-FSDP reference classes; it is not FSDP model performance. Retain labelsTr 1/9 as the final target and do not use these folders to infer FSDP inter-rater agreement or replace the reference.

The correspondence audit reports every nonzero expert raw value against final labels 1–17 and the bilateral 1+9 union. `reference_overlap.csv` distinguishes equal-patient mean Dice (both-empty omitted) from pooled voxel Dice, and reports raw/reference coverage and presence counts. `pooled_contingency.csv` includes background overlaps; `inventory.csv` records per-case raw-value volumes. `summary.json` ranks the top three single-reference-label overlaps per raw expert value and separately reports FSDP comparisons. **These are overlap candidates, not verified semantic mappings**; zero-overlap labels are excluded from the ranked candidates. Expert-specific label definitions or annotation provenance are needed before supplying `EXPERT_MAPPING_JSON` to the agreement launcher. No labels are remapped or replaced, and no training is started.

The audit runs under nohup on CPU and reads image/mask content only for the **54 frozen training patients**. It lists candidate NIfTI filenames in the four source folders to match numeric patient prefixes, but never opens the validation/test scans or masks. It accepts `<id>.nii[.gz]` or a numeric ID followed by a `-`/`_` suffix; multiple candidates are reported as ambiguous. It verifies source CT headers, scalar 3D masks, per-patient final-reference ColorTables, integer label values, and image/mask physical geometry. It does not resample or edit masks. Missing/ambiguous files, invalid values, and geometry mismatches appear in `issues.json` and prevent a complete result. All-rater comparisons use the common set of patients with all four valid annotations; any partial cohort is explicitly marked incomplete.

After expert-specific semantic confirmation, `EXPERT_MAPPING_JSON=/absolute/path/to/confirmed_mapping.json bash scripts/launch_expert_audit_nohup.sh` runs the agreement audit. Each expert must have an object with distinct positive integer `RFSDP` and `LFSDP` IDs. If an asserted target ID is absent across the whole training cohort, the audit exits nonzero, marks `mapping_unsupported`, suppresses endorsed agreement/heatmaps, and preserves arithmetic only as explicitly unverified diagnostics. Presence alone does not establish anatomical identity.

There are three inter-expert pairs plus three final-reference-versus-expert pairs, each scored separately for RFSDP, LFSDP, and bilateral union. Dice averages weight each nonempty patient pair equally. A pair with one empty target gets Dice 0; a pair with both targets empty is omitted from Dice averages and counted separately. Per-pair empty counts, compared-patient counts, median/range, physical target volumes, and centroid distances help expose missing annotations or spatial disagreement. Agreement is evidence about consistency, not a mathematical ceiling on achievable model Dice, and no annotator or consensus mask is automatically declared ground truth.

Agreement artifacts include:

- `summary.json`: label values, completion status, and pair/target aggregate metrics; no case IDs are printed in the console.
- `pairwise_summary.csv` and `case_agreement.csv`: aggregate and per-case pairwise scores/volumes/distances.
- `agreement_heatmap.png`: the three target-specific agreement matrices, with nonempty case counts (only generated for complete audits).
- `inventory.csv`, `issues.json`, and `protocol.json`: pairing/geometry issues, actual label mapping, mapping/script/manifest checksums, and local provenance. Patient IDs remain local in the ignored protocol file; case CSVs use `train_XX` aliases in sorted training-ID order.

`PRIVATE_SOURCE_ROOT` overrides the default external source `~/raid/normal case_86`; `PRIVATE_DATA_DIR` overrides the prepared dataset root. `EXPERT_MAPPING_JSON` supplies an explicitly confirmed per-expert mapping; no tracked mapping is assumed. Direct use of `audit_expert_agreement.py` without a confirmation flag or mapping file is inventory-only and deliberately does not infer label semantics. The audit creates no consensus labels, does not change patient splits, and does not launch training. Read the agreement and representative annotations before choosing the next side-specific/limited-parameter experiment.

## Two-patient side-specific memorization diagnostic

This is a training-only learnability check before another full-cohort experiment. It selects exactly two patients from the frozen 54-patient training list using seed 2026, extracts original final RFSDP=1 and LFSDP=9 separately, and checks each source against its ColorTable, geometry, manifest side counts, and prepared bilateral target. It reads no other patients' image/mask content and leaves all source files, prepared data and patient splits unchanged. Both sides must be nonempty and fully retained in their native 128³ GT-centered crop; unsupported geometry or target truncation stops the run.

```bash
cd ~/raid/Clara/fsdp-train
git pull origin main
conda activate fsdp-med3d
bash scripts/launch_private_tiny_nohup.sh
tail -n 80 -F logs/tiny_training/latest.log
```

The launcher uses nohup, a separate tiny-training log/PID directory, a new timestamped run, and a separate report symlink. Default initialization is the original pretrained `SAM-Med3D/ckpt/sam_med3d_turbo.pth`; it does not replace a previous private checkpoint. The diagnostic uses all model parameters, constant AdamW LR 1e-4, zero weight decay, no augmentation, one optimizer step per side ROI, and five averaged interactive losses with detached mask feedback. Optimization and scoring both use eval mode; gradients remain enabled during optimization. Per-side prompt seeds and inference behavior are shared between training and scoring. CUDA AMP uses the existing inference policy and skipped updates are logged.

The budget is at most 200 epochs (four attempted updates per epoch), with frozen-model evaluation after epoch 1 and every 5 epochs. Training stops when **every side and every combined bilateral patient** reaches Dice ≥0.90 at five oracle clicks per side. This threshold tests memorization, not generalization. A completed run with `memorization_check_passed: false` did not reach the configured minimum-Dice threshold; this does not mean no learning occurred or that training crashed. The best diagnostic checkpoint is selected by the minimum of these held-in scores, reloaded and checked for score reproduction, and exported with before/after overlays and NIfTI prediction crops. No resume is provided; reruns create new isolated runs.

```bash
# During the run:
cat outputs/private/tiny_training/latest/progress.json

# After completion:
cat outputs/private/tiny_training/latest/summary.json
cat outputs/private/tiny_training/latest/best_training/sides.csv
cat outputs/private/tiny_training/latest/best_training/patients.csv
```

The bilateral score restores both predictions into their original voxel grid, removes padding predictions, unions overlapping predictions once, and scores the entire final 1+9 target. Click budgets are **per side**: bilateral @1 consumes two clicks total and bilateral @5 consumes ten, plus GT-derived side ROIs. These scores are not directly comparable with the earlier single-ROI five-total-click validation protocol. High tiny-training Dice does not establish an improvement on the 10 validation patients or a claim of 80% FSDP performance; the 20 test patients remain unopened.

Overrides: `GPU_ID` (default 0), `TINY_EPOCHS` (200), `TINY_EVAL_EVERY` (5), `TINY_LEARNING_RATE` (1e-4), `TINY_TARGET_DICE` (0.90), `SEED` (2026), `INITIAL_CHECKPOINT`, `SAM_MED3D_DIR`, `PRIVATE_SOURCE_ROOT`, and `PRIVATE_DATA_DIR`. Do not tune these on the test set. Artifacts and local patient provenance stay in ignored run/output/log directories. A script failure writes `failure.json` inside its new run directory; inspect that and the nohup log before retrying. The launcher does not stop or overwrite another training experiment.

The first actual DGX tiny run completed 200 epochs, selecting epoch 190. Held-in bilateral patient Dice improved from 0.02022 to 0.79269 at five oracle clicks per side; mean side Dice was 0.79412 and minimum side Dice 0.77193. The configured 0.90 threshold was not reached. There were 798 successful updates and two AMP skips; validation/test were untouched. These are training-only memorization results. A fair next comparison needs the same side-ROI protocol on the 10 validation patients for original turbo, the prior full-cohort best checkpoint, and tiny best before choosing full-cohort retraining settings.

## Compare side-ROI validation before full-cohort retraining

This read-only comparison evaluates original Turbo, the previous v2 validation-best checkpoint, and tiny best on **all 10 frozen validation patients (20 side ROIs)**. Preprocessing, native-coordinate prompts, fixed per-side seeds, threshold and original-grid bilateral scoring are identical across checkpoints. It reads no training/test image content, creates no optimizer, and never replaces any checkpoint or changes annotations/splits.

```bash
cd ~/raid/Clara/fsdp-train
git pull origin main
conda activate fsdp-med3d
GPU_ID=0 bash scripts/launch_side_validation_nohup.sh
tail -n 80 -F logs/side_validation/latest.log
```

After completion:

```bash
cat outputs/private/side_validation/latest/summary.json
```

The default tiny source is `outputs/private/tiny_training/latest`. The previous v2 checkpoint is selected from `outputs/private/best_checkpoint_audit/latest/summary.json`, with its recorded SHA256 checked. Override `TINY_RUN_DIR` or `PRIVATE_V2_RUN_DIR` with explicit completed run directories if needed; `V2_AUDIT_SUMMARY` can point to another independent audit. `SAM_MED3D_DIR`, `TURBO_CHECKPOINT`, `PRIVATE_DATA_DIR`, `PRIVATE_SOURCE_ROOT`, `GPU_ID` and `SEED` are also supported. Both trained checkpoints must match the frozen manifest and the original Turbo initialization. Tiny best must match its completed summary/checksum and two-training-patient provenance; v2 best must match its completed epoch, saved config and declared training path. The source CT/final masks, per-patient ColorTables, full side-target retention, and prepared binary validation masks are verified before inference. Any missing/empty/truncated side or geometry mismatch stops the comparison rather than silently removing difficult validation patients.

Artifacts include `comparison.csv`, `side_cases.csv`, `patient_cases.csv`, patient-matched `paired_changes_vs_turbo.csv`, and checkpoint/config/data provenance in `protocol.json`. Each checkpoint has per-side/per-patient CSVs and worst/middle/best patient overlays with physical prediction crops. Representative re-inference must reproduce its recorded score. A runtime failure marks `summary.json` as `complete: false` and writes `failure.json`; partial scores must not be interpreted as a complete cohort comparison.

The primary comparison is **mean original-grid bilateral patient Dice** at a fixed click budget; per-side and minimum patient scores reveal missed sides. The tiny train/validation gap uses the held-in training reference copied from the previous completed run; training scans are not re-read. This is GT-assisted interactive validation: @1 uses two total clicks and @5 uses ten total clicks, plus GT-derived side crops. It is not a direct comparison with the old 0.12696 result using one bilateral ROI/five total clicks, and does not measure automatic deployment performance. Keep the 20 test patients reserved. After reviewing the actual validation results, choose the next full-cohort training configuration; this command does not launch training or promise a particular validation Dice.

The actual first side-protocol comparison completed on all 10 validation patients: bilateral patient Dice @5 per side was **5.99% Turbo / 10.17% v2 best / 1.17% tiny best**. Tiny's held-in training result was 79.27%, a 78.10-point gap. Its mean validation recall was 0.71%; the high two-patient training score did not generalize. All side targets were fully retained, but this does not establish that all preprocessing or anatomical assumptions are correct. The old 12.70% v2 validation score used a different crop/click protocol and is not directly comparable. The next proposed experiment is side-specific training on all 54 training patients with selection on the fixed 10 validation patients, starting from original Turbo; no such new full-cohort run has been launched yet.

## Train all 54 patients with side-specific targets (v3)

This is the first full-cohort side-specific training experiment: **54 training patients / 108 side ROIs**, **10 validation patients / 20 side ROIs**, with the existing 20 test patients reserved. Initialization must be the same original Turbo checkpoint used by the completed side-validation comparison. The small two-patient checkpoint is not used. The maximum budget is 200 epochs, validation every 5 epochs, and early stopping after 50 epochs without improvement greater than 0.001 Dice. These are initial experiment settings, not a guarantee of 80% validation Dice.

```bash
cd ~/raid/Clara/fsdp-train
git pull origin main
conda activate fsdp-med3d
GPU_ID=0 bash scripts/launch_private_sides_nohup.sh
tail -n 80 -F logs/side_training/latest.log
```

The nohup process includes preparation, source checks, initial validation and training. Source scans/final labels, prepared binary targets, ColorTables, physical geometry, side counts and full-target crop retention are checked on all 64 train/validation patients; no test scans are opened and no source/prepared files are edited. Derivatives and local patient provenance are written only to a new run. The original Turbo must reproduce **every saved side and bilateral validation Dice** at one/five clicks per side within 1e-4 before the first optimizer step. A mismatch stops training and must be investigated. Preparation may take time before epoch logs appear.

Defaults preserve the v2 optimizer recipe while changing the training targets to individual sides: all parameters, AdamW LR 2e-5, weight decay 0.1, five averaged interactive losses with detached feedback, independent XYZ flips, shuffled side ROIs and epoch-varying oracle prompts. Accumulation is 8 ROIs, including the final 4-ROI window: **14 attempted optimizer updates per epoch**. This doubles ROI/update exposure relative to the earlier 54-batch v2 run; it is not an isolated single-variable ablation. CUDA AMP successes/skips are logged. Five-epoch warmup and cosine LR decay are used.

```bash
# During training:
cat outputs/private/side_training/latest/progress.json

# Once the complete pipeline finishes:
cat outputs/private/side_training/latest/summary.json
```

The primary selection metric is mean **validation full-target bilateral patient Dice at five clicks per side**. The original Turbo at epoch 0 remains eligible if training never improves validation. Side scores and the worst patient score are also recorded. Checkpoints are `sam_model_side_val_best.pth` and `sam_model_side_latest.pth`. Latest is saved at validation intervals, on improvements, and at stopping/completion; a crash may require recomputing up to four epochs. At the end, the selected checkpoint is reloaded, its validation score must reproduce, and its full training cohort is scored in the same frozen inference mode without augmentation. This provides a train/validation gap that does not confuse moving augmented training statistics with frozen validation. Representative validation overlays, NIfTI predictions and per-case CSVs are exported. The summary also reports the Dice change against v2 under the saved side-validation protocol when that comparison is available.

The protocol remains GT-assisted: native 128³ side-centered ROIs, native prompt order, and one/five oracle clicks **per side** (two/ten bilateral total). No automatic clinical-performance claim or direct comparison with the earlier single-ROI/five-total-click result is justified. Keep test reserved for the final selected protocol.

Defaults can be overridden with `GPU_ID`, `NUM_EPOCHS`, `VAL_EVERY`, `ACCUMULATION_STEPS`, `LEARNING_RATE`, `PATIENCE_EPOCHS`, `MIN_DELTA`, `SEED`, `SAM_MED3D_DIR`, `PRIVATE_SOURCE_ROOT`, `PRIVATE_DATA_DIR`, `TURBO_CHECKPOINT` and `SIDE_BASELINE_DIR`. The baseline defaults to the completed `outputs/private/side_validation/latest` comparison. Logs/PIDs use `logs/side_training`; prior run files and older training logs are preserved.

To resume after a process failure, use the run name printed by the launcher:

```bash
RESUME=1 SIDE_RUN_NAME=fsdp_private_side_v3_YOUR_RUN_NAME GPU_ID=0 bash scripts/launch_private_sides_nohup.sh
```

Resume restores optimizer, scheduler, scaler, history, early-stopping state and CPU/CUDA/Python/NumPy RNG states. Code (including SAM model source), data hashes, initialization, baseline and settings must match the frozen run. Retain the original overrides and set `SIDE_BASELINE_DIR` to its original resolved comparison directory if the `latest` link has since changed. Changed epoch budgets/schedules require a new run; configuration changes are not silently applied during resume. Failed runs write `failure.json`; inspect that and the log before retrying. No existing process is killed by this launcher.

The first full-cohort v3 run completed via early stopping at epoch 75, selecting epoch 25. Under the unchanged side protocol, mean validation bilateral Dice was **19.25%**, versus v2 **10.17%** (+9.08 percentage points); selected-checkpoint frozen full-training Dice was **30.09%**, a 10.84-point gap. Worst validation patient Dice was 9.55% at five oracle clicks per side. This is improvement on the same validation cohort/protocol, not 80% performance or a test result. Inspect the existing validation history and precision/recall/volume errors before choosing the next change; low scores on both splits at the selected checkpoint do not identify a unique failure mechanism.

To inspect the completed v3 run's existing precision/recall, predicted/target voxel ratios and validation trajectory without loading images or weights:

```bash
python scripts/inspect_side_training.py
```

This defaults to `outputs/private/side_training/latest`; `--run-dir` selects an explicit completed v3 run. The command checks full cohort coverage and reproduces the summary's mean Dice from its CSVs before printing aggregates. Training-curve values are moving augmented scores; only the selected-checkpoint aggregates compare train/validation under frozen inference. The report does not modify artifacts or launch another experiment.

The completed v3 inspection shows validation precision 17.07% and recall 24.15%, with mean predicted/target volume ratio 1.42; training precision 25.82% / recall 37.05%. Median bilateral target sizes are about704 voxels (train) and710 (validation). From epoch25 to75, moving augmented training Dice rises 28.81%→48.27% while validation falls 19.25%→13.23%, supporting late overfitting-like behavior. The approved next trial is v4 ROI zoom for these small targets, with its fixed ROI shape selected on training and original-grid full-target scoring. An improved Dice remains a hypothesis until the controlled DGX run completes.

### Run the controlled v4 ROI-zoom experiment

On DGX, from the repository root after activating the project environment:

```bash
git pull origin main
conda activate fsdp-med3d
bash scripts/launch_private_zoom_nohup.sh
```

The launcher requires the completed v3 checkpoint, the existing validation baseline, prepared private data, and original Turbo weights. It writes a new isolated run under `runs/`, logs under `logs/zoom_training/`, and an `outputs/private/zoom_training/latest` link. It does not stop or overwrite an existing run. Before model updates, it selects the minimum fixed native crop shape independently on each XYZ tensor axis, using only the 54 training patients, that retains every side target with a four-voxel margin. CT is interpolated trilinearly and labels with nearest-neighbor sampling to the model's 128³ input. Any validation target outside the crop remains in the original full-target denominator and is penalized. If no centered shape meets the train-only rule, it stops before training and records `roi_policy.json`/`failure.json`.

The initial cubic 64³/96³ policy stopped before optimizer updates. The saved train-only audit showed the minimum crop sizes on SimpleITK array axes Z/Y/X were 120/90/38, equivalent to model tensor axes X/Y/Z of 38/90/120. The revised policy selects those axis sizes from the actual training masks rather than hard-coding them; it also re-evaluates v3 with that same fixed shape.

Monitor the log and early ROI report in another terminal:

```bash
tail -f logs/zoom_training/latest.log
cat outputs/private/zoom_training/latest/roi_policy.json
cat outputs/private/zoom_training/latest/progress.json
```

When complete, inspect `outputs/private/zoom_training/latest/summary.json`, `roi_policy.json`, `roi_controls.json`, `best_validation/patients.csv`, `best_validation/representatives/`, and `selected_v4_native_patients.csv`. The run reports crop retention, full-target Dice after restoration to the original grid, v3 under the matching zoom protocol, and the native/zoom results for the selected v4 checkpoint. Native128 and anisotropic zoom are different GT-assisted protocols; preserve the 20-patient test split until a final protocol is selected.

The first completed v4 run selected epoch15 and stopped at65/200. At five clicks per side, its frozen training bilateral Dice is26.46% and validation Dice22.45%, with full target retention on both splits. Validation controls:

| Checkpoint | Native128 ROI | Zoom XYZ38×90×120 ROI |
| --- | ---: | ---: |
| v3, selected epoch25 | 19.25% | 9.03% |
| v4, selected epoch15 | 9.75% | 22.45% |

V4 improves over v3 under matching zoom inference by13.42 percentage points. Comparing their respective training/inference pipelines, the best validation result rises3.20 points from19.25% to22.45%; this is a changed ROI protocol, not an isolated inference benefit. Both models are sensitive to the ROI used at inference. These are GT-assisted validation scores on ten patients with ten total oracle clicks per patient; they do not establish80% Dice or independent test performance.

Inspect existing v4 precision/recall, volume errors and curve without starting another training run:

```bash
python scripts/inspect_side_training.py --run-dir outputs/private/zoom_training/latest
```

The supplied v4 inspection reports validation precision21.52%, recall25.13%, mean predicted/target volume ratio1.259 (pooled1.179); frozen training precision25.62%, recall28.48%. Both missed target and false-positive overlap remain substantial. Validation peaks at epoch15 (22.45%) and drops to15.48% at65, while moving augmented model-grid training Dice rises26.35% to50.70%; those moving scores are not the selected checkpoint's frozen native-grid training Dice. The next approved experiment is v5 below.

### V5: train only the mask decoder

Start a fresh run on DGX:

```bash
cd ~/raid/Clara/fsdp-train
git pull origin main
conda activate fsdp-med3d
bash scripts/launch_private_decoder_nohup.sh
tail -f logs/decoder_training/latest.log
```

V5 initializes from original Turbo. It keeps the v4 ROI, loss, five-click interactions, patient split, seed, augmentation, accumulation8, AdamW LR2e-5/decay0.1 and warmup/cosine schedule. Only mask decoder parameters are optimized; image and prompt encoders have gradients disabled and stay in eval mode. The default is a200-epoch cap with validation every5 and early stopping after50 epochs without an improvement larger than0.001. This tests the effect of restricting adaptation; it does not assume higher validation Dice.

Before updates, all v4 saved validation side/patient Dice and initial Turbo zoom scores must reproduce within1e-4. V4 supplies a control checkpoint, its train-derived ROI policy, and source/software fingerprints. Any difference stops preflight. The initial Turbo remains eligible for selection. The run checks frozen encoder parameters and persistent buffers against original Turbo at initialization/resume, validation checkpoints and completion.

The launcher defaults to completed v3 and v4 report links. Override `PRIVATE_V3_RUN_DIR`, `PRIVATE_V4_RUN_DIR`, `SIDE_BASELINE_DIR`, or `GPU_ID` when needed. The recipe must match v4; changing LR/epochs/schedule is a separate experiment and is refused by this comparison. New logs/PIDs/report links use `decoder_training`; checkpoints retain names `sam_model_side_val_best.pth` and `sam_model_side_latest.pth` inside the new run. Existing outputs are retained.

Read the parameter/preflight reports and progress from another terminal (files appear as each stage completes):

```bash
cat outputs/private/decoder_training/latest/trainable_parameters.json
cat outputs/private/decoder_training/latest/frozen_encoder_check.json
cat outputs/private/decoder_training/latest/preflight.json
cat outputs/private/decoder_training/latest/progress.json
```

On completion:

```bash
python -m json.tool outputs/private/decoder_training/latest/summary.json
python scripts/inspect_side_training.py --run-dir outputs/private/decoder_training/latest
```

The summary reports validation/frozen training scores and the difference versus v4's reproduced validation Dice, plus frozen-encoder checks. Representative overlays and CSVs are under the summary's `export_directory`. If a run fails, inspect its log and `failure.json`; resume only with identical code, settings and resolved control directories:

```bash
RESUME=1 DECODER_RUN_NAME=fsdp_private_decoder_v5_YOUR_RUN_NAME bash scripts/launch_private_decoder_nohup.sh
```

Selection uses the same GT-assisted native-grid bilateral full-target Dice at five clicks per side (ten total). The20 test patients remain reserved for the final frozen protocol.

V5 completed110/200 epochs via early stopping, selecting epoch60. Its five-per-side frozen training Dice is22.38%, validation19.09%, and train/validation gap3.29 points. V4's reproduced validation is22.45%, so v5 loses3.36 points. Encoder state matches original Turbo;7,575,636 decoder parameters were trainable of100,509,232 total. Both splits retain100% of their targets; test remains unread. V4 remains the best recorded pipeline. These results do not support decoder-only adaptation with this recipe as a replacement for v4; a smaller gap alone does not establish better learning. The approved partial image-encoder adaptation experiment is implemented as v6 below; its effect on validation remains unmeasured.

### V6: final two image encoder blocks plus mask decoder

The approved v6 experiment starts from original Turbo and optimizes only image encoder blocks10/11 (zero-based indices, of12) and the mask decoder. All other image encoder regions, including the neck, and the prompt encoder are frozen and kept in eval mode. Gradients still pass through the frozen neck to the selected blocks. The v4 ROI/loss/LR/augmentation/split/click/schedule recipe and validation selection are retained. Two blocks is a fixed experiment choice; this setup is not a promised improvement.

Start a fresh run on DGX:

```bash
cd ~/raid/Clara/fsdp-train
git pull origin main
conda activate fsdp-med3d
bash scripts/launch_private_partial_nohup.sh
tail -f logs/partial_training/latest.log
```

V6 uses the same control directories and overrides as v5 (`PRIVATE_V3_RUN_DIR`, `PRIVATE_V4_RUN_DIR`, `SIDE_BASELINE_DIR`, `GPU_ID`). It first reproduces v4 and Turbo zoom validation before model updates. The default cap is200 epochs, validation every5, with50-epoch early-stopping patience/min_delta0.001. The primary comparison is v4's22.45% validation with the same native-grid full-target GT-assisted ROI and five-click-per-side protocol. The20 test patients remain reserved.

Inspect parameter regions, frozen-state checks, progress and the final result as files become available:

```bash
cat outputs/private/partial_training/latest/trainable_parameters.json
cat outputs/private/partial_training/latest/frozen_regions_check.json
cat outputs/private/partial_training/latest/progress.json
python -m json.tool outputs/private/partial_training/latest/summary.json
```

Controls retain the inherited filename `decoder_controls.json`, and checkpoints are `sam_model_side_val_best.pth` / `sam_model_side_latest.pth`. The summary reports `validation_dice_change_vs_v4`, native-grid frozen train/validation scores, parameter-region counts and whether all frozen state matches Turbo. The trainable blocks are excluded from that frozen-state claim. Reports and logs use isolated `partial_training` links.

```bash
python scripts/inspect_side_training.py --run-dir outputs/private/partial_training/latest
```

After a process failure, inspect the log and `failure.json`; resume requires identical code/settings and the original resolved control paths:

```bash
RESUME=1 PARTIAL_RUN_NAME=fsdp_private_partial_v6_YOUR_RUN_NAME bash scripts/launch_private_partial_nohup.sh
```

## Supervised 3D U-Net comparator: start with the tiny diagnostic

The approved comparator uses MONAI's volumetric residual U-Net trained from scratch, with no oracle-click input. CT extraction, labels1/9, fixed v4 ROI, normalization,128³ model input, inverse-logit resizing and original-grid full-target bilateral Dice follow v4. Both approaches still use GT-derived crops. SAM and U-Net differ in architecture, initialization, prompting and optimization, so this compares pipelines rather than isolating one architecture change. It does not measure automatic scan-level localization.

Run only the two-patient train diagnostic first:

```bash
cd ~/raid/Clara/fsdp-train
git pull origin main
conda activate fsdp-med3d
bash scripts/launch_private_unet_nohup.sh
tail -f logs/unet_tiny/latest.log
```

The default `UNET_STAGE=tiny` reads two frozen training patients/four sides and uses the saved train-derived v4 ROI policy. It does not open other patient images or validation/test images. Model channels are16/32/64/128/256, strides2/2/2/2, with two residual units and instance normalization. Tiny training uses LR0.001, no weight decay/augmentation, eval mode with gradients, one update per ROI, cap200 and scoring every5/first/final. It can finish earlier if every side and bilateral patient Dice reaches0.90. This is a learnability target, not validation performance. No cohort run starts automatically.

Check progress and the final diagnostic:

```bash
cat outputs/private/unet_tiny/latest/progress.json
python -m json.tool outputs/private/unet_tiny/latest/summary.json
```

The summary's `diagnostic_passed` must be true before the cohort stage. If it is false, the diagnostic still completed normally; inspect its native overlays, resize roundtrip and metrics before changing the recipe. If the process crashes, inspect the log and `failure.json` instead. Cohort launch verifies the tiny source/config fingerprints, checkpoint and supporting per-side/per-patient CSVs. This prevents a manually changed pass flag from bypassing a failed diagnostic.

The first actual U-Net tiny run finished200 epochs, selecting epoch35 by minimum Dice. Its selected bilateral mean Dice is0.76% and minimum side/patient score0.61%, so the cohort gate remains closed. Full retention does not identify the cause of the low score. Audit best/latest weights, grid restoration, numeric precision and optimizer history before another training change:

```bash
git pull origin main
bash scripts/launch_unet_audit_nohup.sh
tail -f logs/unet_audit/latest.log
```

After the audit finishes:

```bash
python -m json.tool outputs/private/unet_audit/latest/overview.json
```

`UNET_TINY_RUN_DIR` selects an explicit completed tiny run and `GPU_ID` selects the GPU. The audit reads only those two train image/mask pairs and their existing checkpoints, computes gradients without an optimizer, and saves new reports under `outputs/private/unet_audit` without editing training artifacts. `summary.json` contains per-side model/native FP32/AMP scores, label roundtrip, probability statistics, update counts and Dice/BCE gradient probes; `overview.json` is the compact report. Gradient probes use FP32 on the first ROI per checkpoint. The revised audit also reports train-only threshold candidates as described below, without changing the original gate or retraining. Failure writes an isolated `failure.json`.

The first actual audit found800/800 updates, no AMP skips, very similar FP32/AMP scores and a98.51% label-transform roundtrip. At epoch200 hard Dice at0.5 is0, but mean model-grid foreground probability is0.456 versus background0.00195. The updated audit tests a calibration hypothesis by scoring fixed probability thresholds0.05–0.95 on the same two training patients, with full native-target scoring. It does not change weights or the original0.5 gate. After pulling the revision, rerun the audit launcher and read:

```bash
python -m json.tool outputs/private/unet_audit/latest/threshold_overview.json
```

`thresholds.json` contains all threshold curves and per-case scores; the compact overview compares0.5 with the candidate maximizing the minimum held-in side/patient Dice (ties prefer proximity to0.5), separately in FP32/AMP for best/latest checkpoints. Choosing thresholds on these two train patients is optimistic calibration; it does not establish validation performance or authorize a gate override. The cohort stage remains blocked until the fixed-protocol diagnostic passes or a reviewed new protocol is implemented.

After reviewing a passed diagnostic, launch the cohort baseline:

```bash
UNET_STAGE=cohort bash scripts/launch_private_unet_nohup.sh
tail -f logs/unet_cohort/latest.log
```

This uses54train/10validation patients and leaves20test patients reserved. It starts from the same random initialization as the tiny run, not from tiny-learned weights. Defaults: AdamW LR0.001/decay0.0001, DiceCE, XYZ flips, accumulation8 including the final four ROIs, warmup/cosine, cap200, validation every5, early-stop patience50/min_delta0.001. Best selection uses native full-target bilateral validation Dice. These are initial baseline settings, not an optimized recipe.

```bash
cat outputs/private/unet_cohort/latest/progress.json
python -m json.tool outputs/private/unet_cohort/latest/summary.json
```

The cohort summary includes selected validation/frozen train Dice, precision/recall, train/validation gap and a comparison with v4's five-click-per-side validation. Actual per-patient CSVs and representative PNG/NIfTI predictions are in its `export_directory`. U-Net CSVs use `oracle_clicks=0`; the older SAM-only inspection script should not be applied to these reports.

Overrides: `GPU_ID`, `PRIVATE_SOURCE_ROOT`, `PRIVATE_DATA_DIR`, `PRIVATE_V4_RUN_DIR`, `UNET_TINY_RUN_DIR`, `UNET_STAGE`, `UNET_RUN_NAME`, `NUM_EPOCHS`, `VAL_EVERY`, `LEARNING_RATE`, `SEED`, `ACCUMULATION_STEPS`, `PATIENCE_EPOCHS`, `MIN_DELTA`. Matching v4 package versions/source helpers and seed are required. Keep the original resolved v4/tiny control directories if their latest links move. Checkpoints are `unet_best.pth` and `unet_latest.pth`; strict resume restores optimizer/scheduler/scaler/RNG/history and refuses code/config/source changes:

```bash
RESUME=1 UNET_STAGE=tiny UNET_RUN_NAME=fsdp_private_unet_tiny_YOUR_RUN_NAME bash scripts/launch_private_unet_nohup.sh
```

## DGX workflow

1. Clone the repository.
2. Create and activate the Conda environment, then install `requirements.txt`.
3. Confirm DGX GPU visibility from PyTorch.
4. Download/extract public data into the ignored `data/` path.
5. Run the NIfTI audit and inspect representative scans/masks.
6. Package accepted pseudo-labels in SAM-Med3D format and launch the public pseudo-label baseline.

Do not place private scans, private subject IDs, checkpoints, or generated masks in Git.
