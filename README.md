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

The completed threshold audit found epoch200 at0.3 reaches87.71% mean bilateral training Dice,84.25% minimum side Dice and85.02% minimum patient Dice, with90.12% mean precision/85.44% mean recall. FP32 and AMP agree. The same weights at0.5 predict an empty native mask and score0. Epoch35 remains poor under the tested thresholds. This supports a calibration explanation for the latest checkpoint's zero hard Dice on these two patients, and demonstrates substantial held-in learning; it does not establish validation performance. The original minimum90% gate is still unmet. A proposed new tiny continuation would freeze0.3, start from the epoch200 latest weights, and test LR0.0001 for at most100 additional epochs with the same90% minimum target. It is pending approval/implementation; no existing gate or old artifact was changed.

The original threshold0.5 cohort launcher below still requires a passed diagnostic. It is not the command for the newly authorized train-calibrated experiment in the next section:

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

## Full-cohort 3D U-Net with the train-calibrated threshold

The user authorized cohort training after reviewing the tiny threshold audit and visualizations. This is an explicit exploratory protocol despite the original minimum90% tiny gate remaining unmet. No previous summary, diagnostic gate, checkpoint or source dataset is changed. The two-patient87.71% training Dice is not a prediction of cohort validation performance.

Run on DGX:

```bash
cd ~/raid/Clara/fsdp-train
git pull --ff-only origin main
conda activate fsdp-med3d
bash scripts/launch_unet_cohort_nohup.sh
tail -f logs/unet_cohort/latest.log
```

This separate launcher uses `train_unet_cohort.py`:54 train patients/108 side ROIs,10 validation patients/20 side ROIs,20 test patients reserved. It verifies the completed tiny run, matching train-only threshold audit, checkpoint hashes, frozen v4 package versions/ROI policy, manifest, source CT/final-label fingerprints and original random initialization. Default inputs are `outputs/private/zoom_training/latest`, `outputs/private/unet_tiny/latest`, and `outputs/private/unet_audit/latest`; set `PRIVATE_V4_RUN_DIR`, `UNET_TINY_RUN_DIR`, and `UNET_AUDIT_DIR` to their original run directories if these links move. Old training/audit/visual code stays unchanged so previous fingerprinted workflows remain usable.

Threshold0.3 is frozen from the latest tiny checkpoint's train-only audit **before validation image preparation**. If the matching audit selects a different threshold, the launcher refuses this protocol. The cohort starts from fresh seeded random U-Net weights; tiny/SAM learned weights are never loaded. Settings: AdamW LR0.001/weight decay0.0001, unchanged squared DiceCE, XYZ flips, accumulation8 with a four-ROI tail (14 attempted updates/epoch),5-epoch warmup/cosine, maximum200 epochs, validation every5 epochs plus first/final, early-stop patience50 epochs/min_delta0.001. These are an initial experiment recipe, not an estimated optimum or an80% validation guarantee.

Checkpoint selection uses equal-patient native bilateral full-target validation Dice at fixed0.3. Native metrics at0.5 are reported from the same logits as a control and never select checkpoints or thresholds. Moving augmented model-grid training Dice remains explicitly labeled `moving_model_grid_train_dice_at_0_5`; compare final frozen train/validation scores at0.3 for the generalization gap. The v4 GT-side-centered ROI and128³ model input remain fixed; logits are restored before thresholding. This is assisted segmentation validation, not automatic full-scan localization or test performance.

Monitor or inspect completion:

```bash
python -m json.tool outputs/private/unet_cohort/latest/progress.json
python -m json.tool outputs/private/unet_cohort/latest/summary.json
```

`summary.json` exists only after selected-checkpoint verification and exports finish. It contains `best_validation_metrics`, `best_frozen_training_metrics`, train/validation gap, coverage, original failed tiny status, threshold provenance,0.5 controls and checkpoint SHA256. `best_exports` contains side/patient CSVs, frozen-training CSVs and worst/middle/best validation overlays/NIfTI crops. All outputs are ignored by git. Logs are isolated under `logs/unet_cohort`; reports link from `outputs/private/unet_cohort/latest`; checkpoints are `runs/<run>/unet_best.pth` and `unet_latest.pth`. Latest is saved every epoch.

`GPU_ID`, `NUM_EPOCHS`, `VAL_EVERY`, `LEARNING_RATE`, `ACCUMULATION_STEPS`, `PATIENCE_EPOCHS`, `MIN_DELTA`, `PRIVATE_SOURCE_ROOT`, and `PRIVATE_DATA_DIR` are supported. Use the same settings and resolved control/calibration directories when resuming an unfinished run:

```bash
RESUME=1 UNET_RUN_NAME=fsdp_private_unet_cohort_YOUR_ORIGINAL_RUN bash scripts/launch_unet_cohort_nohup.sh
```

Strict resume verifies code/config/source/calibration, restores model/optimizer/scheduler/scaler/RNG/history and refuses completed runs. An error writes isolated `failure.json` and a traceback in the log. This workspace does not have DGX access; actual cohort launch/results must be confirmed on the server.

## Physical preprocessing followed directly by U-Net retraining

The user authorized preprocessing and a new full-cohort run after the training audit. Use this separate launcher on DGX:

```bash
cd ~/raid/Clara/fsdp-train
git pull --ff-only origin main
conda activate fsdp-med3d
bash scripts/launch_unet_physical_nohup.sh
tail -f logs/unet_physical/latest.log
```

One nohup process performs the training-only physical planning, preprocessing checks, validation preparation and then training automatically. Initial log lines describe source checks and resampling; epoch lines appear after preflight. Original CTs/annotations, prior checkpoints and previous scripts are unchanged.54train/10validation patients remain fixed;20test images are never read. This is still a GT-assisted side ROI protocol, not automatic full-scan localization.

`physical_fsdp.py` uses original SimpleITK spatial metadata to resample CT/label together onto an axis-aligned physical LPS reference. It supports the LIP case and oblique source directions; no tensor-only orientation assumption is used. Each side's physical target bounding box defines its center. ROI field-of-view/normalization/grid selection use all54train patients only and are frozen before any validation image preparation. The starting field is at least22.4×39.2×50.4mm, expanded if actual training world-space bounds require it, with at least5mm margin. Dimensions round up to multiples of16 for the residual U-Net. Initial spacing0.35mm typically gives64×112×144 input. No anisotropic cuboid-to128cube stretch is applied.

Before training, every training side must retain its target, have at least5mm physical margin and reach minimum label-resampling roundtrip Dice0.95. If0.35mm fails, the prespecified training-only candidates0.30/0.25mm are tried with suitably larger tensor dimensions; if none pass, processing stops with `failure.json` and `preprocessing/policy.json`. No case is removed or annotation rewritten. This diagnostic is not a model Dice ceiling. Upsampling cannot recover unacquired thin-slice detail. Validation never refits the grid/normalization: all10patients are included, native full targets remain in the denominator, and truncation/roundtrip statistics are reported without excluding difficult cases.

Normalization clips raw CT values to training-FSDP foreground percentiles0.5/99.5 then uses shared training mean/std. It does not assume positive CT values are the foreground or recalibrate scanner HU metadata. Interpolation is linear for CT/logits, nearest neighbor for masks. Outside-CT padding is explicitly marked and excluded from DiceCE loss. Resampled logits are mapped back through physical coordinates to the original CT grid before thresholding; side predictions are unioned without duplicate voxels for patient-macro bilateral scoring.

Default augmentation is mild and paired: affine probability0.5 with rotations±5°, scaling±5%, translation±1mm; LR flip probability0.5 (no AP/IS flips); normalized intensity gain0.9–1.1/shift±0.1 at probability0.3; Gaussian noise sigma0.02 at probability0.15. A spatial transform producing empty or border-touching targets falls back to its untransformed pair. CT/target/validity share the transform, and padding stays excluded. Set `AUGMENTATION=none` for a future separately named control; the first requested pipeline uses `light`.

Training starts from fresh seeded U-Net weights matching the previous architecture/initial-state hash. No tiny/SAM learned weights are loaded. Defaults: AdamW LR0.001/decay0.0001, squared DiceCE in FP32 over valid CT voxels, accumulation8 including the four-ROI tail (14attempted updates/epoch), gradient clipping1, five-epoch warmup/cosine, cap200epochs, evaluation every5 plus first/final, early-stop patience50epochs/min_delta0.001. Preprocessing, padding loss, augmentation and threshold protocol change together: compare pipelines without claiming a one-variable causal ablation or guaranteed80% validation.

Thresholds are reassessed on the changed inputs. At each evaluation, the model first runs frozen/no-augmentation inference on **all54training patients** and selects from prespecified native-logit probability thresholds0.05–0.95 by mean training bilateral Dice (ties prefer0.5 proximity, then lower threshold). That threshold is fixed for the subsequent validation call and saved with the checkpoint. Validation never chooses thresholds. The old0.3 and conventional0.5 remain secondary same-logit validation controls; they do not independently select checkpoints. Native validation Dice at the train-chosen threshold selects the best checkpoint and early stopping. Moving augmented model-grid training Dice at0.5 is labeled separately.

Monitor and inspect completion:

```bash
python -m json.tool outputs/private/unet_physical/latest/preprocessing/summary.json
python -m json.tool outputs/private/unet_physical/latest/progress.json
python -m json.tool outputs/private/unet_physical/latest/summary.json
```

The preprocessing summary exists after train/validation preparation. The final summary exists only after best-checkpoint/threshold reproduction and exports. It includes selected grid/normalization, coverage/roundtrip/padding, threshold source, frozen train/validation metrics, gap, prior-cohort comparison and hashes. `best_exports` contains full side/patient CSVs for selected validation and frozen training, the train-only threshold curve, and score-verified native PNG/NIfTI overlays for worst/middle/best validation patients. Original geometry, native slice indices and actual selected threshold are retained in exports.

New output paths: `logs/unet_physical/latest.log`, `outputs/private/unet_physical/latest`, and `runs/fsdp_private_unet_physical_<timestamp>/`. `physical_data.pt` caches private preprocessed crops/metadata and is ignored by git. Strict resume verifies cache/source image/label/ColorTable hashes, code/package/Monai implementation, control/config/settings, and restores model/optimizer/scheduler/scaler/RNG/history. Keep the same resolved control and original settings:

```bash
RESUME=1 UNET_RUN_NAME=fsdp_private_unet_physical_YOUR_ORIGINAL_RUN bash scripts/launch_unet_physical_nohup.sh
```

Overrides: `UNET_COHORT_RUN_DIR` (default prior `outputs/private/unet_cohort/latest`), `GPU_ID`, `NUM_EPOCHS`, `VAL_EVERY`, `LEARNING_RATE`, `ACCUMULATION_STEPS`, `PATIENCE_EPOCHS`, `MIN_DELTA`, `AUGMENTATION`, `PREPROCESS_THREADS` (default4), `OMP_NUM_THREADS` (default4). Seed must match the frozen prior cohort. Completed runs cannot be resumed or overwritten. This workspace has no DGX connection; launch the command above on the server to obtain actual results.

## Profile the training dataset before changing preprocessing/augmentation

The actual cohort completed115/200 epochs with early stopping, selecting epoch65. At the frozen threshold0.3 its training/validation mean bilateral Dice is32.50%/18.64%; the0.5 control is empty on both splits. Tiny87.71% remains historical held-in calibration. Full target retention does not establish clinical annotation correctness, physical scale consistency or normalization suitability.

Run the read-only training dataset profiler on DGX:

```bash
cd ~/raid/Clara/fsdp-train
git pull --ff-only origin main
conda activate fsdp-med3d
bash scripts/launch_dataset_profile_nohup.sh
tail -f logs/dataset_profile/latest.log
```

When complete:

```bash
python -m json.tool outputs/private/dataset_profile/latest/overview.json
```

`UNET_COHORT_RUN_DIR` overrides the completed cohort control, default `outputs/private/unet_cohort/latest`. The profiler uses CPU and reads only54training image/final-label pairs; it performs no training and leaves source/cohort artifacts unchanged. Code, package versions, original source hashes, v4 ROI and manifest must match. Validation/test image files are not opened. `summary.json` contains all distributions; `overview.json` is the compact shareable report. `patients.csv` and `sides.csv` use train aliases; intermediate prepared crops/provenance remain in the ignored server output directory.

Measurements cover source orientation counts, spacing/anisotropy, ROI extents in millimetres, intensity distributions and clipping, target volumes/bounding boxes/26-connected components, native/model-grid foreground fractions, padding and label transform roundtrip. Identical source pixel arrays are flagged within training only, not automatically removed. No across-split image duplicate check is claimed. Numerical CT ranges alone do not verify HU calibration, and disconnected annotations may be legitimate.

Current implementation facts to evaluate: physical affine is discarded before tensor-only ToCanonical; native voxel-size crops are resized without common physical spacing; per-crop normalization estimates mean/std from values>0 before tighter zoom; augmentation currently flips three axes independently. None alone proves the cause of low Dice. Candidate changes are common physical orientation/spacing preserving thin targets, train-derived CT normalization, foreground/hard-background patch sampling and mild anatomy-aware spatial/intensity augmentation. [nnU-Net's CT normalization](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/explanation_normalization.md) derives shared clipping/normalization statistics from training foreground. [MONAI transforms](https://monai.readthedocs.io/en/stable/transforms.html) support paired image/label spatial transforms and positive/negative crop sampling. Adopt parameter values only after training-only profiling; keep validation deterministic, test reserved, originals unchanged, and compare preprocessing first with a later augmentation ablation. This profiler does not implement or start that new training recipe.

## Export segmentation PNGs on DGX and download with scp

Generate all visualizations on the server from the completed tiny training run and matching threshold audit:

```bash
cd ~/raid/Clara/fsdp-train
git pull origin main
conda activate fsdp-med3d
bash scripts/launch_unet_visuals_nohup.sh
tail -f logs/unet_visuals/latest.log
```

The exporter verifies original source/checkpoint fingerprints and reproduces side/patient scores before declaring success. It does not train or open validation/test images. PNGs include metric bars, Dice matrices for all four sides/two patients, threshold and training curves, native ROI-domain confusion matrices, GT/prediction/error panels for both checkpoints at0.5 and the train-selected candidate (currently0.3), and all native Z slices containing GT or candidate predictions for the latest checkpoint. Physical voxel spacing and slice indices are preserved in the views. GT is cyan; error panels show TP green, FP magenta, FN yellow. These remain two-patient training results, suitable for visual review, not held-out validation claims.

After the log reports `complete: true`:

```bash
ls outputs/private/unet_visuals/latest/figures
ls -lh outputs/private/unet_visuals/latest.tar.gz
```

The tar.gz contains only generated PNGs, CSV metrics, a provenance manifest and a legend/README. It excludes preparation files and original NIfTI volumes. From the laptop, use the DGX IP or SSH hostname that normally works for your account (replace `DGXH100` if necessary):

```bash
scp 'D13K48009@DGXH100:~/raid/Clara/fsdp-train/outputs/private/unet_visuals/latest.tar.gz' .
```

Unpack the downloaded archive locally with `tar -xzf latest.tar.gz`. The figures are already rendered by DGX; local rendering is unnecessary. Override `UNET_TINY_RUN_DIR`, `UNET_AUDIT_DIR`, or `GPU_ID` to select explicit sources/GPU. Failures write an isolated `failure.json`; old data, checkpoints and reports are preserved.

## DGX workflow

1. Clone the repository.
2. Create and activate the Conda environment, then install `requirements.txt`.
3. Confirm DGX GPU visibility from PyTorch.
4. Download/extract public data into the ignored `data/` path.
5. Run the NIfTI audit and inspect representative scans/masks.
6. Package accepted pseudo-labels in SAM-Med3D format and launch the public pseudo-label baseline.

Do not place private scans, private subject IDs, checkpoints, or generated masks in Git.

## SAM Turbo + CNN 32³: unattended DGX workflow (2026-11-23)

This experiment adds a learned CNN detail branch and residual refinement to the original SAM-Med3D Turbo. The Transformer remains at 8³ tokens with a 128³ context image; the CNN fusion grid is 32³ with half/full detail-resolution skips. It does not change Turbo patch size or claim that 32³ global Transformer tokens fit the old checkpoint.

The original Turbo checkpoint SHA256 is checked, its entire state loaded strictly, and its state verified unchanged when wrapped. New CNN/fusion layers are randomly initialized, with a zero residual output so initial refined logits reproduce the mapped SAM component. Epochs 1–5 train the new modules only; subsequent epochs fine-tune image blocks 10/11 and the mask decoder at LR 1e-5 while new modules use LR 1e-4. Earlier image blocks, neck and prompt encoder stay frozen. This architecture experiment has not yet produced DGX validation results.

Run this single command in an initialized DGX terminal:

```bash
cd ~/raid/Clara/fsdp-train && git pull --ff-only origin main && conda activate fsdp-med3d && bash scripts/launch_sam_refine32_nohup.sh
```

It uses the existing SAM source/checkpoint and completed `outputs/private/unet_cohort/latest` as the frozen data/provenance control. It preprocesses original CT/labels into train-selected physical LPS detail grids, creates a centered isotropic 128³ SAM context, checks target retention and roundtrip quality, evaluates initialization, trains up to 200 epochs, selects a checkpoint on validation, verifies selected scores on reload, and exports CSV metrics, a training/patient Dice figure, all 20 validation-side three-plane PNG overlays and native NIfTI patches. Results are bundled into `results.tar.gz` automatically. A preprocessing or training error stops the workflow and records `failure.json`; it does not relax checks or remove patients automatically.

The frozen split remains 54 train, 10 validation, 20 reserved test patients. Context/detail views share physical centers and normalization derived only from train. LR reflections and intensity changes are paired. Loss excludes CT padding; refined and auxiliary SAM losses use FP32 squared DiceCE. There are five oracle error clicks per side, ten total bilaterally; one-click metrics are also reported. Clicks use probability 0.5; the final output threshold is calibrated on all 54 training patients before validation. Early stopping uses a 50-epoch patience with 0.001 minimum validation gain. The same-refined-click coarse SAM component is a component diagnostic, not an independent vanilla-SAM ablation. ROI centers are GT-derived: scores describe interactive GT-assisted segmentation, not automatic full-scan localization.

Monitoring and next-day inspection:

```bash
tail -n 80 -f logs/sam_refine32/latest.log
python -m json.tool outputs/private/sam_refine32/latest/progress.json
python -m json.tool outputs/private/sam_refine32/latest/summary.json
```

`progress.json` starts after the first training epoch; during CPU preprocessing and initial evaluation, use the log. `summary.json` with `complete: true` appears only after selected-checkpoint verification and export/archive completion. Ctrl+C while running `tail` only exits the monitor. No existing training job is stopped by this command; choose a free GPU with `GPU_ID=1 bash scripts/launch_sam_refine32_nohup.sh` if needed. GPU memory/time have not been measured on DGX for this architecture.

An interrupted training run can resume from its original settings and latest epoch checkpoint:

```bash
RESUME=1 SAM_REFINE_RUN_NAME=fsdp_private_sam_refine32_ORIGINAL_NAME bash scripts/launch_sam_refine32_nohup.sh
```

From a local terminal, download the completed bundle using the DGX hostname/IP normally used for SSH:

```bash
scp 'D13K48009@DGXH100:~/raid/Clara/fsdp-train/outputs/private/sam_refine32/latest/results.tar.gz' ./sam_refine32_results.tar.gz
```

The archive contains private validation masks and images; keep it with the private research data. It excludes model weights, full preprocessing caches and raw source volumes. No test CT is read by this workflow. The earlier ~87.7% U-Net tiny result remains training-only; this experiment does not imply 80% validation Dice.

If hybrid preprocessing fails all grids and the remaining 0.25 mm failures are train_26, run this train-only reconstruction diagnostic before restarting training:

```bash
python -u scripts/audit_physical_roundtrip.py --run-dir outputs/private/sam_refine32/latest --train-alias train_26
```

The command reports source geometry, existing-grid reconstruction and a fixed subvoxel phase diagnostic. Full results are saved under `outputs/private/roundtrip_audit/<timestamp>/summary.json`. It does not launch training or change the quality gate.

The SAM preprocessing workflow now checks two shared grid-origin phases (original and +half voxel XYZ) at each spacing, selected on all training sides before validation. All quality checks remain required. After a preprocessing-only failure, pull the correction and start a new run:

```bash
cd ~/raid/Clara/fsdp-train && git pull --ff-only origin main && conda activate fsdp-med3d && RESUME=0 bash scripts/launch_sam_refine32_nohup.sh
```

The launcher creates a new timestamped run and log; the old failed run remains available. Successful preprocessing automatically continues to Turbo+CNN training, evaluation and export. Monitor `logs/sam_refine32/latest.log`. Do not resume a checkpoint created by an earlier source revision.

## Train all annotated structures with learned detail64³ and32³

This new workflow predicts17anatomical classes plus background using the pretrained Turbo **image encoder** and a new multiclass CNN decoder. It runs64³learned detail first, then32³with matched initialization/recipe. Global Turbo tokens remain8³. Inputs are local128³patches with high-resolution skips and full-resolution output. The broader anatomical field is GT-union-centered; this is a research segmentation protocol, not automatic full-CT localization.

Run on DGX in one command:

```bash
cd ~/raid/Clara/fsdp-train && git pull --ff-only origin main && conda activate fsdp-med3d && bash scripts/launch_multiclass_nohup.sh
```

The background process audits all labels/ColorTables, selects preprocessing on train only, verifies feature/gradient flow, trains each model up to200epochs with early stopping, evaluates selected checkpoints and exports a private result bundle. It uses existing Turbo source/checkpoint and `outputs/private/unet_cohort/latest` as the completed frozen data control; no dependency upgrade or existing training run is needed. All17classes require training examples and consistent mapping; failed checks stop with diagnostics. Clinical annotation completeness still needs expert confirmation. The held-out20test images remain unread.

Monitor and inspect:

```bash
tail -n 80 -f logs/multiclass/latest.log
python -m json.tool outputs/private/multiclass/latest/progress.json
python -m json.tool outputs/private/multiclass/latest/detail64/summary.json
python -m json.tool outputs/private/multiclass/latest/summary.json
```

`detail64/summary.json` means the64³experiment is complete; the parent `summary.json` appears only after all requested variants and archive export finish. Each reports macro anatomy Dice **and FSDP separately**. Scores differ in protocol from the earlier binary interactive model. Runtime depends on anatomical field size/GPU and is not guaranteed to finish overnight.

Defaults:64then32,216train patches/epoch,5decoder warmup epochs, original Turbo blocks10/11fine-tuned after warmup, validation every10epochs,50epoch patience. Optional controls:

```bash
GPU_ID=1 DETAIL_GRIDS="64" bash scripts/launch_multiclass_nohup.sh
RESUME=1 MULTICLASS_RUN_NAME=fsdp_multiclass_ORIGINAL_NAME bash scripts/launch_multiclass_nohup.sh
```

Resume requires completed preprocessing and exactly the original settings/source revision. Keep originals and previous runs intact. Once complete, download from your local terminal using the usual DGX hostname/IP:

```bash
scp 'D13K48009@DGXH100:~/raid/Clara/fsdp-train/outputs/private/multiclass/latest/results.tar.gz' ./multiclass_results.tar.gz
```

The bundle includes validation masks/overlays, per-class and per-patient CSVs, curves and confusion matrices. Preserve it as private research data.

## CNN64 with Turbo encoder16³

The encoder16 experiment keeps the same GT-all-class physical field and local128³CNN patches with64³learned detail. Its context is sampled directly from the physical field at256³, producing16³real patch tokens with the original16³kernel/stride. There is no intermediate128³context upsampling. Sampling at256³cannot recover details already absent from the physical source grid.

The original Turbo checkpoint is checksum-verified and loaded strictly before adaptation. Absolute positions are interpolated8³to16³. Patch, attention and relative-position table weights are retained; the official encoder resizes relative positions during attention. Original window/global attention layers remain. The new semantic CNN decoder is randomly initialized. Decoder warmup and later block10/11fine-tuning follow the same recipe as the8³baseline. Nonreentrant block checkpointing reduces saved activations; it does not remove the higher peak cost of attention itself. GPU preflight checks actual feature sizes, gradient flow and peak allocated memory before training. Full Turbo16³has not been profiled locally on a DGX GPU.

Launch a new run on DGX:

```bash
cd ~/raid/Clara/fsdp-train && git pull --ff-only origin main && conda activate fsdp-med3d && bash scripts/launch_multiclass_tokens16_nohup.sh
```

This runs only CNN64/encoder16, with a200epoch cap, early stopping, evaluation, native-grid masks/overlays and archive export. It uses separate log/report links:

```bash
tail -n 80 -f logs/multiclass_tokens16/latest.log
python -m json.tool outputs/private/multiclass_tokens16/latest/detail64/preflight.json
python -m json.tool outputs/private/multiclass_tokens16/latest/progress.json
python -m json.tool outputs/private/multiclass_tokens16/latest/summary.json
```

Use `GPU_ID=1 bash scripts/launch_multiclass_tokens16_nohup.sh` to select another GPU if the baseline is still using GPU0. Separate channels do not prevent cross-workflow GPU contention. Existing jobs are not stopped by the launcher. This source revision changes resume hashes: start a new experiment; do not resume an older revision. Resume this revision using the original name and settings:

```bash
RESUME=1 MULTICLASS_RUN_NAME=fsdp_multiclass_tokens16_ORIGINAL_NAME bash scripts/launch_multiclass_tokens16_nohup.sh
```

Compare the final16³run with CNN64/encoder8³using identical split, seed, physical preprocessing and training settings. Record all adaptation/context changes with the scores; macro17Dice and FSDP1/9Dice remain separate. The20test images remain unread. GPU memory suitability, runtime and validation improvement require DGX measurements.

## Continue the successful FSDP45.63% foundation

The current priority is FSDP segmentation using the successful binary interactive hybrid as the foundation. `recover_fsdp_foundation_nohup.sh` first verifies the original completed run, checkpoint/cache hashes, model/data implementations, package versions and frozen54/10/20split. It then stops verified active multiclass jobs from this project's tracked PID files, checks that they exited, and launches a separate FSDP continuation with nohup. Saved multiclass checkpoints/results and the successful foundation are retained. Run this on the Linux DGX:

```bash
cd ~/raid/Clara/fsdp-train && git pull --ff-only origin main && conda activate fsdp-med3d && bash scripts/recover_fsdp_foundation_nohup.sh
```

The foundation is pinned to `runs/fsdp_private_sam_refine32_20261123T144159Z_3745292`, selected checkpoint SHA256 `81b3e436c25062d25544e6a0b89d1c7887e7935cbf087ce3676fea08f1144a39`. Its full trained hybrid is loaded strictly: encoder, prompt encoder, SAM mask decoder and CNN refiner. Reuse the exact `hybrid_data.pt` physical ROI/context/normalization cache; do not regenerate preprocessing. CNN detail remains32³and Turbo tokens8³. Sampling remains108side ROIs/epoch with5oracle interactions/side, the same loss/auxiliary supervision, paired light augmentation, seed, accumulation, train-only threshold calibration and FSDP-based checkpoint selection.

Before any optimizer step, recompute train and validation@1/@5and the fixed0.5validation control. They must reproduce the foundation within0.0001absolute Dice, including the train-selected threshold0.35, or the job stops. Successful preflight establishes reproduction of the baseline, not improvement. Epoch0in the new run is the unchanged foundation, included among best-checkpoint candidates.

Fine-tuning uses a fresh optimizer with both source learning rates multiplied by0.2(defaultSAM2e-6andCNN2e-5), up to100additional epochs and50epoch patience. Continue the source-best active groups (image blocks10/11,mask decoder,CNN) with no new head-only phase. A new five-epoch learning-rate warmup/cosine schedule begins; sampling seeds continue after the source best epoch30. This is a controlled continuation, not an exact optimizer/scheduler resume. Checkpoint/history epoch fields count additional epochs; `source_plus_additional_epoch` records the sampling epoch. Original test images remain unread.

```bash
tail -n 80 -f logs/fsdp_foundation/latest.log
python -m json.tool outputs/private/fsdp_foundation/latest/preflight.json
python -m json.tool outputs/private/fsdp_foundation/latest/progress.json
python -m json.tool outputs/private/fsdp_foundation/latest/summary.json
```

Final `validation_change_vs_foundation` reports gain or numerical reproduction tolerance; `best_additional_epoch:0` means no better checkpoint was found and the unchanged foundation was retained. The new run exports native validation overlays/masks, train/validation CSVs, curves and `results.tar.gz`. All scores retain the GT-side-ROI/oracle-click protocol and are Dice, not voxel accuracy or automatic full-scan performance. Higher Dice is not guaranteed.

To stop multiclass jobs only, without launching anything:

```bash
python scripts/stop_multiclass_training.py --project-dir "$PWD"
```

Stopping verifies process owner, script, run directory/config, PID start time and command. It uses SIGTERM, waits up to30seconds and refuses to launch if a tracked job remains active. It does not discover manually launched jobs without these PID files. Existing checkpoint files/logs remain; work since the most recent completed checkpoint may be lost. On another GPU or a relocated verified foundation, use `GPU_ID` or `FSDP_FOUNDATION_DIR`. Resume a new unfinished continuation with the original settings/name:

```bash
RESUME=1 FSDP_FOUNDATION_RUN_NAME=fsdp_private_foundation_ORIGINAL_NAME bash scripts/launch_fsdp_foundation_nohup.sh
```

## FSDP resolution audit: retain the trained SAM decoder and CNN

`launch_fsdp_resolution_nohup.sh` tests four binary FSDP variants initialized from the complete45.63%hybrid checkpoint. The default performs readiness audits only, with zero optimizer updates:

| Variant | Encoder tokens | CNN fusion | SAM coarse mask |
| --- | --- | --- | --- |
| `baseline` |8³|32³|32³|
| `detail` |8³|Native half ROI (48×80×104for the successful96×160×208ROI)|32³|
| `tokens` |16³|32³|64³|
| `combined` |16³|Native half ROI|64³|

Each child first loads the original complete trained hybrid and reproduces its full54train/10validation@1/@5scores, selected0.35threshold and fixed0.5control. It then constructs the variant, strictly reloads all trained parameter names, and verifies every tensor is identical except the declared absolute-position8³→16³interpolation. No decoder is replaced with a random semantic head. Native-half fusion uses existing learned features directly; the full/half-resolution skips and trained residual output remain. Point maps retain1mmphysical Gaussian scale on rectangular grids.

Token16variants resample the original CT directly at256³over the same side-context center and physical field. Detail CT, masks, validity, spacing, source labels, normalization and ROI cache remain unchanged. The prompt coordinates normalize to256, dense positional encoding and mask embeddings align at16³, and actual mask feedback/output becomes64³. Original patch16³kernel, attention/relative-position parameters, SAM prompt/mask decoder and all CNN weights are inherited. Encoder block checkpointing reduces saved activations, while larger attention/fusion still require actual GPU profiling. Denser sampling cannot restore detail absent from the source CT.

After adaptation, each audit measures train/validation@1/@5under the same oracle policy, train-calibrated and fixed0.5validation scores, feature dimensions, physical detail-cell spacing, finite gradients through selected encoder blocks/mask decoder/CNN, GPU peak allocated memory, and an unchanged model hash before/after the gradient probe. It exports all20validation-side overlays/native masks and CSVs, and verifies exported scores. This uses validation images for diagnostics, never the20test images. It is not a train-only or automatic segmentation evaluation. Candidate audit scores are before fine-tuning, and changes are not reported as established training improvements.

Audit on a free DGX GPU:

```bash
cd ~/raid/Clara/fsdp-train && git pull --ff-only origin main && conda activate fsdp-med3d && bash scripts/launch_fsdp_resolution_nohup.sh
```

Use `GPU_ID=1`before`bash`to select another GPU. This launcher does not stop other jobs. Monitor:

```bash
tail -n 80 -f logs/fsdp_resolution/latest.log
python -m json.tool outputs/private/fsdp_resolution/latest/progress.json
python -m json.tool outputs/private/fsdp_resolution/latest/audit/detail/resolution_probe.json
python -m json.tool outputs/private/fsdp_resolution/latest/summary.json
```

SIGTERM or SIGINT sent to the workflow PID is forwarded to its active child; the workflow waits up to30seconds and starts no further stage after interruption.

Child folders also contain `foundation_reproduction.json`, `weight_preservation.json`, `config.json`, initial metrics and `overlays/`. Baseline reproduction happens before adaptation; a changed-resolution model is not required or assumed to immediately retain45.63%Dice. A failed reproduction, invalid gradient, OOM or other audit failure is recorded and prevents the subsequent training stage. The workflow collects the other audit variants so failures can be compared. It never lowers resolution, drops cases or selects a variant from validation automatically.

To perform all audits and then the matched fine-tuning comparison in one unattended workflow:

```bash
RUN_TRAINING=1 bash scripts/launch_fsdp_resolution_nohup.sh
```

Only after all four audits pass does this start four independent continuations, including an unchanged-resolution lower-LR control. Each inherits the same original45.63%checkpoint, uses the same source recipe, rates×0.2,100additional epoch cap,50epoch patience and FSDP-based selection. Each training child reruns its own preflight. Both delta versus the original foundation and delta versus the fine-tuned baseline control are reported. Adapted epoch0is a candidate for its own variant; the original foundation remains in its original directory and a worse variant is not promoted over it. Full workflow runtime and full-size DGX memory are not established locally.

The parent `results.tar.gz`contains audit metadata/overlays and, when requested, per-variant training result archives; weights/caches are excluded. Download from the local terminal:

```bash
scp 'D13K48009@DGXH100:~/raid/Clara/fsdp-train/outputs/private/fsdp_resolution/latest/results.tar.gz' ./fsdp_resolution_results.tar.gz
```

Start a new parent run for this revision. Individual unfinished training children can resume through `train_fsdp_foundation.py --resume`with the original child path, foundation, resolution variant and exact settings/source. The parent workflow itself has no resume mode. An unfinished continuation from an earlier source revision fails its existing resume guard; completed successful foundation artifacts remain reusable.

### Missing original source file

The DGX audit stopped on a missing original final label (`labelsTr/<patient_id>-label.nii.gz`) during source verification, before model training. This is an input-path availability problem; it does not establish failure of the new architecture. Source paths can become stale after a move/rename, or files can be absent. Do not replace final labels with expert or newly generated masks to get past verification.

Foundation and resolution workflows now check train/validation source paths before loading the large cache or launching all four child audits. The parent writes `source_precheck.json`. Presence checks are explicitly distinguished from SHA256 verification. After cache loading, verification uses the provenance inside the checksum-verified cache as authority and writes `source_verification.json`; all original content checks remain required.

Diagnose files on DGX without images/GPU or data changes:

```bash
cd ~/raid/Clara/fsdp-train && git pull --ff-only origin main
python scripts/audit_foundation_sources.py \
  --foundation-run-dir runs/fsdp_private_sam_refine32_20261123T144159Z_3745292 \
  --source-root "/raid/normal case_86" --verify \
  --output outputs/private/source_audit/latest.json
```

The report lists unavailable/changed inputs, expected checksums, directory existence and same-patient candidate files in the corresponding `imagesTr`, `labelsTr`, or `ColorTable`folder. `--source-root` explicitly binds original filenames to the supplied dataset root. All expected SHA256 values remain those of the original foundation. `--search-root` only searches for alternatives and does not bind them. Expert folders and test patient IDs are not searched. Candidate paths are diagnostic only, even when `sha256_matches:true`; no files are copied, linked, renamed, replaced or ignored automatically. Missing or changed inputs keep readiness false and produce exit2. A checksum mismatch alone does not establish differing clinical annotations, but does not meet the foundation's byte identity requirement.

For the original dataset at `/raid/normal case_86`, launch a new timestamped audit with an explicit root:

```bash
FSDP_SOURCE_ROOT="/raid/normal case_86" bash scripts/launch_fsdp_resolution_nohup.sh
tail -n 80 -f logs/fsdp_resolution/latest.log
```

The same environment variable is supported by the foundation continuation and recovery launchers. It is forwarded to every child, used for source verification and direct original-CT context sampling, and recorded in new run metadata. Role folders may be symlinks. Original cache tensors, source checksums, manifests, checkpoints and historical metadata are not rewritten. Preparation checks actual bytes against provenance inside the checksum-verified cache; an empty or different root stops before training. Keep the failed run for diagnostics. This command audits only by default; training still requires `RUN_TRAINING=1` and every audit passing.

### Continue from verified cached FSDP targets when original annotations are unavailable

DGX subsequently confirmed that `~/raid/normal case_86` resolves to `/raid/<account>/normal case_86`, with84CTfiles but empty `labelsTr` and `ColorTable`folders. The original foundation cache checksum still matches its recorded config, with108train and20validation side targets intact. This establishes cache availability, not restoration of the original annotation files.

Explicit `FSDP_VERIFIED_CACHE=1` uses that checksum-verified cache as the binary FSDP target authority. Default mode still requires the original annotations. Recovery verifies the pinned full hybrid checkpoint, cache checksum, frozen54/10/20manifest, versions, policy and original model/data implementations. It then checks exact patient/side membership, cached model/context target support, full native foreground and equality of the two native side masks with every cached bilateral patient target. All64original CTs must still match the source hashes inside the verified cache. No expert annotations or pseudo-labels replace the targets, and no dataset/cache/checkpoint files are rewritten.

For16³token variants, cached native side masks are placed at their original voxel offsets in a transient original-CT grid, then resampled to the same physical256³context. CT is sampled directly from its verified original file. This avoids upsampling the old128³context or reconstructing a native target from a resampled mask. Cached detail tensors, native scoring targets, normalization, ROI, split and five-click protocol remain unchanged. Every child must still reproduce the45.63%foundation before adaptation or optimizer updates.

Run audit followed automatically by all four controlled training variants:

```bash
cd ~/raid/Clara/fsdp-train && git pull --ff-only origin main && conda activate fsdp-med3d && \
FSDP_SOURCE_ROOT="$HOME/raid/normal case_86" FSDP_VERIFIED_CACHE=1 RUN_TRAINING=1 \
bash scripts/launch_fsdp_resolution_nohup.sh
```

Monitor `logs/fsdp_resolution/latest.log`. Omit `RUN_TRAINING=1` for audit only. The same cache option is supported by foundation/recovery launchers and by `--verified-cache` on their Python entry points. Resume requires the original cache mode/root/config/source revision. `cache_target_verification.json`, `source_verification.json`, configs and summaries record the target authority and explicitly state that original annotation files were not verified. A missing/changed CT, bad cache, target mismatch, failed foundation reproduction or failed audit prevents subsequent training. No test CTs/targets are loaded; this mode does not recover the original17-class annotation files, ColorTables or the20reserved test annotations.

### Research plan for the next FSDP experiments

The 24 November research note is in `docs/FSDP_RESEARCH_2026-11-24.md`. It relates primary papers and official implementations to the current hybrid, distinguishes measured results from hypotheses, and defines controlled prompt-history, boundary-loss and residual-refiner experiments. The independent implementation is documented below; the active resolution workflow remains unchanged.

### Run a second research pipeline on another GPU

`launch_fsdp_research_nohup.sh` is an independent pipeline. The implementation adds new Python files; existing resolution/foundation/model/data scripts are unchanged, preserving the active resolution parent's code checks. It does not stop, resume or write into the resolution run. Logs, PID files and latest links use `fsdp_research`.

Default physical GPU is1. In default exclusive-start mode, the launcher queries `nvidia-smi`, refuses a GPU with an existing compute process, records its index/UUID, and sets `CUDA_VISIBLE_DEVICES` to that UUID. The child uses logical `cuda:0` inside that single-device mapping. Select another idle physical index with `GPU_ID`; the check is a launch-time snapshot, not a system-wide reservation. No jobs are killed or GPUs reset. Explicit shared-capacity mode is described below.

On DGX, while the resolution pipeline continues on its other GPU:

```bash
cd ~/raid/Clara/fsdp-train && git pull --ff-only origin main && conda activate fsdp-med3d && \
GPU_ID=1 FSDP_SOURCE_ROOT="$HOME/raid/normal case_86" FSDP_VERIFIED_CACHE=1 \
bash scripts/launch_fsdp_research_nohup.sh
```

The second GPU runs four arms sequentially, each independently initialized from the original full45.63%hybrid:

- `control`: original8³tokens, CNN32³fusion, last sparse point and original objective.
- `history`: all1…kpoints passed to SAM; previous mask, CNN click maps, sampler, objective and budget retained.
- `boundary`: original prompts and DiceCE/context loss plus0.05times normalized signed-distance boundary loss;5mmclipping, original physical spacing, padding excluded as a background reference. Distance fields use training targets only and receive exactly the paired LR reflection.
- `refiner`: original prompts/objective and existing trained paths plus a native-half residual block. Inputs are16half features,8pooled full features,1mask prior and2physical click maps. A27-channel depthwise5³convolution and pointwise27→16mixing feed a zero-initialized16→1output, preserving predictions before learning.

Each arm first verifies checkpoint/cache/source/package/manifest integrity and reproduces the original54train/10validation foundation scores. Existing learned tensors must stay identical during construction. A gradient-only probe verifies finite nonzero gradients through encoder10/11, mask decoder and existing residual head; the refiner arm additionally requires gradients on its new output projection. Inner new-branch gradients may initially be zero because that output projection starts at zero. Control/boundary/refiner must also reproduce foundation metrics after construction; history initial scores are measured separately because the prompt inputs change. Any failed arm stops later arms.

Default screening is20additional epochs per arm,14optimizer updates/epoch on108training ROIs, with the same source rates×0.2, five-epoch LR warmup, cosine schedule over that20epoch cap, original paired augmentation and auxiliary0.25loss. Source epoch30is an offset, not progress toward the20additional cap. This matched20epoch control is distinct from the active100epoch resolution control; their schedules differ. `ADDITIONAL_EPOCHS`, `LR_MULTIPLIER`, `PATIENCE_EPOCHS`, `BOUNDARY_WEIGHT`, `BOUNDARY_CLIP_MM`, `RESEARCH_CPU_THREADS` and `FSDP_FOUNDATION_DIR` are supported. These objective/screening choices are experimental, not guaranteed improvements.

Monitor without interrupting either pipeline:

```bash
tail -n 80 -f logs/fsdp_research/latest.log
python -m json.tool outputs/private/fsdp_research/latest/progress.json
```

Parent progress names the active arm; that arm's `progress.json` and `history.csv` contain current/best validation scores. Every child writes its own config, initialization/reproduction proof, gradient/VRAM preflight, checkpoint history, native validation NIfTI masks/GT, selected-threshold overlays and curves. The selected checkpoint is reloaded and exported Dice is verified on both side and bilateral native masks. Primary PNGs show the selected threshold even when a separate0.5control is also exported. Reports include MONAI symmetric HD95 in mm and subvoxel area-weighted surface Dice at0.5/1mm; these tolerances are diagnostic, not clinical acceptance thresholds. Empty predictions score0surface Dice and have undefined HD95, with explicit counts; they are not silently removed from the report.

`summary.json` with `complete:true` appears only after all four arms and archive creation. It compares against both the original foundation and the matched research control; history also reports change from its own initial prompt-policy score. No resolution or research model is promoted automatically. The original test split remains unread. Per-arm and parent archives exclude weights/caches; they contain private masks and images for SCP.

Resume an interrupted research pipeline with the original run name and identical budget/settings:

```bash
GPU_ID=1 FSDP_RESEARCH_RUN_NAME=fsdp_research_ORIGINAL_NAME RESUME=1 \
FSDP_SOURCE_ROOT="$HOME/raid/normal case_86" FSDP_VERIFIED_CACHE=1 \
bash scripts/launch_fsdp_research_nohup.sh
```

Completed arms are skipped only after best-checkpoint checksum verification. An unfinished arm requires `research_latest.pth`; resume restores model/optimizer/scheduler/scaler/RNG/history and refuses code/config/data changes. Failure before the first checkpoint requires inspection and a new timestamped workflow. Stop signals sent to this parent are forwarded only to its current research child.

Copy the completed bundle from a local terminal:

```bash
scp 'D13K48009@DGXH100:~/raid/Clara/fsdp-train/outputs/private/fsdp_research/latest/results.tar.gz' ./fsdp_research_results.tar.gz
```

### When every GPU has a compute process

An existing PID does not establish that VRAM is full. `ALLOW_GPU_SHARING=1` explicitly permits sharing a GPU if enough memory is free; the default remains strict. `GPU_ID=auto` scans candidates and prefers an eligible idle GPU, otherwise the eligible device with the most free VRAM. `EXCLUDE_GPU_IDS` defaults to0 and can contain comma-separated physical indices. The launcher also detects readable Linux processes from the current resolution run and excludes their GPUs even when sharing is enabled. Use explicit exclusions for any additional GPU that must remain dedicated.

Shared mode defaults to a40GiB PyTorch allocator limit plus8GiB headroom, requiring at least48GiBfree. The effective free threshold is the maximum of `MIN_FREE_GPU_GIB` and allocator budget + `GPU_RESERVE_GIB`; lowering the former does not silently remove headroom. These are launch budgets, not a measured guarantee that a model fits. CUDA library allocations outside the PyTorch allocator and other jobs' subsequent memory growth are not reserved by this check; shared GPU compute also competes for throughput.

On DGX, allow sharing explicitly and choose automatically while protecting the original GPU:

```bash
cd ~/raid/Clara/fsdp-train && git pull --ff-only origin main && conda activate fsdp-med3d && \
GPU_ID=auto ALLOW_GPU_SHARING=1 MIN_FREE_GPU_GIB=48 EXCLUDE_GPU_IDS=0 \
FSDP_RESEARCH_GPU_MEMORY_GIB=40 FSDP_SOURCE_ROOT="$HOME/raid/normal case_86" FSDP_VERIFIED_CACHE=1 \
bash scripts/launch_fsdp_research_nohup.sh
```

`FSDP_RESEARCH_GPU_MEMORY_GIB` is applied with `torch.cuda.set_per_process_memory_fraction` to the single visible device before model allocation; it does not alter existing jobs. The selected GPU, free memory, existing PIDs, capacity policy and allocator limit are recorded. A full-size gradient preflight can still fail if the model requires more than its budget. If no candidate qualifies, no research job starts and the error lists each candidate's free VRAM and rejection reason. No global GPU reservation is taken.

Inspect all capacity without launching a job:

```bash
python scripts/fsdp_research_gpu.py --list
```

### Research child stops without a traceback

The parent records each completed child process's return code and terminating signal in `<arm>/process_exit.json` and updates `active_child.json` with its PID, command and elapsed time. A negative subprocess return code denotes a signal; for example, `-9` means SIGKILL. SIGKILL alone does not prove an OOM kill: check kernel or cgroup evidence for that PID. An uncatchable signal cannot create the child's Python `failure.json`, so the parent reports the exit record even when that file is absent. Subsequent arms remain stopped on failure.

Runs launched before this exit-record change did not save the return code. For those runs, preserve `active_child.json` and inspect the server's kernel logs around the failure time. Do not infer model quality, label failure or a VRAM shortage from a missing traceback. A changed workflow source requires a new workflow directory under the existing strict resume checks.

If kernel logs are unavailable to the account, the research parent also samples its own child's readable `/proc` memory fields every 5 seconds into `active_child.resources.jsonl`. `active_child.resources.json` records parent snapshots before/after, the last child sample and cgroup memory-event deltas. The reader supports v1 memory and v2 controllers, mount subtrees and ancestor limits; missing/denied files are reported rather than treated as zero or unlimited memory. Counters cover a shared cgroup scope and do not prove which process was an OOM victim. Global MemAvailable does not establish the job's own limit. These diagnostics read files only; no sudo, limit changes or signals to other jobs are used. `PYTHONFAULTHANDLER=1` enables Python fault traces where supported; it cannot catch SIGKILL.

Read current account diagnostics without starting training:

```bash
python scripts/fsdp_research_resources.py --output outputs/private/fsdp_research/resources_now.json
```

This snapshot cannot recover the dead process's historical exit code. After pulling the diagnostic changes, use a new default research run name with `RESUME=0`; the frozen foundation/cache/ROI and scientific recipe remain unchanged. An existing resolution run continues independently.

### Switch the active resolution workflow to sequential research

To prioritize research immediately, the switch helper verifies the tracked resolution parent against its owner, PID start time, exact project script, run directory and workflow config. It records the physical GPU from that parent's CUDA mapping, sends SIGTERM only to the verified parent (which forwards stop signals to its child), and waits for both parent and scoped children to exit. It does not send SIGKILL or stop other jobs. If shutdown cannot be verified, research is not launched. Existing resolution checkpoints/logs remain; work since the last completed checkpoint is not saved by this stop helper. The interrupted resolution workflow does not automatically resume later.

Run on DGX:

```bash
cd ~/raid/Clara/fsdp-train && git pull --ff-only origin main && conda activate fsdp-med3d && \
FSDP_SOURCE_ROOT="$HOME/raid/normal case_86" FSDP_VERIFIED_CACHE=1 \
bash scripts/switch_fsdp_resolution_to_research.sh
```

The switch creates a fresh research workflow on the released GPU, using the same45.63%foundation and default20additional epochs per arm. It explicitly clears `EXCLUDE_GPU_IDS` for this launch so the released GPU0can be reused; unset exclusions still default to0for ordinary research launches. Existing memory headroom and40GiBallocator safeguards remain, and shared-capacity mode defaults to enabled for this switch without stopping any co-resident jobs. It can be set to strict mode with `ALLOW_GPU_SHARING=0`. No successful launch is guaranteed if that GPU lacks the required free capacity. A timestamped stop record under `outputs/private/stops` includes the target, children, GPU and retained checkpoint file paths/sizes. If the tracked resolution workflow already exited, the helper sends no signal and stops; use the ordinary research launcher after inspecting GPU capacity.

### Continue the selected trained history checkpoint

The completed GPU7 research screen reported validation Dice47.75%control,49.18%history,47.51%boundary and47.20%refiner, each with20additional epochs. History's selected checkpoint is epoch5. These are the user's DGX results, rounded from its summary, under GT-assisted ROI and5oracle clicks per side. The source is `runs/fsdp_research_gpu7_20261124T132503Z_3250370/history/research_best.pth`.

`launch_fsdp_history_nohup.sh` creates one history-only continuation from that completed best checkpoint. It inherits the full trained SAM/CNN state, fixed8³/32³architecture, original verified cache/ROI,54/10split, click budget, augmentation, seed and evaluation rules. It does not replay all four arms or resume the interrupted resolution workflow. Resolution checkpoints and the completed research artifacts remain intact.

Default budget is100additional epochs, a fresh optimizer/scheduler/scaler, LR multiplier0.1relative to the original foundation (half the screening LR multiplier0.2), five-epoch warmup/cosine schedule, and early stopping after25epochs without gain beyond the configured patience threshold. This is a new fine-tuning run, not an exact continuation of the old optimizer. The epoch0candidate is the inherited history model, so the new best-selection pool includes the original49.18%result. The budget is a cap and does not promise improved Dice or100completed epochs.

The source's completed status, variant, checkpoint/config/summary hashes, original foundation, cache/manifest/policy, model/data/metric implementation hashes, package versions and recipe must match. Before any new optimizer step, reproduce source train/val@1/@5, its train-calibrated threshold and fixed0.5validation control within1e-4absolute Dice; failure stops training. Model weights load strictly. A source reproduction proof, its provenance, cumulative source epoch and delta versus the source are exported. A single continued history arm has no matched control for its extended budget.

On DGX:

```bash
cd ~/raid/Clara/fsdp-train && git pull --ff-only origin main && conda activate fsdp-med3d && \
GPU_ID=7 ALLOW_GPU_SHARING=1 MIN_FREE_GPU_GIB=48 FSDP_RESEARCH_GPU_MEMORY_GIB=40 \
FSDP_SOURCE_ROOT="$HOME/raid/normal case_86" FSDP_VERIFIED_CACHE=1 \
bash scripts/launch_fsdp_history_nohup.sh
```

The wrapper's default source is the immutable run path above; override `FSDP_RESEARCH_SOURCE_DIR` for another compatible completed history arm. The source path is resolved before updating the research latest link. Log/report locations remain `logs/fsdp_research/latest.log` and `outputs/private/fsdp_research/latest`. The new log counter starts at1; cumulative epoch reporting begins at36for the original source30+history-best5. Look for `Loaded trained history best checkpoint` and `Trained history source reproduced before optimizer updates` before the new training loop. GPU capacity and unprivileged memory/exit diagnostics remain enabled. Strict resume applies to an unfinished new continuation only; retain its run name and all original arguments/source files.

### Finish resolution on GPU0 while history uses GPU7

Parallel execution on separate GPUs can be attempted when server policy and capacity permit. The previous research SIGTERM did not identify its sender and does not establish a server ban on concurrent jobs. The new launcher does not disable signals, stop another job or override scheduler policy.

`launch_fsdp_resolution_remaining_nohup.sh` continues the original `fsdp_resolution_20261124T095541Z_3048501` study. It verifies its original trainer/model/data source hashes and passed audits, skips checksum-verified completed variants, starts tokens, resumes detail from its actual saved checkpoint (reported at epoch3), then starts combined. All use the original45.63%foundation and recorded100epoch cap,0.2LR multiplier and50epoch patience. It preserves the completed70epoch baseline46.51%and does not change history's49.18%source or its active runtime files.

On DGX, explicitly choose physical GPU0:

```bash
cd ~/raid/Clara/fsdp-train && git pull --ff-only origin main && conda activate fsdp-med3d && \
RESUME=0 GPU_ID=0 EXCLUDE_GPU_IDS=7 ALLOW_GPU_SHARING=1 \
MIN_FREE_GPU_GIB=48 FSDP_RESOLUTION_GPU_MEMORY_GIB=40 \
bash scripts/launch_fsdp_resolution_remaining_nohup.sh
```

GPU7 is excluded explicitly. The selector also protects GPUs belonging to readable active research parents/compute processes from this project, including a research parent between CUDA phases. UUID pinning ensures the original child's logicalcuda:0is the selected physical GPU0. Capacity must meet the maximum of the free-VRAM floor and allocator budget plus8GiBheadroom. If GPU0has less than48GiBfree with these defaults, nothing is launched; no automatic queue or job displacement occurs. The40GiBPyTorch budget is applied by a new wrapper before delegating to the byte-unchanged original trainer with exactly its original arguments. It is not a measured guarantee that every resolution variant fits. New variants retain the original native-grid and gradient preflights.

An exclusive lock on the original study and scoped active-writer checks prevent simultaneous writes to its child directories. Any failed child stops later launches. Saved optimizer/scheduler/scaler/RNG state is restored by the original strict trainer for unfinished variants. Reports and process/resource diagnostics are separate:

```bash
tail -n 80 -f logs/fsdp_research/latest.log
tail -n 80 -f logs/fsdp_resolution_remaining/latest.log
```

The original per-variant files remain under `outputs/private/fsdp_resolution/latest/train/`; aggregate continuation results appear under `outputs/private/fsdp_resolution_remaining/latest/summary.json`. The old root's interrupted parent progress is not rewritten. The new archive references the completed child exports without including model weights/cache. To resume an interrupted new parent, retain its `FSDP_RESOLUTION_REMAINING_NAME` and set `RESUME=1`; completed children are verified and skipped again. Actual DGX parallel execution and GPU0capacity have not been verified from the local workspace.

### Resolution SIGTERM sender diagnostics

The GPU3attempt skipped baseline and began tokens, then the child exited with-15during foundation verification. That identifies SIGTERM, not its sender or reason. The new child wrapper can log the immediate sender PID/UID/signal code and readable executable/comm metadata without sudo. It uses Linux [sigwaitinfo](https://man7.org/linux/man-pages/man2/sigwaitinfo.2.html), installed before Torch/CUDA/native threads, when the inherited SIGTERM disposition is default. After recording the received SIGTERM, it terminates itself with that signal; there is no automatic retry or continuation after termination. It does not replace an existing custom/ignored disposition or an inherited blocked mask. If observer initialization fails, the diagnostic worker exits rather than continuing with broken signal handling.

Logs are under the separate report directory: `tokens_signal_setup.json` records whether setup was enabled, and `tokens_signal.json` appears if a SIGTERM is captured. They are not written inside the legacy trainer's child run directory, so a new child is not accidentally made to look like an existing run. The sender may be a forwarding parent/supervisor rather than the original initiator; a missing record cannot prove no signal was sent (for example, termination before setup or an uncatchable signal). Readable metadata excludes full sender arguments/environment. Original trainer/config/checkpoint semantics and active history source files stay unchanged.

After pulling this diagnostic revision, use a fresh continuation report name/`RESUME=0` because its own source signature changed. The original resolution child checkpoints still use unchanged legacy code and remain eligible for strict per-variant resume. A manual diagnostic attempt may identify a supervisor; it does not establish permission to bypass that supervisor. If it stops again, inspect:

```bash
python -m json.tool outputs/private/fsdp_resolution_remaining/latest/tokens_signal.json
```

### Audit the remaining Dice bottleneck without another GPU run

The history continuation stopped25/100with selected epoch0and unchanged49.18%validation Dice. The artifact audit below separates that selected score from the measured learning curve:

```bash
cd ~/raid/Clara/fsdp-train && git pull --ff-only origin main && \
python scripts/inspect_fsdp_bottleneck.py --run-dir outputs/private/fsdp_research/latest
```

The stdlib-only inspector reads summary/config/history and selected native patient CSVs, verifies confusion counts and summary means, separates selected metrics from the measured learning curve, reports FP/FN/volume ratios and per-patient original history/control changes, and shows the physical spacing of fusion/token cells. It does not load images/weights/cache tensors, allocate a GPU, modify training artifacts or read test data. Parent reports may be supplied directly; the history arm is resolved within that parent. A descriptive pattern flag is a hypothesis rather than a causal diagnosis or an automated training decision. The updated research note records conditional experiments and primary sources in `docs/FSDP_RESEARCH_2026-11-24.md`. No40-point gain is promised by increasing epochs, changing threshold or selecting a different metric.

### Test train-only physical translations from the49.18%history source

The DGX artifact audit now shows frozen train53.02→55.76%while validation49.18→45.25%over25additional epochs. Best remains the inherited epoch0. This is consistent with overfitting under that recipe; it does not identify one cause. Several errors are spatial/shape errors even when predicted volume is close to GT, so changing threshold or predicted volume alone is insufficient.

`launch_fsdp_history_jitter_nohup.sh` tests one training change: paired physical translations with probability0.5and maximum1mm per axis, in integer detail voxels. CT/GT/valid support in both detail and SAM context use the same physical displacement. It samples within the existing cached FOV rather than fetching anatomy outside it. Local target margin and foreground/support checks reject unsafe transforms for both views together. CT uses trilinear sampling; labels/support use nearest. Native targets, source cache and evaluation stay unchanged. New prompts are drawn from the translated train targets; the5-click-per-side budget and accumulated history policy are preserved.

It inherits the original completed history best at49.18%, not the last25epoch continuation. The completed `fsdp_research_history_20261125T022714Z_2766483/history` continuation is the matched unjittered control: same source, versions, computational model/data code, physical policy, seed, loss, light augmentation,100epoch cap, LR multiplier0.1and25epoch patience. Different budget/source/recipe/checksum causes refusal. Preflight must reproduce the source scores before updates; source epoch0remains eligible for best selection. This is a single-seed exploratory ablation on repeatedly inspected validation, not a promise of80%Dice.

Run on DGX in the activated environment; GPU7is checked at launch and must have at least48GiBfree for the40GiBallocator budget plus8GiBheadroom. No existing jobs are stopped:

```bash
cd ~/raid/Clara/fsdp-train && git pull --ff-only origin main && \
GPU_ID=7 EXCLUDE_GPU_IDS=0,3 ALLOW_GPU_SHARING=1 \
MIN_FREE_GPU_GIB=48 FSDP_RESEARCH_GPU_MEMORY_GIB=40 \
FSDP_SOURCE_ROOT="$HOME/raid/normal case_86" FSDP_VERIFIED_CACHE=1 \
bash scripts/launch_fsdp_history_jitter_nohup.sh
```

Read `logs/fsdp_research/latest.log` and `outputs/private/fsdp_research/latest/progress.json`. Each measured training epoch includes `roi_jitter_counts` (applied translations, unselected samples, unsafe rejections), so an ineffective/no-op augmentation is visible. Child summary adds `validation_change_vs_matched_control` (Dice units, multiply100for percentage points), policy and control hashes; the parent entry also contains `delta_vs_matched_control`. For strict resume retain the printed run name, set `RESUME=1 FSDP_RESEARCH_RUN_NAME=<original_name>`, and use the same source/control/settings. This launcher starts one history arm, not the old four-arm screen. Test images remain unread.
