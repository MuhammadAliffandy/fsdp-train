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

## DGX workflow

1. Clone the repository.
2. Create and activate the Conda environment, then install `requirements.txt`.
3. Confirm DGX GPU visibility from PyTorch.
4. Download/extract public data into the ignored `data/` path.
5. Run the NIfTI audit and inspect representative scans/masks.
6. Package accepted pseudo-labels in SAM-Med3D format and launch the public pseudo-label baseline.

Do not place private scans, private subject IDs, checkpoints, or generated masks in Git.
