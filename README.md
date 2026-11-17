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

## DGX workflow

1. Clone the repository.
2. Create and activate the Conda environment, then install `requirements.txt`.
3. Confirm DGX GPU visibility from PyTorch.
4. Download/extract public data into the ignored `data/` path.
5. Run the NIfTI audit and inspect representative scans/masks.
6. Package accepted pseudo-labels in SAM-Med3D format and launch the public pseudo-label baseline.

Do not place private scans, private subject IDs, checkpoints, or generated masks in Git.
