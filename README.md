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

## Launch the first SAM-Med3D run

Clone the official implementation on DGX and obtain its recommended `sam_med3d_turbo.pth` checkpoint. Keep the checkpoint outside this project repository. In the SAM-Med3D checkout, point `img_datas` at the prepared directory and lower the loader foreground threshold to `-1`; its default threshold of 1000 excludes small masks and can recurse indefinitely if all masks are below threshold.

```bash
cd ~/raid/Clara
git clone https://github.com/uni-medical/SAM-Med3D.git
cd SAM-Med3D
conda activate fsdp-med3d
mkdir -p ckpt
curl -L --fail --show-error --retry 5 \
  -o ckpt/sam_med3d_turbo.pth \
  https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth
python - <<'PY'
from pathlib import Path

paths = Path("utils/data_paths.py")
paths.write_text('''import os.path as osp\n\nPROJ_DIR = osp.dirname(osp.dirname(__file__))\nimg_datas = [osp.expanduser("~/raid/Clara/fsdp-train/data/medical_preprocessed/fsdp/ct_nifti")]\n''')

trainer = Path("train.py")
text = trainer.read_text()
old = "threshold=1000)"
if text.count(old) != 1:
    raise SystemExit("Expected one threshold=1000 in train.py; inspect upstream file before editing")
trainer.write_text(text.replace(old, "threshold=-1)", 1))
PY
```

Then launch a conservative single-GPU run; adjust `--gpu_ids` to a GPU visible to this job and lower batch size if memory is insufficient:

```bash
python train.py \
  --task_name fsdp_nasalseg_ts_pseudolabel_baseline \
  --model_type vit_b_ori \
  --checkpoint ckpt/sam_med3d_turbo.pth \
  --gpu_ids 0 \
  --num_workers 4 \
  --batch_size 1 \
  --accumulation_steps 8 \
  --img_size 128 \
  --num_epochs 50 \
  --lr 8e-5
```

This official training script trains on `labelsTr` and reports training-set metrics; it does not provide a held-out validation split in its standard `train.py` path. Treat those metrics as agreement with TotalSegmentator pseudo-labels, not anatomical accuracy. The first run is a pipeline/adaptation baseline. Do not use it to claim FSDP clinical performance; evaluation against the private expert-labeled holdout is a later, separate stage.

## DGX workflow

1. Clone the repository.
2. Create and activate the Conda environment, then install `requirements.txt`.
3. Confirm DGX GPU visibility from PyTorch.
4. Download/extract public data into the ignored `data/` path.
5. Run the NIfTI audit and inspect representative scans/masks.
6. Package accepted pseudo-labels in SAM-Med3D format and launch the public pseudo-label baseline.

Do not place private scans, private subject IDs, checkpoints, or generated masks in Git.
