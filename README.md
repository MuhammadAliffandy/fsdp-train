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

NasalSeg v2 contains 130 paired NRRD CT/mask files. Its masks label maxillary sinuses, nasal cavities, and nasopharynx, not the frontal sinus. For the public experiment, run TotalSegmentator's `craniofacial_structures` task on the CT images and use its `sinus_frontal` output as pseudo-labels.

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

## DGX workflow

1. Clone the repository.
2. Create and activate the Conda environment, then install `requirements.txt`.
3. Confirm DGX GPU visibility from PyTorch.
4. Download/extract public data into the ignored `data/` path.
5. Run the NIfTI audit and inspect representative scans/masks.
6. After the data layout and label mapping are agreed, implement pseudo-label generation and SAM-Med3D training commands.

Do not place private scans, private subject IDs, checkpoints, or generated masks in Git.
