# FSDP / Frontal Sinus Segmentation Experiments

Research workspace for the public frontal-sinus pseudo-label experiment and later supervised fine-tuning with the available private NIfTI labels. See [PROJECT.md](PROJECT.md) for the research plan and open decisions.

## Create the Conda environment

The environment follows the official SAM-Med3D Python 3.10 and PyTorch 2.6.0 setup, with NIfTI and dataset-audit dependencies added.

```bash
conda env create -f environment.yml
conda activate fsdp-med3d
```

On DGX, confirm that PyTorch sees the assigned GPU before launching an experiment:

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
mkdir -p data/raw/public/nasalseg
cd data/raw/public/nasalseg
curl -L --fail --retry 5 --retry-delay 2 -C - \
  -o NasalSeg.zip \
  'https://zenodo.org/records/13893419/files/NasalSeg.zip?download=1'
echo 'ba83df47974d907798542101fa43ac7d  NasalSeg.zip' | md5sum -c -
unzip -l NasalSeg.zip | head -40
mkdir -p extracted
unzip -q NasalSeg.zip -d extracted
```

The Zenodo v2 record lists the archive as 224 MB. Review the extracted layout before writing dataset conversion code; do not assume image or mask extensions from the archive name.

## DGX workflow

1. Clone or sync this repository onto DGX.
2. Create the Conda environment from `environment.yml`.
3. Download/extract public data and model checkpoints into ignored `data/` and `checkpoints/` paths.
4. Run the NIfTI audit and inspect representative scans/masks.
5. After the data layout and label mapping are agreed, implement pseudo-label generation and SAM-Med3D training commands.

Do not place private scans, private subject IDs, checkpoints, or generated masks in Git.
