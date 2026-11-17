#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM_MED3D_DIR="${SAM_MED3D_DIR:-$HOME/raid/Clara/SAM-Med3D}"
DATA_DIR="$PROJECT_DIR/data/medical_preprocessed/fsdp/ct_nifti"
CHECKPOINT="$SAM_MED3D_DIR/ckpt/sam_med3d_turbo.pth"

if [[ "${CONDA_DEFAULT_ENV:-}" != "fsdp-med3d" ]]; then
  echo "Activate the project environment first: conda activate fsdp-med3d" >&2
  exit 1
fi

if [[ ! -d "$DATA_DIR/imagesTr" || ! -d "$DATA_DIR/labelsTr" ]]; then
  echo "Prepared dataset not found under $DATA_DIR; run prepare_sam_med3d.py first." >&2
  exit 1
fi

if [[ ! -d "$SAM_MED3D_DIR/.git" ]]; then
  mkdir -p "$(dirname "$SAM_MED3D_DIR")"
  git clone https://github.com/uni-medical/SAM-Med3D.git "$SAM_MED3D_DIR"
fi

mkdir -p "$SAM_MED3D_DIR/ckpt"
if [[ ! -s "$CHECKPOINT" ]]; then
  curl -L --fail --show-error --retry 5 \
    -o "$CHECKPOINT" \
    https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth
fi
echo "899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c  $CHECKPOINT" | sha256sum -c -

export PROJECT_DATA_DIR="$DATA_DIR"
export SAM_MED3D_DIR
python - <<'PY'
import os
from pathlib import Path

repo = Path(os.environ["SAM_MED3D_DIR"])
data_dir = Path(os.environ["PROJECT_DATA_DIR"]).resolve()
paths_file = repo / "utils/data_paths.py"
paths_file.write_text(
    "import os.path as osp\n\n"
    "PROJ_DIR = osp.dirname(osp.dirname(__file__))\n"
    f"img_datas = [{str(data_dir)!r}]\n",
    encoding="utf-8",
)

trainer = repo / "train.py"
source = trainer.read_text(encoding="utf-8")
old = "threshold=1000)"
new = "threshold=-1)"
if source.count(old) == 1:
    trainer.write_text(source.replace(old, new, 1), encoding="utf-8")
elif source.count(new) == 1:
    pass
else:
    raise SystemExit("Could not safely set the foreground threshold in train.py")
print(f"Configured SAM-Med3D data path: {data_dir}")
print("Configured foreground threshold: -1 (include all non-empty masks)")
PY

echo "SAM-Med3D is ready at $SAM_MED3D_DIR"
echo "Checkpoint: $CHECKPOINT"
