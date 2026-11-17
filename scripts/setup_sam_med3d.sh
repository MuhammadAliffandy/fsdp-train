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
    "import os\nimport os.path as osp\n\n"
    "PROJ_DIR = osp.dirname(osp.dirname(__file__))\n"
    f"img_datas = [os.environ.get('FSDP_DATA_DIR', {str(data_dir)!r})]\n",
    encoding="utf-8",
)

trainer = repo / "train.py"
source = trainer.read_text(encoding="utf-8")
old = "threshold=1000)"
new = "threshold=-1)"
if source.count(old) == 1:
    source = source.replace(old, new, 1)
elif source.count(new) == 1:
    pass
else:
    raise SystemExit("Could not safely set the foreground threshold in train.py")

old_loss = "self.seg_loss(prev_masks, gt3D)"
new_loss = "self.seg_loss(prev_masks, gt3D.float())"
if source.count(old_loss) == 1:
    source = source.replace(old_loss, new_loss, 1)
elif source.count(new_loss) != 1:
    raise SystemExit("Could not safely configure float targets for DiceCELoss in train.py")
metric_start = source.index("    def get_dice_score(")
metric_end = source.find("\n    def ", metric_start + 1)
if metric_end == -1:
    raise SystemExit("Could not locate the end of get_dice_score in train.py")
metric = source[metric_start:metric_end]
old_empty = "return np.NaN"
new_empty = "return volume_sum.new_tensor(1.0, dtype=torch.float32)"
if metric.count(old_empty) == 1:
    metric = metric.replace(old_empty, new_empty, 1)
elif metric.count(new_empty) != 1:
    raise SystemExit("Could not safely configure empty-pair Dice in train.py")
source = source[:metric_start] + metric + source[metric_end:]
trainer.write_text(source, encoding="utf-8")

click_file = repo / "utils/click_method.py"
click_source = click_file.read_text(encoding="utf-8")
start = click_source.index("def get_next_click3D_torch_2(")
end = click_source.find("\ndef ", start + 1)
if end == -1:
    end = len(click_source)
click_function = click_source[start:end]
old_points = """        points = torch.argwhere(to_point_mask[i])
        point = points[np.random.randint(len(points))]"""
new_points = """        points = torch.argwhere(to_point_mask[i])
        if len(points) == 0:
            points = torch.argwhere(true_masks[i])
        if len(points) > 0:
            point = points[np.random.randint(len(points))]
        else:
            point = torch.tensor(
                [0] + [np.random.randint(size) for size in true_masks.shape[2:]],
                dtype=torch.long, device=true_masks.device,
            )"""
old_label = "if fn_masks[i, 0, point[1], point[2], point[3]]:"
new_label = "if true_masks[i, 0, point[1], point[2], point[3]]:"
if click_function.count(old_points) == 1 and click_function.count(old_label) == 1:
    click_function = click_function.replace(old_points, new_points, 1).replace(old_label, new_label, 1)
elif click_function.count(new_points) != 1 or click_function.count(new_label) != 1:
    raise SystemExit("Could not safely configure empty-region handling in click_method.py")
click_file.write_text(click_source[:start] + click_function + click_source[end:], encoding="utf-8")
print(f"Configured SAM-Med3D data path: {data_dir}")
print("Configured foreground threshold: -1 (include all non-empty masks)")
print("Configured DiceCELoss targets: float32 (integer labels retained for prompts)")
print("Configured prompt fallback: target foreground or negative point when the error region is empty")
print("Configured Dice: both-empty masks score 1.0, returned as a tensor")
PY

echo "SAM-Med3D is ready at $SAM_MED3D_DIR"
echo "Checkpoint: $CHECKPOINT"
