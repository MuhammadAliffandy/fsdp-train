#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM_MED3D_DIR="${SAM_MED3D_DIR:-$HOME/raid/Clara/SAM-Med3D}"
PRIVATE_DATA_DIR="${PRIVATE_DATA_DIR:-$PROJECT_DIR/data/processed/private/fsdp}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "fsdp-med3d" ]]; then
  echo "Activate the project environment first: conda activate fsdp-med3d" >&2
  exit 1
fi
if [[ "${RESUME:-0}" != "0" ]]; then
  echo "This starts a new private fine-tuning run. To resume a private run, use TASK_NAME=<private_run> RESUME=1 TRAIN_DATA_DIR=<private_train_dir> with launch_training_nohup.sh." >&2
  exit 2
fi

export SAM_MED3D_DIR PRIVATE_DATA_DIR
PUBLIC_CHECKPOINT="$(python - <<'PY'
import json
import os
from pathlib import Path
import torch

data = Path(os.environ['PRIVATE_DATA_DIR']).resolve()
manifest = json.loads((data / 'manifest.json').read_text())
if manifest.get('target') != {'1': 'RFSDP', '9': 'LFSDP'} or manifest['counts']['test'] != 20:
    raise SystemExit('Expected the frozen bilateral FSDP manifest with exactly 20 test patients')
parts = [manifest['splits'][name] for name in ('train', 'val', 'test')]
flat = sum(parts, [])
if len(set(flat)) != len(flat) or any(len(manifest['splits'][name]) != manifest['counts'][name] for name in ('train', 'val', 'test')):
    raise SystemExit('Private split is overlapping or inconsistent')
for split in ('train', 'val', 'test'):
    expected = {key + '.nii.gz' for key in manifest['splits'][split]}
    for folder in ('imagesTr', 'labelsTr'):
        found = {p.name for p in (data / split / 'ct_private' / folder).glob('*.nii.gz')}
        if found != expected:
            raise SystemExit('Private data files differ from the frozen split manifest')
explicit = os.environ.get('PUBLIC_CHECKPOINT')
if explicit:
    checkpoint = Path(explicit).expanduser().resolve()
else:
    candidates = list((Path(os.environ['SAM_MED3D_DIR']) / 'work_dir').glob('fsdp_nasalseg_ts_pseudolabel_*/sam_model_latest.pth'))
    if not candidates:
        raise SystemExit('No public checkpoint found; set PUBLIC_CHECKPOINT explicitly')
    checkpoint = max(candidates, key=lambda path: path.stat().st_mtime).resolve()
state = torch.load(checkpoint, map_location='cpu', weights_only=False)
if 'model_state_dict' not in state or state.get('epoch', 0) < 50:
    raise SystemExit('Expected a public checkpoint after 50 completed epochs; set PUBLIC_CHECKPOINT to that file')
print(checkpoint)
PY
)"

export TRAIN_DATA_DIR="$PRIVATE_DATA_DIR/train/ct_private"
export INITIAL_CHECKPOINT="$PUBLIC_CHECKPOINT"
export TASK_NAME="${TASK_NAME:-fsdp_private_binary_$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$TASK_NAME" =~ ^fsdp_private_binary_[A-Za-z0-9._-]+$ ]]; then
  echo "Private TASK_NAME must begin with fsdp_private_binary_." >&2
  exit 2
fi
export RESUME=0
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
echo "Private initialization: $PUBLIC_CHECKPOINT"
exec bash "$PROJECT_DIR/scripts/launch_training_nohup.sh"
