#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM_MED3D_DIR="${SAM_MED3D_DIR:-$HOME/raid/Clara/SAM-Med3D}"
DATA_DIR="$PROJECT_DIR/data/medical_preprocessed/fsdp/ct_nifti"
CHECKPOINT="$SAM_MED3D_DIR/ckpt/sam_med3d_turbo.pth"
GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
IMG_SIZE="${IMG_SIZE:-128}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-8e-5}"
RESUME="${RESUME:-0}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "fsdp-med3d" ]]; then
  echo "Activate the project environment first: conda activate fsdp-med3d" >&2
  exit 1
fi
if [[ ! -f "$SAM_MED3D_DIR/train.py" || ! -s "$CHECKPOINT" ]]; then
  echo "SAM-Med3D setup/checkpoint missing; run scripts/setup_sam_med3d.sh first." >&2
  exit 1
fi
if [[ ! -d "$DATA_DIR/imagesTr" || ! -d "$DATA_DIR/labelsTr" ]]; then
  echo "Prepared training data not found at $DATA_DIR" >&2
  exit 1
fi

IMAGE_COUNT="$(find "$DATA_DIR/imagesTr" -maxdepth 1 -type f -name '*.nii.gz' | wc -l | tr -d ' ')"
LABEL_COUNT="$(find "$DATA_DIR/labelsTr" -maxdepth 1 -type f -name '*.nii.gz' | wc -l | tr -d ' ')"
if [[ "$IMAGE_COUNT" -eq 0 || "$IMAGE_COUNT" -ne "$LABEL_COUNT" ]]; then
  echo "Image/label count mismatch: images=$IMAGE_COUNT labels=$LABEL_COUNT" >&2
  exit 1
fi

RUN_ID="${TASK_NAME:-fsdp_nasalseg_ts_pseudolabel_$(date -u +%Y%m%dT%H%M%SZ)}"
RESUME_ARGS=()
case "$RESUME" in
  latest)
    RUN_ID="$(python - "$SAM_MED3D_DIR" <<'PY'
import sys
from pathlib import Path

checkpoints = list((Path(sys.argv[1]) / "work_dir").glob("fsdp_nasalseg_ts_pseudolabel_*/sam_model_latest.pth"))
if not checkpoints:
    raise SystemExit("No completed-epoch checkpoint found; use RESUME=0 for a new run.")
print(max(checkpoints, key=lambda path: path.stat().st_mtime).parent.name)
PY
)"
    RESUME_ARGS=(--resume)
    ;;
  1)
    if [[ -z "${TASK_NAME:-}" ]]; then
      echo "RESUME=1 requires TASK_NAME to identify the run." >&2
      exit 2
    fi
    RESUME_ARGS=(--resume)
    ;;
  0) ;;
  *) echo "RESUME must be 0, 1, or latest." >&2; exit 2 ;;
esac
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "TASK_NAME may contain only letters, numbers, dot, underscore, and hyphen." >&2
  exit 2
fi
LOG_DIR="$PROJECT_DIR/logs/training"
LOG_FILE="$LOG_DIR/$RUN_ID.log"
if [[ "$RESUME" != "0" ]]; then
  if [[ ! -s "$SAM_MED3D_DIR/work_dir/$RUN_ID/sam_model_latest.pth" ]]; then
    echo "Cannot resume: completed-epoch checkpoint missing for $RUN_ID" >&2
    exit 1
  fi
  LOG_FILE="$LOG_DIR/$RUN_ID.resume_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
elif [[ -e "$LOG_FILE" ]]; then
  echo "Log already exists for $RUN_ID; choose a new TASK_NAME or use RESUME=1." >&2
  exit 1
fi
PID_FILE="$LOG_DIR/$RUN_ID.pid"
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Training already running: PID $OLD_PID ($(cat "$PID_FILE"))" >&2
    exit 1
  fi
  rm -f "$PID_FILE"
fi

read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
if ((${#GPU_ID_ARRAY[@]} != 1)); then
  echo "This launcher is configured for one GPU; set GPU_IDS to a single ID (default: 0)." >&2
  exit 2
fi
cd "$SAM_MED3D_DIR"
nohup python -u train.py \
  --task_name "$RUN_ID" \
  --model_type vit_b_ori \
  --checkpoint "$CHECKPOINT" \
  --gpu_ids "${GPU_ID_ARRAY[@]}" \
  --num_workers "$NUM_WORKERS" \
  --batch_size "$BATCH_SIZE" \
  --accumulation_steps "$ACCUMULATION_STEPS" \
  --img_size "$IMG_SIZE" \
  --num_epochs "$NUM_EPOCHS" \
  --lr "$LEARNING_RATE" \
  "${RESUME_ARGS[@]}" \
  > "$LOG_FILE" 2>&1 < /dev/null &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
ln -sfn "$LOG_FILE" "$LOG_DIR/latest.log"

sleep 2
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
  echo "Training process exited during startup. Recent log output:" >&2
  tail -80 "$LOG_FILE" >&2 || true
  rm -f "$PID_FILE"
  exit 1
fi

echo "Training started (PID $TRAIN_PID)"
echo "Cases: $IMAGE_COUNT | GPUs: ${GPU_ID_ARRAY[*]} | Epochs: $NUM_EPOCHS"
echo "Run: $RUN_ID | Resume: $RESUME"
echo "Log: $LOG_FILE"
echo "Monitor: bash '$PROJECT_DIR/scripts/monitor_training.sh' tail"
