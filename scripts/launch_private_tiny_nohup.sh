#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != "fsdp-med3d" ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
SAM_DIR="${SAM_MED3D_DIR:-$HOME/raid/Clara/SAM-Med3D}"
DATA_DIR="${PRIVATE_DATA_DIR:-$PROJECT_DIR/data/processed/private/fsdp}"
CHECKPOINT="${INITIAL_CHECKPOINT:-$SAM_DIR/ckpt/sam_med3d_turbo.pth}"
if [[ ! -f "$CHECKPOINT" || ! -f "$DATA_DIR/manifest.json" || ! -f "$SAM_DIR/segment_anything/build_sam3D.py" ]]; then
  echo "Initial weights, prepared manifest or SAM-Med3D source missing." >&2; exit 1
fi
NAME="fsdp_private_tiny_$(date -u +%Y%m%dT%H%M%SZ)_$$"
OUTPUT="$PROJECT_DIR/runs/$NAME"
LOG_DIR="$PROJECT_DIR/logs/tiny_training"
REPORT_DIR="$PROJECT_DIR/outputs/private/tiny_training"
mkdir -p "$LOG_DIR" "$REPORT_DIR"
if [[ -f "$LOG_DIR/latest.pid" ]] && kill -0 "$(cat "$LOG_DIR/latest.pid")" 2>/dev/null; then
  echo "Previous tiny-run PID is still active; inspect its log before launching another." >&2; exit 1
fi
COMMAND=(python -u "$PROJECT_DIR/scripts/train_private_tiny.py"
  --sam-dir "$SAM_DIR" --data-dir "$DATA_DIR"
  --source-root "${PRIVATE_SOURCE_ROOT:-$HOME/raid/normal case_86}"
  --checkpoint "$CHECKPOINT" --run-dir "$OUTPUT"
  --epochs "${TINY_EPOCHS:-200}" --eval-every "${TINY_EVAL_EVERY:-5}"
  --lr "${TINY_LEARNING_RATE:-1e-4}" --target-dice "${TINY_TARGET_DICE:-0.90}"
  --seed "${SEED:-2026}" --device cuda:0)
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
nohup "${COMMAND[@]}" > "$LOG_DIR/$NAME.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG_DIR/$NAME.pid"
ln -sfn "$LOG_DIR/$NAME.pid" "$LOG_DIR/latest.pid"
ln -sfn "$LOG_DIR/$NAME.log" "$LOG_DIR/latest.log"
ln -sfn "$OUTPUT" "$REPORT_DIR/latest"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  if wait "$PID"; then
    echo "Tiny diagnostic completed during startup."
  else
    tail -80 "$LOG_DIR/$NAME.log" >&2; exit 1
  fi
else
  echo "Tiny training diagnostic launched: PID $PID; GPU ${CUDA_VISIBLE_DEVICES}"
fi
echo "Two training patients; four side ROIs. No validation/test images read."
echo "Log: $LOG_DIR/latest.log"
echo "Summary: $REPORT_DIR/latest/summary.json"
