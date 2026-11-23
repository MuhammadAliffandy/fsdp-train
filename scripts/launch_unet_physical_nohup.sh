#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
CONTROL="${UNET_COHORT_RUN_DIR:-$PROJECT_DIR/outputs/private/unet_cohort/latest}"
NAME="${UNET_RUN_NAME:-fsdp_private_unet_physical_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ ! "$NAME" =~ ^fsdp_private_unet_physical_[A-Za-z0-9._-]+$ ]]; then
  echo "UNET_RUN_NAME must begin with fsdp_private_unet_physical_." >&2; exit 2
fi
RUN="$PROJECT_DIR/runs/$NAME"
LOG="$PROJECT_DIR/logs/unet_physical"
REPORT="$PROJECT_DIR/outputs/private/unet_physical"
for FILE in "$CONTROL/config.json" "$CONTROL/summary.json" "$CONTROL/source_provenance.json"; do
  if [[ ! -f "$FILE" ]]; then echo "Completed prior U-Net cohort control missing: $FILE" >&2; exit 1; fi
done
COMMAND=(python -u "$PROJECT_DIR/scripts/train_unet_physical.py" --cohort-run-dir "$CONTROL" --run-dir "$RUN"
  --epochs "${NUM_EPOCHS:-200}" --val-every "${VAL_EVERY:-5}" --accumulation "${ACCUMULATION_STEPS:-8}"
  --lr "${LEARNING_RATE:-0.001}" --seed "${SEED:-2026}" --augmentation "${AUGMENTATION:-light}"
  --patience-epochs "${PATIENCE_EPOCHS:-50}" --min-delta "${MIN_DELTA:-0.001}" --device cuda:0)
case "${RESUME:-0}" in
  0) if [[ -e "$RUN" ]]; then echo "Run exists; choose a new name or RESUME=1." >&2; exit 1; fi ;;
  1)
    if [[ -z "${UNET_RUN_NAME:-}" || ! -f "$RUN/unet_latest.pth" || -f "$RUN/summary.json" ]]; then
      echo "Resume requires original UNET_RUN_NAME and an unfinished physical run/latest checkpoint." >&2; exit 1
    fi
    COMMAND+=(--resume)
    ;;
  *) echo "RESUME must be0 or1." >&2; exit 2 ;;
esac
mkdir -p "$LOG" "$REPORT"
if [[ -f "$LOG/latest.pid" ]] && kill -0 "$(cat "$LOG/latest.pid")" 2>/dev/null; then
  echo "A physical training PID is active; inspect its log before launching another run." >&2; exit 1
fi
LOG_FILE="$LOG/$NAME.$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${PREPROCESS_THREADS:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$LOG/.mpl-cache}"
mkdir -p "$MPLCONFIGDIR"
nohup "${COMMAND[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG/$NAME.pid"
ln -sfn "$LOG/$NAME.pid" "$LOG/latest.pid"
ln -sfn "$LOG_FILE" "$LOG/latest.log"
ln -sfn "$RUN" "$REPORT/latest"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  if wait "$PID"; then echo "Physical U-Net training finished during startup."; else tail -80 "$LOG_FILE" >&2; exit 1; fi
else
  echo "Physical preprocessing + U-Net launched: PID $PID; GPU $CUDA_VISIBLE_DEVICES"
fi
echo "54train /10validation /20test reserved. Preprocessing quality checks run before training."
echo "Log: $LOG/latest.log"
echo "Preprocessing: $REPORT/latest/preprocessing/summary.json"
echo "Progress: $REPORT/latest/progress.json"
echo "Final result: $REPORT/latest/summary.json"
