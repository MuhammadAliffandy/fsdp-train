#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != "fsdp-med3d" ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
STAGE="${UNET_STAGE:-tiny}"
case "$STAGE" in tiny|cohort) ;; *) echo "UNET_STAGE must be tiny or cohort." >&2; exit 2 ;; esac
DATA_DIR="${PRIVATE_DATA_DIR:-$PROJECT_DIR/data/processed/private/fsdp}"
V4_DIR="${PRIVATE_V4_RUN_DIR:-$PROJECT_DIR/outputs/private/zoom_training/latest}"
TINY_DIR="${UNET_TINY_RUN_DIR:-$PROJECT_DIR/outputs/private/unet_tiny/latest}"
NAME="${UNET_RUN_NAME:-fsdp_private_unet_${STAGE}_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ ! "$NAME" =~ ^fsdp_private_unet_${STAGE}_[A-Za-z0-9._-]+$ ]]; then
  echo "UNET_RUN_NAME must begin with fsdp_private_unet_${STAGE}_." >&2; exit 2
fi
RUN_DIR="$PROJECT_DIR/runs/$NAME"
LOG_DIR="$PROJECT_DIR/logs/unet_$STAGE"
REPORT_DIR="$PROJECT_DIR/outputs/private/unet_$STAGE"
if [[ ! -f "$DATA_DIR/manifest.json" || ! -f "$V4_DIR/summary.json" || ! -f "$V4_DIR/roi_policy.json" ]]; then
  echo "Prepared manifest or completed v4 control missing." >&2; exit 1
fi
COMMAND=(python -u "$PROJECT_DIR/scripts/train_private_unet.py"
  --source-root "${PRIVATE_SOURCE_ROOT:-$HOME/raid/normal case_86}"
  --data-dir "$DATA_DIR" --v4-run-dir "$V4_DIR" --run-dir "$RUN_DIR" --stage "$STAGE"
  --epochs "${NUM_EPOCHS:-200}" --val-every "${VAL_EVERY:-5}" --accumulation "${ACCUMULATION_STEPS:-8}"
  --lr "${LEARNING_RATE:-0.001}" --seed "${SEED:-2026}"
  --patience-epochs "${PATIENCE_EPOCHS:-50}" --min-delta "${MIN_DELTA:-0.001}" --device cuda:0)
if [[ "$STAGE" == cohort ]]; then
  if [[ ! -f "$TINY_DIR/summary.json" || ! -f "$TINY_DIR/unet_best.pth" ]]; then
    echo "Complete the tiny diagnostic first; cohort requires its passed summary/checkpoint." >&2; exit 1
  fi
  COMMAND+=(--tiny-run-dir "$TINY_DIR")
fi
case "${RESUME:-0}" in
  0) if [[ -e "$RUN_DIR" ]]; then echo "Run exists; choose a new name or RESUME=1." >&2; exit 1; fi ;;
  1)
    if [[ -z "${UNET_RUN_NAME:-}" || ! -f "$RUN_DIR/unet_latest.pth" ]]; then
      echo "Resume requires UNET_RUN_NAME and its latest checkpoint." >&2; exit 1
    fi
    COMMAND+=(--resume)
    ;;
  *) echo "RESUME must be 0 or 1." >&2; exit 2 ;;
esac
mkdir -p "$LOG_DIR" "$REPORT_DIR"
if [[ -f "$LOG_DIR/latest.pid" ]] && kill -0 "$(cat "$LOG_DIR/latest.pid")" 2>/dev/null; then
  echo "A U-Net $STAGE PID is active; inspect its log before launching another run." >&2; exit 1
fi
LOG_FILE="$LOG_DIR/$NAME.$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
nohup "${COMMAND[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG_DIR/$NAME.pid"
ln -sfn "$LOG_DIR/$NAME.pid" "$LOG_DIR/latest.pid"
ln -sfn "$LOG_FILE" "$LOG_DIR/latest.log"
ln -sfn "$RUN_DIR" "$REPORT_DIR/latest"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  if wait "$PID"; then echo "U-Net stage completed during startup."; else tail -80 "$LOG_FILE" >&2; exit 1; fi
else
  echo "U-Net $STAGE launched: PID $PID; GPU $CUDA_VISIBLE_DEVICES"
fi
echo "Run: $NAME"
echo "Log: $LOG_DIR/latest.log"
echo "Summary: $REPORT_DIR/latest/summary.json"
