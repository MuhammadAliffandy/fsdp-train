#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
NAME="${UNET_RUN_NAME:-fsdp_private_unet_cohort_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ ! "$NAME" =~ ^fsdp_private_unet_cohort_[A-Za-z0-9._-]+$ ]]; then
  echo "UNET_RUN_NAME must begin with fsdp_private_unet_cohort_." >&2; exit 2
fi
RUN="$PROJECT_DIR/runs/$NAME"
DATA="${PRIVATE_DATA_DIR:-$PROJECT_DIR/data/processed/private/fsdp}"
V4="${PRIVATE_V4_RUN_DIR:-$PROJECT_DIR/outputs/private/zoom_training/latest}"
TINY="${UNET_TINY_RUN_DIR:-$PROJECT_DIR/outputs/private/unet_tiny/latest}"
AUDIT="${UNET_AUDIT_DIR:-$PROJECT_DIR/outputs/private/unet_audit/latest}"
LOG="$PROJECT_DIR/logs/unet_cohort"
REPORT="$PROJECT_DIR/outputs/private/unet_cohort"
for FILE in "$DATA/manifest.json" "$V4/summary.json" "$V4/roi_policy.json" \
  "$TINY/summary.json" "$TINY/unet_latest.pth" "$AUDIT/summary.json" "$AUDIT/thresholds.json"; do
  if [[ ! -f "$FILE" ]]; then echo "Required control/calibration artifact missing: $FILE" >&2; exit 1; fi
done
COMMAND=(python -u "$PROJECT_DIR/scripts/train_unet_cohort.py" --exploratory-cohort
  --source-root "${PRIVATE_SOURCE_ROOT:-$HOME/raid/normal case_86}"
  --data-dir "$DATA" --v4-run-dir "$V4" --tiny-run-dir "$TINY" --audit-dir "$AUDIT" --run-dir "$RUN"
  --threshold 0.3 --epochs "${NUM_EPOCHS:-200}" --val-every "${VAL_EVERY:-5}"
  --accumulation "${ACCUMULATION_STEPS:-8}" --lr "${LEARNING_RATE:-0.001}" --seed "${SEED:-2026}"
  --patience-epochs "${PATIENCE_EPOCHS:-50}" --min-delta "${MIN_DELTA:-0.001}" --device cuda:0)
case "${RESUME:-0}" in
  0) if [[ -e "$RUN" ]]; then echo "Run exists; select a new name or RESUME=1." >&2; exit 1; fi ;;
  1)
    if [[ -z "${UNET_RUN_NAME:-}" || ! -f "$RUN/unet_latest.pth" || -f "$RUN/summary.json" ]]; then
      echo "Resume requires the original UNET_RUN_NAME and an unfinished run/latest checkpoint." >&2; exit 1
    fi
    COMMAND+=(--resume)
    ;;
  *) echo "RESUME must be 0 or 1." >&2; exit 2 ;;
esac
mkdir -p "$LOG" "$REPORT"
if [[ -f "$LOG/latest.pid" ]] && kill -0 "$(cat "$LOG/latest.pid")" 2>/dev/null; then
  echo "A cohort PID is active; inspect its log before launching another run." >&2; exit 1
fi
LOG_FILE="$LOG/$NAME.$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
nohup "${COMMAND[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG/$NAME.pid"
ln -sfn "$LOG/$NAME.pid" "$LOG/latest.pid"
ln -sfn "$LOG_FILE" "$LOG/latest.log"
ln -sfn "$RUN" "$REPORT/latest"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  if wait "$PID"; then echo "U-Net cohort finished during startup."; else tail -80 "$LOG_FILE" >&2; exit 1; fi
else
  echo "Exploratory U-Net cohort launched: PID $PID; GPU $CUDA_VISIBLE_DEVICES"
fi
echo "Run: $NAME"
echo "54 train / 10 validation; 20 test reserved; fixed threshold0.3; original tiny gate remains unmet."
echo "Log: $LOG/latest.log"
echo "Summary: $REPORT/latest/summary.json"
