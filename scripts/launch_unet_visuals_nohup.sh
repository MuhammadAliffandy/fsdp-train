#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
RUN="${UNET_TINY_RUN_DIR:-$PROJECT_DIR/outputs/private/unet_tiny/latest}"
AUDIT="${UNET_AUDIT_DIR:-$PROJECT_DIR/outputs/private/unet_audit/latest}"
NAME="unet_visuals_$(date -u +%Y%m%dT%H%M%SZ)_$$"
REPORT_ROOT="$PROJECT_DIR/outputs/private/unet_visuals"
LOG_ROOT="$PROJECT_DIR/logs/unet_visuals"
OUTPUT="$REPORT_ROOT/$NAME"
if [[ ! -f "$RUN/summary.json" || ! -f "$AUDIT/thresholds.json" || ! -f "$AUDIT/threshold_overview.json" ]]; then
  echo "Completed tiny run and threshold audit are required." >&2; exit 1
fi
mkdir -p "$REPORT_ROOT" "$LOG_ROOT"
if [[ -f "$LOG_ROOT/latest.pid" ]] && kill -0 "$(cat "$LOG_ROOT/latest.pid")" 2>/dev/null; then
  echo "A visual export is active; inspect its log first." >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
nohup python -u "$PROJECT_DIR/scripts/create_unet_visuals.py" --run-dir "$RUN" --audit-dir "$AUDIT" \
  --output-dir "$OUTPUT" --device cuda:0 > "$LOG_ROOT/$NAME.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG_ROOT/$NAME.pid"
ln -sfn "$LOG_ROOT/$NAME.pid" "$LOG_ROOT/latest.pid"
ln -sfn "$LOG_ROOT/$NAME.log" "$LOG_ROOT/latest.log"
ln -sfn "$OUTPUT" "$REPORT_ROOT/latest"
ln -sfn "$OUTPUT.tar.gz" "$REPORT_ROOT/latest.tar.gz"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  if wait "$PID"; then echo "Visual export completed during startup."; else tail -80 "$LOG_ROOT/$NAME.log" >&2; exit 1; fi
else
  echo "U-Net visual export launched: PID $PID; GPU $CUDA_VISIBLE_DEVICES"
fi
echo "Figures: $REPORT_ROOT/latest/figures"
echo "Archive: $REPORT_ROOT/latest.tar.gz"
echo "Log: $LOG_ROOT/latest.log"
