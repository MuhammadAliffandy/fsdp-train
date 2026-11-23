#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
RUN="${UNET_TINY_RUN_DIR:-$PROJECT_DIR/outputs/private/unet_tiny/latest}"
NAME="unet_audit_$(date -u +%Y%m%dT%H%M%SZ)_$$"
REPORT_ROOT="$PROJECT_DIR/outputs/private/unet_audit"
LOG_ROOT="$PROJECT_DIR/logs/unet_audit"
OUTPUT="$REPORT_ROOT/$NAME"
if [[ ! -f "$RUN/summary.json" || ! -f "$RUN/unet_best.pth" || ! -f "$RUN/unet_latest.pth" ]]; then
  echo "Completed U-Net tiny diagnostic and both checkpoints are required." >&2; exit 1
fi
mkdir -p "$REPORT_ROOT" "$LOG_ROOT"
if [[ -f "$LOG_ROOT/latest.pid" ]] && kill -0 "$(cat "$LOG_ROOT/latest.pid")" 2>/dev/null; then
  echo "A U-Net audit is already active; inspect its log first." >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
nohup python -u "$PROJECT_DIR/scripts/audit_unet_tiny.py" --run-dir "$RUN" --output-dir "$OUTPUT" --device cuda:0 \
  > "$LOG_ROOT/$NAME.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG_ROOT/$NAME.pid"
ln -sfn "$LOG_ROOT/$NAME.pid" "$LOG_ROOT/latest.pid"
ln -sfn "$LOG_ROOT/$NAME.log" "$LOG_ROOT/latest.log"
ln -sfn "$OUTPUT" "$REPORT_ROOT/latest"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  if wait "$PID"; then echo "U-Net audit completed during startup."; else tail -80 "$LOG_ROOT/$NAME.log" >&2; exit 1; fi
else
  echo "U-Net audit launched: PID $PID; GPU $CUDA_VISIBLE_DEVICES"
fi
echo "Log: $LOG_ROOT/latest.log"
echo "Report: $REPORT_ROOT/latest/summary.json"
