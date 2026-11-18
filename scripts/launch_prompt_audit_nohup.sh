#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != "fsdp-med3d" ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
NAME="prompt_audit_$(date -u +%Y%m%dT%H%M%SZ)_$$"
OUTPUT="$PROJECT_DIR/outputs/private/prompt_audit/$NAME"
LOG_DIR="$PROJECT_DIR/logs/prompt_audit"
mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT")"
COMMAND=(python -u "$PROJECT_DIR/scripts/audit_prompt_coordinates.py"
  --sam-dir "${SAM_MED3D_DIR:-$HOME/raid/Clara/SAM-Med3D}"
  --data-dir "${PRIVATE_DATA_DIR:-$PROJECT_DIR/data/processed/private/fsdp}"
  --output-dir "$OUTPUT" --device cuda:0)
if [[ -n "${PRIVATE_V2_RUN_DIR:-}" ]]; then COMMAND+=(--run-dir "$PRIVATE_V2_RUN_DIR"); fi
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
nohup "${COMMAND[@]}" > "$LOG_DIR/$NAME.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG_DIR/$NAME.pid"
ln -sfn "$LOG_DIR/$NAME.log" "$LOG_DIR/latest.log"
ln -sfn "$OUTPUT" "$PROJECT_DIR/outputs/private/prompt_audit/latest"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then tail -80 "$LOG_DIR/$NAME.log" >&2; exit 1; fi
echo "Prompt-coordinate audit launched: PID $PID"
echo "Log: $LOG_DIR/latest.log"
echo "Summary: $OUTPUT/summary.json"
