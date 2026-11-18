#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM_MED3D_DIR="${SAM_MED3D_DIR:-$HOME/raid/Clara/SAM-Med3D}"
PRIVATE_DATA_DIR="${PRIVATE_DATA_DIR:-$PROJECT_DIR/data/processed/private/fsdp}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "fsdp-med3d" ]]; then
  echo "Activate the project environment first: conda activate fsdp-med3d" >&2
  exit 1
fi
if [[ ! -f "$PRIVATE_DATA_DIR/manifest.json" || ! -f "$SAM_MED3D_DIR/segment_anything/build_sam3D.py" ]]; then
  echo "Prepared private data or SAM-Med3D code not found." >&2
  exit 1
fi
RUN_ID="private_validation_$(date -u +%Y%m%dT%H%M%SZ)_$$"
if [[ "${COMPARE_STAGES:-0}" == "1" ]]; then
  RUN_ID="stage_comparison_$(date -u +%Y%m%dT%H%M%SZ)_$$"
fi
OUTPUT_DIR="$PROJECT_DIR/outputs/private/diagnostics/$RUN_ID"
LOG_DIR="$PROJECT_DIR/logs/validation"
mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_DIR")"
LOG_FILE="$LOG_DIR/$RUN_ID.log"
COMMAND=(python -u "$PROJECT_DIR/scripts/evaluate_private_validation.py"
  --sam-dir "$SAM_MED3D_DIR" --data-dir "$PRIVATE_DATA_DIR"
  --output-dir "$OUTPUT_DIR" --device cuda:0)
if [[ -n "${PRIVATE_RUN_DIR:-}" ]]; then
  COMMAND+=(--run-dir "$PRIVATE_RUN_DIR")
fi
if [[ "${COMPARE_STAGES:-0}" == "1" ]]; then
  COMMAND+=(--compare-stages)
  if [[ -n "${TURBO_CHECKPOINT:-}" ]]; then
    COMMAND+=(--turbo-checkpoint "$TURBO_CHECKPOINT")
  fi
  if [[ -n "${PUBLIC_CHECKPOINT:-}" ]]; then
    COMMAND+=(--public-checkpoint "$PUBLIC_CHECKPOINT")
  fi
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
nohup "${COMMAND[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG_DIR/$RUN_ID.pid"
ln -sfn "$LOG_FILE" "$LOG_DIR/latest.log"
ln -sfn "$OUTPUT_DIR" "$PROJECT_DIR/outputs/private/diagnostics/latest"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  echo "Validation exited during startup; inspect $LOG_FILE" >&2
  tail -80 "$LOG_FILE" >&2 || true
  exit 1
fi
echo "Validation launched (PID $PID); startup only, not completion."
echo "Log: $LOG_FILE"
echo "Artifacts: $OUTPUT_DIR"
echo "Monitor: tail -n 80 -F '$LOG_DIR/latest.log'"
