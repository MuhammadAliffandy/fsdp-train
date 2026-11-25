#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
ORIGINAL="${FSDP_RESOLUTION_ORIGINAL_DIR:-$PROJECT_DIR/runs/fsdp_resolution_20261124T095541Z_3048501}"
NAME="${FSDP_RESOLUTION_REMAINING_NAME:-fsdp_resolution_remaining_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ ! "$NAME" =~ ^fsdp_resolution_remaining_[A-Za-z0-9._-]+$ ]]; then echo "Invalid continuation name." >&2; exit 2; fi
LOG="$PROJECT_DIR/logs/fsdp_resolution_remaining"
REPORT="$PROJECT_DIR/outputs/private/fsdp_resolution_remaining"
RUN="$PROJECT_DIR/runs/$NAME"
if [[ -f "$LOG/latest.pid" ]] && kill -0 "$(cat "$LOG/latest.pid")" 2>/dev/null; then
  echo "A resolution continuation is already active; no jobs stopped." >&2; exit 1
fi
COMMAND=(python -u "$PROJECT_DIR/scripts/resume_fsdp_resolution.py" --project-dir "$PROJECT_DIR"
  --original-run-dir "$ORIGINAL" --run-dir "$RUN")
case "${RESUME:-0}" in
  0) if [[ -e "$RUN" ]]; then echo "Choose a new continuation name." >&2; exit 1; fi ;;
  1) if [[ -z "${FSDP_RESOLUTION_REMAINING_NAME:-}" ]]; then echo "Resume needs original continuation name." >&2; exit 1; fi
     COMMAND+=(--resume) ;;
  *) echo "RESUME must be 0 or 1." >&2; exit 2 ;;
esac
mkdir -p "$LOG" "$REPORT"
GPU_RECORD="$LOG/$NAME.gpu.json"
export FSDP_RESOLUTION_GPU_MEMORY_GIB="${FSDP_RESOLUTION_GPU_MEMORY_GIB:-40}"
CHECK=(python "$PROJECT_DIR/scripts/resume_fsdp_resolution.py" --gpu-check-only --project-dir "$PROJECT_DIR"
  --original-run-dir "$ORIGINAL" --gpu-id "${GPU_ID:-auto}" --exclude "${EXCLUDE_GPU_IDS-7}"
  --min-free-gib "${MIN_FREE_GPU_GIB:-48}" --memory-budget-gib "$FSDP_RESOLUTION_GPU_MEMORY_GIB" --gpu-record "$GPU_RECORD")
case "${ALLOW_GPU_SHARING:-0}" in
  0) ;;
  1) CHECK+=(--allow-sharing) ;;
  *) echo "ALLOW_GPU_SHARING must be 0 or 1." >&2; exit 2 ;;
esac
SELECTED_UUID="$("${CHECK[@]}")"
export CUDA_VISIBLE_DEVICES="$SELECTED_UUID"
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${RESOLUTION_CPU_THREADS:-4}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${RESOLUTION_CPU_THREADS:-4}"
export MPLCONFIGDIR="$LOG/.mpl-cache"
mkdir -p "$MPLCONFIGDIR"
COMMAND+=(--gpu-record "$GPU_RECORD")
LOG_FILE="$LOG/$NAME.$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
nohup "${COMMAND[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG/$NAME.pid"
ln -sfn "$LOG/$NAME.pid" "$LOG/latest.pid"
ln -sfn "$LOG_FILE" "$LOG/latest.log"
ln -sfn "$RUN" "$REPORT/latest"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  if wait "$PID"; then echo "Continuation finished during startup."; else tail -80 "$LOG_FILE" >&2; exit 1; fi
fi
echo "Resolution continuation PID $PID; UUID $SELECTED_UUID. Research jobs were not stopped."
echo "Completed variants are skipped; unfinished detail resumes; tokens/combined use original recipe."
echo "Log: $LOG/latest.log"
echo "Report: $REPORT/latest/progress.json"
