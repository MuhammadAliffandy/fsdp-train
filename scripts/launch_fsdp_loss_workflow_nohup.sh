#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo 'Activate fsdp-med3d first.' >&2; exit 1; fi
SOURCE="${FSDP_LOSS_SOURCE_DIR:-$PROJECT_DIR/outputs/private/fsdp_research/latest}"
SOURCE="$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$SOURCE")"
NAME="fsdp_loss_$(date -u +%Y%m%dT%H%M%SZ)_$$"
REPORT="$PROJECT_DIR/outputs/private/fsdp_loss_workflow"
LOG="$PROJECT_DIR/logs/fsdp_loss_workflow"
mkdir -p "$REPORT" "$LOG"
if [[ -f "$LOG/latest.pid" ]] && kill -0 "$(cat "$LOG/latest.pid")" 2>/dev/null; then
  echo 'A loss workflow is already active.' >&2; exit 1
fi
GPU_CHECK=(python "$PROJECT_DIR/scripts/fsdp_research_gpu.py" --gpu-id "${GPU_ID:-7}" --output "$LOG/$NAME.gpu.json"
  --min-free-gib 48 --exclude "${EXCLUDE_GPU_IDS:-0,3}" --protect-resolution-run "$PROJECT_DIR/outputs/private/fsdp_resolution/latest")
case "${ALLOW_GPU_SHARING:-0}" in
  0) ;;
  1) GPU_CHECK+=(--allow-sharing --memory-budget-gib 40 --reserve-gib 8) ;;
  *) echo 'ALLOW_GPU_SHARING must be0or1.' >&2; exit 2 ;;
esac
SELECTED_UUID="$("${GPU_CHECK[@]}")"
export CUDA_VISIBLE_DEVICES="$SELECTED_UUID"
export FSDP_RESEARCH_GPU_MEMORY_GIB=40
export OMP_NUM_THREADS="${RESEARCH_CPU_THREADS:-4}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="$LOG/.mpl-cache"
export PYTHONFAULTHANDLER=1
mkdir -p "$MPLCONFIGDIR"
COMMAND=(python -u "$PROJECT_DIR/scripts/run_fsdp_loss_workflow.py" --run-dir "$SOURCE" --output-dir "$REPORT/$NAME"
  --device cuda:0 --epochs "${TINY_EPOCHS:-100}")
if [[ -n "${FSDP_SOURCE_ROOT:-}" ]]; then COMMAND+=(--source-root "$FSDP_SOURCE_ROOT"); fi
nohup "${COMMAND[@]}" > "$LOG/$NAME.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG/$NAME.pid"
ln -sfn "$LOG/$NAME.pid" "$LOG/latest.pid"
ln -sfn "$LOG/$NAME.log" "$LOG/latest.log"
ln -sfn "$REPORT/$NAME" "$REPORT/latest"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  if wait "$PID"; then echo 'Diagnostic finished during startup.'; else tail -80 "$LOG/$NAME.log" >&2; exit 1; fi
fi
echo "Loss workflow PID=$PID; source=$SOURCE; log=$LOG/latest.log"
echo "Stages: read-only gradient audit, tiny control, tiny balanced BCE, tiny auxiliary0.1; target85%TRAIN only."
echo "Summary: $REPORT/latest/summary.json"
