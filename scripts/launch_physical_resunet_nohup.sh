#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo 'Activate fsdp-med3d first.' >&2; exit 1; fi
NAME="${RESUNET_RUN_NAME:-physical_resunet_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ ! "$NAME" =~ ^physical_resunet_[A-Za-z0-9._-]+$ ]]; then echo 'Invalid run name.' >&2; exit 2; fi
RUN="$PROJECT_DIR/runs/$NAME"
LOG="$PROJECT_DIR/logs/physical_resunet"
REPORT="$PROJECT_DIR/outputs/private/physical_resunet"
mkdir -p "$LOG" "$REPORT"
if [[ -f "$LOG/latest.pid" ]] && kill -0 "$(cat "$LOG/latest.pid")" 2>/dev/null; then echo 'A ResUNet job is active.' >&2; exit 1; fi
COMMAND=(python -u "$PROJECT_DIR/scripts/train_physical_resunet.py" --run-dir "$RUN"
  --foundation-run-dir "${FSDP_FOUNDATION_DIR:-$PROJECT_DIR/runs/fsdp_private_sam_refine32_20261123T144159Z_3745292}"
  --sam-comparison-dir "${SAM_COMPARISON_DIR:-$PROJECT_DIR/runs/fsdp_loss_cohort_20261125T070213Z_3564637/control}"
  --epochs "${RESUNET_EPOCHS:-300}" --patience "${PATIENCE_EPOCHS:-60}" --lr "${RESUNET_LR:-0.0003}"
  --accumulation "${ACCUMULATION_STEPS:-2}" --device cuda:0)
if [[ -n "${FSDP_SOURCE_ROOT:-}" ]]; then COMMAND+=(--source-root "$FSDP_SOURCE_ROOT"); fi
case "${RESUME:-0}" in
  0) if [[ -e "$RUN" ]]; then echo 'Choose a new run name.' >&2; exit 1; fi ;;
  1)
    if [[ -z "${RESUNET_RUN_NAME:-}" || ! -f "$RUN/resunet_latest.pth" || -f "$RUN/summary.json" ]]; then
      echo 'Resume requires the original unfinished ResUNet run name.' >&2; exit 1
    fi
    COMMAND+=(--resume) ;;
  *) echo 'RESUME must be0or1.' >&2; exit 2 ;;
esac
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
LOG_FILE="$LOG/$NAME.$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
nohup "${COMMAND[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG/$NAME.pid"
ln -sfn "$LOG/$NAME.pid" "$LOG/latest.pid"
ln -sfn "$LOG_FILE" "$LOG/latest.log"
ln -sfn "$RUN" "$REPORT/latest"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  if wait "$PID"; then echo 'Baseline finished during startup.'; else tail -80 "$LOG_FILE" >&2; exit 1; fi
fi
echo "ResUNet FULL54train/10val PID=$PID; GPU=$SELECTED_UUID; run=$NAME;0oracle clicks; GT-assisted physical ROI."
echo "Log: $LOG/latest.log"
echo "Summary: $REPORT/latest/summary.json"
