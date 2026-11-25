#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo 'Activate fsdp-med3d first.' >&2; exit 1; fi
NAME="${FSDP_COHORT_RUN_NAME:-fsdp_loss_cohort_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ ! "$NAME" =~ ^fsdp_loss_cohort_[A-Za-z0-9._-]+$ ]]; then echo 'Invalid cohort name.' >&2; exit 2; fi
RUN="$PROJECT_DIR/runs/$NAME"
LOG="$PROJECT_DIR/logs/fsdp_loss_cohort"
REPORT="$PROJECT_DIR/outputs/private/fsdp_loss_cohort"
mkdir -p "$LOG" "$REPORT"
if [[ -f "$LOG/latest.pid" ]] && kill -0 "$(cat "$LOG/latest.pid")" 2>/dev/null; then echo 'A cohort workflow is active.' >&2; exit 1; fi
SOURCE="${FSDP_COHORT_SOURCE_DIR:-$PROJECT_DIR/runs/fsdp_research_jitter_20261125T042908Z_3087949/history}"
DIAGNOSTIC="${FSDP_DIAGNOSTIC_DIR:-$PROJECT_DIR/outputs/private/fsdp_loss_workflow/fsdp_loss_20261125T062514Z_3438178}"
COMMAND=(python -u "$PROJECT_DIR/scripts/run_fsdp_loss_cohort.py"
  --foundation-run-dir "${FSDP_FOUNDATION_DIR:-$PROJECT_DIR/runs/fsdp_private_sam_refine32_20261123T144159Z_3745292}"
  --research-source-dir "$SOURCE" --diagnostic-dir "$DIAGNOSTIC" --run-dir "$RUN"
  --epochs "${COHORT_EPOCHS:-150}" --lr-multiplier "${LR_MULTIPLIER:-0.3}" --patience-epochs "${PATIENCE_EPOCHS:-30}"
  --accumulation "${ACCUMULATION_STEPS:-4}" --device cuda:0)
if [[ -n "${FSDP_SOURCE_ROOT:-}" ]]; then COMMAND+=(--source-root "$FSDP_SOURCE_ROOT"); fi
case "${RESUME:-0}" in
  0) if [[ -e "$RUN" ]]; then echo 'Choose a new cohort run name.' >&2; exit 1; fi ;;
  1)
    if [[ -z "${FSDP_COHORT_RUN_NAME:-}" || ! -f "$RUN/workflow_config.json" || -f "$RUN/summary.json" ]]; then
      echo 'Resume requires the original unfinished cohort run name.' >&2; exit 1
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
  if wait "$PID"; then echo 'Cohort completed during startup.'; else tail -80 "$LOG_FILE" >&2; exit 1; fi
fi
echo "FULL cohort PID=$PID; source=$SOURCE; GPU=$SELECTED_UUID; run=$NAME"
echo "Sequential arms: control then balanced;54train/10validation; cap=${COHORT_EPOCHS:-150}/arm; early stop=${PATIENCE_EPOCHS:-30}."
echo "Log: $LOG/latest.log"
echo "Summary/archive: $REPORT/latest"
