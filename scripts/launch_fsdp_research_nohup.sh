#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
BASE="${FSDP_FOUNDATION_DIR:-$PROJECT_DIR/runs/fsdp_private_sam_refine32_20261123T144159Z_3745292}"
NAME="${FSDP_RESEARCH_RUN_NAME:-fsdp_research_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ ! "$NAME" =~ ^fsdp_research_[A-Za-z0-9._-]+$ ]]; then echo "Invalid research run name." >&2; exit 2; fi
RUN="$PROJECT_DIR/runs/$NAME"
LOG="$PROJECT_DIR/logs/fsdp_research"
REPORT="$PROJECT_DIR/outputs/private/fsdp_research"
for FILE in config.json summary.json hybrid_best.pth hybrid_data.pt; do
  if [[ ! -f "$BASE/$FILE" ]]; then echo "Foundation artifact missing: $BASE/$FILE" >&2; exit 1; fi
done
COMMAND=(python -u "$PROJECT_DIR/scripts/run_fsdp_research.py" --foundation-run-dir "$BASE" --run-dir "$RUN"
  --epochs "${ADDITIONAL_EPOCHS:-20}" --lr-multiplier "${LR_MULTIPLIER:-0.2}"
  --patience-epochs "${PATIENCE_EPOCHS:-50}" --boundary-weight "${BOUNDARY_WEIGHT:-0.05}"
  --boundary-clip-mm "${BOUNDARY_CLIP_MM:-5}" --device cuda:0)
if [[ -n "${FSDP_SOURCE_ROOT:-}" ]]; then COMMAND+=(--source-root "$FSDP_SOURCE_ROOT"); fi
case "${FSDP_VERIFIED_CACHE:-1}" in
  0) ;;
  1) COMMAND+=(--verified-cache) ;;
  *) echo "FSDP_VERIFIED_CACHE must be0or1." >&2; exit 2 ;;
esac
case "${RESUME:-0}" in
  0) if [[ -e "$RUN" ]]; then echo "Research run exists; choose a new name." >&2; exit 1; fi ;;
  1)
    if [[ -z "${FSDP_RESEARCH_RUN_NAME:-}" || ! -f "$RUN/workflow_config.json" || -f "$RUN/summary.json" ]]; then
      echo "Resume needs the original unfinished research run name." >&2; exit 1
    fi
    COMMAND+=(--resume) ;;
  *) echo "RESUME must be0or1." >&2; exit 2 ;;
esac
mkdir -p "$LOG" "$REPORT"
if [[ -f "$LOG/latest.pid" ]] && kill -0 "$(cat "$LOG/latest.pid")" 2>/dev/null; then
  echo "A research workflow is already active." >&2; exit 1
fi
GPU_RECORD="$LOG/$NAME.gpu.json"
SELECTED_UUID="$(python "$PROJECT_DIR/scripts/fsdp_research_gpu.py" --gpu-id "${GPU_ID:-1}" --output "$GPU_RECORD")"
COMMAND+=(--gpu-record "$GPU_RECORD")
export CUDA_VISIBLE_DEVICES="$SELECTED_UUID"
export OMP_NUM_THREADS="${RESEARCH_CPU_THREADS:-4}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${RESEARCH_CPU_THREADS:-4}"
export MPLCONFIGDIR="$LOG/.mpl-cache"
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
  if wait "$PID"; then echo "Research workflow finished during startup."; else tail -80 "$LOG_FILE" >&2; exit 1; fi
fi
echo "Research workflow PID$PID; physical GPU${GPU_ID:-1}; UUID$SELECTED_UUID"
echo "Sequential arms: control/history/boundary/refiner; ${ADDITIONAL_EPOCHS:-20}additional epochs each."
echo "Log: $LOG/latest.log"
echo "Progress: $REPORT/latest/progress.json"
echo "Archive: $REPORT/latest/results.tar.gz"
