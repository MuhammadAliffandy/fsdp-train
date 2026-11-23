#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
SAM="${SAM_MED3D_DIR:-$HOME/raid/Clara/SAM-Med3D}"
CHECKPOINT="${TURBO_CHECKPOINT:-$SAM/ckpt/sam_med3d_turbo.pth}"
CONTROL="${UNET_COHORT_RUN_DIR:-$PROJECT_DIR/outputs/private/unet_cohort/latest}"
NAME="${SAM_REFINE_RUN_NAME:-fsdp_private_sam_refine32_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ ! "$NAME" =~ ^fsdp_private_sam_refine32_[A-Za-z0-9._-]+$ ]]; then
  echo "SAM_REFINE_RUN_NAME must begin with fsdp_private_sam_refine32_." >&2; exit 2
fi
RUN="$PROJECT_DIR/runs/$NAME"
LOG="$PROJECT_DIR/logs/sam_refine32"
REPORT="$PROJECT_DIR/outputs/private/sam_refine32"
for FILE in "$SAM/segment_anything/build_sam3D.py" "$CHECKPOINT" "$CONTROL/config.json" "$CONTROL/summary.json" "$CONTROL/source_provenance.json"; do
  if [[ ! -f "$FILE" ]]; then echo "Required Turbo/source/data control missing: $FILE" >&2; exit 1; fi
done
COMMAND=(python -u "$PROJECT_DIR/scripts/train_sam_refine32.py"
  --sam-dir "$SAM" --checkpoint "$CHECKPOINT" --cohort-run-dir "$CONTROL" --run-dir "$RUN"
  --head-warmup-epochs "${HEAD_WARMUP_EPOCHS:-5}" --epochs "${NUM_EPOCHS:-200}" --val-every "${VAL_EVERY:-5}" --accumulation "${ACCUMULATION_STEPS:-8}"
  --sam-lr "${SAM_LEARNING_RATE:-0.00001}" --head-lr "${HEAD_LEARNING_RATE:-0.0001}"
  --auxiliary-weight "${AUXILIARY_WEIGHT:-0.25}" --precision "${PRECISION:-bf16}" --augmentation "${AUGMENTATION:-light}"
  --seed "${SEED:-2026}" --patience-epochs "${PATIENCE_EPOCHS:-50}" --min-delta "${MIN_DELTA:-0.001}" --device cuda:0)
case "${RESUME:-0}" in
  0) if [[ -e "$RUN" ]]; then echo "Run exists; choose a new name or RESUME=1." >&2; exit 1; fi ;;
  1)
    if [[ -z "${SAM_REFINE_RUN_NAME:-}" || ! -f "$RUN/hybrid_latest.pth" || -f "$RUN/summary.json" ]]; then
      echo "Resume requires original SAM_REFINE_RUN_NAME and unfinished hybrid/latest checkpoint." >&2; exit 1
    fi
    COMMAND+=(--resume)
    ;;
  *) echo "RESUME must be0 or1." >&2; exit 2 ;;
esac
mkdir -p "$LOG" "$REPORT"
if [[ -f "$LOG/latest.pid" ]] && kill -0 "$(cat "$LOG/latest.pid")" 2>/dev/null; then
  echo "A refine32 PID is active; inspect its log before launching another run." >&2; exit 1
fi
LOG_FILE="$LOG/$NAME.$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${PREPROCESS_THREADS:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$LOG/.mpl-cache}"
mkdir -p "$MPLCONFIGDIR"
nohup "${COMMAND[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG/$NAME.pid"
ln -sfn "$LOG/$NAME.pid" "$LOG/latest.pid"
ln -sfn "$LOG_FILE" "$LOG/latest.log"
ln -sfn "$RUN" "$REPORT/latest"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  if wait "$PID"; then echo "SAM refine32 finished during startup."; else tail -80 "$LOG_FILE" >&2; exit 1; fi
else
  echo "SAM Turbo + CNN32³refinement launched: PID $PID; GPU $CUDA_VISIBLE_DEVICES"
fi
echo "SAM tokens remain8³;32³is the learned CNN detail fusion.1/5oracle clicks per side."
echo "Log: $LOG/latest.log"
echo "Progress: $REPORT/latest/progress.json"
echo "Summary: $REPORT/latest/summary.json"

echo "Results archive after completion: $REPORT/latest/results.tar.gz"
