#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
SAM="${SAM_MED3D_DIR:-$HOME/raid/Clara/SAM-Med3D}"
CHECKPOINT="${TURBO_CHECKPOINT:-$SAM/ckpt/sam_med3d_turbo.pth}"
CONTROL="${UNET_COHORT_RUN_DIR:-$PROJECT_DIR/outputs/private/unet_cohort/latest}"
NAME="${MULTICLASS_RUN_NAME:-fsdp_multiclass_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ ! "$NAME" =~ ^fsdp_multiclass_[A-Za-z0-9._-]+$ ]]; then echo "Invalid MULTICLASS_RUN_NAME." >&2; exit 2; fi
RUN="$PROJECT_DIR/runs/$NAME"
LOG="$PROJECT_DIR/logs/multiclass"
REPORT="$PROJECT_DIR/outputs/private/multiclass"
for FILE in "$SAM/segment_anything/build_sam3D.py" "$CHECKPOINT" "$CONTROL/config.json" "$CONTROL/summary.json" "$CONTROL/source_provenance.json"; do
  if [[ ! -f "$FILE" ]]; then echo "Required source/checkpoint/control missing: $FILE" >&2; exit 1; fi
done
read -r -a GRIDS <<< "${DETAIL_GRIDS:-64 32}"
COMMAND=(python -u "$PROJECT_DIR/scripts/train_multiclass_turbo.py"
  --sam-dir "$SAM" --checkpoint "$CHECKPOINT" --cohort-run-dir "$CONTROL" --run-dir "$RUN"
  --detail-grids "${GRIDS[@]}" --epochs "${NUM_EPOCHS:-200}" --val-every "${VAL_EVERY:-10}"
  --patches-per-patient "${PATCHES_PER_PATIENT:-4}" --accumulation "${ACCUMULATION_STEPS:-4}"
  --head-warmup-epochs "${HEAD_WARMUP_EPOCHS:-5}" --encoder-lr "${ENCODER_LEARNING_RATE:-0.00001}"
  --head-lr "${HEAD_LEARNING_RATE:-0.0001}" --patience-epochs "${PATIENCE_EPOCHS:-50}"
  --min-delta "${MIN_DELTA:-0.001}" --precision "${PRECISION:-bf16}" --device cuda:0 --seed "${SEED:-2026}")
case "${RESUME:-0}" in
  0) if [[ -e "$RUN" ]]; then echo "Run exists; use new timestamped name." >&2; exit 1; fi ;;
  1)
    if [[ -z "${MULTICLASS_RUN_NAME:-}" || ! -f "$RUN/preprocessing/prepared.json" || -f "$RUN/summary.json" ]]; then
      echo "Resume requires original MULTICLASS_RUN_NAME and completed preprocessing of unfinished run." >&2; exit 1
    fi
    COMMAND+=(--resume) ;;
  *) echo "RESUME must be0 or1." >&2; exit 2 ;;
esac
mkdir -p "$LOG" "$REPORT"
if [[ -f "$LOG/latest.pid" ]] && kill -0 "$(cat "$LOG/latest.pid")" 2>/dev/null; then
  echo "A multiclass process is active; inspect logs before launching another." >&2; exit 1
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
  if wait "$PID"; then echo "Workflow finished during startup."; else tail -80 "$LOG_FILE" >&2; exit 1; fi
else
  echo "All-structure workflow launched: PID $PID; GPU $CUDA_VISIBLE_DEVICES; learned detail grids ${GRIDS[*]}"
fi
echo "GT-all-class field; original Turbo image encoder;18class output; no clicks;20test untouched."
echo "Log: $LOG/latest.log"
echo "Progress: $REPORT/latest/progress.json"
echo "Final summary: $REPORT/latest/summary.json"
echo "Archive after both experiments: $REPORT/latest/results.tar.gz"
