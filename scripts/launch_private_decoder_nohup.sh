#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != "fsdp-med3d" ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
SAM_DIR="${SAM_MED3D_DIR:-$HOME/raid/Clara/SAM-Med3D}"
DATA_DIR="${PRIVATE_DATA_DIR:-$PROJECT_DIR/data/processed/private/fsdp}"
BASELINE="${SIDE_BASELINE_DIR:-$PROJECT_DIR/outputs/private/side_validation/latest}"
CHECKPOINT="${TURBO_CHECKPOINT:-$SAM_DIR/ckpt/sam_med3d_turbo.pth}"
V3_DIR="${PRIVATE_V3_RUN_DIR:-$PROJECT_DIR/outputs/private/side_training/latest}"
V4_DIR="${PRIVATE_V4_RUN_DIR:-$PROJECT_DIR/outputs/private/zoom_training/latest}"
NAME="${DECODER_RUN_NAME:-fsdp_private_decoder_v5_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ ! "$NAME" =~ ^fsdp_private_decoder_v5_[A-Za-z0-9._-]+$ ]]; then echo "DECODER_RUN_NAME must begin with fsdp_private_decoder_v5_." >&2; exit 2; fi
RUN_DIR="$PROJECT_DIR/runs/$NAME"
LOG_DIR="$PROJECT_DIR/logs/decoder_training"
REPORT_DIR="$PROJECT_DIR/outputs/private/decoder_training"
if [[ ! -f "$CHECKPOINT" || ! -f "$DATA_DIR/manifest.json" || ! -f "$BASELINE/summary.json" || ! -f "$BASELINE/protocol.json" ]]; then
  echo "Turbo weights, frozen data or completed side-validation baseline missing." >&2; exit 1
fi
if [[ ! -f "$V3_DIR/summary.json" || ! -f "$V3_DIR/sam_model_side_val_best.pth" ]]; then
  echo "Completed v3 run missing; set PRIVATE_V3_RUN_DIR." >&2; exit 1
fi
if [[ ! -f "$V4_DIR/summary.json" || ! -f "$V4_DIR/sam_model_side_val_best.pth" ]]; then
  echo "Completed v4 run missing; set PRIVATE_V4_RUN_DIR." >&2; exit 1
fi
COMMAND=(python -u "$PROJECT_DIR/scripts/train_private_decoder.py"
  --sam-dir "$SAM_DIR" --source-root "${PRIVATE_SOURCE_ROOT:-$HOME/raid/normal case_86}"
  --v3-run-dir "$V3_DIR" --v4-run-dir "$V4_DIR" --data-dir "$DATA_DIR" --checkpoint "$CHECKPOINT" --baseline-dir "$BASELINE" --run-dir "$RUN_DIR"
  --epochs "${NUM_EPOCHS:-200}" --val-every "${VAL_EVERY:-5}" --accumulation "${ACCUMULATION_STEPS:-8}"
  --lr "${LEARNING_RATE:-2e-5}" --seed "${SEED:-2026}"
  --patience-epochs "${PATIENCE_EPOCHS:-50}" --min-delta "${MIN_DELTA:-0.001}" --device cuda:0)
case "${RESUME:-0}" in
  0) if [[ -e "$RUN_DIR" ]]; then echo "Run exists; choose a new name or RESUME=1." >&2; exit 1; fi ;;
  1)
    if [[ -z "${DECODER_RUN_NAME:-}" || ! -f "$RUN_DIR/sam_model_side_latest.pth" ]]; then
      echo "Resume requires DECODER_RUN_NAME and its latest checkpoint." >&2; exit 1
    fi
    COMMAND+=(--resume)
    ;;
  *) echo "RESUME must be 0 or 1." >&2; exit 2 ;;
esac
mkdir -p "$LOG_DIR" "$REPORT_DIR"
if [[ -f "$LOG_DIR/latest.pid" ]] && kill -0 "$(cat "$LOG_DIR/latest.pid")" 2>/dev/null; then
  echo "Another decoder-training PID is still active; inspect its log before starting a second run." >&2; exit 1
fi
LOG_FILE="$LOG_DIR/$NAME.$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
nohup "${COMMAND[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG_DIR/$NAME.pid"
ln -sfn "$LOG_DIR/$NAME.pid" "$LOG_DIR/latest.pid"
ln -sfn "$LOG_FILE" "$LOG_DIR/latest.log"
ln -sfn "$RUN_DIR" "$REPORT_DIR/latest"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  if wait "$PID"; then echo "Decoder training completed during startup."; else tail -80 "$LOG_FILE" >&2; exit 1; fi
else
  echo "Decoder controls/training launched: PID $PID; GPU ${CUDA_VISIBLE_DEVICES}"
fi
echo "Run: $NAME"
echo "Training: 54 patients / 108 side ROIs; validation: 10 patients / 20 side ROIs."
echo "Log: $LOG_DIR/latest.log"
echo "Progress: $REPORT_DIR/latest/progress.json"
echo "Summary: $REPORT_DIR/latest/summary.json"
