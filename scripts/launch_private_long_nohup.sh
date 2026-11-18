#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM_MED3D_DIR="${SAM_MED3D_DIR:-$HOME/raid/Clara/SAM-Med3D}"
DATA_DIR="${PRIVATE_DATA_DIR:-$PROJECT_DIR/data/processed/private/fsdp}"
DIAGNOSTICS="${DIAGNOSTICS_DIR:-$PROJECT_DIR/outputs/private/diagnostics/latest}"
CHECKPOINT="${INITIAL_CHECKPOINT:-$SAM_MED3D_DIR/ckpt/sam_med3d_turbo.pth}"
RUN_NAME="${LONG_RUN_NAME:-fsdp_private_v2_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "fsdp-med3d" ]]; then
  echo "Activate fsdp-med3d first." >&2; exit 1
fi
if [[ ! "$RUN_NAME" =~ ^fsdp_private_v2_[A-Za-z0-9._-]+$ ]]; then
  echo "LONG_RUN_NAME must begin with fsdp_private_v2_." >&2; exit 2
fi
if [[ ! -f "$DIAGNOSTICS/validation_cases.csv" || ! -f "$CHECKPOINT" ]]; then
  echo "Saved validation diagnostics or initial weights are missing." >&2; exit 1
fi
RUN_DIR="$PROJECT_DIR/runs/$RUN_NAME"
LOG_DIR="$PROJECT_DIR/logs/training"
PREFLIGHT="$PROJECT_DIR/outputs/private/preflight/$RUN_NAME-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$LOG_DIR" "$(dirname "$PREFLIGHT")"
case "${RESUME:-0}" in
  0)
    if [[ -e "$RUN_DIR" ]]; then echo "Run already exists; use RESUME=1 and LONG_RUN_NAME." >&2; exit 1; fi
    ;;
  1)
    if [[ -z "${LONG_RUN_NAME:-}" || ! -f "$RUN_DIR/sam_model_latest.pth" ]]; then
      echo "Resume requires LONG_RUN_NAME and its latest checkpoint." >&2; exit 1
    fi
    ;;
  *) echo "RESUME must be 0 or 1." >&2; exit 2 ;;
esac
PID_FILE="$LOG_DIR/$RUN_NAME.pid"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Run is already active." >&2; exit 1
fi
LOG_FILE="$LOG_DIR/$RUN_NAME.$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
COMMAND=(bash "$PROJECT_DIR/scripts/run_private_long_worker.sh" "$DATA_DIR" "$DIAGNOSTICS" "$PREFLIGHT"
  --sam-dir "$SAM_MED3D_DIR" --data-dir "$DATA_DIR" --checkpoint "$CHECKPOINT" --run-dir "$RUN_DIR"
  --epochs "${NUM_EPOCHS:-800}" --val-every "${VAL_EVERY:-10}" --accumulation "${ACCUMULATION_STEPS:-8}"
  --clicks "${TRAIN_CLICKS:-5}" --lr "${LEARNING_RATE:-2e-5}" --seed "${SEED:-2026}" --device cuda:0)
if [[ "${RESUME:-0}" == "1" ]]; then COMMAND+=(--resume); fi
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
nohup "${COMMAND[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"
ln -sfn "$LOG_FILE" "$LOG_DIR/latest.log"
ln -sfn "$PREFLIGHT" "$PROJECT_DIR/outputs/private/preflight/latest"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then tail -80 "$LOG_FILE" >&2; exit 1; fi
echo "Preflight/training pipeline launched: PID $PID"
echo "Run name: $RUN_NAME"
echo "Run directory: $RUN_DIR"
echo "Log: $LOG_FILE"
echo "Monitor: tail -n 80 -F '$LOG_FILE'"
