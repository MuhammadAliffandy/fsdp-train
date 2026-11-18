#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != "fsdp-med3d" ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
SAM_DIR="${SAM_MED3D_DIR:-$HOME/raid/Clara/SAM-Med3D}"
TINY_DIR="${TINY_RUN_DIR:-$PROJECT_DIR/outputs/private/tiny_training/latest}"
V2_AUDIT="${V2_AUDIT_SUMMARY:-$PROJECT_DIR/outputs/private/best_checkpoint_audit/latest/summary.json}"
NAME="side_validation_$(date -u +%Y%m%dT%H%M%SZ)_$$"
OUTPUT="$PROJECT_DIR/outputs/private/side_validation/$NAME"
LOG_DIR="$PROJECT_DIR/logs/side_validation"
mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT")"
COMMAND=(python -u "$PROJECT_DIR/scripts/compare_side_validation.py"
  --sam-dir "$SAM_DIR" --source-root "${PRIVATE_SOURCE_ROOT:-$HOME/raid/normal case_86}"
  --data-dir "${PRIVATE_DATA_DIR:-$PROJECT_DIR/data/processed/private/fsdp}"
  --tiny-run-dir "$TINY_DIR" --turbo-checkpoint "${TURBO_CHECKPOINT:-$SAM_DIR/ckpt/sam_med3d_turbo.pth}"
  --output-dir "$OUTPUT" --seed "${SEED:-2026}" --device cuda:0)
if [[ -n "${PRIVATE_V2_RUN_DIR:-}" ]]; then
  COMMAND+=(--v2-run-dir "$PRIVATE_V2_RUN_DIR")
else
  if [[ ! -f "$V2_AUDIT" ]]; then
    echo "Previous best-checkpoint audit missing; set PRIVATE_V2_RUN_DIR to the completed v2 run." >&2; exit 1
  fi
  COMMAND+=(--v2-audit-summary "$V2_AUDIT")
fi
if [[ ! -f "$TINY_DIR/summary.json" || ! -f "$TINY_DIR/sam_model_tiny_best.pth" ]]; then
  echo "Completed tiny run missing; set TINY_RUN_DIR to its run directory." >&2; exit 1
fi
if [[ -f "$LOG_DIR/latest.pid" ]] && kill -0 "$(cat "$LOG_DIR/latest.pid")" 2>/dev/null; then
  echo "Previous comparison PID is still active; inspect its log first." >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
nohup "${COMMAND[@]}" > "$LOG_DIR/$NAME.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG_DIR/$NAME.pid"
ln -sfn "$LOG_DIR/$NAME.pid" "$LOG_DIR/latest.pid"
ln -sfn "$LOG_DIR/$NAME.log" "$LOG_DIR/latest.log"
ln -sfn "$OUTPUT" "$PROJECT_DIR/outputs/private/side_validation/latest"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  if wait "$PID"; then echo "Side validation completed during startup."; else tail -80 "$LOG_DIR/$NAME.log" >&2; exit 1; fi
else
  echo "Side validation launched: PID $PID; GPU ${CUDA_VISIBLE_DEVICES}"
fi
echo "Validation only: ten patients, twenty side ROIs, three frozen checkpoints."
echo "Log: $LOG_DIR/latest.log"
echo "Summary: $OUTPUT/summary.json"
