#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
BASE="${FSDP_FOUNDATION_DIR:-$PROJECT_DIR/runs/fsdp_private_sam_refine32_20261123T144159Z_3745292}"
NAME="${FSDP_FOUNDATION_RUN_NAME:-fsdp_private_foundation_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ ! "$NAME" =~ ^fsdp_private_foundation_[A-Za-z0-9._-]+$ ]]; then echo "Invalid continuation run name." >&2; exit 2; fi
RUN="$PROJECT_DIR/runs/$NAME"
LOG="$PROJECT_DIR/logs/fsdp_foundation"
REPORT="$PROJECT_DIR/outputs/private/fsdp_foundation"
for FILE in config.json summary.json hybrid_best.pth hybrid_data.pt; do
  if [[ ! -f "$BASE/$FILE" ]]; then echo "Foundation artifact missing: $BASE/$FILE" >&2; exit 1; fi
done
COMMAND=(python -u "$PROJECT_DIR/scripts/train_fsdp_foundation.py"
  --foundation-run-dir "$BASE" --run-dir "$RUN" --epochs "${ADDITIONAL_EPOCHS:-100}"
  --lr-multiplier "${LR_MULTIPLIER:-0.2}" --patience-epochs "${PATIENCE_EPOCHS:-50}" --device cuda:0)
if [[ -n "${FSDP_EXCLUDE_TRAIN_ALIASES:-}" ]]; then
  IFS=',' read -r -a EXCLUDED_ALIASES <<< "$FSDP_EXCLUDE_TRAIN_ALIASES"
  for ALIAS in "${EXCLUDED_ALIASES[@]}"; do
    if [[ ! "$ALIAS" =~ ^train_[0-9]{2}$ ]]; then echo "Invalid training alias exclusion: $ALIAS" >&2; exit 2; fi
    COMMAND+=(--exclude-train-alias "$ALIAS")
  done
fi
if [[ -n "${FSDP_SOURCE_ROOT:-}" ]]; then COMMAND+=(--source-root "$FSDP_SOURCE_ROOT"); fi
case "${FSDP_VERIFIED_CACHE:-0}" in
  0) ;;
  1) COMMAND+=(--verified-cache) ;;
  *) echo "FSDP_VERIFIED_CACHE must be0or1." >&2; exit 2 ;;
esac
case "${RESUME:-0}" in
  0) if [[ -e "$RUN" ]]; then echo "Run exists; choose a new name." >&2; exit 1; fi ;;
  1)
    if [[ -z "${FSDP_FOUNDATION_RUN_NAME:-}" || ! -f "$RUN/hybrid_latest.pth" || -f "$RUN/summary.json" ]]; then
      echo "Resume requires original run name and an unfinished latest checkpoint." >&2; exit 1
    fi
    COMMAND+=(--resume) ;;
  *) echo "RESUME must be0or1." >&2; exit 2 ;;
esac
mkdir -p "$LOG" "$REPORT"
if [[ -f "$LOG/latest.pid" ]] && kill -0 "$(cat "$LOG/latest.pid")" 2>/dev/null; then
  echo "A foundation continuation is already active." >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${PREPROCESS_THREADS:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$LOG/.mpl-cache}"
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
  if wait "$PID"; then echo "Continuation finished during startup."; else tail -80 "$LOG_FILE" >&2; exit 1; fi
else
  echo "FSDP foundation continuation launched: PID$PID; GPU$CUDA_VISIBLE_DEVICES"
fi
echo "First: reproduce foundation validation before updates. Excluded aliases are omitted only from continuation training; cache/manifest remain unchanged."
echo "Log: $LOG/latest.log"
echo "Progress: $REPORT/latest/progress.json"
echo "Final archive: $REPORT/latest/results.tar.gz"
