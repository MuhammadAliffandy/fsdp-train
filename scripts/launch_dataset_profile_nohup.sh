#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
RUN="${UNET_COHORT_RUN_DIR:-$PROJECT_DIR/outputs/private/unet_cohort/latest}"
NAME="dataset_profile_$(date -u +%Y%m%dT%H%M%SZ)_$$"
LOG="$PROJECT_DIR/logs/dataset_profile"
REPORT="$PROJECT_DIR/outputs/private/dataset_profile"
if [[ ! -f "$RUN/summary.json" || ! -f "$RUN/config.json" || ! -f "$RUN/source_provenance.json" ]]; then
  echo "Completed cohort summary/config/source provenance required." >&2; exit 1
fi
mkdir -p "$LOG" "$REPORT"
if [[ -f "$LOG/latest.pid" ]] && kill -0 "$(cat "$LOG/latest.pid")" 2>/dev/null; then
  echo "A dataset profile PID is active; inspect its log first." >&2; exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
nohup python -u "$PROJECT_DIR/scripts/profile_fsdp_training_data.py" --run-dir "$RUN" --output-dir "$REPORT/$NAME" \
  > "$LOG/$NAME.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG/$NAME.pid"
ln -sfn "$LOG/$NAME.pid" "$LOG/latest.pid"
ln -sfn "$LOG/$NAME.log" "$LOG/latest.log"
ln -sfn "$REPORT/$NAME" "$REPORT/latest"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  if wait "$PID"; then echo "Dataset profile completed during startup."; else tail -80 "$LOG/$NAME.log" >&2; exit 1; fi
else
  echo "Train-only dataset profile launched: PID $PID; CPU; no training or dataset changes."
fi
echo "Log: $LOG/latest.log"
echo "Compact report: $REPORT/latest/overview.json"
