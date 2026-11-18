#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != "fsdp-med3d" ]]; then
  echo "Activate fsdp-med3d first." >&2; exit 1
fi
COMMAND=(python -u "$PROJECT_DIR/scripts/stop_private_800.py" --project-dir "$PROJECT_DIR")
if [[ -n "${LONG_RUN_NAME:-}" ]]; then COMMAND+=(--run-name "$LONG_RUN_NAME"); fi
"${COMMAND[@]}"
unset LONG_RUN_NAME
export NUM_EPOCHS=200
export PATIENCE_EPOCHS=50
export MIN_DELTA=0.001
export RESUME=0
exec bash "$PROJECT_DIR/scripts/launch_private_long_nohup.sh"
