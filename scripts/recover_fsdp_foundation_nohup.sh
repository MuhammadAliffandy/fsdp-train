#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
BASE="${FSDP_FOUNDATION_DIR:-$PROJECT_DIR/runs/fsdp_private_sam_refine32_20261123T144159Z_3745292}"
python -u "$PROJECT_DIR/scripts/train_fsdp_foundation.py" --foundation-run-dir "$BASE" --check-only
python -u "$PROJECT_DIR/scripts/stop_multiclass_training.py" --project-dir "$PROJECT_DIR"
exec bash "$PROJECT_DIR/scripts/launch_fsdp_foundation_nohup.sh"
