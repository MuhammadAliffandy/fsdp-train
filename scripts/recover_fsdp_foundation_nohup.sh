#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
BASE="${FSDP_FOUNDATION_DIR:-$PROJECT_DIR/runs/fsdp_private_sam_refine32_20261123T144159Z_3745292}"
CHECK=(python -u "$PROJECT_DIR/scripts/train_fsdp_foundation.py" --foundation-run-dir "$BASE" --check-only)
if [[ -n "${FSDP_SOURCE_ROOT:-}" ]]; then CHECK+=(--source-root "$FSDP_SOURCE_ROOT"); fi
case "${FSDP_VERIFIED_CACHE:-0}" in
  0) ;;
  1) CHECK+=(--verified-cache) ;;
  *) echo "FSDP_VERIFIED_CACHE must be0or1." >&2; exit 2 ;;
esac
"${CHECK[@]}"
python -u "$PROJECT_DIR/scripts/stop_multiclass_training.py" --project-dir "$PROJECT_DIR"
exec bash "$PROJECT_DIR/scripts/launch_fsdp_foundation_nohup.sh"
