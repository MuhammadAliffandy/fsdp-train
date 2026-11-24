#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ENCODER_GRID=16
export DETAIL_GRIDS=64
export MULTICLASS_CHANNEL=multiclass_tokens16
export MULTICLASS_RUN_NAME="${MULTICLASS_RUN_NAME:-fsdp_multiclass_tokens16_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
exec bash "$PROJECT_DIR/scripts/launch_multiclass_nohup.sh"
