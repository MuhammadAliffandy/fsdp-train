#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FSDP_RESEARCH_SOURCE_DIR="${FSDP_RESEARCH_SOURCE_DIR:-$PROJECT_DIR/runs/fsdp_research_gpu7_20261124T132503Z_3250370/history}"
export ADDITIONAL_EPOCHS="${ADDITIONAL_EPOCHS:-100}"
export LR_MULTIPLIER="${LR_MULTIPLIER:-0.1}"
export PATIENCE_EPOCHS="${PATIENCE_EPOCHS:-25}"
export FSDP_RESEARCH_RUN_NAME="${FSDP_RESEARCH_RUN_NAME:-fsdp_research_history_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
exec bash "$PROJECT_DIR/scripts/launch_fsdp_research_nohup.sh"
