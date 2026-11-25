#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FSDP_HISTORY_ROI_JITTER=1
export FSDP_MATCHED_CONTROL_DIR="${FSDP_MATCHED_CONTROL_DIR:-$PROJECT_DIR/runs/fsdp_research_history_20261125T022714Z_2766483/history}"
export FSDP_RESEARCH_RUN_NAME="${FSDP_RESEARCH_RUN_NAME:-fsdp_research_jitter_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
exec bash "$PROJECT_DIR/scripts/launch_fsdp_history_nohup.sh"
