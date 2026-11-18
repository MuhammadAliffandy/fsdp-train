#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMPARE_STAGES=1
exec bash "$SCRIPT_DIR/launch_validation_nohup.sh"
