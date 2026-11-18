#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$1"
DIAGNOSTICS="$2"
PREFLIGHT_DIR="$3"
shift 3
python -u "$SCRIPT_DIR/verify_private_pipeline.py" \
  --data-dir "$DATA_DIR" --diagnostics "$DIAGNOSTICS" --output-dir "$PREFLIGHT_DIR"
exec python -u "$SCRIPT_DIR/train_private_long.py" "$@"
