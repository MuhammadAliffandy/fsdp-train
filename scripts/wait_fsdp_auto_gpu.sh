#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_RECORD="${AUTO_GPU_RECORD:?Set AUTO_GPU_RECORD}"
WAIT_SECONDS="${AUTO_GPU_POLL_SECONDS:-120}"
MAX_WAIT_SECONDS="${AUTO_GPU_MAX_WAIT_SECONDS:-86400}"
if [[ ! "$WAIT_SECONDS" =~ ^[0-9]+$ || ! "$MAX_WAIT_SECONDS" =~ ^[0-9]+$ ]] \
   || (( WAIT_SECONDS < 30 || MAX_WAIT_SECONDS < WAIT_SECONDS )); then
  echo 'GPU wait settings require poll >=30 seconds and max wait >= poll.' >&2
  exit 2
fi
STARTED_AT="$(date +%s)"
echo "Waiting up to $MAX_WAIT_SECONDS seconds for an idle GPU with at least 48 GiB free." \
  "Request=${GPU_ID:-auto}; excluded=${EXCLUDE_GPU_IDS:-0}." \
  'No other processes will be stopped.'
while true; do
  STATUS=0
  SELECTED_UUID="$(python "$PROJECT_DIR/scripts/fsdp_research_gpu.py" \
    --gpu-id "${GPU_ID:-auto}" --output "$GPU_RECORD" --min-free-gib 48 \
    --exclude "${EXCLUDE_GPU_IDS:-0}" 2>"${GPU_RECORD}.stderr")" || STATUS=$?
  if (( STATUS == 0 )); then
    export CUDA_VISIBLE_DEVICES="$SELECTED_UUID"
    echo "GPU selected at $(date -u +%Y-%m-%dT%H:%M:%SZ): $SELECTED_UUID"
    cat "${GPU_RECORD}.stderr"
    exec bash "$PROJECT_DIR/scripts/run_fsdp_auto_nnunet.sh"
  fi
  if (( STATUS != 2 )) || ! python - "$GPU_RECORD" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file() or json.loads(p.read_text()).get('ready') is not False:
    raise SystemExit(1)
PY
  then
    cat "${GPU_RECORD}.stderr" >&2
    exit "$STATUS"
  fi
  NOW="$(date +%s)"
  if (( NOW - STARTED_AT >= MAX_WAIT_SECONDS )); then
    echo 'No eligible GPU became available before the waiting limit.' >&2
    cat "${GPU_RECORD}.stderr" >&2
    exit 2
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ): all eligible GPUs are occupied; checking again in $WAIT_SECONDS seconds."
  sleep "$WAIT_SECONDS"
done
