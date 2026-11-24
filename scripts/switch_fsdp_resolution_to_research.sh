#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
BASE="${FSDP_FOUNDATION_DIR:-$PROJECT_DIR/runs/fsdp_private_sam_refine32_20261123T144159Z_3745292}"
for FILE in config.json summary.json hybrid_best.pth hybrid_data.pt; do
  if [[ ! -f "$BASE/$FILE" ]]; then echo "Foundation artifact missing: $BASE/$FILE" >&2; exit 1; fi
done
if [[ -f "$PROJECT_DIR/logs/fsdp_research/latest.pid" ]] && kill -0 "$(cat "$PROJECT_DIR/logs/fsdp_research/latest.pid")" 2>/dev/null; then
  echo "A research workflow is already active; no resolution process stopped." >&2; exit 1
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$"
RECORD="$PROJECT_DIR/outputs/private/stops/resolution_to_research_$STAMP.json"
python "$PROJECT_DIR/scripts/stop_fsdp_resolution.py" --project-dir "$PROJECT_DIR" --output "$RECORD"
GPU_ID="$(python - "$RECORD" <<'PY'
import json
from pathlib import Path
import sys
record = json.loads(Path(sys.argv[1]).read_text())
if not record['complete']:
    raise SystemExit('Resolution stop is incomplete; research blocked')
print(int(record['gpu']['index']))
PY
)"
export GPU_ID
export EXCLUDE_GPU_IDS=""
export ALLOW_GPU_SHARING="${ALLOW_GPU_SHARING:-1}"
export RESUME=0
export FSDP_RESEARCH_RUN_NAME="fsdp_research_switch_$STAMP"
echo "Starting research on released GPU $GPU_ID; stop record: $RECORD"
exec bash "$PROJECT_DIR/scripts/launch_fsdp_research_nohup.sh"
