#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != "fsdp-med3d" ]]; then echo "Activate fsdp-med3d first." >&2; exit 1; fi
NAME="expert_agreement_$(date -u +%Y%m%dT%H%M%SZ)_$$"
OUTPUT="$PROJECT_DIR/outputs/private/expert_agreement/$NAME"
LOG_DIR="$PROJECT_DIR/logs/expert_agreement"
mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT")"
COMMAND=(python -u "$PROJECT_DIR/scripts/audit_expert_agreement.py"
  --root "${PRIVATE_SOURCE_ROOT:-$HOME/raid/normal case_86}"
  --manifest "${PRIVATE_DATA_DIR:-$PROJECT_DIR/data/processed/private/fsdp}/manifest.json"
  --output-dir "$OUTPUT")
if [[ -n "${EXPERT_MAPPING_JSON:-}" ]]; then
  COMMAND+=(--mapping-json "$EXPERT_MAPPING_JSON")
elif [[ "${EXPERT_IDS_CONFIRMED:-0}" == "1" ]]; then
  COMMAND+=(--confirm-shared-label-ids)
else
  echo "Expert mapping not confirmed: inventory-only mode (no agreement Dice yet)."
fi
nohup "${COMMAND[@]}" > "$LOG_DIR/$NAME.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG_DIR/$NAME.pid"
ln -sfn "$LOG_DIR/$NAME.log" "$LOG_DIR/latest.log"
ln -sfn "$OUTPUT" "$PROJECT_DIR/outputs/private/expert_agreement/latest"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  if wait "$PID"; then
    echo "Expert audit completed during startup."
  else
    tail -80 "$LOG_DIR/$NAME.log" >&2; exit 1
  fi
else
  echo "Expert audit launched: PID $PID (CPU only)."
fi
echo "Log: $LOG_DIR/latest.log"
echo "Summary: $OUTPUT/summary.json"
