#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo 'Activate fsdp-med3d first.' >&2; exit 1; fi
LOG="$PROJECT_DIR/logs/system_diagnosis"
REPORT="$PROJECT_DIR/outputs/private/system_diagnosis"
NAME="diagnosis_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$LOG" "$REPORT"
if [[ -f "$LOG/latest.pid" ]] && kill -0 "$(cat "$LOG/latest.pid")" 2>/dev/null; then echo 'A diagnosis is already active.' >&2; exit 1; fi
COMMAND=(python -u "$PROJECT_DIR/scripts/diagnose_fsdp_system.py" --project "$PROJECT_DIR" --output-dir "$REPORT/$NAME")
if [[ -n "${FSDP_SOURCE_ROOT:-}" ]]; then COMMAND+=(--source-root "$FSDP_SOURCE_ROOT"); fi
if [[ -n "${FSDP_FOUNDATION_DIR:-}" ]]; then COMMAND+=(--foundation-run-dir "$FSDP_FOUNDATION_DIR"); fi
if [[ -n "${RESUNET_AUDIT_DIR:-}" ]]; then COMMAND+=(--resunet-run "$RESUNET_AUDIT_DIR"); fi
case "${BUILD_CANDIDATE:-1}" in 0) ;; 1) COMMAND+=(--build-candidate) ;; *) echo 'BUILD_CANDIDATE must be0or1.' >&2; exit 2 ;; esac
case "${INCLUDE_PROBE:-1}" in 0) ;; 1) COMMAND+=(--include-probe) ;; *) echo 'INCLUDE_PROBE must be0or1.' >&2; exit 2 ;; esac
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${DIAGNOSIS_CPU_THREADS:-4}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="$LOG/.mpl-cache"
export XDG_CACHE_HOME="$LOG/.cache"
mkdir -p "$MPLCONFIGDIR"
nohup "${COMMAND[@]}" > "$LOG/$NAME.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG/$NAME.pid"
ln -sfn "$LOG/$NAME.pid" "$LOG/latest.pid"
ln -sfn "$LOG/$NAME.log" "$LOG/latest.log"
ln -sfn "$REPORT/$NAME" "$REPORT/latest"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  if wait "$PID"; then echo 'Diagnosis finished during startup.'; else tail -80 "$LOG/$NAME.log" >&2; exit 1; fi
fi
echo "CPU-only diagnosis PID=$PID; no optimizer updates or test images; source artifacts retained."
echo "Log: $LOG/latest.log"
echo "Report/candidate: $REPORT/latest"
