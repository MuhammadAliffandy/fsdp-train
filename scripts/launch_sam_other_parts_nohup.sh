#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo 'Activate fsdp-med3d first.' >&2; exit 2; fi
LOG="$PROJECT_DIR/logs/sam_other_parts"
mkdir -p "$LOG"
if [[ -f "$LOG/latest.pid" ]] && kill -0 "$(cat "$LOG/latest.pid")" 2>/dev/null; then
  echo 'An other-parts SAM trainer is already active.' >&2; exit 2
fi
NAME="sam_other_parts_$(date -u +%Y%m%dT%H%M%SZ)_$$"
RUN="$PROJECT_DIR/runs/$NAME"
AUDIT="$LOG/$NAME.audit.json"
SOURCE="${FSDP_PARTS_SOURCE_DIR:-$PROJECT_DIR/runs/sam_augmentation_20261126T084626Z_4096150/affine}"
AUDIT_ARGS=(--project "$PROJECT_DIR" --source-run "$SOURCE" --output "$AUDIT")
if [[ -n "${FSDP_MULTICLASS_PREPARED:-}" ]]; then AUDIT_ARGS+=(--prepared "$FSDP_MULTICLASS_PREPARED"); fi
# CPU audit runs first. Missing original multiclass labels never trigger GPU training.
python -u scripts/train_sam_other_parts.py audit "${AUDIT_ARGS[@]}"
GPU_ARGS=(--gpu-id "${GPU_ID:-auto}" --output "$LOG/$NAME.gpu.json" --min-free-gib 48
  --exclude "${EXCLUDE_GPU_IDS:-0}" --memory-budget-gib 24 --reserve-gib 16)
if [[ "${ALLOW_GPU_SHARING:-0}" == 1 ]]; then GPU_ARGS+=(--allow-sharing); fi
CUDA_VISIBLE_DEVICES="$(python scripts/fsdp_research_gpu.py "${GPU_ARGS[@]}")"
export CUDA_VISIBLE_DEVICES
export FSDP_RESEARCH_GPU_MEMORY_GIB=24 OMP_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 PYTHONFAULTHANDLER=1
export MPLCONFIGDIR="$LOG/.mpl-cache"
mkdir -p "$MPLCONFIGDIR"
LOG_FILE="$LOG/$NAME.log"
nohup python -u scripts/train_sam_other_parts.py train --audit "$AUDIT" --run-dir "$RUN" \
  --epochs "${PARTS_EPOCHS:-20}" --patience "${PARTS_PATIENCE:-10}" --device cuda:0 > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG/$NAME.pid"
ln -sfn "$LOG/$NAME.pid" "$LOG/latest.pid"
ln -sfn "$LOG_FILE" "$LOG/latest.log"
echo "$RUN" > "$LOG/latest_run.txt"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then wait "$PID" || { tail -n 60 "$LOG_FILE"; exit 1; }; fi
echo "Full learned SAM hybrid;15non-FSDPstructures; GTROI+5points/class; PID=$PID; run=$RUN"
echo "Log: $LOG/latest.log"
