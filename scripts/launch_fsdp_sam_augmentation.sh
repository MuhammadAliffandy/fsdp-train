#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo 'Activate fsdp-med3d first.' >&2; exit 2; fi
LOG="$PROJECT_DIR/logs/fsdp_sam_augmentation"
mkdir -p "$LOG"
if [[ -f "$LOG/latest.pid" ]] && kill -0 "$(cat "$LOG/latest.pid")" 2>/dev/null; then
  echo 'SAM augmentation workflow already active.' >&2; exit 2
fi
SOURCE="${FSDP_AUG_SOURCE_DIR:-$PROJECT_DIR/runs/fsdp_loss_cohort_20261125T070213Z_3564637/control}"
if [[ ! -f "$SOURCE/research_best.pth" ]]; then echo "Missing source checkpoint: $SOURCE" >&2; exit 2; fi
NAME="sam_augmentation_$(date -u +%Y%m%dT%H%M%SZ)_$$"
RUN="$PROJECT_DIR/runs/$NAME"
GPU_ARGS=(--gpu-id "${GPU_ID:-auto}" --output "$LOG/$NAME.gpu.json" --min-free-gib 48
  --exclude "${EXCLUDE_GPU_IDS:-0}" --memory-budget-gib 24 --reserve-gib 16)
if [[ "${ALLOW_GPU_SHARING:-0}" == 1 ]]; then GPU_ARGS+=(--allow-sharing); fi
export CUDA_VISIBLE_DEVICES="$(python scripts/fsdp_research_gpu.py "${GPU_ARGS[@]}")"
export FSDP_RESEARCH_GPU_MEMORY_GIB=24
export FSDP_AUG_SOURCE_DIR="$SOURCE"
export FSDP_FOUNDATION_DIR="${FSDP_FOUNDATION_DIR:-$PROJECT_DIR/runs/fsdp_private_sam_refine32_20261123T144159Z_3745292}"
export FSDP_SOURCE_ROOT="${FSDP_SOURCE_ROOT:-$HOME/raid/normal case_86}"
export OMP_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 PYTHONFAULTHANDLER=1
LOG_FILE="$LOG/$NAME.log"
nohup bash scripts/run_fsdp_sam_augmentation.sh "$RUN" > "$LOG_FILE" 2>&1 < /dev/null &
echo "$!" > "$LOG/$NAME.pid"
ln -sfn "$LOG/$NAME.pid" "$LOG/latest.pid"
ln -sfn "$LOG_FILE" "$LOG/latest.log"
echo "$RUN" > "$LOG/latest_run.txt"
echo "SAM control then affine queued; log=$LOG/latest.log; run=$RUN"
