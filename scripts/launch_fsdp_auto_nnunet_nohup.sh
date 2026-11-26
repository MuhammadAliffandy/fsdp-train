#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then
  echo 'Activate fsdp-med3d first.' >&2; exit 2
fi
NAME="auto_nnunet_$(date -u +%Y%m%dT%H%M%SZ)_$$"
LOG="$PROJECT_DIR/logs/fsdp_auto_nnunet"
mkdir -p "$LOG" "$PROJECT_DIR/runs"
if [[ -f "$LOG/latest.pid" ]] && kill -0 "$(cat "$LOG/latest.pid")" 2>/dev/null; then
  echo 'An automatic nnU-Net job is already active.' >&2; exit 1
fi
python -c 'import nnunetv2; from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_100epochs'
for REQUIRED_COMMAND in nnUNetv2_plan_and_preprocess nnUNetv2_train nnUNetv2_predict; do
  command -v "$REQUIRED_COMMAND" >/dev/null || { echo "Missing nnU-Net command: $REQUIRED_COMMAND" >&2; exit 2; }
done
export AUTO_GPU_RECORD="$LOG/$NAME.gpu.json"
export AUTO_RUN_DIR="$PROJECT_DIR/runs/$NAME"
export FSDP_FOUNDATION_DIR="${FSDP_FOUNDATION_DIR:-$PROJECT_DIR/runs/fsdp_private_sam_refine32_20261123T144159Z_3745292}"
export FSDP_SOURCE_ROOT="${FSDP_SOURCE_ROOT:-$HOME/raid/normal case_86}"
if [[ ! -f "$FSDP_FOUNDATION_DIR/hybrid_data.pt" || ! -d "$FSDP_SOURCE_ROOT/imagesTr" ]]; then
  echo 'Foundation cache or CT source folder is unavailable.' >&2; exit 2
fi
export OMP_NUM_THREADS="${AUTO_CPU_THREADS:-4}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$OMP_NUM_THREADS"
export PYTHONFAULTHANDLER=1
LOG_FILE="$LOG/$NAME.log"
nohup bash "$PROJECT_DIR/scripts/wait_fsdp_auto_gpu.sh" > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG/$NAME.pid"
ln -sfn "$LOG/$NAME.pid" "$LOG/latest.pid"
ln -sfn "$LOG_FILE" "$LOG/latest.log"
echo "$AUTO_RUN_DIR" > "$LOG/latest_run.txt"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  wait "$PID" || { tail -n 80 "$LOG_FILE" >&2; exit 1; }
fi
echo "Automatic FSDP experiment queued PID=$PID; GPU request=${GPU_ID:-auto}; run=$AUTO_RUN_DIR"
echo "Log: $LOG/latest.log"
echo "Final comparison: $AUTO_RUN_DIR/comparison.json"
