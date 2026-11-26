#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${AUTO_RUN_DIR:?Set AUTO_RUN_DIR to a new run directory}"
FOUNDATION_DIR="${FSDP_FOUNDATION_DIR:?Set FSDP_FOUNDATION_DIR}"
SOURCE_ROOT="${FSDP_SOURCE_ROOT:?Set FSDP_SOURCE_ROOT}"
DATASET_ID="${AUTO_DATASET_ID:-901}"
TRAINER=nnUNetTrainer_100epochs
SHARING="${AUTO_GPU_ALLOW_SHARING:-0}"
BUDGET="${AUTO_GPU_MEMORY_BUDGET_GIB:-24}"

if [[ "$RUN_DIR" == "$SOURCE_ROOT"* || "$RUN_DIR" == "$FOUNDATION_DIR"* ]]; then
  echo 'Run directory must be outside source and foundation directories.' >&2
  exit 2
fi
python -c 'import nnunetv2; from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_100epochs'
for command in nnUNetv2_plan_and_preprocess nnUNetv2_train nnUNetv2_predict; do
  command -v "$command" >/dev/null || { echo "Missing nnU-Net command: $command" >&2; exit 2; }
done

python -u "$PROJECT_DIR/scripts/fsdp_auto_nnunet.py" prepare \
  --foundation-run-dir "$FOUNDATION_DIR" --source-root "$SOURCE_ROOT" \
  --output "$RUN_DIR" --dataset-id "$DATASET_ID"

export nnUNet_raw="$RUN_DIR/nnUNet_raw"
export nnUNet_preprocessed="$RUN_DIR/nnUNet_preprocessed"
export nnUNet_results="$RUN_DIR/nnUNet_results"
nnUNetv2_plan_and_preprocess -d "$DATASET_ID" -c 2d 3d_fullres --verify_dataset_integrity

for configuration in 2d 3d_fullres; do
  echo "Starting $configuration with exactly 100 epochs; 53 train; 10 external validation."
  if [[ "$SHARING" == 1 ]]; then
    python -u "$PROJECT_DIR/scripts/fsdp_auto_nnunet_capped.py" train \
      --memory-budget-gib "$BUDGET" "$DATASET_ID" "$configuration" all -tr "$TRAINER"
  else
    nnUNetv2_train "$DATASET_ID" "$configuration" all -tr "$TRAINER"
  fi
  model_dir="$nnUNet_results/Dataset$(printf '%03d' "$DATASET_ID")_FSDPAuto/${TRAINER}__nnUNetPlans__${configuration}/fold_all"
  if [[ ! -f "$model_dir/checkpoint_final.safetensors" && ! -f "$model_dir/checkpoint_final.pth" ]]; then
    echo "Training finished without a final checkpoint: $model_dir" >&2
    exit 1
  fi
  predictions="$RUN_DIR/predictions_${configuration}"
  mkdir -p "$predictions"
  if [[ "$SHARING" == 1 ]]; then
    python -u "$PROJECT_DIR/scripts/fsdp_auto_nnunet_capped.py" predict \
      --memory-budget-gib "$BUDGET" -i "$RUN_DIR/validation_images" -o "$predictions" \
      -d "$DATASET_ID" -c "$configuration" -f all -tr "$TRAINER"
  else
    nnUNetv2_predict -i "$RUN_DIR/validation_images" -o "$predictions" \
      -d "$DATASET_ID" -c "$configuration" -f all -tr "$TRAINER"
  fi
  python -u "$PROJECT_DIR/scripts/fsdp_auto_nnunet.py" score \
    --output "$RUN_DIR" --predictions "$predictions" --configuration "$configuration"
done

python - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path
run = Path(sys.argv[1])
scores = {name: json.loads((run / f'predictions_{name}' / 'summary.json').read_text())
          for name in ('2d', '3d_fullres')}
comparison = {'complete': True, 'train_patients': 53, 'validation_patients': 10,
              'epochs_per_arm': 100, 'same_patient_split': True,
              'same_epoch_count_not_same_optimizer_updates_or_compute': True,
              'no_gt_roi_or_clicks_at_inference': True, 'test_images_read': False,
              'results': {name: {'dice': row['mean_bilateral_patient_dice'],
                                 'minimum_dice': row['minimum_bilateral_patient_dice']}
                          for name, row in scores.items()}}
(run / 'comparison.json').write_text(json.dumps(comparison, indent=2))
print(json.dumps(comparison, indent=2), flush=True)
PY
