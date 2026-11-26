#!/usr/bin/env bash
set -euo pipefail
RUN="${1:?New output directory required}"
mkdir -p "$RUN"
for ARM in control affine; do
  python -u scripts/fsdp_sam_augmentation.py \
    --foundation-run-dir "$FSDP_FOUNDATION_DIR" --research-source-dir "$FSDP_AUG_SOURCE_DIR" \
    --source-root "$FSDP_SOURCE_ROOT" --verified-cache --run-dir "$RUN/$ARM" \
    --variant history --augmentation-arm "$ARM" --epochs "${AUG_EPOCHS:-100}" \
    --patience-epochs "${AUG_PATIENCE:-30}" --lr-multiplier .2 --device cuda:0
 done
python - "$RUN" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = {arm: json.loads((root / arm / 'summary.json').read_text()) for arm in ('control', 'affine')}
scores = {arm: row['best_validation_metrics']['5']['mean_bilateral_patient_dice'] for arm, row in rows.items()}
result = dict(complete=all(row['complete'] for row in rows.values()), validation_dice_5=scores,
              affine_minus_control=scores['affine'] - scores['control'], test_images_read=False,
              protocol='53 train/10 val; cached GT ROI;5oracle clicks/side; train-only threshold; same source/recipe')
(root / 'comparison.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
PY
