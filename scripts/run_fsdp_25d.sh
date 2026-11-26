#!/usr/bin/env bash
set -euo pipefail
RUN="${1:?New run directory required}"
mkdir -p "$RUN"
for NEIGHBORS in 1 5; do
  python -u scripts/train_physical_resunet.py --neighbor-slices "$NEIGHBORS" \
    --foundation-run-dir "$FSDP_FOUNDATION_DIR" --sam-comparison-dir "$SAM_COMPARISON_DIR" \
    --source-root "$FSDP_SOURCE_ROOT" --run-dir "$RUN/slices_$NEIGHBORS" \
    --epochs "${UNET25D_EPOCHS:-100}" --patience "${UNET25D_PATIENCE:-30}" \
    --val-every 5 --lr .0003 --accumulation 2 --device cuda:0
 done
python - "$RUN" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = {name: json.loads((root / name / 'summary.json').read_text()) for name in ('slices_1','slices_5')}
report = dict(complete=all(r['complete'] for r in rows.values()), test_images_read=False,
    results={name: dict(volume_dice=r['best_validation_metrics']['mean_bilateral_patient_dice'],
                       surface_dice_0_5mm=r['best_validation_metrics']['mean_surface_dice_0_5mm'],
                       surface_dice_1mm=r['best_validation_metrics']['mean_surface_dice_1mm'],
                       hd95_mm_median=r['best_validation_metrics']['median_hd95_mm'], best_epoch=r['best_epoch'])
             for name,r in rows.items()},
    protocol='53train/10val; native GTROI; zero clicks; new random 2D weights;1vs5slice comparison; not SAM finetuning')
(root / 'comparison.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
PY
