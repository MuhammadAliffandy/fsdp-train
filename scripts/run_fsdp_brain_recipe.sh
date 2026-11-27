#!/usr/bin/env bash
set -euo pipefail
RUN="${1:?New run directory required}"
mkdir -p "$RUN"
for ARM in control focal_boundary; do
  python -u scripts/train_physical_resunet.py --neighbor-slices 5 --slice-loss "$ARM" \
    --slice-batch-size "${BRAIN_SLICE_BATCH:-4}" \
    --foundation-run-dir "$FSDP_FOUNDATION_DIR" --sam-comparison-dir "$SAM_COMPARISON_DIR" \
    --source-root "$FSDP_SOURCE_ROOT" --run-dir "$RUN/$ARM" \
    --epochs "${BRAIN_EPOCHS:-100}" --patience "${BRAIN_PATIENCE:-30}" \
    --val-every 5 --lr .0001 --accumulation 2 --device cuda:0
 done
python - "$RUN" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
rows={name:json.loads((root/name/'summary.json').read_text()) for name in ('control','focal_boundary')}
configs={name:json.loads((root/name/'config.json').read_text()) for name in rows}
for key in ('cache_sha256','manifest_sha256','architecture','normalization','training_sampler'):
    if configs['control'][key]!=configs['focal_boundary'][key]:
        raise ValueError(f'Arms differ in {key}')
report=dict(complete=all(r['complete'] for r in rows.values()),test_images_read=False,
    target_volume_dice=.8,target_achieved=any(r['best_validation_metrics']['mean_bilateral_patient_dice']>=.8 for r in rows.values()),
    results={name:dict(metrics=r['best_validation_metrics'],best_epoch=r['best_epoch'],completed_epochs=r['completed_epochs']) for name,r in rows.items()},
    delta_focal_boundary_vs_control=rows['focal_boundary']['best_validation_metrics']['mean_bilateral_patient_dice']-rows['control']['best_validation_metrics']['mean_bilateral_patient_dice'],
    protocol='53train/10val; native GTROI;0clicks; fresh residual2Dweights;Brain-inspired sampler/loss;C8not enabled')
(root/'comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
PY
