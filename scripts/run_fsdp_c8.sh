#!/usr/bin/env bash
set -euo pipefail
RUN="${1:?New run directory required}"
mkdir -p "$RUN"
for BACKBONE in plain4 c8; do
  python -u scripts/train_physical_resunet.py --neighbor-slices 5 --slice-loss focal_boundary \
    --rotation-backbone "$BACKBONE" --c8-base-fields "${C8_BASE_FIELDS:-2}" --slice-batch-size "${C8_SLICE_BATCH:-4}" \
    --foundation-run-dir "$FSDP_FOUNDATION_DIR" --sam-comparison-dir "$SAM_COMPARISON_DIR" \
    --source-root "$FSDP_SOURCE_ROOT" --run-dir "$RUN/$BACKBONE" \
    --epochs "${C8_EPOCHS:-100}" --patience "${C8_PATIENCE:-30}" \
    --val-every 5 --lr .0001 --accumulation 2 --device cuda:0
 done
python - "$RUN" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
rows={key:json.loads((root/key/'summary.json').read_text()) for key in ('plain4','c8')}
configs={key:json.loads((root/key/'config.json').read_text()) for key in rows}
for key in ('cache_sha256','manifest_sha256','normalization','training_sampler','loss'):
    if configs['plain4'][key]!=configs['c8'][key]:
        raise ValueError(f'Comparison differs in {key}')
report=dict(complete=all(r['complete'] for r in rows.values()),test_images_read=False,target_volume_dice=.8,
    results={key:dict(metrics=r['best_validation_metrics'],best_epoch=r['best_epoch'],completed_epochs=r['completed_epochs']) for key,r in rows.items()},
    delta_c8_vs_plain4=rows['c8']['best_validation_metrics']['mean_bilateral_patient_dice']-rows['plain4']['best_validation_metrics']['mean_bilateral_patient_dice'],
    protocol='53train/10val; native GTROI;0clicks;four-level topology and scalar channels matched;different group constraints/parameter counts;noSAMweights')
(root/'comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
PY
