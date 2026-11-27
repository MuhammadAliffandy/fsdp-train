"""Recover verified multiclass caches and construct training-derived per-structure SAM views."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import SimpleITK as sitk
import physical_fsdp as physical

LABELS = tuple(i for i in range(1,18) if i not in (1,9))


def digest(path):
    sha=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            sha.update(block)
    return sha.hexdigest()


def validate_case(case, split, alias, coverage):
    if case['case']!=alias or not alias.startswith(split+'_'):
        raise ValueError('Cache patient/split identity differs')
    raw=np.asarray(case['raw_native']); target=np.asarray(case['native_target'])
    if raw.shape!=target.shape or tuple(case['native']['size'])!=raw.shape or not np.isfinite(raw).all():
        raise ValueError(f'{alias}: invalid native CT/label geometry')
    if not np.isin(target,range(18)).all() or not np.array_equal(target,target.astype(np.uint8)):
        raise ValueError(f'{alias}: expected full integer labels0..17')
    geometry=np.asarray([*case['native']['spacing'],*case['native']['origin'],*case['native']['direction']],float)
    if not np.isfinite(geometry).all() or min(case['native']['spacing'])<=0:
        raise ValueError('Invalid native physical geometry')
    for row in coverage:
        count=int((target==int(row['label'])).sum())
        if count!=int(row['target_voxels']):
            raise ValueError(f'{alias}: native class target count differs from original coverage')
    if np.array_equal(raw,target):
        raise ValueError(f'{alias}: input CT equals the label map')
    if float(raw.std())<1e-5:
        raise ValueError(f'{alias}: degenerate CT')


def audit(prepared_path, expected_manifest):
    path=Path(prepared_path).resolve()
    prepared=json.loads(path.read_text()); config=json.loads((path.parent.parent/'config.json').read_text())
    if prepared.get('test_images_read') is not False or config['manifest_sha256']!=expected_manifest:
        raise ValueError('Cache manifest differs from SAM foundation or test was accessed')
    mapping=prepared['policy']['mapping']
    if set(map(int,mapping))!=set(range(18)) or mapping['1'].upper()!='RFSDP' or mapping['9'].upper()!='LFSDP':
        raise ValueError('Original class mapping must identify RFSDP1/LFSDP9 and labels0..17')
    if len(set(mapping.values()))!=18:
        raise ValueError('Class names must be unique')
    rows=[]; values=[]; extents={i:[] for i in LABELS}; missing=[]
    for split,count in (('train',54),('val',10)):
        if len(prepared['paths'][split])!=count:
            raise ValueError('Expected original54/10multiclass cache')
        seen=set()
        for old_path in prepared['paths'][split]:
            alias=Path(old_path).stem
            if alias in seen:
                raise ValueError('Duplicate cache alias')
            seen.add(alias)
            if alias=='train_28':
                continue
            expected=prepared['cache_sha256'][old_path]
            candidates=[Path(old_path)] if Path(old_path).is_file() else sorted(path.parent.glob('*/'+Path(old_path).name))
            good=[p for p in candidates if p.is_file() and digest(p)==expected]
            if not good:
                raise ValueError(f'Missing or changed full multiclass cache: {old_path}')
            case=torch.load(good[0],map_location='cpu',weights_only=False)
            coverage=[r for r in prepared['coverage'][split] if r['case']==alias]
            if {int(r['label']) for r in coverage}!=set(range(1,18)):
                raise ValueError('Coverage must include all17original classes')
            validate_case(case,split,alias,coverage)
            raw=np.asarray(case['raw_native'],dtype=np.float32); target=np.asarray(case['native_target'])
            counts={str(i):int((target==i).sum()) for i in LABELS}
            if not any(counts.values()):
                raise ValueError(f'{alias}: no annotated non-FSDP structures to train/evaluate')
            if split=='train':
                rng=np.random.default_rng(2026+int(alias.split('_')[-1]))
                flat=raw.ravel();values.append(flat if len(flat)<=65536 else rng.choice(flat,65536,replace=False))
                for label in LABELS:
                    if counts[str(label)]:
                        lo,hi=physical.world_bounds(target==label,case['native']);extents[label].append(hi-lo)
            missing.extend({'case':alias,'label':i} for i in LABELS if not counts[str(i)])
            rows.append(dict(split=split,case=alias,path=str(good[0].resolve()),sha256=expected,counts=counts))
            print(f'Audit {alias}: full native labels verified',flush=True)
        if seen!={f'{split}_{i:02d}' for i in range(1,count+1)}:
            raise ValueError('Unexpected patient aliases in frozen split')
    if sum(r['split']=='train' for r in rows)!=53 or sum(r['split']=='val' for r in rows)!=10:
        raise ValueError('Cache split identities do not yield53/10')
    if any(not extents[i] for i in LABELS):
        raise ValueError('A requested structure has no training annotations')
    raw=np.concatenate(values).astype(np.float64)
    lower,upper=np.percentile(raw,[.1,99.9]);clipped=np.clip(raw,lower,upper)
    norm=dict(clip_lower=float(lower),clip_upper=float(upper),mean=float(clipped.mean()),std=float(clipped.std()),source='53training native CT contexts; no val statistics')
    if not np.isfinite(list(norm[key] for key in ('clip_lower','clip_upper','mean','std'))).all() or norm['std']<=1e-6:
        raise ValueError('Invalid train-only normalization')
    if any(not np.isfinite(np.asarray(v)).all() for v in extents.values()):
        raise ValueError('Nonfinite training anatomy extents')
    shape=np.array([96,160,208])
    plans={str(i):dict(shape=shape.tolist(),spacing=np.maximum(.25,(np.max(extents[i],axis=0)+12)/(shape-1)).tolist()) for i in LABELS}
    return dict(ready=True,prepared_path=str(path),prepared_sha256=digest(path),manifest_sha256=expected_manifest,
                mapping=mapping,labels=list(LABELS),excluded_labels=[1,9],excluded_training=['train_28'],
                rows=rows,normalization=norm,plans=plans,missing_annotations=missing,test_images_read=False,
                protocol='All available non-FSDP annotated structures; GT class ROI and oracle points; absent classes not localized/evaluated')


def make_view(case,label,plan,norm):
    target=np.asarray(case['native_target'])==label
    if not target.any():
        raise ValueError('No class annotation; absent-class localization is outside this protocol')
    image=physical.image_xyz(np.asarray(case['raw_native'],np.float32),case['native'])
    sample=physical.make_sample(image,np.asarray(case['raw_native']),target,case['case'],f'C{label:02d}',plan,norm)
    detail=sample['grid'];extent=np.asarray(detail['size'])*detail['spacing']
    context_extent=float(extent.max());step=context_extent/128
    center=np.asarray(detail['origin'])+(np.asarray(detail['size'])-1)*np.asarray(detail['spacing'])/2
    context=dict(size=[128]*3,spacing=[step]*3,origin=(center-127*step/2).tolist(),direction=np.eye(3).ravel().tolist())
    binary=physical.image_xyz(target.astype(np.uint8),case['native'])
    ones=sitk.Image(image.GetSize(),sitk.sitkUInt8)+1;ones.CopyInformation(image)
    mask=physical.xyz(physical.resample(binary,context,sitk.sitkNearestNeighbor,pixel=sitk.sitkUInt8))
    valid=physical.xyz(physical.resample(ones,context,sitk.sitkNearestNeighbor,pixel=sitk.sitkUInt8)).astype(bool)
    raw=physical.xyz(physical.resample(image,context,sitk.sitkLinear,norm['clip_lower']))
    normalized=(np.clip(raw,norm['clip_lower'],norm['clip_upper'])-norm['mean'])/norm['std'];normalized[~valid]=0
    if not mask.any() or not valid[mask.astype(bool)].all():
        raise ValueError(f'{case["case"]}/class{label}: target lost in SAM context')
    sample.update(sam_image=torch.from_numpy(normalized.astype(np.float32))[None,None],
                  sam_target=torch.from_numpy(mask)[None,None],sam_valid=torch.from_numpy(valid)[None,None],
                  sam_grid=context,fov_ratio=torch.tensor(extent/context_extent,dtype=torch.float32),
                  detail_spacing=torch.tensor(detail['spacing'],dtype=torch.float32),class_id=label)
    return sample
