#!/usr/bin/env python3
"""Shared promptable SAM hybrid for non-FSDP structures from verified full-label caches."""
import argparse
from functools import lru_cache
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import sam_other_parts_data as views


def source_info(folder):
    config=json.loads((folder/'config.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    if not summary['complete'] or summary['test_images_read'] or summary.get('variant')!='history':
        raise ValueError('Expected completed trained SAM history hybrid')
    if summary['detail_grid']!=[32]*3 or summary['token_grid']!=[8]*3:
        raise ValueError('Expected original learned CNN32/Turbo8 architecture')
    if views.digest(folder/'research_best.pth')!=summary['best_checkpoint_sha256']:
        raise ValueError('Source best checkpoint hash differs')
    return config,summary


def find_audit(project,source,prepared=None):
    config,_=source_info(source)
    candidates=[prepared] if prepared else sorted((project/'runs').glob('*/preprocessing/prepared.json'))
    failures=[]
    for path in candidates:
        try:
            proof=views.audit(path,config['manifest_sha256'])
            proof.update(source_run=str(source),source_config_sha256=views.digest(source/'config.json'),
                         source_checkpoint_sha256=views.digest(source/'research_best.pth'))
            return proof
        except (ValueError,KeyError,FileNotFoundError) as error:
            failures.append(dict(path=str(path),error=str(error)))
    return dict(ready=False,test_images_read=False,failures=failures,
                reason='No intact17class cache found. Binary FSDP cache cannot reconstruct other structures. Restore original labels/ColorTables or verified multiclass caches; no training started.')


def stage_views(proof,output):
    minimum=sum(sum(int(v)>0 for v in row['counts'].values()) for row in proof['rows'])*32*2**20+8*2**30
    if shutil.disk_usage(output.parent).free<minimum:
        raise ValueError(f'Insufficient space for class view caches; need at least {minimum/2**30:.1f}GiB before staging')
    output.mkdir(parents=True)
    entries={'train':[],'val':[]}
    for row in proof['rows']:
        if views.digest(row['path'])!=row['sha256']:
            raise ValueError('Source multiclass cache changed after audit')
        case=torch.load(row['path'],map_location='cpu',weights_only=False)
        for label in proof['labels']:
            if not row['counts'][str(label)]:
                continue
            sample=views.make_view(case,label,proof['plans'][str(label)],proof['normalization'])
            if sample['metadata']['target_retention']<1-1e-6:
                raise ValueError(f'{sample["name"]}: class target truncated')
            if row['split']=='train' and sample['roundtrip_dice']<.95:
                raise ValueError(f'{sample["name"]}: roundtrip={sample["roundtrip_dice"]:.4f}<0.95; inspect training grid before proceeding')
            sample.pop('raw_native',None)
            if shutil.disk_usage(output).free<5*2**30:
                raise ValueError('Stopping staging before free space falls below5GiB')
            path=output/f'{sample["name"]}.pt';torch.save(sample,path)
            entries[row['split']].append(dict(path=str(path),sha256=views.digest(path),case=row['case'],label=label))
            print(f'Prepared {sample["name"]}: roundtrip={sample["roundtrip_dice"]:.4f}',flush=True)
    return entries


@lru_cache(maxsize=2)
def sample_at(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def evaluate(model,proof,entries,device,seed,export=None):
    import physical_fsdp as physical
    from fsdp_research_model import run_sample
    from fsdp_research_metrics import surface_metrics
    from evaluate_private_validation import write_csv
    model.eval();rows=[];patients=[]
    for row in (r for r in proof['rows'] if r['split']=='val'):
        case=torch.load(row['path'],map_location='cpu',weights_only=False)
        truth=np.asarray(case['native_target']);best=np.zeros(truth.shape,np.float32);pred=np.zeros(truth.shape,np.uint8)
        for entry in (e for e in entries['val'] if e['case']==row['case']):
            sample=sample_at(entry['path']);label=entry['label']
            with torch.inference_mode():
                _,outputs=run_sample(model,sample,device,seed+int(row['case'].split('_')[-1])*100+label)
            restored=physical.restore_logits(outputs[5]['refined'][0,0].float().cpu().numpy(),sample)
            full=physical.xyz(physical.resample(physical.image_xyz(restored,sample['native']),case['native'],physical.sitk.sitkLinear,-30.))
            take=full>best;pred[take]=label;best[take]=full[take]
        metrics=[]
        # Score final mutually exclusive15-class map, not overlapping prompted binary masks.
        for label in proof['labels']:
            reference=truth==label;prediction=pred==label
            if not reference.any():
                continue  # Explicit conditional evaluation; absent-class detection is not measured.
            tp=int((reference&prediction).sum());p=int(prediction.sum());g=int(reference.sum())
            surface=surface_metrics(np.flatnonzero(prediction),np.flatnonzero(reference),truth.shape,case['native']['spacing'])
            record=dict(case=row['case'],label=label,name=proof['mapping'][str(label)],volume_dice=2*tp/(p+g),
                        target_voxels=g,predicted_voxels=p,precision=tp/p if p else 0.,recall=tp/g,**surface)
            rows.append(record);metrics.append(record)
        patients.append(dict(case=row['case'],classes=len(metrics),volume_dice=float(np.mean([m['volume_dice'] for m in metrics])),
                             surface_dice_1mm=float(np.mean([m['surface_dice_1mm'] for m in metrics])),
                             surface_dice_0_5mm=float(np.mean([m['surface_dice_0_5mm'] for m in metrics]))))
        if export:
            export.mkdir(parents=True,exist_ok=True)
            physical.sitk.WriteImage(physical.image_xyz(pred,case['native']),str(export/f'{row["case"]}_prediction.nii.gz'),True)
            physical.sitk.WriteImage(physical.image_xyz(np.where(np.isin(truth,proof['labels']),truth,0).astype(np.uint8),case['native']),str(export/f'{row["case"]}_target.nii.gz'),True)
        print(f'{row["case"]}: composed non-FSDP map evaluated',flush=True)
    classes=[]
    for label in proof['labels']:
        subset=[r for r in rows if r['label']==label]
        classes.append(dict(label=label,name=proof['mapping'][str(label)],patients=len(subset),
                            volume_dice=float(np.mean([r['volume_dice'] for r in subset])) if subset else None,
                            surface_dice_1mm=float(np.mean([r['surface_dice_1mm'] for r in subset])) if subset else None))
    result=dict(volume_dice=float(np.mean([r['volume_dice'] for r in patients])),
                surface_dice_1mm=float(np.mean([r['surface_dice_1mm'] for r in patients])),
                surface_dice_0_5mm=float(np.mean([r['surface_dice_0_5mm'] for r in patients])),
                patients=patients,classes=classes,case_classes=rows,
                metric_scope='Macro over GT-present non-FSDP classes within patient, then10patients; absent-class localization not evaluated')
    result['classes_reaching_surface_dice_1mm_80']=[r['label'] for r in classes if r['surface_dice_1mm'] is not None and r['surface_dice_1mm']>=.8]
    if export:
        for key in ('patients','classes','case_classes'):
            write_csv(export/f'{key}.csv',result[key])
    return result


def train(args):
    from monai.losses import DiceCELoss
    import physical_fsdp as physical
    import train_sam_refine32 as base
    from fsdp_research_model import ResearchModel,run_sample
    from train_fsdp_research import allocator_limit
    from train_private_long import atomic_save,accumulation_window,early_stop_update
    from train_private_unet import state_hash
    proof=json.loads(args.audit.read_text())
    if not proof['ready'] or proof['test_images_read']:
        raise ValueError('Full label audit must pass before training')
    source=Path(proof['source_run']);config,summary=source_info(source)
    if views.digest(source/'config.json')!=proof['source_config_sha256'] or views.digest(source/'research_best.pth')!=proof['source_checkpoint_sha256']:
        raise ValueError('SAM source changed after audit')
    parameters=config['parameters']
    from sam_refine_data import source_hashes
    foundation_config_path=Path(config['foundation']['run'])/'config.json'
    if views.digest(foundation_config_path)!=config['foundation']['config_sha256']:
        raise ValueError('Original SAM foundation configuration changed')
    foundation_config=json.loads(foundation_config_path.read_text())
    if source_hashes(Path(parameters['sam_dir']))!=foundation_config['sam_source_sha256']:
        raise ValueError('SAM package source differs from the trained foundation')
    from train_fsdp_research import inspect_source_code
    inspect_source_code(config['code_sha256'])
    import importlib.metadata
    if any(importlib.metadata.version(name)!=version for name,version in config['versions'].items()):
        raise ValueError('Use the SAM source environment package versions')
    protected=[source.resolve(),Path(proof['prepared_path']).resolve().parent.parent]
    if any(args.run_dir.resolve().is_relative_to(p) for p in protected):
        raise ValueError('Training outputs must be outside source SAM and original multiclass runs')
    if args.device.startswith('cuda') and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError('CUDA/BF16 unavailable')
    args.run_dir.mkdir(parents=True,exist_ok=False)
    prepared=stage_views(proof,args.run_dir/'samples')
    physical.save(args.run_dir/'prepared.json',prepared)
    np.random.seed(2026);torch.manual_seed(2026)
    device=torch.device(args.device);allocator_limit(device)
    model,_=base.build(SimpleNamespace(sam_dir=Path(parameters['sam_dir']),checkpoint=Path(parameters['checkpoint'])))
    model=ResearchModel.from_foundation(model,'history')
    checkpoint=torch.load(source/'research_best.pth',map_location='cpu',weights_only=False)
    if checkpoint['config']!=config:
        raise ValueError('Source checkpoint/config differs')
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    initial_hash=state_hash(model.state_dict());del checkpoint
    model=model.to(device);model.amp_dtype=torch.bfloat16
    criterion=DiceCELoss(sigmoid=True,squared_pred=True)
    optimizer=torch.optim.AdamW([dict(params=model.sam.parameters(),lr=2e-6),
        dict(params=[p for n,p in model.named_parameters() if not n.startswith('sam.')],lr=2e-5)],weight_decay=1e-4)
    run_config=dict(source_checkpoint_sha256=proof['source_checkpoint_sha256'],initial_state_sha256=initial_hash,
        audit_sha256=views.digest(args.audit),labels=proof['labels'],mapping=proof['mapping'],epochs=args.epochs,
        architecture='Inherited full SAMTurbo encoder/prompt/maskdecoder+CNN32refiner; shared binary class prompting',
        oracle_clicks_per_present_class=5,threshold_fixed=.5,train_patients=53,validation_patients=10,test_images_read=False,
        selection='Validation macro surfaceDice@1mm; volumeDice independently reported; present-class GTROI/points',
        code_sha256={name:views.digest(Path(__file__).with_name(name)) for name in ('train_sam_other_parts.py','sam_other_parts_data.py','sam_refine32.py','fsdp_research_model.py')})
    physical.save(args.run_dir/'config.json',run_config)
    initial=evaluate(model,proof,prepared,device,2026)
    best=initial['surface_dice_1mm'];best_epoch=0;history=[];early=dict(anchor=best,last_improvement_epoch=0)
    def state(epoch):
        return dict(model_state_dict=model.state_dict(),epoch=epoch,config=run_config,metrics=best,best_epoch=best_epoch)
    atomic_save(args.run_dir/'model_best.pth',state(0))
    for epoch in range(args.epochs):
        base.set_stage(model,epoch+30,0)
        order=np.random.default_rng(2026+epoch).permutation(len(prepared['train']))
        optimizer.zero_grad(set_to_none=True);losses=[];updates=0
        for step,index in enumerate(order):
            entry=prepared['train'][int(index)];sample=sample_at(entry['path'])
            loss,_=run_sample(model,sample,device,2026+epoch*10000+step,criterion,augment=True)
            if not loss.isfinite():
                raise ValueError('Nonfinite SAM multi-structure loss')
            window,update=accumulation_window(step,len(order),4)
            (loss/window).backward();losses.append(float(loss.detach()))
            if update:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not norm.isfinite():
                    raise ValueError('Nonfinite SAM multi-structure gradients')
                optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
            if step%10==0 or step==len(order)-1:
                print(f'SAM other-parts epoch={epoch+1}/{args.epochs} classROI={step+1}/{len(order)} loss={float(loss):.5f} updates={updates}',flush=True)
        row=dict(epoch=epoch+1,loss=float(np.mean(losses)),optimizer_updates=updates,volume_dice=None,surface_dice_1mm=None)
        stop=False
        if epoch==0 or (epoch+1)%5==0 or epoch+1==args.epochs:
            result=evaluate(model,proof,prepared,device,2026)
            row.update(volume_dice=result['volume_dice'],surface_dice_1mm=result['surface_dice_1mm'])
            early,stop=early_stop_update(early,result['surface_dice_1mm'],epoch+1,args.patience,.001)
            if result['surface_dice_1mm']>best:
                best=result['surface_dice_1mm'];best_epoch=epoch+1
                atomic_save(args.run_dir/'model_best.pth',state(epoch+1))
        history.append(row);physical.save(args.run_dir/'progress.json',dict(complete=False,best_surface_dice_1mm=best,best_epoch=best_epoch,latest=row))
        physical.save(args.run_dir/'history.json',history)
        print(json.dumps(row),flush=True)
        if stop:
            break
    checkpoint=torch.load(args.run_dir/'model_best.pth',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    result=evaluate(model,proof,prepared,device,2026,args.run_dir/'best_exports')
    if abs(result['surface_dice_1mm']-best)>1e-4:
        raise ValueError('Selected checkpoint exports do not reproduce surfaceDice')
    report=dict(complete=True,completed_epochs=len(history),best_epoch=best_epoch,initial_metrics=initial,best_metrics=result,
                target_surface_dice_1mm=.8,target_achieved=best>=.8,oracle_clicks_per_present_class=5,test_images_read=False,
                volume_dice_target_achieved=result['volume_dice']>=.8,checkpoint_sha256=views.digest(args.run_dir/'model_best.pth'),
                protocol=proof['protocol'],absent_classes=proof['missing_annotations'])
    physical.save(args.run_dir/'summary.json',report);print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    audit=sub.add_parser('audit');audit.add_argument('--project',type=Path,required=True)
    audit.add_argument('--source-run',type=Path,required=True);audit.add_argument('--prepared',type=Path);audit.add_argument('--output',type=Path,required=True)
    fit=sub.add_parser('train');fit.add_argument('--audit',type=Path,required=True);fit.add_argument('--run-dir',type=Path,required=True)
    fit.add_argument('--epochs',type=int,default=20);fit.add_argument('--patience',type=int,default=10);fit.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.action=='audit':
        proof=find_audit(args.project.resolve(),args.source_run.resolve(),args.prepared)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(proof,indent=2))
        display={key:proof[key] for key in ('ready','reason','failures','prepared_path','excluded_labels','missing_annotations') if key in proof}
        if proof['ready']:
            display['classes']={str(i):proof['mapping'][str(i)] for i in proof['labels']}
            display['patients']={'train':53,'val':10}
        display['audit_report']=str(args.output)
        print(json.dumps(display,indent=2));raise SystemExit(0 if proof['ready'] else 2)
    if args.epochs<1 or args.patience<1:
        parser.error('Positive epochs/patience required')
    existed=args.run_dir.exists()
    try:
        train(args)
    except Exception as error:
        if not existed and args.run_dir.is_dir():
            (args.run_dir/'failure.json').write_text(json.dumps({'error':str(error),'type':type(error).__name__},indent=2))
        raise


if __name__=='__main__':
    main()
