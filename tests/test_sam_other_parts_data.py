import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import sam_other_parts_data as data


def case_fixture():
    rng=np.random.default_rng(2026)
    raw=rng.normal(-500,250,(32,40,48)).astype(np.float32)
    labels=np.zeros_like(raw,np.uint8);labels[12:20,15:25,18:30]=2
    native=dict(size=list(raw.shape),spacing=[.5,.5,.5],origin=[0.,0.,0.],direction=np.eye(3).ravel().tolist())
    return dict(case='train_01',raw_native=raw,native_target=labels,native=native)


class PartTests(unittest.TestCase):
    def test_label_scope_excludes_both_fsdp_classes(self):
        self.assertEqual(len(data.LABELS),15)
        self.assertFalse({1,9}&set(data.LABELS))

    def test_cache_count_mismatch_rejected(self):
        case=case_fixture()
        counts=[dict(label=i,target_voxels=int((case['native_target']==i).sum())) for i in range(1,18)]
        data.validate_case(case,'train','train_01',counts)
        counts[1]['target_voxels']+=1
        with self.assertRaisesRegex(ValueError,'target count differs'):
            data.validate_case(case,'train','train_01',counts)

    def test_views_preserve_native_targets_and_physical_context(self):
        case=case_fixture();before=case['native_target'].copy()
        norm=dict(clip_lower=-2000.,clip_upper=3000.,mean=-500.,std=300.)
        sample=data.make_view(case,2,dict(shape=[64,64,64],spacing=[.25]*3),norm)
        self.assertEqual(int(sample['native_target'].sum()),int((before==2).sum()))
        self.assertGreaterEqual(sample['metadata']['target_retention'],1-1e-6)
        self.assertEqual(tuple(sample['sam_image'].shape),(1,1,128,128,128))
        self.assertTrue(torch.isfinite(sample['image']).all())
        self.assertTrue(sample['sam_valid'][sample['sam_target'].bool()].all())
        detail_center=np.array(sample['grid']['origin'])+(np.array(sample['grid']['size'])-1)*np.array(sample['grid']['spacing'])/2
        context_center=np.array(sample['sam_grid']['origin'])+127*np.array(sample['sam_grid']['spacing'])/2
        np.testing.assert_allclose(detail_center,context_center)
        np.testing.assert_array_equal(before,case['native_target'])
        with self.assertRaisesRegex(ValueError,'No class annotation'):
            data.make_view(case,3,dict(shape=[64]*3,spacing=[.25]*3),norm)


class AuditGateTests(unittest.TestCase):
    def test_missing_multiclass_cache_does_not_start_training(self):
        import tempfile,json
        from train_sam_other_parts import find_audit
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'sam';source.mkdir()
            weights=source/'research_best.pth';weights.write_bytes(b'fixture')
            (source/'config.json').write_text(json.dumps({'manifest_sha256':'frozen'}))
            (source/'summary.json').write_text(json.dumps(dict(complete=True,test_images_read=False,variant='history',
                detail_grid=[32]*3,token_grid=[8]*3,best_checkpoint_sha256=data.digest(weights))))
            proof=find_audit(root,source)
            self.assertFalse(proof['ready']);self.assertFalse(proof['test_images_read'])
            self.assertIn('Binary FSDP cache cannot',proof['reason'])

    def test_launcher_stops_when_gpu_selection_fails(self):
        import os, shutil, subprocess, tempfile
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);scripts=root/'scripts';scripts.mkdir();binary=root/'bin';binary.mkdir()
            launcher=Path(__file__).resolve().parents[1]/'scripts/launch_sam_other_parts_nohup.sh'
            shutil.copy(launcher,scripts/launcher.name)
            python=binary/'python'
            python.write_text('#!/usr/bin/env bash\n'
                'if [[ "$*" == *fsdp_research_gpu.py* ]]; then exit 2; fi\n'
                'if [[ "$*" == *" train "* ]]; then touch train_called; fi\n'
                'exit 0\n')
            python.chmod(0o755)
            env={**os.environ,'PATH':str(binary)+os.pathsep+os.environ['PATH'],'CONDA_DEFAULT_ENV':'fsdp-med3d'}
            result=subprocess.run(['bash',str(scripts/launcher.name)],env=env,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertFalse((root/'train_called').exists())
            self.assertFalse((root/'logs/sam_other_parts/latest.pid').exists())


class CompositionTests(unittest.TestCase):
    def test_native_export_resolves_overlaps_and_does_not_score_absent_classes(self):
        import tempfile
        from unittest.mock import patch
        import train_sam_other_parts as trainer
        import fsdp_research_model
        import fsdp_research_metrics
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            native=dict(size=[4]*3,spacing=[.5]*3,origin=[-2.,3.,7.],direction=np.eye(3).ravel().tolist())
            truth=np.zeros((4,4,4),np.uint8);truth[1,1,1]=2;truth[2,2,2]=3;truth[0,0,0]=1
            case_path=root/'val_01.pt';torch.save(dict(native_target=truth,native=native),case_path)
            entries={'val':[]}
            for label in (2,3):
                path=root/f'class{label}.pt'
                torch.save(dict(class_id=label,native=native,grid=native,valid=torch.ones((1,1,4,4,4),dtype=torch.bool)),path)
                entries['val'].append(dict(path=str(path),case='val_01',label=label))
            proof=dict(rows=[dict(split='val',case='val_01',path=str(case_path))],labels=[2,3,4],
                       mapping={'2':'PART2','3':'PART3','4':'ABSENT'})
            def prediction(model,sample,device,seed):
                logits=torch.full((1,1,4,4,4),-5.)
                logits[0,0,1,1,1]=5. if sample['class_id']==2 else 4.
                logits[0,0,2,2,2]=2. if sample['class_id']==2 else 5.
                return None,{5:{'refined':logits}}
            surface=dict(hd95_mm=0.,surface_dice_0_5mm=1.,surface_dice_1mm=1.,empty_prediction=False)
            trainer.sample_at.cache_clear()
            with patch.object(fsdp_research_model,'run_sample',side_effect=prediction), \
                 patch.object(fsdp_research_metrics,'surface_metrics',return_value=surface):
                result=trainer.evaluate(torch.nn.Identity(),proof,entries,torch.device('cpu'),2026,root/'exports')
            self.assertEqual(result['volume_dice'],1.)
            self.assertEqual(len(result['case_classes']),2)
            self.assertIsNone(result['classes'][2]['surface_dice_1mm'])
            exported=data.physical.sitk.ReadImage(str(root/'exports/val_01_prediction.nii.gz'))
            np.testing.assert_array_equal(data.physical.xyz(exported),np.where(truth==1,0,truth))
            np.testing.assert_allclose(exported.GetSpacing(),native['spacing'])
            np.testing.assert_allclose(exported.GetOrigin(),native['origin'])
            trainer.sample_at.cache_clear()


if __name__=='__main__':
    unittest.main()
