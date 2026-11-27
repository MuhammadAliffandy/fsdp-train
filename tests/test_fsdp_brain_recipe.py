import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from physical_unet25d import slice_stack
from fsdp_brain_recipe import collate_slices, slice_loss, boundary_weights, shape_batches


class RecipeTests(unittest.TestCase):
    def test_slice_targets_neighbors_and_shape_rejection(self):
        samples=[]
        for h,w in ((12,16),(12,16)):
            image=torch.arange(6.).reshape(1,1,1,1,6).expand(1,1,h,w,6).clone()
            mask=torch.zeros_like(image).byte(); mask[...,2]=1
            samples.append(dict(image=image,target=mask,native={'spacing':[.5,.7,1.]}))
        image,target,valid,spacing=collate_slices(samples,[(0,2),(1,4)],99,augment=False)
        self.assertEqual(image.shape,(2,5,12,16))
        self.assertEqual(image[0,:,0,0].tolist(),[0,1,2,3,4])
        self.assertTrue(target[0,:,:12,:16].all()); self.assertFalse(target[1].any())
        self.assertTrue(valid.all())
        samples[1]['image']=torch.zeros(1,1,16,20,6)
        with self.assertRaisesRegex(ValueError,'share native XY shape'):
            collate_slices(samples,[(0,2),(1,4)],99,augment=False)

    def test_shape_batches_cover_every_slice_once_and_are_deterministic(self):
        samples=[{'image':torch.zeros(1,1,h,w,z)} for h,w,z in ((12,16,5),(16,20,3),(12,16,4))]
        batches=shape_batches(samples,4,2026)
        self.assertEqual(batches,shape_batches(samples,4,2026))
        entries=[entry for batch in batches for entry in batch]
        self.assertEqual(sorted(entries),[(i,z) for i,s in enumerate(samples) for z in range(s['image'].shape[-1])])
        for batch in batches:
            self.assertLessEqual(len(batch),4)
            self.assertEqual(len({tuple(samples[i]['image'].shape[2:4]) for i,_ in batch}),1)

    def test_same_shape_collation_matches_unbatched_inference(self):
        from physical_unet25d import PhysicalUNet25D
        torch.manual_seed(2026)
        samples=[dict(image=torch.randn(1,1,24,32,6),target=torch.zeros(1,1,24,32,6).byte(),
                      native={'spacing':[.5,.5,1.]}) for _ in range(2)]
        model=PhysicalUNet25D(5,channels=(8,8,16,16)).eval()
        batch,_,valid,_=collate_slices(samples,[(0,2),(1,4)],2026,augment=False)
        with torch.no_grad():
            actual=model.plane(batch)[0]
            expected=torch.cat([model.plane(slice_stack(samples[i]['image'],[z],5))[0] for i,z in ((0,2),(1,4))])
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        self.assertTrue(valid.all())

    def test_focal_boundary_gradients_ignore_invalid_padding(self):
        target=torch.zeros(2,1,16,24);target[0,:,6:10,8:12]=1
        valid=torch.ones_like(target).bool();valid[:,:,:2]=False
        outputs=[torch.zeros_like(target,requires_grad=True) for _ in range(3)]
        loss=slice_loss(outputs,target,valid,[(.5,.8),(.7,.7)],'focal_boundary')
        loss.backward()
        self.assertTrue(loss.isfinite())
        for output in outputs:
            self.assertFalse(output.grad[~valid].any())
            self.assertTrue(output.grad[valid].isfinite().all())
        weights=boundary_weights(target,valid,[(.5,.8),(.7,.7)])
        self.assertGreater(float(weights.max()),1)
        self.assertFalse(weights[~valid].any())

    def test_empty_slice_and_control_loss_finite(self):
        target=torch.zeros(1,1,16,24);valid=torch.ones_like(target).bool()
        for arm in ('control','focal_boundary'):
            output=torch.zeros_like(target,requires_grad=True)
            loss=slice_loss((output,output,output),target,valid,[(.5,.5)],arm)
            loss.backward()
            self.assertTrue(loss.isfinite());self.assertGreater(float(output.grad.sum()),0)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
