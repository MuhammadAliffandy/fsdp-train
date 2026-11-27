import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsdp_brain_recipe import collate_slices, slice_loss, boundary_weights


class RecipeTests(unittest.TestCase):
    def test_slice_targets_neighbors_and_padded_support(self):
        samples=[]
        for h,w in ((12,16),(16,20)):
            image=torch.arange(6.).reshape(1,1,1,1,6).expand(1,1,h,w,6).clone()
            mask=torch.zeros_like(image).byte(); mask[...,2]=1
            samples.append(dict(image=image,target=mask,native={'spacing':[.5,.7,1.]}))
        image,target,valid,spacing=collate_slices(samples,[(0,2),(1,4)],99,augment=False)
        self.assertEqual(image.shape,(2,5,16,20))
        self.assertEqual(image[0,:,0,0].tolist(),[0,1,2,3,4])
        self.assertTrue(target[0,:,:12,:16].all()); self.assertFalse(target[1].any())
        self.assertFalse(valid[0,:,12:,:].any())
        self.assertEqual(int(valid.sum()),12*16+16*20)

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
