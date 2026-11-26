import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from physical_unet25d import slice_stack, PhysicalUNet25D


class SliceTests(unittest.TestCase):
    def test_order_center_and_edge_replication(self):
        image = torch.arange(6.).reshape(1,1,1,1,6).expand(1,1,12,15,6)
        stacks = slice_stack(image, [0,3,5], 5)
        self.assertEqual(stacks[:, :, 0, 0].tolist(), [[0,0,0,1,2],[1,2,3,4,5],[3,4,5,5,5]])
        self.assertEqual(slice_stack(image,[3],1)[0,0,0,0],3)

    def test_native_reconstruction_and_chunk_independence(self):
        torch.manual_seed(8)
        model = PhysicalUNet25D(5, chunk_size=2, channels=(8,8,16,16)).eval()
        image = torch.randn(1,1,19,23,7)
        with torch.no_grad():
            first = model(image)
            model.chunk_size=7
            second = model(image)
        for a,b in zip(first,second):
            self.assertEqual(a.shape,image.shape)
            torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

    def test_checkpointed_training_passes_gradients_to_neighbor_channels(self):
        model=PhysicalUNet25D(5, chunk_size=2, channels=(8,8,16,16)).train()
        image=torch.randn(1,1,16,24,5)
        outputs=model(image)
        sum(t.square().mean() for t in outputs).backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(parameter.grad.isfinite().all())
        gradient=model.encoder[0].layers[0].weight.grad
        self.assertTrue((gradient.abs().sum((0,2,3))>0).all())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
