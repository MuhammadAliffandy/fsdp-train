import importlib.util
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsdp_c8_model import BrainUNet25D


class PlainTests(unittest.TestCase):
    def test_shapes_backward_and_bn_preflight_recovery(self):
        torch.manual_seed(42)
        model=BrainUNet25D('plain4',base_fields=1).train()
        initial={k:v.clone() for k,v in model.state_dict().items()}
        outputs=model.plane(torch.randn(2,5,29,35))
        self.assertTrue(all(t.shape==(2,1,29,35) for t in outputs))
        sum(p.square().mean() for p in outputs).backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
        self.assertTrue(any(not torch.equal(initial[k],v) for k,v in model.state_dict().items()))
        model.load_state_dict(initial)
        self.assertTrue(all(torch.equal(initial[k],v) for k,v in model.state_dict().items()))


@unittest.skipUnless(importlib.util.find_spec('escnn'),'ESCNN not installed')
class C8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.model=BrainUNet25D('c8',base_fields=1,chunk_size=2)

    def test_01_native_shapes_backward_all_heads(self):
        model=self.model.train()
        outputs=model.plane(torch.randn(2,5,29,35))
        self.assertTrue(all(t.shape==(2,1,29,35) for t in outputs))
        sum(p.square().mean() for p in outputs).backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
        model.zero_grad(set_to_none=True)

    def test_02_eval_checkpoint_roundtrip_and_volume_reconstruction(self):
        model=self.model.eval()
        value=torch.randn(1,1,29,35,3)
        with torch.no_grad():
            expected=model(value)
        state={k:v.clone() for k,v in model.state_dict().items()}
        self.assertFalse(any(k.endswith('.filter') or k.endswith('.expanded_bias') for k in state))
        model.load_state_dict(state,strict=True)
        with torch.no_grad():
            actual=model(value)
        for a,b in zip(expected,actual):
            self.assertEqual(a.shape,value.shape)
            torch.testing.assert_close(a,b)

    def test_03_first_equivariant_block_tracks_quarter_rotation(self):
        model=self.model.eval()
        group=model.input_type.gspace.fibergroup.element(2)
        value=model.enn.GeometricTensor(torch.randn(2,5,49,49),model.input_type)
        with torch.no_grad():
            actual=model.encoder[0](value.transform(group)).tensor[...,8:-8,8:-8]
            expected=model.encoder[0](value).transform(group).tensor[...,8:-8,8:-8]
        error=(actual-expected).abs().mean()/(expected.abs().mean()+1e-6)
        self.assertLess(float(error),.05)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
