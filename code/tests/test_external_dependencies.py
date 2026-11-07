"""Optional checks against legally obtained private pinned source dependencies."""
import os
from pathlib import Path
import sys
import unittest
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from common.upstream import source_directory


class ExternalDependencyTests(unittest.TestCase):
    @unittest.skipUnless((source_directory('ars')/'train_ars/train_ARS.py').exists(), 'Optional private ARS source not installed')
    def test_pinned_ars_definitions(self):
        from baselines.ars.kernel import LinearMap, icm_sameid_ars_loss
        from baselines.ars.detector_kernel import NonLinearClassifier
        torch.manual_seed(42)
        model = LinearMap(5, 3)
        x = model(torch.randn(8, 5))
        loss = icm_sameid_ars_loss(x, torch.tensor([0,0,1,1,2,2,3,3]), torch.tensor([True,False]*4), temperature=.1, lambda_reg=.001, W=model.W)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        probe = NonLinearClassifier(3)
        self.assertEqual(tuple(probe(x.detach()).shape), (8,1))

    @unittest.skipUnless((source_directory('hami')/'hami.py').exists(), 'Optional private HaMI source not installed')
    def test_pinned_hami_definitions(self):
        from baselines.hami.kernel import HaMI, MIL_loss
        torch.manual_seed(42)
        model = HaMI(5, 2)
        scores, nl, al = model([torch.randn(4,1,5),torch.randn(5,1,5)], [torch.randn(6,1,5),torch.randn(7,1,5)], n_layer=0, device='cpu')
        loss = MIL_loss(scores, nl, al, 'cpu', .1)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()


if __name__ == '__main__':
    unittest.main()
