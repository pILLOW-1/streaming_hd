from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
from baselines.rhd.kernel import token_depth,components
from baselines.rede.pipeline import fit_projection,retained_indices,proxy_indices
from baselines.rede.kernel import perplexity_weighted_embedding


class AddedKernelTests(unittest.TestCase):
    def test_rhd_original_reverse_kl_equation(self):
        torch.manual_seed(8)
        mature=torch.randn(5,11);anchor=torch.randn(3,5,11)
        depth,entropy=token_depth(mature,anchor)
        reference=[]
        for i in range(5):
            p=F.softmax(mature[i:i+1],-1)
            q=F.softmax(anchor[:,i:i+1],-1)
            mixture=(p.unsqueeze(0)+q)/2
            kl1=F.kl_div(F.log_softmax(mature[i:i+1],-1).unsqueeze(0),mixture,reduction='none').mean(-1)
            kl2=F.kl_div(F.log_softmax(anchor[:,i:i+1],-1),mixture,reduction='none').mean(-1)
            reference.append(((kl1+kl2)*.5).mean()*1e5)
        torch.testing.assert_close(depth,torch.stack(reference),rtol=0,atol=0)
        self.assertTrue(bool(((entropy>=0)&(entropy<=1)).all()))

    def test_rhd_components_and_undefined_case(self):
        r=np.array([1.,2.,.5,1.7,3.2]);e=np.array([.4,.2,.8,.5,.6]);a=np.tril(np.ones((5,5)),-1)
        x=components(r,e,a)
        self.assertAlmostEqual(x[1],np.std(r[:2])/np.mean(r[:2]))
        self.assertAlmostEqual(x[2],np.mean(r))
        self.assertAlmostEqual(x[3],np.corrcoef(np.diff(r),np.diff(e))[0,1])
        with self.assertRaises(ValueError):components([0.],[.2],[[0.]])

    def test_rede_weighting_and_projection_filter(self):
        x=torch.tensor([[1.,0.],[0.,1.]])
        out=perplexity_weighted_embedding(x,torch.tensor([np.log(.5),np.log(.25)]))
        torch.testing.assert_close(out,torch.tensor([1/3,2/3]))
        self.assertIsNone(proxy_indices(torch.arange(3.).float(),.2))
        torch.manual_seed(42)
        traces=[(torch.randn(20,4),torch.rand(20)) for _ in range(3)]
        model,history=fit_projection(traces,hidden_width=6,output_width=3,epochs=2,
            proxy_ratio=.2,lambda_disperse=1.,lambda_separate=1.)
        self.assertEqual(len(history),2)
        keep=retained_indices(model,traces[0][0],drop_ratio=.2,short_trace_policy='keep')
        self.assertEqual(len(keep),16);self.assertEqual(keep,sorted(set(keep)))
        self.assertEqual(retained_indices(model,torch.randn(3,4),drop_ratio=.2,short_trace_policy='keep'),[0,1,2])
        with self.assertRaises(ValueError):retained_indices(model,torch.randn(3,4),drop_ratio=.2,short_trace_policy='error')
