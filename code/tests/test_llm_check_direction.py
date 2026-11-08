"""Source-regression checks for the official hallucination-positive score."""
import ast
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from baselines.llm_check.kernel import attention_score
from baselines.run import raw_scores
from sklearn.metrics import roc_auc_score

class LLMCheckDirectionTests(unittest.TestCase):
    def test_official_raw_statistic_and_negative_orientation(self):
        # Same head aggregation as upstream get_attn_eig_prod, including ties.
        diagonal=torch.tensor([[1.,.5,.25],[1.,.2,.1]],dtype=torch.float64)
        reference=sum(torch.log(torch.diag(diagonal[h])).diagonal().mean() for h in range(2))
        torch.testing.assert_close(attention_score(diagonal),reference,rtol=0,atol=1e-14)
        cfg=json.loads((ROOT/'configs/baselines/llm_check.prefix.json').read_text())
        self.assertEqual(cfg['score_sign'],-1)
        raw=np.array([-1.,-4.,-2.,-5.]);labels=np.array([0,1,0,1])
        self.assertEqual(roc_auc_score(labels,raw_scores('llm_check',{'score':raw},cfg)),1.)
    def test_invalid_diagonals_raise(self):
        for d in [torch.tensor([[0.]]),torch.tensor([[-1.]]),torch.tensor([[float('nan')]])]:
            with self.assertRaises(ValueError):attention_score(d)
if __name__=='__main__':unittest.main()
