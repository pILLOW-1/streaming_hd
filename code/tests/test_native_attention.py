from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
from common.native_attention import attention_blocks
from baselines.icr_probe.native import prefix_vector
from baselines.icr_probe.attention import ICRScore


class NativeAttentionTests(unittest.TestCase):
    def test_hybrid_mapping_uses_actual_decoder_indices(self):
        cfg=SimpleNamespace(num_hidden_layers=4,layer_types=['linear_attention','full_attention','linear_attention','full_attention'])
        a=torch.ones(1,8,20,20)/20
        self.assertEqual(list(attention_blocks(cfg,[a,a])),[1,3])
        self.assertEqual(list(attention_blocks(cfg,[None,a,None,a])),[1,3])
        with self.assertRaises(ValueError):attention_blocks(cfg,[a])

    def test_native_icr_matches_official_contiguous_case(self):
        torch.manual_seed(42);prompt=20;end=24;layers=2
        hidden=[torch.randn(end,4) for _ in range(layers+1)]
        attention=[]
        for _ in range(layers):
            score=torch.randn(8,end,end).masked_fill(torch.ones(end,end,dtype=torch.bool).triu(1),-torch.inf)
            attention.append(score.softmax(-1))
        hs=[tuple(h[:prompt][None] for h in hidden)]
        ats=[tuple(a[:,:prompt,:prompt][None] for a in attention)]
        for token in range(prompt,end):
            hs.append(tuple(h[token:token+1][None] for h in hidden))
            ats.append(tuple(a[:,token:token+1,:token+1][None] for a in attention))
        core=dict(user_prompt_start=1,user_prompt_end=prompt-1,response_start=prompt)
        scorer=ICRScore(hs,ats,core_positions=core,icr_device=torch.device('cpu'))
        scores,_=scorer.compute_icr(top_k=None,top_p=.1,pooling='mean',attention_uniform=False,hidden_uniform=False,use_induction_head=True)
        expected=np.asarray(scores).mean(1)
        actual=prefix_vector(hidden,dict(enumerate(attention)),prompt,end,core_positions=core)
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
        # A skipped hybrid block must select its own incoming/outgoing states.
        mapped=prefix_vector(hidden,{1:attention[1]},prompt,end,core_positions=core)
        np.testing.assert_array_equal(mapped,actual[1:])
