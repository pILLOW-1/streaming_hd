from pathlib import Path
import ast
import hashlib
import json
import sys
import unittest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from ours.forward import STEP_MODELS,PREFIX_MODELS,step_forward


class ResearchModelTests(unittest.TestCase):
    def test_exported_model_equations_match_preserved_sources(self):
        root=ROOT/'code/ours';text=(root/'models.py').read_text()
        classes={n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.ClassDef)}
        for row in json.loads((root/'MODEL_BINDINGS.json').read_text()):
            self.assertEqual(hashlib.sha256(classes[row['class_name']].encode()).hexdigest(),row['export_SHA256'])
            original=(root/row['source']).read_text()
            node=next(n for n in ast.parse(original).body if isinstance(n,ast.ClassDef) and n.name==row['class_name'])
            self.assertEqual(hashlib.sha256(ast.get_source_segment(original,node).encode()).hexdigest(),row['source_SHA256'])

    def test_all_step_architectures_forward_and_backward(self):
        for name,constructor in STEP_MODELS.items():
            with self.subTest(name=name):
                torch.manual_seed(42);model=constructor(8)
                bags=[torch.randn(n,8) for n in (2,4,1)]
                logits,z=step_forward(model,bags,torch.device('cpu'))
                self.assertEqual(logits.shape,(3,));self.assertEqual(len(z),3)
                torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([0.,1.,0.])).backward()
                self.assertTrue(any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))

    def test_all_prefix_architectures_reset_and_no_future(self):
        for name,constructor in PREFIX_MODELS.items():
            with self.subTest(name=name):
                torch.manual_seed(42);model=constructor(8).eval()
                z=torch.nn.functional.normalize(torch.randn(5,8),dim=1)
                states=[]
                hook=model.probe.register_forward_pre_hook(lambda module,args: states.append(args[0]))
                score=model.forward_sequence(z)[0]
                hook.remove()
                changed=z.clone();changed[3:]=torch.randn_like(changed[3:])
                causal=model.forward_sequence(changed)[0]
                # Same matrix shape isolates future-token leakage from GEMM rounding.
                torch.testing.assert_close(score[:3],causal[:3],rtol=0,atol=0)
                torch.testing.assert_close(states[0][0],z[0],rtol=0,atol=0)
                score.sum().backward()
