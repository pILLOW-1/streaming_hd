"""Numerical gates: averaged sigmoid CE, top-k, causal memory and gradients."""
import ast
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def main():
    torch.set_num_threads(1)
    names = {'top_count', 'selected', 'log_mean_probability', 'Model', 'objective'}
    tree = ast.parse(Path(__file__).with_name('worker.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    ns = dict(torch=torch, nn=nn, F=F, np=np, math=math, ARMS=('H1', 'H2'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'worker.py', 'exec'), ns)
    class Bank:
        def __init__(self):
            self.bags = [[torch.randn(13, 8), torch.randn(20, 8), torch.randn(11, 8)],
                         [torch.randn(12, 8), torch.randn(21, 8)]]
            self.tid = np.array([0, 0, 0, 1, 1]); self.local = np.array([0, 1, 2, 0, 1])
            self.prefix = torch.randn(5, 8)
        def trajectory(self, t, device): return self.bags[t]
        def step(self, t, i, device): return self.bags[t][i]
    for arm in ('H1', 'H2'):
        torch.manual_seed(42)
        model = ns['Model'](arm, width=8).eval()
        with torch.no_grad(): model.memory.weight.normal_(std=.1)
        bank = Bank(); memories = model.historical(bank, [0, 1], torch.device('cpu'))
        assert torch.count_nonzero(memories[0][0]) == 0
        assert torch.count_nonzero(memories[1][0]) == 0
        h = bank.bags[0][0]; s = model.scorer(h).flatten(); idx = ns['selected'](s)
        if arm == 'H2': m = h[idx].mean(0)
        else:
            a = torch.exp(3*torch.linspace(0,1,len(h))); m = (h*(a/a.sum())[:,None]).sum(0)
        assert torch.allclose(memories[0][1], s[idx].sigmoid().mean()*m, atol=1e-6)
        before = memories[0][1].clone(); bank.bags[0][2].add_(100)
        after = model.historical(bank, [0, 1], torch.device('cpu'))[0][1]
        assert torch.equal(before, after), 'future Step entered memory'
        z, lq, l1q = model.forward_positions(bank, np.arange(5))
        y = torch.tensor([1,0,0,1,0]); c = torch.tensor([1,1,0,1,0])
        loss = ns['objective'](z,lq,l1q,y,c)
        ref = -(y*lq+(1-y)*l1q).mean()+((1-y)*F.binary_cross_entropy_with_logits(z,c.float(),reduction='none')).mean()
        assert torch.allclose(loss,ref,atol=1e-6)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        print(arm, 'PASS: C18 loss, exclusive full-history memory, top-k features, gradients')
    s = torch.tensor([1000.,-1000.,0.,500.,-500.,2.,-2.,1.,-1.,3.,-3.],requires_grad=True)
    lq,l1q = ns['log_mean_probability'](s)
    assert torch.isfinite(lq+l1q); (-lq-l1q).backward(); assert torch.isfinite(s.grad).all()
    tied = ns['selected'](torch.zeros(20)); assert torch.equal(tied,torch.arange(3))
    print('PASS: extreme-logit stability and deterministic top-k ties')


if __name__ == '__main__': main()
