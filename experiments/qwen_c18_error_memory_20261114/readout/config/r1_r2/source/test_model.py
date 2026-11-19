"""Synthetic numerical gate: C18 objective, exclusive memory, both widths."""
import ast
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

def main():
 tree=ast.parse((Path(__file__).parent/'worker.py').read_text(encoding='utf-8'))
 selected=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Model','objective')]
 ns=dict(torch=torch,nn=nn,F=F)
 exec(compile(ast.Module(body=selected,type_ignores=[]),'worker.py','exec'),ns)
 for arm,width in [('R1',5120),('R2',10240)]:
  torch.manual_seed(42);net=ns['Model'](arm)
  with torch.no_grad():net.memory.weight.normal_(std=.001)
  e=torch.randn(7,width);c=torch.randn(5,5120)
  positions=torch.tensor([0,2,3,5,6]);starts=torch.tensor([0,0,0,3,3,3,3]);x=dict(e=e,c=c,positions=positions,starts=starts)
  z,s,q=net(x);expected=net.prefix(c).flatten().clone()
  for n,pos in enumerate(positions):
   indices=torch.arange(starts[pos],pos)
   if len(indices):expected[n]+=net.memory((net.step(e[indices]).sigmoid().detach()*e[indices]).sum(0)).squeeze()
  assert torch.allclose(z,expected,atol=2e-6,rtol=1e-5)
  y=torch.tensor([1,0,0,1,0]);label=torch.tensor([1,1,0,1,0]);loss=ns['objective'](arm,z,s,q,y,label)
  reference=F.binary_cross_entropy_with_logits(s,y.float())+((1-y)*F.binary_cross_entropy_with_logits(z,label.float(),reduction='none')).mean()
  assert torch.allclose(loss,reference,atol=1e-6)
  loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
  print(arm,'PASS: dimensions, strictly earlier within-trajectory memory, C18 CE and finite gradients')

if __name__=='__main__':main()
