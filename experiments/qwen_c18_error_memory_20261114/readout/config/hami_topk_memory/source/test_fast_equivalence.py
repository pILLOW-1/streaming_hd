"""Synthetic equivalence checks; never runs TRAIN/VAL/TEST model replay."""
import ast
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def extract(path,names,namespace):
    tree=ast.parse(path.read_text())
    nodes=[x for x in tree.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)


def main():
    torch.set_num_threads(1)
    device=torch.device('cuda' if '--cuda' in sys.argv else 'cpu')
    width=5120 if '--wide' in sys.argv else 8
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    folder=Path(__file__).resolve().parent
    ns=dict(torch=torch,nn=nn,F=F,np=np,math=math,ARMS=('H1','H2'))
    extract(folder/'worker.py',{'top_count','selected','log_mean_probability','Model','objective','predict'},ns)
    original_predict=ns['predict']
    extract(folder/'worker_fast.py',{'metadata','raw_positions','gather','top_grid','historical','fast_forward','fast_predict'},ns)
    class Bank:
        def __init__(self):
            lengths=[1,13,20,5,33,17,2,9]
            h=torch.randn(sum(lengths),width).to(torch.bfloat16)
            self.raw=h.view(torch.int16).numpy().view(np.uint16).copy()
            self.spans=[];offset=0
            for length in lengths:self.spans.append((offset,offset+length));offset+=length
            self.tid=np.asarray([0,0,0,0,1,1,1,1]);self.local=np.asarray([0,1,2,3,0,1,2,3])
            self.prefix=torch.randn(8,width).to(device)
            histories=[(0,0,self.spans[:4],slice(0,4),slice(0,4),[0,1,2,3]),
                       (0,0,self.spans[4:],slice(4,8),slice(4,8),[0,1,2,3])]
            self.data=SimpleNamespace(history=histories,starts=np.zeros(8),y=np.asarray([0,1,0,0,1,0,0,1]))
            self.fixed_e=[]
            for a,b in self.spans:
                bag=torch.from_numpy(self.raw[a:b].view(np.int16)).view(torch.bfloat16).float()
                w=torch.exp(3*torch.linspace(0,1,len(bag)));self.fixed_e.append((bag*(w/w.sum())[:,None]).sum(0).numpy())
            self.fixed_e=np.stack(self.fixed_e)
        def step(self,tid,i,device):
            a,b=self.data.history[tid][2][i]
            return torch.from_numpy(self.raw[a:b].view(np.int16)).view(torch.bfloat16).float().to(device)
        def trajectory(self,tid,device):return [self.step(tid,i,device) for i in range(4)]
    maxima={}
    for seed in (42,7):
        for arm in ('H1','H2'):
            torch.manual_seed(seed);bank=Bank();slow=ns['Model'](arm,width=width).to(device)
            with torch.no_grad():slow.memory.weight.normal_(std=.1)
            fast=copy.deepcopy(slow)
            for train in (False,True):
                slow.train(train);fast.train(train)
                ids=np.asarray([5,0,3,6,2,7])
                slow.zero_grad();fast.zero_grad()
                rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None
                old=slow.forward_positions(bank,ids)
                torch.set_rng_state(rng)
                if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
                new=ns['fast_forward'](fast,bank,ids)
                for a,b in zip(old,new):torch.testing.assert_close(a,b,atol=3e-6,rtol=2e-5)
                y=torch.tensor([1,0,0,0,1,1],device=device);c=torch.tensor([1,1,0,0,1,1],device=device)
                loss_old=ns['objective'](*old,y,c);loss_new=ns['objective'](*new,y,c)
                torch.testing.assert_close(loss_old,loss_new,atol=1e-6,rtol=1e-6)
                loss_old.backward();loss_new.backward()
                for (name,a),(_,b) in zip(slow.named_parameters(),fast.named_parameters()):
                    torch.testing.assert_close(a.grad,b.grad,atol=5e-6,rtol=5e-5)
                for a,b in zip(slow.scorer[1].buffers(),fast.scorer[1].buffers()):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-7)
            old=original_predict(slow,bank);new=ns['fast_predict'](fast,bank)
            for a,b in zip(old,new):np.testing.assert_allclose(a,b,atol=3e-6,rtol=2e-5)
            # A request for earlier positions must not score or use future Steps.
            slow.eval();ids=np.asarray([1,5]);old=slow.forward_positions(bank,ids)
            new=ns['fast_forward'](fast.eval(),bank,ids)
            for a,b in zip(old,new):torch.testing.assert_close(a,b,atol=3e-6,rtol=2e-5)
            maxima[f'{arm}_seed{seed}']='PASS'
    for lengths in ([1,2,3],[10,20,31,64],[17,17,17]):
        v=torch.zeros(sum(lengths),device=device);grid,mask,counts=ns['top_grid'](v,lengths)
        offset=0
        for i,n in enumerate(lengths):
            expected=torch.arange(offset,offset+int(n*.1)+1,device=device)
            assert torch.equal(grid[i][mask[i]],expected);offset+=n
    out=dict(status='PASS',checks=maxima,width=width,forward_loss_gradient_BN_Dropout_and_causal_prefix='PASS',
        deterministic_topk_ties='PASS',fast_sha256=hashlib.sha256((folder/'worker_fast.py').read_bytes()).hexdigest(),
        contract_sha256=hashlib.sha256((folder/'worker.py').read_bytes()).hexdigest(),
        scope='synthetic '+device.type+' gates, not a scientific result or independent VAL/TEST inference')
    target='FAST_CUDA_EQUIVALENCE_GATE.json' if device.type=='cuda' else 'FAST_EQUIVALENCE_GATE.json'
    (folder/target).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))


if __name__=='__main__':main()
