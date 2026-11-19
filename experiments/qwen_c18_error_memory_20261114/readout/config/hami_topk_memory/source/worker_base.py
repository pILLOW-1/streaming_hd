"""Fixed A2 history readout; frozen original Streaming Step; Prefix BCE only."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss, precision_recall_curve, roc_auc_score
from torch import nn
from torch.nn import functional as F

from evaluate_test import baseline_root
from history import HistorySplit
from pooling import pools
from run import save_json

NAME = 'qwen_a2_prefix_only_frozen_step_20261119'
TEACHERS = ('streaming', 'learnable_aggr')
BASE = Path('/workspace/qwen_relation/results')
INPUT = BASE/'qwen_baseline_prefix_decomposition_20261116/input'
INDEX = Path('/workspace/qwen_relation/indexes/qwen_c18_frozen_history_20261114')
SOURCE = BASE/'qwen_step_only_learnable_20261115/legacy_recipe/code/inputs.json'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def checkpoint(p, value):
    temp=p.with_suffix('.pending')
    torch.save(value,temp)
    temp.replace(p)


def split(domain, part):
    if part!='test':
        return HistorySplit(domain,part,read(SOURCE)[domain][part],INDEX/f'{domain}.{part}.json')
    reference=baseline_root(domain)/'ce'
    p=reference/'test_predictions.npz'
    assert sha(p)==read(reference/'TEST_RESULT.json')['predictions_SHA256']
    with np.load(p,allow_pickle=False) as bank:
        labels={k:(int(y),int(c)) for k,y,c in zip(bank['position_id'],bank['y'],bank['c'],strict=True)}
    index=read(INDEX/f'{domain}.test.json')
    rows=[]
    for row in index['rows']:
        sid=row['sample_id']
        chosen=[(i,span,*labels[f'{sid}:{i}']) for i,span in enumerate(row['step_token_spans']) if f'{sid}:{i}' in labels]
        ids,spans,y,c=zip(*chosen) if chosen else ([],[],[],[])
        rows.append(dict(sample_id=sid,token_sha=row['token_sha'],origin=row['origin'],retained_original_step_ids=ids,step_token_spans=spans,step_hallucination=y,prefix_hallucination=c))
    data=HistorySplit(domain,'test',{'full_sha':index['full_sha']},INDEX/f'{domain}.test.json',labels=rows)
    assert set(data.keys)==set(labels)
    return data


def raw_pools(data, layer, need_prefix=True, only=None):
    """Read existing raw BF16 banks; no backbone inference or persistent pooled bank."""
    name=f'layer_{layer:02d}/reasoning_answer.bf16.npy'
    path=data.root/name
    assert path.stat().st_size==data.receipt['files'][name]['bytes']
    raw=np.load(path,mmap_mode='r',allow_pickle=False)
    assert raw.dtype==np.dtype('<u2') and raw.shape==(data.receipt['tokens'],5120)
    e=np.zeros((len(data.starts),5120),np.float32)
    c=np.empty((len(data.y),5120),np.float32) if need_prefix else None
    for offset,origin,spans,history,slots,selected in data.history:
        visible=(raw[offset:offset+spans[-1][1]-origin].astype(np.uint32)<<16).view(np.float32)
        if only is None:
            e[history]=pools(visible,spans,origin,('step_exp',))['step_exp']
        else:
            missing=[(n,s) for n,s in enumerate(spans) if only[history.start+n]]
            if missing:
                ids,subset=zip(*missing)
                e[history.start+np.asarray(ids)]=pools(visible,subset,origin,('step_exp',))['step_exp']
        if need_prefix:
            c[slots]=pools(visible,[spans[n] for n in selected],origin,('prefix_exp',))['prefix_exp']
    return e,c


def learned_scores(data, domain, part, out):
    from legacy_step_train import Detector
    root=BASE/'qwen_step_only_learnable_20261115/legacy_recipe'/domain/'aggr'
    selected=read(root/'TEST_STARTED.json')['selection']
    assert (selected['layer'],selected['epoch'])=={'Logic':(35,3),'Factuality':(49,1)}[domain]
    side=root/('lower' if selected['layer']<=32 else 'upper')
    teacher=side/selected['checkpoint']
    assert sha(teacher)==selected['checkpoint_sha256']
    binding=dict(inputs=data.pins,teacher_sha256=sha(teacher),teacher_layer=selected['layer'])
    path=out/f'{part}.frozen_step_all_history.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as b:
            assert json.loads(b['binding'].item())==binding
            return b['q'].copy()
    if part=='train':
        p=BASE/'qwen_aggr_kappa_gru_memory_20261117'/domain/'q_train.npz'
        pin=read(p.with_suffix('.json'))
        assert pin['checkpoint_sha256']==sha(teacher) and pin['input']==data.pins
        expected=pin['sha256']
    elif part=='val':
        p=side/selected['prediction'];expected=selected['prediction_sha256']
    else:
        p=root/'test.npz';expected=read(root/'TEST_COMPLETE.json')['sha256']
    assert sha(p)==expected
    with np.load(p,allow_pickle=False) as b:
        lookup={k:i for i,k in enumerate(b['position_id'])}
        ids=[lookup[k] for k in data.keys]
        assert np.array_equal(b['y'][ids],data.y)
        current=b['probability'][ids].copy()
    allq=np.full(len(data.starts),np.nan,np.float32)
    allq[data.positions.numpy()]=current
    missing=~np.isfinite(allq)
    if missing.any():
        name=f"layer_{selected['layer']:02d}/reasoning_answer.bf16.npy"
        assert (data.root/name).stat().st_size==data.receipt['files'][name]['bytes']
        raw=np.load(data.root/name,mmap_mode='r',allow_pickle=False)
        detector=Detector('aggr').eval().requires_grad_(False)
        detector.load_state_dict(torch.load(teacher,map_location='cpu',weights_only=True)['model'])
        with torch.inference_mode():
            for offset,origin,spans,hist,_,_ in data.history:
                for n,(a,b) in enumerate(spans):
                    slot=hist.start+n
                    if missing[slot]:
                        hidden=torch.from_numpy((raw[offset+a-origin:offset+b-origin].astype(np.uint32)<<16).view(np.float32))
                        weights=detector.scorer(hidden).flatten().softmax(0)
                        allq[slot]=detector.step((hidden*weights[:,None]).sum(0)).sigmoid().item()
    assert np.isfinite(allq).all() and np.all((allq>=0)&(allq<=1))
    np.savez_compressed(path,q=allq,binding=json.dumps(binding),saved_current_scores_sha256=sha(p),additional_historical_positions=int(missing.sum()))
    print('FROZEN_Q',domain,part,'learnable_aggr',len(allq),int(missing.sum()),binding,flush=True)
    return allq


def frozen_scores(data, domain, part, out, teacher='streaming'):
    if teacher=='learnable_aggr':return learned_scores(data,domain,part,out)
    manifest=read(INPUT/'manifest.json')
    teacher=INPUT/domain/'streaming/frozen_step.pt'
    assert sha(teacher)==manifest['files'][f'{domain}/streaming/frozen_step.pt']
    binding=dict(inputs=data.pins,teacher_sha256=sha(teacher),teacher_layer=manifest['cells'][domain]['streaming']['layer'])
    path=out/f'{part}.frozen_step_all_history.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as b:
            assert json.loads(b['binding'].item())==binding
            allq=b['q'].copy()
            assert allq.shape==(len(data.starts),)
            return allq
    if part=='train':
        p=BASE/'qwen_baseline_prefix_decomposition_20261116'/domain/'streaming/train_step.npz'
        with np.load(p,allow_pickle=False) as b:
            assert b['frozen_sha'].item()==binding['teacher_sha256'] and b['positions_sha'].item()==data.pins['positions']
            current=b['q'].copy()
    else:
        p=INPUT/domain/'streaming'/f'step_{part}.npz'
        assert sha(p)==manifest['files'][f'{domain}/streaming/step_{part}.npz']
        with np.load(p,allow_pickle=False) as b:
            lookup={key:i for i,key in enumerate(b['sample_id'])}
            order=[lookup[k] for k in data.keys]
            assert np.array_equal(b['y'][order],data.y)
            current=b['scores'][order].copy()
    allq=np.full(len(data.starts),np.nan,np.float32)
    allq[data.positions.numpy()]=current
    missing=~np.isfinite(allq)
    if missing.any():
        e,_=raw_pools(data,binding['teacher_layer'],False,missing)
        state=torch.load(teacher,map_location='cpu',weights_only=True)
        assert state['layer']==binding['teacher_layer']
        model=nn.Linear(5120,1)
        model.load_state_dict(state['model'])
        model.eval()
        with torch.inference_mode():
            allq[missing]=model(F.normalize(torch.from_numpy(e[missing]),dim=1)).sigmoid().flatten().numpy()
    assert np.isfinite(allq).all() and np.all((allq>=0)&(allq<=1))
    np.savez_compressed(path,q=allq,binding=json.dumps(binding),saved_current_scores_sha256=sha(p),additional_historical_positions=int(missing.sum()))
    print('FROZEN_Q',domain,part,len(allq),int(missing.sum()),binding,flush=True)
    return allq


def inputs(data, layer, q, device, pooled=None):
    e,c=raw_pools(data,layer) if pooled is None else pooled
    pos=data.positions.numpy()
    g=np.empty_like(c)
    for _,_,_,hist,slots,selected in data.history:
        weighted=e[hist].astype(np.float64)*q[hist][:,None]
        exclusive=np.concatenate((np.zeros((1,5120)),weighted.cumsum(axis=0)[:-1]),axis=0)
        g[slots]=exclusive[np.asarray(selected)]
    return torch.from_numpy(c).to(device),torch.from_numpy(g).to(device),torch.from_numpy(q[pos]).to(device)


class A2(nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix=nn.Linear(5120,1)
        self.memory=nn.Linear(5120,1,bias=False)
        nn.init.zeros_(self.memory.weight)

    def forward(self,c,g):
        return (self.prefix(c)+self.memory(g)).flatten()


def probability(logit,q):
    return q+(1-q)*logit.sigmoid()


def loss(logit,q,label):
    q=q.detach().double()
    z=logit.double()
    clean=torch.log1p(-q)+F.logsigmoid(-z)
    error=torch.logaddexp(torch.log(q),torch.log1p(-q)+F.logsigmoid(z))
    return -torch.where(label.bool(),error,clean).clamp_min(-100).mean()


def metrics(label,p):
    return dict(AUROC=float(roc_auc_score(label,p)),AP=float(average_precision_score(label,p)),BCE=float(log_loss(label,p.astype(np.float64),labels=[0,1])))


def rank(m):
    return m['AUROC'],m['AP'],-m['BCE']


def fit(domain,layer,out,train,val,x,v,device,binding):
    report=out/f'layer_{layer:02d}.json'
    if report.exists():
        rec=read(report)
        assert rec['binding']==binding
        return rec
    torch.manual_seed(42)
    rng=torch.Generator().manual_seed(42)
    net=A2().to(device)
    opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001)
    y=torch.tensor(train.c,device=device)
    resume=out/f'resume_{layer:02d}.pt'
    best=None; best_state=None; best_p=None; start=1
    if resume.exists():
        state=torch.load(resume,map_location=device,weights_only=True)
        assert state['binding']==binding
        net.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        rng.set_state(state['shuffle'].cpu())
        best=state['best'];best_state=state['best_model'];best_p=state['best_p'].cpu().numpy();start=state['epoch']+1
    curve=[]
    for epoch in range(start,101):
        if best is not None and epoch-best['epoch']>10:break
        for batch in torch.randperm(len(train.c),generator=rng).split(256):
            ids=batch.to(device)
            opt.zero_grad(set_to_none=True)
            value=loss(net(x[0][ids],x[1][ids]),x[2][ids],y[ids])
            assert torch.isfinite(value)
            value.backward();opt.step()
        with torch.inference_mode():
            p=probability(net(v[0],v[1]),v[2]).cpu().numpy()
        m=metrics(val.c,p)
        curve.append(dict(epoch=epoch,**m))
        if best is None or rank(m)>rank(best['metrics']):
            best=dict(epoch=epoch,metrics=m)
            best_state={k:t.detach().cpu().clone() for k,t in net.state_dict().items()}
            best_p=p.copy()
        checkpoint(resume,dict(binding=binding,model=net.state_dict(),optimizer=opt.state_dict(),shuffle=rng.get_state(),epoch=epoch,best=best,best_model=best_state,best_p=torch.from_numpy(best_p)))
        print('VAL',domain,layer,epoch,m,flush=True)
    w=out/f'best_{layer:02d}.pt';vp=out/f'val_{layer:02d}.npz'
    checkpoint(w,dict(model=best_state,binding=binding,selection=best))
    np.savez_compressed(vp,probability=best_p,position_id=val.keys,c=val.c)
    rec=dict(domain=domain,layer=layer,binding=binding,**best,checkpoint=w.name,checkpoint_sha256=sha(w),prediction=vp.name,prediction_sha256=sha(vp),trained_epochs=epoch,curve=curve)
    save_json(report,rec);resume.unlink()
    return rec


def threshold(y,p):
    precision,recall,cuts=precision_recall_curve(y,p)
    f1=np.divide(2*precision[:-1]*recall[:-1],precision[:-1]+recall[:-1],out=np.zeros_like(cuts),where=precision[:-1]+recall[:-1]>0)
    cuts=np.nextafter(cuts,np.float32(-np.inf))
    return float(cuts[np.lexsort((cuts,precision[:-1],f1))[-1]])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('domain',choices=('Logic','Factuality'));a=parser.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda');root=BASE/NAME/a.domain;root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if all((root/t/'TEST_COMPLETE.json').exists() for t in TEACHERS):return
        train,val=(split(a.domain,s) for s in ('train','val'))
        assert not set(train.group)&set(val.group)
        banks={};bindings={};records={t:[] for t in TEACHERS}
        for t in TEACHERS:
            out=root/t;out.mkdir(exist_ok=True)
            banks[t]=tuple(frozen_scores(d,a.domain,s,out,t) for d,s in ((train,'train'),(val,'val')))
            bindings[t]=dict(experiment=NAME,teacher=t,worker_sha256=sha(Path(__file__)),train=train.pins,val=val.pins,q_train_sha256=sha(out/'train.frozen_step_all_history.npz'),q_val_sha256=sha(out/'val.frozen_step_all_history.npz'))
        for layer in range(1,65):
            if all((root/t/f'layer_{layer:02d}.json').exists() for t in TEACHERS):
                for t in TEACHERS:
                    item=read(root/t/f'layer_{layer:02d}.json');assert item['binding']==bindings[t];records[t].append(item)
                continue
            print('POOL',a.domain,layer,flush=True)
            pooled_train=raw_pools(train,layer);pooled_val=raw_pools(val,layer)
            for t in TEACHERS:
                x=inputs(train,layer,banks[t][0],device,pooled_train);v=inputs(val,layer,banks[t][1],device,pooled_val)
                records[t].append(fit(a.domain,layer,root/t,train,val,x,v,device,bindings[t]))
                del x,v
            del pooled_train,pooled_val
        for t in TEACHERS:
            out=root/t
            if (out/'TEST_COMPLETE.json').exists():continue
            finish(a.domain,t,out,records[t],train,val,device)
        save_json(root/'DOMAIN_COMPLETE.json',dict(domain=a.domain,teachers=TEACHERS))


def finish(domain,teacher,out,records,train,val,device):
        winner=max(records,key=lambda r:rank(r['metrics']))
        assert sha(out/winner['checkpoint'])==winner['checkpoint_sha256']
        with np.load(out/winner['prediction'],allow_pickle=False) as b:
            assert np.array_equal(b['position_id'],val.keys) and np.array_equal(b['c'],val.c)
            cut=threshold(val.c,b['probability'])
        selection=dict(winner,threshold=cut);save_json(out/'VAL_LOCK.json',selection)
        test=split(domain,'test');assert not set(test.group)&(set(train.group)|set(val.group))
        with (out/'TEST_STARTED.json').open('x') as f:json.dump(selection,f)
        qtest=frozen_scores(test,domain,'test',out,teacher)
        tx=inputs(test,winner['layer'],qtest,device)
        model=A2().to(device);model.load_state_dict(torch.load(out/winner['checkpoint'],map_location=device,weights_only=True)['model'])
        with torch.inference_mode():
            k=model(tx[0],tx[1]).sigmoid().cpu().numpy();q=tx[2].cpu().numpy();p=q+(1-q)*k
        np.savez_compressed(out/'test.npz',probability=p,kappa=k,q=q,position_id=test.keys,y=test.y,c=test.c,prior=test.prior,group=test.group)
        m=metrics(test.c,p);pred=p>cut;m['F1']=float(2*np.sum(pred*test.c)/(pred.sum()+test.c.sum()))
        cohort=(test.y==0)&((test.c==1)|test.prior)
        recovery=float(roc_auc_score(test.c[cohort],p[cohort]))
        save_json(out/'TEST_COMPLETE.json',dict(domain=domain,teacher=teacher,selection=selection,inputs=test.pins,metrics=m,inherited_vs_recovered_AUROC=recovery,prediction_sha256=sha(out/'test.npz')))
        print('TEST_COMPLETE',domain,teacher,m,recovery,flush=True)


if __name__=='__main__':main()
