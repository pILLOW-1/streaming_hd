"""Four additional A2/A1 loss and Step initialization controls."""
import argparse,fcntl,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import worker_base as w
import matched_repr as matched

UNCACHED_RAW_POOLS=w.raw_pools

NAME='qwen_a2_step_readout_r1_r2_20261119'
ARMS=('R1','R2')


class Model(nn.Module):
    def __init__(self,arm,initial=None):
        super().__init__();self.arm=arm
        width=5120 if arm=='R1' else 10240
        self.step=nn.Linear(width,1);self.prefix=nn.Linear(5120,1)
        self.memory=nn.Linear(width,1,bias=False);nn.init.zeros_(self.memory.weight)

    def forward(self,x,ids=None):
        if ids is None:ids=torch.arange(len(x['c']),device=x['c'].device)
        z=self.prefix(x['c'][ids]).flatten()
        s_all=self.step(x['e']).flatten();q_all=s_all.sigmoid()
        weighted=self.memory(x['e']).flatten()*q_all.detach()
        cum=F.pad(weighted.cumsum(0,dtype=torch.float64),(1,0));pos=x['positions'][ids]
        z=z+(cum[pos]-cum[x['starts'][pos]]).to(z.dtype)
        return z,s_all[pos],q_all[pos]

def objective(arm,z,s,q,y,c):
    no_step=F.logsigmoid(-s)
    logp=torch.stack((F.logsigmoid(s),no_step+F.logsigmoid(z),no_step+F.logsigmoid(-z)),dim=1)
    target=torch.where(y.bool(),0,torch.where(c.bool(),1,2)).long()
    return F.nll_loss(logp,target)


def tensors(data,c,e,device):
    return dict(c=torch.from_numpy(c).to(device),e=torch.from_numpy(e).to(device),positions=data.positions.to(device),starts=data.starts.to(device))


def joint_metrics(val,p,q):
    result=w.metrics(val.c,p)
    step=w.metrics(val.y,q)
    result.update(step_AUROC=step['AUROC'],step_AP=step['AP'],step_BCE=step['BCE'])
    result.update(mean_AUROC=(result['AUROC']+step['AUROC'])/2,mean_AP=(result['AP']+step['AP'])/2,mean_BCE=(result['BCE']+step['BCE'])/2)
    return result


def joint_rank(m):
    return m['mean_AUROC'],m['mean_AP'],-m['mean_BCE']


def fit(domain,arm,layer,out,train,val,x,v,initial,binding,device):
    report=out/f'layer_{layer:02d}.json'
    if report.exists():
        r=w.read(report);assert r['binding']==binding;return r
    torch.manual_seed(42);rng=torch.Generator().manual_seed(42)
    model=Model(arm,initial).to(device);opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    y=torch.tensor(train.y,device=device);c=torch.tensor(train.c,device=device)
    resume=out/f'resume_{layer:02d}.pt';best=None;best_state=None;best_p=None;best_q=None;start=1
    if resume.exists():
        s=torch.load(resume,map_location=device,weights_only=True);assert s['binding']==binding
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);rng.set_state(s['shuffle'].cpu())
        start=s['epoch']+1;best=s['best'];best_state=s['best_model'];best_p=s['best_p'].cpu().numpy();best_q=s['best_q'].cpu().numpy()
    curve=[]
    for epoch in range(start,101):
        if best is not None and epoch-best['epoch']>10:break
        for ids in torch.randperm(len(train.c),generator=rng).split(256):
            ids=ids.to(device);opt.zero_grad(set_to_none=True)
            z,s,q=model(x,ids);loss=objective(arm,z,s,q,y[ids],c[ids]);assert torch.isfinite(loss)
            loss.backward();opt.step()
        with torch.inference_mode():z,_,q=model(v);p=w.probability(z,q).cpu().numpy()
        q_val=q.detach().cpu().numpy();m=joint_metrics(val,p,q_val);curve.append(dict(epoch=epoch,**m))
        if best is None or joint_rank(m)>joint_rank(best['metrics']):
            best=dict(epoch=epoch,metrics=m);best_state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()};best_p=p.copy();best_q=q_val.copy()
        w.checkpoint(resume,dict(binding=binding,model=model.state_dict(),optimizer=opt.state_dict(),shuffle=rng.get_state(),epoch=epoch,best=best,best_model=best_state,best_p=torch.from_numpy(best_p),best_q=torch.from_numpy(best_q)))
        print('VAL',domain,arm,layer,epoch,m,flush=True)
    pt=out/f'best_{layer:02d}.pt';vp=out/f'val_{layer:02d}.npz'
    w.checkpoint(pt,dict(model=best_state,binding=binding,selection=best));np.savez_compressed(vp,probability=best_p,q=best_q,position_id=val.keys,c=val.c,y=val.y)
    r=dict(domain=domain,layer=layer,binding=binding,**best,checkpoint=pt.name,checkpoint_sha256=w.sha(pt),prediction=vp.name,prediction_sha256=w.sha(vp),trained_epochs=epoch,curve=curve)
    w.save_json(report,r);resume.unlink();return r


def setup(domain,arm,data,out,device):
    return {},None,dict(step_input='last_token' if arm=='R1' else 'concat(step_exp,last_token)',normalization='none',width=5120 if arm=='R1' else 10240)

def layer_inputs(data,layer,arm,base,device):
    import feature_cache as cache
    def build_last():
        name=f'layer_{layer:02d}/reasoning_answer.bf16.npy';path=data.root/name
        assert path.stat().st_size==data.receipt['files'][name]['bytes']
        raw=np.load(path,mmap_mode='r',allow_pickle=False)
        assert raw.shape==(data.receipt['tokens'],5120) and raw.dtype==np.dtype('<u2')
        last=np.empty((len(data.starts),5120),np.float32)
        for offset,origin,spans,history,slots,selected in data.history:
            idx=np.asarray([offset+b-origin-1 for a,b in spans])
            last[history]=(raw[idx].astype(np.uint32)<<16).view(np.float32)
        assert np.isfinite(last).all();return last
    last=cache.get(cache.identity(data,layer,'last_token_all_history'),build_last)
    e=last
    if arm=='R2':
        step=cache.get(cache.identity(data,layer,'step_exp_all_history'),lambda:UNCACHED_RAW_POOLS(data,layer,False)[0])
        e=np.concatenate((step,last),axis=1)
    c=matched.prefix(data,layer)
    assert e.shape==(len(data.starts),5120 if arm=='R1' else 10240) and c.shape==(len(data.y),5120)
    return tensors(data,c,e,device)


def main():
    p=argparse.ArgumentParser();p.add_argument('domain',choices=('Logic','Factuality'));p.add_argument('arm',choices=ARMS);a=p.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda');out=w.BASE/NAME/a.domain/a.arm;out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'TEST_COMPLETE.json').exists():return
        train,val=(w.split(a.domain,s) for s in ('train','val'));assert not set(train.group)&set(val.group)
        bt,initial,pin=setup(a.domain,a.arm,train,out,device);bv,_,vpin=setup(a.domain,a.arm,val,out,device);assert pin==vpin
        binding=dict(experiment=NAME,arm=a.arm,worker_sha256=w.sha(Path(__file__)),train=train.pins,val=val.pins,Step=pin,selection_protocol='joint mean Step/Prefix VAL AUROC, AP, lower BCE')
        if a.arm in ('F5','F6'):
            binding.update(q_train_sha256=w.sha(out/'train.frozen_step_all_history.npz'),q_val_sha256=w.sha(out/'val.frozen_step_all_history.npz'))
        print('BOUND',a.domain,a.arm,binding,flush=True)
        records=[]
        for layer in range(1,65):
            if (out/f'layer_{layer:02d}.json').exists():
                r=w.read(out/f'layer_{layer:02d}.json');assert r['binding']==binding;records.append(r);continue
            print('POOL',a.domain,a.arm,layer,flush=True)
            x=layer_inputs(train,layer,a.arm,bt,device);v=layer_inputs(val,layer,a.arm,bv,device)
            records.append(fit(a.domain,a.arm,layer,out,train,val,x,v,initial,binding,device));del x,v
        winner=max(records,key=lambda r:joint_rank(r['metrics']))
        with np.load(out/winner['prediction'],allow_pickle=False) as b:cut=w.threshold(val.c,b['probability'])
        lock=dict(winner,threshold=cut);w.save_json(out/'VAL_LOCK.json',lock)
        test=w.split(a.domain,'test');assert not set(test.group)&(set(train.group)|set(val.group))
        with (out/'TEST_STARTED.json').open('x') as f:json.dump(lock,f)
        btest,_,_=setup(a.domain,a.arm,test,out,device);tx=layer_inputs(test,winner['layer'],a.arm,btest,device)
        net=Model(a.arm,initial).to(device);net.load_state_dict(torch.load(out/winner['checkpoint'],map_location=device,weights_only=True)['model'])
        with torch.inference_mode():z,_,q=net(tx);k=z.sigmoid().cpu().numpy();q=q.cpu().numpy();pred=q+(1-q)*k
        np.savez_compressed(out/'test.npz',probability=pred,kappa=k,q=q,position_id=test.keys,y=test.y,c=test.c,prior=test.prior,group=test.group)
        m=w.metrics(test.c,pred);active=pred>cut;m['F1']=float(2*np.sum(active*test.c)/(active.sum()+test.c.sum()))
        cohort=(test.y==0)&((test.c==1)|test.prior);recovery=float(w.roc_auc_score(test.c[cohort],pred[cohort]))
        w.save_json(out/'TEST_COMPLETE.json',dict(domain=a.domain,teacher=a.arm,selection=lock,inputs=test.pins,metrics=m,inherited_vs_recovered_AUROC=recovery,prediction_sha256=w.sha(out/'test.npz')))
        print('TEST_COMPLETE',a.domain,a.arm,m,recovery,flush=True)


if __name__=='__main__':main()
