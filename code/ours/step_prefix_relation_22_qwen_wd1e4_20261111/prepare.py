"""CPU preparation: exact existing assignment/position identity and scientific gates."""
import itertools
import json
import numpy as np
import torch
from relation_common import *
from model import objective, make_model
from extract import pools


def tests():
    logits=torch.tensor([[.2,-.4,.9],[-2.,1.,.3],[4.,0.,-1.]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([0,0,1]);c=torch.tensor([0,1,1]);p=logits.softmax(1)
    reference=.5*(torch.nn.functional.binary_cross_entropy(p[:,0],y.double())+
                  torch.nn.functional.binary_cross_entropy(p[:,:2].sum(1),c.double()))
    torch.testing.assert_close(objective(logits,y,c,'marginal_bce'),reference)
    objective(logits,y,c,'marginal_bce').backward();assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(objective(logits,y,c,'ce'),torch.nn.functional.cross_entropy(logits,torch.tensor([2,1,0])))
    rng=np.random.default_rng(42);x=rng.normal(size=(137,9)).astype(np.float32);spans=[(0,13),(15,64),(72,137)]
    result=pools(x,spans,0,READOUTS)
    for i,(a,b) in enumerate(spans):
        np.testing.assert_allclose(result['last_token'][i],x[b-1],rtol=0,atol=0)
        np.testing.assert_allclose(result['global_mean'][i],x[:b].astype(np.float64).mean(0),rtol=1e-6,atol=1e-7)
        w=np.exp(.003*np.arange(1,b+1));w/=w.sum()
        np.testing.assert_allclose(result['prefix_exp'][i],(x[:b]*w[:,None]).sum(0),rtol=1e-6,atol=1e-7)
        t=torch.from_numpy(x[a:b]);w=torch.exp(3*torch.linspace(0,1,len(t)));w/=w.sum()
        np.testing.assert_allclose(result['step_exp'][i],(t*w[:,None]).sum(0).numpy(),rtol=0,atol=0)
    # Full-state resume preserves the next optimizer step and permutation sequence.
    torch.manual_seed(42);a=make_model(9,'mlp');opt=torch.optim.Adam(a.parameters(),lr=config()['learning_rate'],weight_decay=config()['weight_decay'])
    q=torch.from_numpy(x[:3]);objective(a(q),y,c,'ce').backward();opt.step()
    import copy
    state=copy.deepcopy(a.state_dict());ostate=copy.deepcopy(opt.state_dict())
    b=make_model(9,'mlp');b.load_state_dict(state);ob=torch.optim.Adam(b.parameters(),lr=config()['learning_rate'],weight_decay=config()['weight_decay']);ob.load_state_dict(ostate)
    for net,o in [(a,opt),(b,ob)]:o.zero_grad();objective(net(q),y,c,'ce').backward();o.step()
    for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa,pb,rtol=0,atol=0)


def main():
    assert config()['weight_decay'] == 0.0001
    tests();summaries=[]
    for domain in DOMAINS:
        ds=dataset(domain,('train','val','test'))
        keys=ds['train']['keys']+ds['val']['keys']
        source_receipt,_,source_hash=source(domain,'last_token')
        native={s:sha(DATA/'native'/MODEL/domain/(s+'.native.jsonl')) for s in ('train','val')}
        assert source_receipt['identity']['native_files']==native
        assert source_receipt['identity']['ranges']=={'train':[0,len(ds['train']['keys'])],
                 'val':[len(ds['train']['keys']),len(keys)]}
        value=dict(domain=domain,position_hash=hashlib.sha256(json.dumps(keys).encode()).hexdigest(),native_files=native,
                   data_hashes=data_hashes(domain),baseline_source_manifest_SHA256=source_hash,
                   splits={s:dict(trajectories=len(v['rows']),positions=len(v['keys']),step_positive=int(v['y'].sum()),
                                 prefix_positive=int(v['c'].sum()),classes={str(k):int((v['k']==k).sum()) for k in range(3)},
                                 conflicts=int((v['y']>v['c']).sum())) for s,v in ds.items()})
        save(STATE/(domain+'.json'),value)
        for readout in ('step_exp', 'prefix_exp'):
            try:
                source(domain, readout)
            except FileNotFoundError:
                pass
        summaries.append(value)
    save(RUN/'reports/PREPARATION.json',dict(status='PASS',config_SHA256=sha(WORK/'config.json'),
         tests=['stable BCE marginals','CE mapping','Streaming four pools reference','full-state optimizer resume'],domains=summaries))
    return 0


if __name__=='__main__':raise SystemExit(main())
