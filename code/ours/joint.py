#!/usr/bin/env python3
"""Joint learned-token Step / scalar-gate Prefix training on local dataset cells."""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--domain',required=True)
    p.add_argument('--layers',default='all');p.add_argument('--device',default='cuda');p.add_argument('--regime',choices=['j_unified','j_dual'],default='j_unified')
    a=p.parse_args()
    import numpy as np
    import torch
    from ours.models import LearnedTokenStep,DynamicPrefixGate
    from ours.forward import step_forward
    from common.checkpoint_selection import best_snapshot
    from common.dataset import load_cell
    from common.token_cache import TokenCache
    from common.baseline_training import seed_all
    from baselines.streaming.run import labels,objectives,evaluate_metrics,choose,save
    from baselines.streaming.stream_metrics import select_threshold
    paths=json.loads(a.paths.read_text());rows=load_cell(paths,a.model,a.domain)
    if not paths.get('output_root') or not paths.get('feature_cache_root'):p.error('Set output_root and feature_cache_root')
    cfg=json.loads((ROOT/'experiments/streaming_622_p70/config.json').read_text())
    cell=next(c for c in cfg['cells'] if (c['model'],c['domain'],c['level'])==(a.model,a.domain,'step'))
    layers=cell['candidate_layers'] if a.layers=='all' else list(map(int,a.layers.split(',')))
    out=Path(paths['output_root'])/'ours'/a.regime/a.model/a.domain;out.mkdir(parents=True,exist_ok=False)
    save(out/'config.json',dict(regime=a.regime,layers=layers,seed=42,epochs=20,batch_trajectories=64,
         checkpoint_selection='VAL Prefix AUROC then AP',final_anchor=30,sync_weight=120))
    class Joint(torch.nn.Module):
        def __init__(self):
            super().__init__();seed_all(42);self.step=LearnedTokenStep(cell['width']);seed_all(42);self.prefix=DynamicPrefixGate(cell['width'])
    def batch(model,cache,parts):
        ids=[i for part in parts for i in range(part.start,part.stop)]
        step_logits,z=step_forward(model.step,cache.bags(ids),torch.device(a.device))
        values=[];cursor=0
        for part in parts:
            n=part.stop-part.start;values.append(model.prefix.forward_sequence(z[cursor:cursor+n])[0]);cursor+=n
        return step_logits,torch.cat(values),ids
    def score(model,cache):
        model.eval();s=[];p=[]
        with torch.no_grad():
            for part in cache.trajectory_slices:
                sl,pl,_=batch(model,cache,[part]);s.append(sl.sigmoid().cpu().numpy());p.append(pl.sigmoid().cpu().numpy())
        return dict(step=np.concatenate(s),prefix=np.concatenate(p))
    candidates=[]
    for layer in layers:
        seed_all(42);model=Joint().to(a.device)
        cache={s:TokenCache(Path(paths['feature_cache_root'])/a.model,rows[s],layer) for s in ('train','val')}
        step_parameters=list(model.step.parameters());prefix_parameters=list(model.prefix.parameters())
        if a.regime=='j_dual':
            optimizers=[torch.optim.Adam(step_parameters,lr=.001,weight_decay=.0005),torch.optim.AdamW(prefix_parameters,lr=.001,weight_decay=.01)]
        else:
            optimizers=[torch.optim.AdamW([dict(params=step_parameters,lr=.001,weight_decay=.0005),dict(params=prefix_parameters,lr=.001,weight_decay=.01)])]
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[0],mode='min',factor=.1,patience=20,min_lr=1e-6)
        ys=torch.tensor(labels(rows['train'],'step'),dtype=torch.float32,device=a.device)
        yp=torch.tensor(labels(rows['train'],'prefix'),dtype=torch.float32,device=a.device);vy=labels(rows['val'],'prefix')
        best=None
        for epoch in range(1,21):
            model.train();order=list(cache['train'].trajectory_slices);random.Random(42000+epoch).shuffle(order);total=0.;nrows=0
            for start in range(0,len(order),64):
                parts=order[start:start+64];sl,pl,indices=batch(model,cache['train'],parts)
                ids=torch.tensor(indices,device=a.device)
                step_loss=torch.nn.functional.binary_cross_entropy_with_logits(sl,ys[ids])
                prefix_loss=objectives(pl,yp[ids],sl.sigmoid().detach(),[x.stop-x.start for x in parts])
                for optimizer in optimizers:optimizer.zero_grad()
                (step_loss+prefix_loss).backward()
                for optimizer in optimizers:optimizer.step()
                total+=float(step_loss.detach())*len(ids);nrows+=len(ids)
            scheduler.step(total/nrows)
            vs=score(model,cache['val']);metric=evaluate_metrics(vy,vs['prefix'],.5)
            best=best_snapshot((metric['AUROC'],metric['AP']),model.state_dict(),best)
        if best['tie']:raise ValueError('Exact best VAL AUROC/AP joint checkpoint tie')
        name=f'layer_{layer}.pt';torch.save(best['state'],out/name)
        candidates.append(dict(layer=layer,path=name,AUROC=best['rank'][0],AP=best['rank'][1]))
    selected=choose(candidates);model=Joint().to(a.device)
    model.load_state_dict(torch.load(out/selected['path'],weights_only=True,map_location=a.device))
    def collect(split):
        cache=TokenCache(Path(paths['feature_cache_root'])/a.model,rows[split],selected['layer']);scores=score(model,cache)
        last=np.cumsum([len(r['retained_original_step_ids']) for r in rows[split]])-1
        scores['answer']=scores['prefix'][last]
        targets={k:(np.asarray([r['answer_hallucination'] for r in rows[split]]) if k=='answer' else labels(rows[split],k)) for k in scores}
        return scores,targets
    vs,vy=collect('val');thresholds={k:select_threshold(vy[k],v) for k,v in vs.items()}
    save(out/'FROZEN_SELECTION.json',dict(selected=selected,thresholds=thresholds))
    ts,ty=collect('test');result={}
    for level,scores in ts.items():
        result[level]=evaluate_metrics(ty[level],scores,thresholds[level]);np.savez(out/(level+'_predictions.npz'),labels=ty[level],scores=scores)
    save(out/'result.json',result);save(out/'candidates.json',candidates)


if __name__=='__main__':main()
