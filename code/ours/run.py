#!/usr/bin/env python3
"""Portable Adam Step training and frozen-teacher recurrent Prefix training."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))


from common.checkpoint_selection import best_snapshot

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths',type=Path,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--domain',required=True)
    parser.add_argument('--step-method',default='learnable_aggregation_representation')
    parser.add_argument('--prefix-method',default='scalar_gate')
    parser.add_argument('--layers',default='all')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    import numpy as np
    import torch
    from common.dataset import load_cell
    from common.token_cache import TokenCache
    from ours.forward import STEP_MODELS,PREFIX_MODELS,step_forward
    from baselines.streaming.run import evaluate_metrics,labels,choose,save,objectives
    from baselines.streaming.stream_metrics import select_threshold
    if args.step_method not in STEP_MODELS or args.prefix_method not in PREFIX_MODELS:
        parser.error('Unknown method; see code/ours/forward.py for supported names')
    paths=json.loads(args.paths.read_text());rows=load_cell(paths,args.model,args.domain)
    if not paths.get('feature_cache_root') or not paths.get('output_root'):parser.error('Configure feature_cache_root and output_root')
    device=torch.device(args.device)
    cfg=json.loads((ROOT/'experiments/streaming_622_p70/config.json').read_text())
    cell=next(c for c in cfg['cells'] if (c['model'],c['domain'],c['level'])==(args.model,args.domain,'step'))
    layers=cell['candidate_layers'] if args.layers=='all' else [int(x) for x in args.layers.split(',')]
    out=Path(paths['output_root'])/'ours'/args.step_method/args.prefix_method/args.model/args.domain
    out.mkdir(parents=True,exist_ok=False)
    save(out/'config.json',dict(step_method=args.step_method,prefix_method=args.prefix_method,layers=layers,
        step_recipe=dict(optimizer='Adam',lr=.001,weight_decay=.0005,batch_steps=128,epochs=20,patience=5),
        prefix_recipe=dict(optimizer='AdamW',lr=.001,weight_decay=.01,batch_trajectories=64,epochs=20,final_weight=30,sync_weight=120),
        selection='own VAL AUROC, then AP; no layer or update preference',seed=42))
    cache_root=Path(paths['feature_cache_root'])/args.model
    def seed():
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    def step_scores(model,cache):
        model.eval();scores=[]
        with torch.no_grad():
            for start in range(0,len(cache),128):
                logits,_=step_forward(model,cache.bags(range(start,min(start+128,len(cache)))),device)
                scores.append(logits.sigmoid().cpu().numpy())
        return np.concatenate(scores)
    candidates=[]
    for layer in layers:
        seed();model=STEP_MODELS[args.step_method](cell['width']).to(device)
        tr=TokenCache(cache_root,rows['train'],layer);va=TokenCache(cache_root,rows['val'],layer)
        y=torch.tensor(labels(rows['train'],'step'),dtype=torch.float32,device=device);vy=labels(rows['val'],'step')
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0005)
        best=None;stale=0
        for epoch in range(1,21):
            model.train();order=torch.randperm(len(tr)).tolist()
            for begin in range(0,len(order),128):
                ids=order[begin:begin+128];optimizer.zero_grad()
                logits,_=step_forward(model,tr.bags(ids),device)
                torch.nn.functional.binary_cross_entropy_with_logits(logits,y[ids]).backward();optimizer.step()
            scores=step_scores(model,va);metric=evaluate_metrics(vy,scores,.5);rank=(metric['AUROC'],metric['AP'])
            old=None if best is None else best['rank'];best=best_snapshot(rank,model.state_dict(),best)
            stale=0 if old is None or rank>old else stale+1
            if stale>=5:break
        if best['tie']:raise ValueError('Exact best VAL AUROC/AP Step checkpoint tie')
        name=f'step_layer_{layer}.pt';torch.save(best['state'],out/name)
        candidates.append(dict(layer=layer,path=name,AUROC=best['rank'][0],AP=best['rank'][1]))
    selected=choose(candidates);save(out/'step_candidates.json',candidates)
    step=STEP_MODELS[args.step_method](cell['width']).to(device)
    step.load_state_dict(torch.load(out/selected['path'],weights_only=True,map_location=device));step.eval();step.requires_grad_(False)
    cache={s:TokenCache(cache_root,r,selected['layer']) for s,r in rows.items()}
    def representations(split):
        z=[];alarm=[]
        with torch.no_grad():
            for start in range(0,len(cache[split]),128):
                logits,value=step_forward(step,cache[split].bags(range(start,min(start+128,len(cache[split])))),device)
                z.append(value.cpu());alarm.append(logits.sigmoid().cpu())
        return torch.cat(z),torch.cat(alarm)
    train_z,train_alarm=representations('train');val_z,val_alarm=representations('val')
    seed();prefix=PREFIX_MODELS[args.prefix_method](train_z.shape[1]).to(device)
    # This portable composition explicitly uses global-mean initialization.
    # It is a new-dataset recipe, not a claim of historical bitwise reproduction.
    initial=[]
    for part in cache['train'].trajectory_slices:
        value=train_z[part].to(device);mean=value.cumsum(0)/torch.arange(1,len(value)+1,device=device)[:,None]
        initial.append(torch.nn.functional.normalize(mean,dim=1))
    x=torch.cat(initial);py=torch.tensor(labels(rows['train'],'prefix'),dtype=torch.float32,device=device)
    opt=torch.optim.LBFGS(prefix.probe.parameters(),lr=1.,max_iter=100)
    def closure():
        opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(prefix.probe(x).flatten(),py)
        loss=loss+1e-4*sum(.5*p.square().sum() for p in prefix.probe.parameters());loss.backward();return loss
    opt.step(closure)
    optimizer=torch.optim.AdamW(prefix.parameters(),lr=.001,weight_decay=.01)
    def prefix_scores(z,slices):
        prefix.eval()
        with torch.no_grad():return np.concatenate([prefix.forward_sequence(z[part].to(device))[0].sigmoid().cpu().numpy() for part in slices])
    best=None;vy=labels(rows['val'],'prefix')
    for epoch in range(1,21):
        prefix.train();order=list(cache['train'].trajectory_slices);random.shuffle(order)
        for begin in range(0,len(order),64):
            parts=order[begin:begin+64]
            logits=torch.cat([prefix.forward_sequence(train_z[p].to(device))[0] for p in parts])
            ids=torch.tensor([i for p in parts for i in range(p.start,p.stop)],device=device)
            alarm=torch.cat([train_alarm[p] for p in parts]).to(device)
            optimizer.zero_grad();loss=objectives(logits,py[ids],alarm,[p.stop-p.start for p in parts]);loss.backward();optimizer.step()
        scores=prefix_scores(val_z,cache['val'].trajectory_slices);m=evaluate_metrics(vy,scores,.5)
        best=best_snapshot((m['AUROC'],m['AP']),prefix.state_dict(),best)
    if best['tie']:raise ValueError('Exact best VAL AUROC/AP Prefix checkpoint tie')
    prefix.load_state_dict(best['state']);torch.save(best['state'],out/'prefix.pt')
    vs_step=val_alarm.numpy();vs_prefix=prefix_scores(val_z,cache['val'].trajectory_slices)
    last_val=np.cumsum([len(r['retained_original_step_ids']) for r in rows['val']])-1
    val_scores={'step':vs_step,'prefix':vs_prefix,'answer':vs_prefix[last_val]}
    val_labels={k:(np.asarray([r['answer_hallucination'] for r in rows['val']]) if k=='answer' else labels(rows['val'],k)) for k in val_scores}
    thresholds={k:select_threshold(val_labels[k],v) for k,v in val_scores.items()}
    save(out/'FROZEN_SELECTION.json',dict(step=selected,prefix_VAL=dict(AUROC=best['rank'][0],AP=best['rank'][1]),thresholds=thresholds))
    test_z,test_alarm=representations('test');ts_prefix=prefix_scores(test_z,cache['test'].trajectory_slices)
    last_test=np.cumsum([len(r['retained_original_step_ids']) for r in rows['test']])-1
    scores={'step':test_alarm.numpy(),'prefix':ts_prefix,'answer':ts_prefix[last_test]}
    result={}
    for level,values in scores.items():
        targets=np.asarray([r['answer_hallucination'] for r in rows['test']]) if level=='answer' else labels(rows['test'],level)
        result[level]=evaluate_metrics(targets,values,thresholds[level]);np.savez(out/(level+'_predictions.npz'),scores=values,labels=targets)
    save(out/'result.json',result)
    print(json.dumps(dict(status='COMPLETE',output=str(out))))


if __name__=='__main__':main()
