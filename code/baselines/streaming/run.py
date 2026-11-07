#!/usr/bin/env python3
"""Portable all-layer Streaming reproduction and evaluation of released detectors."""
import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.dataset import load_cell, load_split, verify_assignment
from baselines.streaming.observed_lbfgs import ObservedLBFGS
from baselines.streaming.stream_metrics import metrics, select_threshold


def evaluate_metrics(y, scores, threshold):
    value = metrics(y, scores, threshold)
    if value['positive'] and value['negative']:
        order = np.argsort(-scores, kind='stable')
        ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0)+1, len(y)]
        tp = np.cumsum(y[order])[ends-1]
        fpr = (ends-tp)/value['negative']
        value['TPR_at_1pct_FPR'] = float(np.r_[0, tp[fpr <= .01]/value['positive']].max())
    else:
        value['TPR_at_1pct_FPR'] = None
    return value


def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def load_rows(root, model, domain, split):
    return load_split(root, model, domain, split)


def matrix(root, rows, kind, layer, device):
    parts = []
    for row in rows:
        directory = Path(root)/row['sample_id']
        manifest = json.loads((directory/'MANIFEST.json').read_text())
        assert manifest['retained_original_step_ids'] == row['retained_original_step_ids']
        name = f'{kind}.layer_{layer}.npy'; path=directory/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['files'][name]['SHA256']
        x=np.load(path, allow_pickle=False)
        assert x.shape[0] == len(row['retained_original_step_ids']) and np.isfinite(x).all()
        parts.append(x)
    x=torch.tensor(np.concatenate(parts), dtype=torch.float32, device=device)
    return torch.nn.functional.normalize(x, dim=1)


def labels(rows, level):
    return np.asarray([v for r in rows for v in r[level+'_hallucination']], dtype=np.int64)


def probabilities(model, x):
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(x[i:i+2048]).flatten()).cpu().numpy() for i in range(0,len(x),2048)])


def choose(candidates):
    best=max((x['AUROC'],x['AP']) for x in candidates)
    winners=[x for x in candidates if (x['AUROC'],x['AP'])==best]
    if len(winners)!=1:
        raise ValueError('Exact VAL AUROC/AP tie; no layer or update preference is permitted.')
    return winners[0]


def objectives(logits,y,alarm,lengths):
    w=torch.ones_like(y);w[torch.tensor(np.cumsum(lengths)-1,device=y.device)]=30
    anchor=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*w).mean()
    p=torch.sigmoid(logits)
    sync=(torch.relu(alarm.detach()-p).square()*alarm.detach().square()).sum()/sum(lengths)
    return anchor+120.0*sync


def fit_step(rows, config, args, out):
    trainrows,valrows=rows['train'],rows['val'];y=torch.tensor(labels(trainrows,'step'),dtype=torch.float32,device=args.device);vy=labels(valrows,'step');candidates=[]
    for layer in config['candidate_layers']:
        train=matrix(args.features,trainrows,'local_exp',layer,args.device);val=matrix(args.features,valrows,'local_exp',layer,args.device)
        torch.manual_seed(42);model=torch.nn.Linear(config['width'],1,device=args.device)
        def accepted(iteration):
            scores=probabilities(model,val);m=metrics(vy,scores,.5)
            name=f'step_layer_{layer}_update_{iteration}.pt';torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},out/name)
            candidates.append(dict(layer=layer,weight=iteration,AUROC=m['AUROC'],AP=m['AP'],path=name))
        optimizer=ObservedLBFGS(model.parameters(),on_accepted=accepted,lr=1.,max_iter=100,line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(train).flatten(),y)+1e-4*sum(.5*p.square().sum() for p in model.parameters())
            loss.backward();return loss
        optimizer.step(closure)
    best=choose(candidates);save(out/'step_candidates.json',candidates);return best


def fit_prefix(rows, config, args, out, step):
    tr,va=rows['train'],rows['val'];y=torch.tensor(labels(tr,'prefix'),dtype=torch.float32,device=args.device);vy=labels(va,'prefix')
    teacher=torch.nn.Linear(config['width'],1,device=args.device);teacher.load_state_dict(torch.load(out/step['path'],map_location=args.device,weights_only=True));teacher.requires_grad_(False)
    local=matrix(args.features,tr,'local_exp',step['layer'],args.device)
    lengths=[len(r['retained_original_step_ids']) for r in tr];offsets=np.r_[0,np.cumsum(lengths)];candidates=[]
    for layer in config['candidate_layers']:
        train=matrix(args.features,tr,'global_exp',layer,args.device);val=matrix(args.features,va,'global_exp',layer,args.device)
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
        model=torch.nn.Linear(config['width'],1).to(args.device)
        optimizer=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=100)
        def closure():
            optimizer.zero_grad()
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(train).flatten(),y)+1e-4*sum(.5*torch.norm(p)**2 for p in model.parameters())
            loss.backward();return loss
        optimizer.step(closure)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        for epoch in range(1,21):
            order=list(range(len(tr)));random.shuffle(order)
            for begin in range(0,len(order),64):
                batch=order[begin:begin+64];indices=np.concatenate([np.arange(offsets[k],offsets[k+1]) for k in batch]);index=torch.tensor(indices,device=args.device)
                with torch.no_grad():alarm=torch.sigmoid(teacher(local[index])).flatten()
                optimizer.zero_grad();loss=objectives(model(train[index]).flatten(),y[index],alarm,[lengths[k] for k in batch]);loss.backward();optimizer.step()
            scores=probabilities(model,val);m=metrics(vy,scores,.5);name=f'prefix_layer_{layer}_epoch_{epoch}.pt'
            torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},out/name)
            candidates.append(dict(layer=layer,weight=epoch,AUROC=m['AUROC'],AP=m['AP'],path=name))
    best=choose(candidates);save(out/'prefix_candidates.json',candidates);return best


def evaluate(rows,args,out,selection,meta,level,threshold=None):
    if level=='answer':
        with np.load(out/'prefix_val_predictions.npz') as z:vs=z['scores']
        vr=rows['val'];vs=vs[np.cumsum([len(r['retained_original_step_ids']) for r in vr])-1];vy=np.asarray([r['answer_hallucination'] for r in vr])
        t=select_threshold(vy,vs) if threshold is None else threshold
        save(out/'answer_frozen.json',dict(selection=selection,threshold=t,VAL=evaluate_metrics(vy,vs,t)))
        with np.load(out/'prefix_predictions.npz') as z:ts=z['scores']
        tr=rows['test'];ts=ts[np.cumsum([len(r['retained_original_step_ids']) for r in tr])-1];ty=np.asarray([r['answer_hallucination'] for r in tr])
        result=evaluate_metrics(ty,ts,t);save(out/'answer_result.json',result);np.savez_compressed(out/'answer_predictions.npz',scores=ts,y=ty)
        return result
    model=torch.nn.Linear(meta['width'],1,device=args.device)
    model.load_state_dict(torch.load(out/selection['path'],map_location=args.device,weights_only=True));model.eval()
    kind='local_exp' if level=='step' else 'global_exp'
    vr=rows['val'];vs=probabilities(model,matrix(args.features,vr,kind,selection['layer'],args.device));vy=labels(vr,'step' if level=='step' else 'prefix')
    if level=='answer':
        vs=vs[np.cumsum([len(r['retained_original_step_ids']) for r in vr])-1];vy=np.asarray([r['answer_hallucination'] for r in vr])
    np.savez_compressed(out/(level+'_val_predictions.npz'),scores=vs,y=vy)
    t=select_threshold(vy,vs) if threshold is None else threshold
    save(out/(level+'_frozen.json'),dict(selection=selection,threshold=t,VAL=evaluate_metrics(vy,vs,t)))
    tr=rows['test'];ts=probabilities(model,matrix(args.features,tr,kind,selection['layer'],args.device));ty=labels(tr,'step' if level=='step' else 'prefix')
    if level=='answer':
        ts=ts[np.cumsum([len(r['retained_original_step_ids']) for r in tr])-1];ty=np.asarray([r['answer_hallucination'] for r in tr])
    result=evaluate_metrics(ty,ts,t);save(out/(level+'_result.json'),result)
    np.savez_compressed(out/(level+'_predictions.npz'),scores=ts,y=ty)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',required=True,type=Path);p.add_argument('--model',required=True);p.add_argument('--domain',required=True)
    p.add_argument('--mode',choices=['train','evaluate'],default='evaluate');p.add_argument('--device',default='cuda')
    args=p.parse_args();paths=json.loads(args.paths.read_text());root=Path(__file__).resolve().parents[3]
    for key in ['feature_cache_root','output_root']:
        if not paths.get(key):p.error('Set '+key+' in your private paths file.')
    args.features=str(Path(paths['feature_cache_root'])/args.model)
    cfg=json.loads((root/'experiments/streaming_622_p70/config.json').read_text());meta=next(c for c in cfg['cells'] if (c['model'],c['domain'],c['level'])==(args.model,args.domain,'step'))
    rows=load_cell(paths,args.model,args.domain)
    if args.mode=='evaluate':
        from common.dataset import verify_archived_files
        verify_assignment(rows,args.model,args.domain,root/'experiments/streaming_622_p70/split_assignments.jsonl')
        verify_archived_files(paths,args.model,args.domain,root/'experiments/streaming_622_p70/data_identity.json')
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not {r['sample_id'] for r in rows[a]} & {r['sample_id'] for r in rows[b]}
        assert not {r['question_family_id'] for r in rows[a]} & {r['question_family_id'] for r in rows[b]}
    out=Path(paths['output_root'])/args.model/args.domain;out.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='train':
        step=fit_step(rows,meta,args,out);prefix=fit_prefix(rows,meta,args,out,step)
        for level,selection in [('step',step),('prefix',prefix),('answer',prefix)]:evaluate(rows,args,out,selection,meta,level)
    else:
        import shutil
        archive=root/'experiments/streaming_622_p70/checkpoint'/args.model/args.domain
        metrics_saved=json.loads((root/'experiments/streaming_622_p70/result/metrics.json').read_text())
        for level in ['step','prefix','answer']:
            match=next(x for x in metrics_saved if (x['model'],x['domain'],x['level'])==(args.model,args.domain,level))
            source_level='prefix' if level=='answer' else level;name=source_level+'.pt'
            if not (out/name).exists():shutil.copyfile(archive/name,out/name)
            evaluate(rows,args,out,dict(layer=match['selected_layer'],path=name),meta,level,threshold=match['threshold'])
    print(json.dumps({'status':'COMPLETE','output':str(out)}))

if __name__=='__main__':main()
