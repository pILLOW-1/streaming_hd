#!/usr/bin/env python3
"""ReDe + supervised probing on the agreed local dataset.

The recipe explicitly supplies projection/search parameters not fixed by the
paper. Auxiliary full native inputs include the generated Answer, never gold.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',type=Path,required=True);p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--domain',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    import numpy as np
    import torch
    from common.dataset import load_cell,read_jsonl
    from common.compact_answer_backend import CompactAnswerBackend as NativeBackend
    from common.native_inputs import validate_full_input, verify_snapshot
    from baselines.rede.pipeline import fit_projection,retained_indices
    from baselines.ars.detector_kernel import NonLinearClassifier
    from baselines.streaming.run import save,choose,evaluate_metrics
    from baselines.streaming.stream_metrics import select_threshold
    from common.baseline_training import seed_all,probabilities
    from common.checkpoint_selection import best_snapshot
    paths=json.loads(a.paths.read_text());cfg=json.loads(a.recipe.read_text())
    for key in ('native_full_input_root','output_root'):
        if not paths.get(key):p.error('Set '+key)
    rows=load_cell(paths,a.model,a.domain)
    out=Path(paths['output_root'])/'rede'/a.model/a.domain;out.mkdir(parents=True,exist_ok=False)
    save(out/'recipe.json',cfg)
    native={}
    for split in ('train','val','test'):
        source=Path(paths['native_full_input_root'])/a.model/a.domain/(split+'.native.jsonl')
        by_id={r['sample_id']:r for r in read_jsonl(source)}
        if by_id.keys()!={r['sample_id'] for r in rows[split]}:raise ValueError('Full native input IDs differ from dataset')
        native[split]=[by_id[r['sample_id']] for r in rows[split]]
        for row,n in zip(rows[split],native[split]):
            validate_full_input(n, row)
            if n['original_steps']!=row['original_steps']:raise ValueError('Full native original Steps differ')
            if n['answer']!=row['answer']:raise ValueError('Full native generated Answer differs')
            if len(n['step_token_spans'])!=len(n['original_steps']):raise ValueError('ReDe requires all original reasoning Step spans')
            start,end=n['answer_token_span']
            if not n['step_token_spans'][-1][1]<=start<end==len(n['input_ids']):raise ValueError('Full input must end at generated Answer')
    verify_snapshot(ROOT, paths['model_paths'][a.model], a.model)
    backend=NativeBackend(paths['model_paths'][a.model],device=a.device,dtype=cfg['dtype'])
    text=backend.text_config
    types=getattr(text,'layer_types',None)
    full=[i+1 for i in range(text.num_hidden_layers) if types is None or types[i] not in ('linear_attention','linear')]
    for candidate in cfg['candidates']:
        if candidate['layer']=='last':candidate['layer']=text.num_hidden_layers
        if candidate['attention_layer']=='last_full':candidate['attention_layer']=full[-1]
    save(out/'effective_recipe.json',cfg)
    cache={}
    def features(split,layer,attention_layer):
        key=(split,layer,attention_layer)
        if key in cache:return cache[key]
        result=[]
        for n in native[split]:
            states,attentions=backend.forward(n['input_ids'],attention=split=='train')
            embeddings=backend.step_embeddings(states,n['input_ids'],n['step_token_spans'],layer)
            if split=='train':
                if attention_layer-1 not in attentions:raise ValueError('Declared ReDe attention layer is not a native attention block')
                last=attentions[attention_layer-1][:,-1].float().mean(0)
                mass=torch.stack([last[start:end].sum().cpu() for start,end in n['step_token_spans']])
            else:mass=None
            result.append((embeddings,mass))
            del states,attentions
        cache[key]=result;return result
    candidates=[];models={}
    for i,recipe in enumerate(cfg['candidates']):
        seed_all(cfg.get('seed',42));layer=recipe['layer'];attention_layer=recipe['attention_layer']
        tr=features('train',layer,attention_layer);va=features('val',layer,attention_layer)
        projection,history=fit_projection(tr,hidden_width=recipe['hidden_width'],output_width=recipe['output_width'],
            epochs=recipe['epochs'],proxy_ratio=recipe['proxy_ratio'],lambda_disperse=recipe['lambda_disperse'],
            lambda_separate=recipe['lambda_separate'],device=a.device,seed=cfg.get('seed',42))
        def vectors(split,source):
            values=[]
            for n,(embedding,_) in zip(native[split],source):
                keep=retained_indices(projection,embedding,drop_ratio=recipe['drop_ratio'],short_trace_policy=cfg['short_trace_policy'])
                values.append(backend.filtered_answer(n,keep,layer))
            return torch.stack(values)
        x,xv=vectors('train',tr),vectors('val',va)
        y=torch.tensor([r['answer_hallucination'] for r in rows['train']],dtype=torch.float32,device=a.device)
        vy=np.asarray([r['answer_hallucination'] for r in rows['val']],dtype=np.int64)
        head=NonLinearClassifier(x.shape[1],hidden=512,p_drop=.6).to(a.device)
        optimizer=torch.optim.SGD(head.parameters(),lr=.008,momentum=.9,weight_decay=.05)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=7,min_lr=1e-4)
        best=None
        for epoch in range(100):
            head.train();order=torch.randperm(len(x))
            for start in range(0,len(x),128):
                index=order[start:start+128]
                if len(index)==1:raise ValueError('ReDe probe BatchNorm singleton batch; no silent row drop')
                optimizer.zero_grad();logits=head(x[index].to(a.device)).flatten()
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y[index]);loss.backward();optimizer.step()
            score=probabilities(head,xv,a.device)
            val_loss=torch.nn.functional.binary_cross_entropy(torch.tensor(score),torch.tensor(vy,dtype=torch.float32))
            scheduler.step(val_loss);m=evaluate_metrics(vy,score,.5)
            best=best_snapshot((m['AUROC'],m['AP']),head.state_dict(),best)
        if best['tie']:raise ValueError('Exact best VAL AUROC/AP probe checkpoint tie')
        head.load_state_dict(best['state'])
        torch.save(dict(projection=projection.state_dict(),head=best['state']),out/f'candidate_{i}.pt')
        candidates.append(dict(layer=i,AUROC=best['rank'][0],AP=best['rank'][1]))
        models[i]=(projection,head,probabilities(head,xv,a.device))
    selected=choose(candidates);i=selected['layer'];recipe=cfg['candidates'][i];projection,head,vs=models[i]
    threshold=select_threshold(vy,vs);save(out/'FROZEN_SELECTION.json',dict(candidate=i,recipe=recipe,threshold=threshold,VAL=evaluate_metrics(vy,vs,threshold)))
    # The chosen projection is now fixed. TEST has not been forwarded before this point.
    xt=[]
    for n,(embedding,_) in zip(native['test'],features('test',recipe['layer'],recipe['attention_layer'])):
        keep=retained_indices(projection,embedding,drop_ratio=recipe['drop_ratio'],short_trace_policy=cfg['short_trace_policy'])
        xt.append(backend.filtered_answer(n,keep,recipe['layer']))
    scores=probabilities(head,torch.stack(xt),a.device);labels=np.asarray([r['answer_hallucination'] for r in rows['test']],dtype=np.int64)
    np.savez(out/'test_predictions.npz',sample_id=np.asarray([r['sample_id'] for r in rows['test']]),labels=labels,scores=scores)
    save(out/'result.json',evaluate_metrics(labels,scores,threshold));save(out/'candidates.json',candidates)


if __name__=='__main__':main()
