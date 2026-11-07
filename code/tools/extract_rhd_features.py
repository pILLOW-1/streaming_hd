#!/usr/bin/env python3
"""Produce RHD four-component feature packs from frozen native reasoning inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--domain',required=True)
    p.add_argument('--anchor-layers',required=True,help='Explicit native hidden-state indices, comma separated.')
    p.add_argument('--attention-layers',required=True,help='Explicit native decoder output indices, comma separated.')
    p.add_argument('--dtype',choices=['float16','bfloat16','float32'],default='bfloat16')
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    import numpy as np
    import torch
    from common.dataset import load_cell,read_jsonl
    from common.native_backend import NativeBackend
    from baselines.rhd.kernel import token_depth,step_statistics,components
    paths=json.loads(a.paths.read_text())
    for key in ('native_input_root','baseline_feature_root'):
        if not paths.get(key):p.error('Set '+key)
    anchors=[int(x) for x in a.anchor_layers.split(',')];attn_layers=[int(x) for x in a.attention_layers.split(',')]
    backend=NativeBackend(paths['model_paths'][a.model],dtype=a.dtype,device=a.device)
    mature=backend.text_config.num_hidden_layers
    if not anchors or any(not 1<=i<mature for i in anchors):raise ValueError('Invalid RHD anchor layer set')
    rows=load_cell(paths,a.model,a.domain)
    out=Path(paths['baseline_feature_root'])/'rhd'/a.model/a.domain/'answer/layer_0'
    out.mkdir(parents=True,exist_ok=False)
    for split,records in rows.items():
        path=Path(paths['native_input_root'])/a.model/a.domain/(split+'.native.jsonl')
        native={r['sample_id']:r for r in read_jsonl(path)}
        if native.keys()!={r['sample_id'] for r in records}:raise ValueError('Native coverage differs from dataset')
        result=[]
        for row in records:
            n=native[row['sample_id']]
            if n['retained_original_step_ids']!=row['retained_original_step_ids']:raise ValueError('Retained-Step identity differs')
            states,attention=backend.forward(n['input_ids'])
            if any(i-1 not in attention for i in attn_layers):raise ValueError('RHD requires true native attention blocks')
            depths=[];entropies=[]
            # Chunk vocabulary projections; never materialize [layers,T,vocab].
            for begin in range(0,len(n['input_ids']),128):
                end=min(len(n['input_ids']),begin+128)
                with torch.no_grad():
                    final=backend.logits(states[mature][begin:end])
                    premature=torch.stack([backend.logits(backend.norm(states[i][begin:end])) for i in anchors])
                    depth,entropy=token_depth(final,premature)
                depths.append(depth.cpu());entropies.append(entropy.cpu())
            r,e=step_statistics(torch.cat(depths),torch.cat(entropies),n['step_token_spans'])
            spans=n['step_token_spans'];matrix=np.zeros((len(spans),len(spans)),dtype=np.float64)
            # Exact mean over native heads, token pairs, and declared layers.
            for i,(left,right) in enumerate(spans):
                for j,(start,stop) in enumerate(spans[:i]):
                    matrix[i,j]=np.mean([float(attention[l-1][:,left:right,start:stop].float().mean()) for l in attn_layers])
            result.append(components(r,e,matrix))
            del states,attention
        target=out/(split+'.npz')
        np.savez(target,components=np.stack(result),position_id=np.asarray([r['sample_id']+':answer' for r in records]))
        receipt=dict(method='rhd',level='answer',layer=0,anchors=anchors,mature=mature,attention_layers=attn_layers,
                     dtype=a.dtype,SHA256=hashlib.sha256(target.read_bytes()).hexdigest(),native_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
        (out/(split+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
