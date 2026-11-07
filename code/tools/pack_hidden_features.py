#!/usr/bin/env python3
"""Reuse native token tensors to build SAPLMA/Global Mean Step or Prefix packs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from common.dataset import load_cell
from baselines.run import target_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths',type=Path,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--domain',required=True)
    parser.add_argument('--method',choices=['saplma','global_mean'],required=True)
    parser.add_argument('--level',choices=['step','prefix'],required=True)
    parser.add_argument('--layers',required=True)
    args=parser.parse_args()
    import numpy as np
    import torch
    paths=json.loads(args.paths.read_text())
    if not paths.get('baseline_feature_root') or not paths.get('feature_cache_root'):
        parser.error('Set baseline_feature_root and feature_cache_root')
    rows=load_cell(paths,args.model,args.domain)
    for layer in map(int,args.layers.split(',')):
        output=Path(paths['baseline_feature_root'])/args.method/args.model/args.domain/args.level/f'layer_{layer}'
        output.mkdir(parents=True,exist_ok=False)
        for split,records in rows.items():
            features=[];source_hashes={}
            for row in records:
                cache=Path(paths['feature_cache_root'])/args.model/row['sample_id']
                manifest=json.loads((cache/'MANIFEST.json').read_text())
                if manifest['retained_original_step_ids'] != row['retained_original_step_ids']:
                    raise ValueError('Cache retained-Step identity mismatch')
                def read(name):
                    p=cache/name;sha=hashlib.sha256(p.read_bytes()).hexdigest()
                    if sha!=manifest['files'][name]['SHA256']:raise ValueError('Cache hash mismatch')
                    source_hashes[row['sample_id']+'/'+name]=sha
                    return torch.load(p,map_location='cpu',weights_only=True)
                base=read(f'layer_{layer}.pt')
                for position,(start,end) in enumerate(base['step_token_spans']):
                    tensor=base
                    if position in manifest['causal_override_positions']:
                        tensor=read(f'override_{position}_layer_{layer}.pt')
                        start,end=tensor['step_token_spans'][0]
                    raw=tensor['hidden_states'];begin=tensor['token_start']
                    if not 0 <= end-1-begin < len(raw):raise ValueError('Hidden-state/token span mismatch')
                    # Global Mean includes every visible original reasoning token,
                    # including Steps excluded from supervision, and no prompt.
                    value=(raw[end-1-begin].float() if args.method=='saplma'
                           else raw[:end-begin].double().mean(0).float())
                    features.append(value.numpy())
            keys,_=target_rows(records,args.level)
            p=output/(split+'.npz')
            np.savez(p,x=np.stack(features),position_id=np.asarray(keys))
            receipt=dict(method=args.method,level=args.level,layer=layer,
                         SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),sources=source_hashes)
            (output/(split+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print('Feature packs written without a backbone forward.')


if __name__=='__main__':main()
