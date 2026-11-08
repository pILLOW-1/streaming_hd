#!/usr/bin/env python3
"""Validate and assemble HaMI/ARS native features and method-specific auxiliaries.

Run the pinned method's sampling/variant and native-embedding scripts first.
This boundary never substitutes labels for semantic probabilities/agreement.
"""
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
    p.add_argument('--method',choices=['ars','hami'],required=True);p.add_argument('--layer',type=int,required=True)
    p.add_argument('--source',type=Path,required=True,help='Directory containing per-split JSON manifests and named .npy arrays.')
    a=p.parse_args()
    import numpy as np
    from common.dataset import load_cell
    from baselines.run import target_rows
    paths=json.loads(a.paths.read_text());rows=load_cell(paths,a.model,a.domain)
    if not paths.get('baseline_feature_root'):p.error('Set baseline_feature_root')
    level='answer' if a.method=='ars' else 'step'
    out=Path(paths['baseline_feature_root'])/a.method/a.model/a.domain/level/f'layer_{a.layer}'
    out.mkdir(parents=True,exist_ok=False)
    for split,records in rows.items():
        metadata=json.loads((a.source/(split+'.json')).read_text());keys,_=target_rows(records,level)
        if metadata['position_id']!=keys:raise ValueError('Auxiliary feature ID/order mismatch')
        if metadata['model']!=a.model or metadata['layer']!=a.layer:raise ValueError('Auxiliary native model/layer mismatch')
        arrays={}
        for name,spec in metadata['arrays'].items():
            path=a.source/spec['file']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['SHA256']:raise ValueError('Auxiliary feature hash mismatch')
            arrays[name]=np.load(path,allow_pickle=False)
        if a.method=='hami':
            mode=metadata.get('hami_mode','se')
            if mode not in ('ori','se'):raise ValueError('Unknown HaMI branch')
            if not {'tokens','offsets'}<=arrays.keys():raise ValueError('Incomplete HaMI token bags')
            if len(arrays['offsets'])!=len(keys)+1 or arrays['offsets'][0]!=0 or arrays['offsets'][-1]!=len(arrays['tokens']) or np.any(np.diff(arrays['offsets'])<=0):raise ValueError('HaMI token bag alignment mismatch')
            if mode=='se':
                if metadata.get('semantic_probability_source')!='same_backbone_sample_clusters':raise ValueError('HaMI SE requires actual sampled semantic clusters')
                if 'semantic_probability' not in arrays:raise ValueError('HaMI SE probability missing')
                probability=arrays['semantic_probability']
                if probability.shape!=(len(keys),) or not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)):raise ValueError('Invalid semantic probability')
            if metadata.get('token_dtype')=='bfloat16_bits' and arrays['tokens'].dtype!=np.uint16:
                raise ValueError('BF16 native token storage must use uint16 bits')
        else:
            if arrays['x'].shape[0]!=len(keys):raise ValueError('ARS original embedding coverage mismatch')
            if split=='train':
                if metadata['variant_supervision_kind']!='answer_agreement_NOT_answer_truth':raise ValueError('ARS agreement supervision required')
                required={'variants_x','variant_original_id','variant_agreement','variant_qid'}
                if not required<=arrays.keys():raise ValueError('Missing ARS variant arrays')
                n=len(arrays['variants_x'])
                if any(len(arrays[k])!=n for k in required):raise ValueError('ARS variant alignment mismatch')
                if not set(map(str,arrays['variant_qid']))<={r['canonical_question_id'] for r in records}:raise ValueError('ARS variants cross TRAIN boundary')
        if any(k in arrays for k in ('labels','gold_answer','answer_hallucination')):raise ValueError('Label contamination in features')
        if any(v.dtype.kind in 'fc' and not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite auxiliary features')
        target=out/(split+'.npz');np.savez(target,position_id=np.asarray(keys),**arrays)
        metadata.update(method=a.method,level=level,SHA256=hashlib.sha256(target.read_bytes()).hexdigest())
        (out/(split+'.json')).write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
