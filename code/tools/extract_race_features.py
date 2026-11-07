#!/usr/bin/env python3
"""Run the pinned full-component RACE scorer on local data and sampled responses."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--domain',required=True)
    a=p.parse_args()
    import numpy as np
    from common.dataset import load_cell,read_jsonl
    from baselines.race.adapter import UPSTREAM,require_complete_scorer,calculate
    paths=json.loads(a.paths.read_text());aux=paths.get('race_assets',{})
    for key in ('embedding_model','nli_model','scoring_model','sample_root'):
        if not aux.get(key):p.error('Set race_assets.'+key)
    if not paths.get('baseline_feature_root'):p.error('Set baseline_feature_root')
    sys.path.insert(0,str(UPSTREAM))
    from RACE import RACEScorer
    # Upstream resolves its fixed prompt relative to CWD; restore CWD afterwards.
    old=os.getcwd()
    try:
        os.chdir(UPSTREAM)
        scorer=RACEScorer(embedding_model_path=aux['embedding_model'],nli_model_path=aux['nli_model'],llm_model_path=aux['scoring_model'],use_gpu=True)
    finally:os.chdir(old)
    require_complete_scorer(scorer)
    rows=load_cell(paths,a.model,a.domain)
    out=Path(paths['baseline_feature_root'])/'race'/a.model/a.domain/'answer/layer_0';out.mkdir(parents=True,exist_ok=False)
    for split,records in rows.items():
        source=Path(aux['sample_root'])/a.model/a.domain/(split+'.jsonl')
        samples={r['sample_id']:r for r in read_jsonl(source)}
        if samples.keys()!={r['sample_id'] for r in records}:raise ValueError('RACE sample coverage mismatch')
        vectors=[]
        for row in records:
            sample=samples[row['sample_id']]
            if sample['model']!=a.model or sample['model_revision']!=row['model_revision']:
                raise ValueError('RACE samples must come from the same generator/revision')
            main=dict(question=row['question'],final_answer=row['answer'],think='\n\n'.join(row['original_steps']))
            value=calculate(scorer,main,sample)
            vectors.append([value['components'][k] for k in ('consistency','cross_uncertainty','sindex','entity_consistency')])
        target=out/(split+'.npz')
        np.savez(target,components=np.asarray(vectors),position_id=np.asarray([r['sample_id']+':answer' for r in records]))
        (out/(split+'.json')).write_text(json.dumps(dict(method='race',level='answer',layer=0,
            SHA256=hashlib.sha256(target.read_bytes()).hexdigest(),samples_SHA256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
