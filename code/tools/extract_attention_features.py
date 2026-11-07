#!/usr/bin/env python3
"""Reference native-attention feature producer for ICR Probe and LLM Check.

Run on an allocated GPU with sufficient memory. This exact dense implementation
never truncates a trajectory or substitutes linear attention. It saves compact
feature packs, not attention matrices. Native source sidecars are immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths',type=Path,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--domain',required=True)
    parser.add_argument('--method',choices=['icr_probe','llm_check'],required=True)
    parser.add_argument('--dtype',choices=['bfloat16','float32'],default='bfloat16')
    args=parser.parse_args()
    import numpy as np
    import torch
    import transformers
    from common.dataset import load_cell,read_jsonl
    from common.native_attention import attention_blocks
    from baselines.icr_probe.native import prefix_vector
    from baselines.llm_check.kernel import attention_score
    from baselines.run import target_rows
    sys.path.insert(0,str(ROOT/'code/generation'))
    from generation import validate_row
    from hash_utils import sha256_tokens
    if not torch.cuda.is_available():raise RuntimeError('Use an allocated GPU')
    paths=json.loads(args.paths.read_text())
    for key in ('native_input_root','baseline_feature_root'):
        if not paths.get(key):parser.error('Set '+key)
    model_path=paths['model_paths'][args.model]
    if not model_path:parser.error('Set local model path')
    pins=json.loads((ROOT/'configs/model_revisions.json').read_text())['models'][args.model]
    for name,info in pins['snapshot_files'].items():
        if hashlib.sha256((Path(model_path)/name).read_bytes()).hexdigest()!=info['SHA256']:
            raise ValueError('Native model/tokenizer snapshot identity mismatch')
    cfg=transformers.AutoConfig.from_pretrained(model_path,local_files_only=True)
    model=getattr(transformers,cfg.architectures[0]).from_pretrained(model_path,local_files_only=True,
        dtype=getattr(torch,args.dtype),attn_implementation='eager',device_map={'':'cuda:0'}).eval()
    model.requires_grad_(False)
    rows=load_cell(paths,args.model,args.domain)
    destination=Path(paths['baseline_feature_root'])/args.method/args.model/args.domain/'prefix'
    if destination.exists():raise FileExistsError(destination)
    stored={}
    for split,records in rows.items():
        native_file=Path(paths['native_input_root'])/args.model/args.domain/(split+'.native.jsonl')
        native={r['sample_id']:r for r in read_jsonl(native_file)}
        if native.keys()!={r['sample_id'] for r in records}:raise ValueError('Native/supervised trajectory coverage mismatch')
        values={}
        for row in records:
            n=native[row['sample_id']];validate_row(n)
            if n['retained_original_step_ids']!=row['retained_original_step_ids']:raise ValueError('Retained Step mismatch')
            if n.get('token_input_SHA256') and sha256_tokens(n['input_ids'])!=n['token_input_SHA256']:raise ValueError('Token hash mismatch')
            def forward(ids):
                tokens=torch.tensor([ids],device='cuda')
                with torch.no_grad():
                    result=model(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False,
                                 output_hidden_states=args.method=='icr_probe',output_attentions=True,logits_to_keep=1)
                mapped=attention_blocks(cfg,result.attentions)
                hidden=[h[0] for h in result.hidden_states] if args.method=='icr_probe' else None
                return result,mapped,hidden
            output,attentions,hidden=forward(n['input_ids'])
            for position,(_,end) in enumerate(n['step_token_spans']):
                current_attentions,current_hidden=attentions,hidden
                corrected=None
                override=n.get('step_causal_overrides',{}).get(str(position))
                if override is not None:
                    corrected,current_attentions,current_hidden=forward(override['input_ids']);end=override['span'][1]
                if args.method=='icr_probe':
                    if 'core_positions' not in n:raise ValueError('ICR requires exact native user/response boundaries in core_positions')
                    value=prefix_vector(current_hidden,current_attentions,n['reasoning_start_token'],end,core_positions=n['core_positions'])
                    values.setdefault(0,[]).append(value)
                else:
                    for block,a in current_attentions.items():
                        diagonal=a[:,:end,:end].diagonal(dim1=-2,dim2=-1)
                        values.setdefault(block+1,[]).append(float(attention_score(diagonal)))
                del corrected
            del output,attentions,hidden
        keys,_=target_rows(records,'prefix')
        for layer,value in values.items():
            directory=destination/f'layer_{layer}';directory.mkdir(parents=True,exist_ok=True)
            path=directory/(split+'.npz')
            field='x' if args.method=='icr_probe' else 'score'
            np.savez(path,position_id=np.asarray(keys),**{field:np.asarray(value)})
            receipt=dict(method=args.method,level='prefix',layer=layer,backbone_dtype=args.dtype,
                         SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),native_input_SHA256=hashlib.sha256(native_file.read_bytes()).hexdigest(),
                         model_revision=pins['model_revision'],native_attention_only=True)
            (directory/(split+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print('Native attention feature extraction complete.')


if __name__=='__main__':main()
