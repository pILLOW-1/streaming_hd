#!/usr/bin/env python3
"""Extract each trajectory once into a model-specific reusable cache."""
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True,type=Path);p.add_argument('--model',required=True);p.add_argument('--domain',required=True);p.add_argument('--layers',default='all');a=p.parse_args();c=json.loads(a.paths.read_text())
 for k in ['native_input_root','feature_cache_root']:
  if not c.get(k):p.error('Set '+k)
 model=c.get('model_paths',{}).get(a.model)
 if not model:p.error('Set model_paths.'+a.model)
 pin=json.loads((ROOT/'configs/model_revisions.json').read_text())['models'][a.model]
 import hashlib
 for name,info in pin['snapshot_files'].items():
  f=Path(model)/name
  if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=info['SHA256']:raise ValueError('Model/tokenizer snapshot mismatch: '+name)
 model_config=json.loads((Path(model)/'config.json').read_text());text_config=model_config.get('text_config',model_config);layer_count=text_config['num_hidden_layers']
 root=Path(c['native_input_root'])/a.model/a.domain;output=Path(c['feature_cache_root'])/a.model;output.mkdir(parents=True,exist_ok=True);pending=[]
 for split in ['train','val','test']:
  for line in (root/(split+'.native.jsonl')).open():
   row=json.loads(line);manifest=output/row['sample_id']/'MANIFEST.json'
   if manifest.exists():
    m=json.loads(manifest.read_text());assert m['token_input_SHA256']==row['token_input_SHA256']
    expected=set(range(1,layer_count+1)) if a.layers=='all' else set(map(int,a.layers.split(',')))
    assert expected<=set(m['layers']), 'Cache lacks requested layers; preserve it and use a new output root.'
    for name, info in m['files'].items():
     cached=manifest.parent/name
     assert cached.is_file() and hashlib.sha256(cached.read_bytes()).hexdigest()==info['SHA256'], 'Corrupt or incomplete cache: '+str(cached)
    continue
   pending.append(line)
 if not pending:print('All requested trajectories cached.');return
 with tempfile.TemporaryDirectory() as td:
  inputs=Path(td)/'inputs.jsonl';inputs.write_text(''.join(pending))
  subprocess.run([sys.executable,str(ROOT/'code/generation/generation.py'),'--model',model,'--inputs',str(inputs),'--output',str(output),'--layers',a.layers,'--pool-only'],check=True)
if __name__=='__main__':main()
