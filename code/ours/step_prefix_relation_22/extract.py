"""Extract only missing Streaming pools; native BF16 context and overrides preserved."""
import json
import mmap
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
from relation_common import *


def pools(visible, spans, reasoning_start, methods):
    """Raw FP32 pools, same local-exp/global-exp equations as Streaming."""
    result = {m: [] for m in methods}
    acc = np.zeros(visible.shape[1], dtype=np.float64)
    weighted = acc.copy(); denominator = 0.0; previous = 0
    for a, b in spans:
        left, end = a-reasoning_start, b-reasoning_start
        assert 0 <= left < end <= len(visible)
        if 'last_token' in result: result['last_token'].append(visible[end-1])
        if 'step_exp' in result:
            # Match torch.float32 Streaming local weighting exactly.
            bag = torch.from_numpy(visible[left:end])
            weight = torch.exp(3*torch.linspace(0, 1, len(bag), dtype=torch.float32))
            weight /= weight.sum()
            result['step_exp'].append((bag*weight[:, None]).sum(0).numpy())
        if 'global_mean' in result:
            acc += visible[previous:end].sum(0, dtype=np.float64)
            result['global_mean'].append((acc/end).astype(np.float32))
        if 'prefix_exp' in result:
            # Rescale by the current endpoint for stable exp at arbitrary native lengths.
            # This is algebraically the same normalized exp(.003*j), not truncation.
            scale = np.exp(-0.003*(end-previous))
            weighted *= scale; denominator *= scale
            for lo in range(previous, end, 1024):
                hi = min(end, lo+1024)
                weight = np.exp(0.003*(np.arange(lo, hi, dtype=np.float64)+1-end))
                weighted += np.einsum('ij,i->j', visible[lo:hi], weight, dtype=np.float64)
                denominator += weight.sum()
            result['prefix_exp'].append((weighted/denominator).astype(np.float32))
        previous = end
    return {k: np.stack(v).astype(np.float32) for k, v in result.items()}


def main(domain, phase):
    assert os.environ.get('PBS_JOBID') and torch.cuda.is_available()
    sys.path[:0] = [str(REPO/'code'), str(REPO/'code/generation'),
                   str(PROJECT/'workspaces/streaming_hd/pipelines/training/baselines_622_p70_20261108')]
    from native_layers import NativeLayers, StopAtLayer
    from generation import validate_row
    from hash_utils import sha256_tokens
    cfg = config(); binding = json.loads((STATE/(domain+'.json')).read_text())
    assert binding['data_hashes'] == data_hashes(domain)
    splits = ('train', 'val') if phase == 'train' else ('test',)
    methods = ('step_exp', 'prefix_exp') if phase == 'train' else READOUTS
    if phase == 'train': layers = cfg['layers']
    else:
        arms = [a for a in cfg['arms'] if a.startswith(domain+'.')]
        assert len(arms) == 16
        frozen = [json.loads((arm_paths(a)[0]/'FROZEN_SELECTION.json').read_text()) for a in arms]
        layers = sorted({x['selected']['layer'] for x in frozen})
        save(RUN/'reports/test_features'/domain/'FROZEN_INPUT.json', {a:sha(arm_paths(a)[0]/'FROZEN_SELECTION.json') for a in arms})
    ds = dataset(domain, splits); data=[]; positions=[]; ranges={}; offset=0
    for split in splits:
        begin=offset; records=ds[split]['rows']
        native_path=DATA/'native'/MODEL/domain/(split+'.native.jsonl')
        native={x['sample_id']:x for x in map(json.loads,native_path.read_text().splitlines()) if x}
        assert set(native)=={x['sample_id'] for x in records}
        for row in records:
            n=native[row['sample_id']];validate_row(n)
            assert n['retained_original_step_ids']==row['retained_original_step_ids']
            assert sha256_tokens(n['input_ids'])==n['token_input_SHA256']
            data.append((n,offset));positions.extend(row['sample_id']+':'+str(i) for i in row['retained_original_step_ids'])
            offset+=len(n['step_token_spans'])
        ranges[split]=[begin,offset]
    root=SCRATCH/'features'/domain/phase;root.mkdir(parents=True,exist_ok=True)
    reports=RUN/'reports'/('features' if phase=='train' else 'test_features')/domain
    if (reports/'FEATURE_MANIFEST.json').exists():return 0
    identity=dict(model=MODEL,domain=domain,layers=64,width=5120,positions=offset,ranges=ranges,
                  native_files={s:sha(DATA/'native'/MODEL/domain/(s+'.native.jsonl')) for s in splits},
                  config_SHA256=sha(WORK/'config.json'),backbone_dtype='bfloat16',feature_L2=False,
                  position_hash=hashlib.sha256(json.dumps(positions).encode()).hexdigest())
    if (root/'IDENTITY.json').exists():assert json.loads((root/'IDENTITY.json').read_text())==identity
    else:
        required=offset*5120*4*len(layers)*len(methods)
        assert shutil.disk_usage(root).free>required+(32<<30),'DISK_ADMISSION no changes to other caches'
        save(root/'IDENTITY.json',identity);save(root/'POSITIONS.json',positions)
    save(reports/'POSITION_IDENTITY.json',dict(position_hash=identity['position_hash'],positions=offset))
    arrays={}
    for layer in layers:
        (root/f'layer_{layer}').mkdir(exist_ok=True)
        for method in methods:
            p=root/f'layer_{layer}'/(method+'.npy')
            arrays[layer,method]=np.lib.format.open_memmap(p,mode='r+' if p.exists() else 'w+',dtype=np.float32,shape=(offset,5120))
    complete=set(json.loads((root/'COMPLETED.json').read_text())) if (root/'COMPLETED.json').exists() else set()
    class Extractor(NativeLayers):
        current=None
        current_offset=0
        def _hook(self,layer):
            original=super()._hook(layer)
            def hook(module,args,output):
                if self.current is None:return original(module,args,output)
                if layer not in self.active:return
                h=output[0] if isinstance(output,(tuple,list)) else output
                n=self.current;start=n['reasoning_start_token'];spans=n['step_token_spans']
                visible=h[0,start:max(b for a,b in spans)].float().cpu().numpy()
                values=pools(visible,spans,start,methods)
                for method,value in values.items():
                    assert np.isfinite(value).all()
                    arrays[layer,method][self.current_offset:self.current_offset+len(spans)]=value
                self.pooled[layer]=True
                if self.stop and layer==self.stop_layer:raise StopAtLayer()
            return hook
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    engine=Extractor(str(PROJECT/'assets/models/Qwen3.8-27B'),64,5120)
    probe=data[0][0]['input_ids'][:128];cut=len(probe)//2
    engine.run(probe,layers,raw_span=(cut-1,cut),api=True)
    _,a=engine.run(probe,layers,raw_span=(cut-1,cut));changed=probe.copy();changed[cut:]=[1]*(len(probe)-cut)
    _,b=engine.run(changed,layers,raw_span=(cut-1,cut))
    assert all(torch.equal(a[l],b[l]) for l in layers),'Native fixed-shape causal gate failed'
    save(reports/'NATIVE_GATE.json',dict(status='PASS',API_identity=True,fixed_shape_causal=True,job=os.environ['PBS_JOBID'],phase=phase))
    deadline=time.monotonic()+float(os.environ.get('EXECUTION_SECONDS','19800')); maximum=0
    try:
        for n,left in sorted(data,key=lambda z:len(z[0]['input_ids']),reverse=True):
            if n['sample_id'] in complete:continue
            if time.monotonic()+max(300,maximum*1.5)>deadline:return 75
            began=time.monotonic();engine.current=n;engine.current_offset=left
            engine.run(n['input_ids'],layers,spans=n['step_token_spans'])
            for pos,override in n.get('step_causal_overrides',{}).items():
                pos=int(pos)
                causal=dict(n,input_ids=override['input_ids'],step_token_spans=[override['span']],
                            retained_original_step_ids=[n['retained_original_step_ids'][pos]],step_causal_overrides={})
                validate_row(causal);engine.current=causal;engine.current_offset=left+pos
                engine.run(causal['input_ids'],layers,spans=causal['step_token_spans'])
            for value in arrays.values():value.flush();value._mmap.madvise(mmap.MADV_DONTNEED)
            complete.add(n['sample_id']);save(root/'COMPLETED.json',sorted(complete))
            maximum=max(maximum,time.monotonic()-began)
            save(reports/'PROGRESS.json',dict(completed=len(complete),total=len(data),host=os.uname().nodename,
                                            feature_root=str(root),job=os.environ['PBS_JOBID'],last_seconds=time.monotonic()-began,
                                            peak_VRAM=torch.cuda.max_memory_allocated()))
    finally:engine.close()
    receipt=dict(status='COMPLETE',identity=identity,feature_root=str(root),host=os.uname().nodename,
                 files={f'layer_{l}/{m}.npy':dict(SHA256=sha(root/f'layer_{l}'/(m+'.npy')),
                        bytes=(root/f'layer_{l}'/(m+'.npy')).stat().st_size) for l in layers for m in methods})
    save(root/'FEATURE_MANIFEST.json',receipt);save(reports/'FEATURE_MANIFEST.json',receipt)
    return 0


if __name__=='__main__':raise SystemExit(main(*sys.argv[1:]))
