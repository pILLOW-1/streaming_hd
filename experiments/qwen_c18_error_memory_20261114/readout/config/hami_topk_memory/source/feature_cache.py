"""Bounded, source-bound shared FP32 readout cache; never modifies raw banks."""
import contextlib,fcntl,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np

ROOT=Path('/workspace/qwen_relation/shared_cache/readouts_v1')
LIMIT=24_000_000_000
POOLING_SHA=None


def digest(a):
    assert a.flags.c_contiguous
    return hashlib.sha256(memoryview(a).cast('B')).hexdigest()


def identity(data,layer,readout,extra=None):
    return dict(domain=data.receipt['domain'],split=data.receipt['split'],layer=layer,readout=readout,dtype='float32',inputs=data.pins,model_revision=data.receipt['model_revision'],tokenizer_revision=data.receipt['tokenizer_revision'],pooling_source_sha256=POOLING_SHA,version='exact_fp32_readouts_v1',parameters=extra or {})


@contextlib.contextmanager
def lock(path,nonblocking=False):
    f=path.open('a')
    try:
        fcntl.flock(f,fcntl.LOCK_EX|(fcntl.LOCK_NB if nonblocking else 0));yield
    finally:f.close()


def load_index():
    p=ROOT/'INDEX.json'
    return json.loads(p.read_text()) if p.exists() else dict(version=1,max_cache_bytes=LIMIT,entries={},score_references={})


def save_index(index):
    p=ROOT/'INDEX.pending';p.write_text(json.dumps(index,indent=2)+'\n');p.replace(ROOT/'INDEX.json')


def get(spec,builder,pinned=False):
    ROOT.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    path=ROOT/(key+'.npy')
    with lock(ROOT/(key+'.lock')):
        with lock(ROOT/'.index.lock'):
            index=load_index();entry=index['entries'].get(key)
        if entry is not None and path.exists():
            a=np.load(path,allow_pickle=False)
            assert a.dtype==np.float32 and list(a.shape)==entry['shape'] and digest(a)==entry['tensor_sha256']
            with lock(ROOT/'.index.lock'):
                index=load_index();entry=index['entries'][key];entry['last_used']=time.time();entry['hits']=entry.get('hits',0)+1
                consumer=' '.join(sys.argv)
                if consumer not in entry['consumers']:entry['consumers'].append(consumer)
                save_index(index)
            print('CACHE_HIT',spec['domain'],spec['split'],spec['layer'],spec['readout'],flush=True)
            return a
        a=np.ascontiguousarray(builder(),dtype=np.float32)
        assert np.isfinite(a).all()
        with lock(ROOT/'.index.lock'):
            index=load_index();used=sum(e['file_bytes'] for e in index['entries'].values())
            needed=a.nbytes+256
            for old,e in sorted(list(index['entries'].items()),key=lambda pair:pair[1]['last_used']):
                if used+needed<=LIMIT:break
                if e.get('pinned') or old==key:continue
                assert len(old)==64 and all(c in '0123456789abcdef' for c in old)
                try:
                    with lock(ROOT/(old+'.lock'),True):
                        (ROOT/(old+'.npy')).unlink(missing_ok=True)
                        used-=e['file_bytes'];del index['entries'][old]
                except BlockingIOError:continue
            if used+needed>LIMIT:
                save_index(index)
                print('CACHE_CAPACITY_BYPASS',key,flush=True)
                return a
            pending=path.with_suffix('.pending')
            with pending.open('wb') as f:np.save(f,a,allow_pickle=False)
            pending.replace(path)
            index['entries'][key]=dict(identity=spec,path=str(path),shape=list(a.shape),dtype=str(a.dtype),tensor_bytes=a.nbytes,file_bytes=path.stat().st_size,tensor_sha256=digest(a),created=time.time(),last_used=time.time(),pinned=pinned,hits=0,consumers=[' '.join(sys.argv)])
            save_index(index)
        print('CACHE_BUILD',spec['domain'],spec['split'],spec['layer'],spec['readout'],a.nbytes,flush=True)
        return a


def attach(w,m):
    global POOLING_SHA
    POOLING_SHA=hashlib.sha256(Path(sys.modules[w.pools.__module__].__file__).read_bytes()).hexdigest()
    orig_raw=w.raw_pools;orig_prefix=m.prefix;orig_step=m.step_repr;orig_memory=m.memory;orig_q=w.frozen_scores

    def prefix(data,layer):
        return get(identity(data,layer,'prefix_exp'),lambda:orig_prefix(data,layer))

    def raw(data,layer,need_prefix=True,only=None):
        manifest=w.read(w.INPUT/'manifest.json')
        selected=manifest['cells'][data.receipt['domain']]['streaming']['layer']
        if only is None and layer==selected:
            e=get(identity(data,layer,'step_exp_all_history'),lambda:orig_raw(data,layer,False)[0],True)
        else:e=orig_raw(data,layer,False,only)[0]
        return e,prefix(data,layer) if need_prefix else None

    def step(data,domain,teacher,device):
        if teacher=='streaming':
            layer=w.read(w.INPUT/'manifest.json')['cells'][domain]['streaming']['layer']
            pin=dict(layer=layer,normalization='original Streaming input L2')
        else:
            root=w.BASE/'qwen_step_only_learnable_20261115/legacy_recipe'/domain/'aggr'
            selected=w.read(root/'TEST_STARTED.json')['selection'];layer=selected['layer']
            pin=dict(layer=layer,epoch=selected['epoch'],checkpoint_sha256=selected['checkpoint_sha256'],normalization='original raw learned aggregate')
        def build():
            e,actual=orig_step(data,domain,teacher,device);assert actual==pin;return e
        params=dict(selection=pin)
        if teacher!='streaming':
            import torch,legacy_step_train
            params.update(scorer_source_sha256=hashlib.sha256(Path(legacy_step_train.__file__).read_bytes()).hexdigest(),numeric_backend=torch.cuda.get_device_name(device),torch_version=torch.__version__)
        return get(identity(data,layer,'frozen_step_detector_input',params),build,True),pin

    def memory(data,e,q):
        params=dict(e_sha256=digest(e),q_sha256=digest(np.ascontiguousarray(q)))
        return get(identity(data,0,'exclusive_predicted_error_memory',params),lambda:orig_memory(data,e,q),True)

    def scores(data,domain,part,out,teacher='streaming'):
        q=orig_q(data,domain,part,out,teacher)
        p=out/f'{part}.frozen_step_all_history.npz'
        if p.exists():
            with np.load(p,allow_pickle=False) as b:binding=json.loads(b['binding'].item())
            spec=dict(domain=domain,split=part,teacher=teacher,binding=binding)
            key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
            ROOT.mkdir(parents=True,exist_ok=True)
            with lock(ROOT/'.index.lock'):
                index=load_index();e=index['score_references'].setdefault(key,dict(identity=spec,paths=[],shape=list(q.shape),dtype=str(q.dtype),tensor_sha256=digest(np.ascontiguousarray(q))))
                assert e['tensor_sha256']==digest(np.ascontiguousarray(q))
                if str(p) not in e['paths']:e['paths'].append(str(p))
                save_index(index)
        return q

    w.raw_pools=raw;w.frozen_scores=scores;m.prefix=prefix;m.step_repr=step;m.memory=memory
    print('SHARED_CACHE_ENABLED',ROOT,LIMIT,flush=True)
