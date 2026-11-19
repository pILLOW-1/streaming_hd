"""Private Cetus import and independent CPU verification of saved A2 predictions."""
import hashlib,json,os,sys,tempfile,shutil
from pathlib import Path
import boto3
import numpy as np

NAME='qwen_a2_hami_topk_memory_20261119'
RUNS=Path('/shared/homes/u26487468/agent_uncertainty_quantification/experiments/streaming_hd/runs')


def roc_auc_score(y,p):
    order=np.argsort(-p,kind='stable');score=p[order];target=y[order]
    ends=np.r_[np.flatnonzero(score[1:]!=score[:-1]),len(score)-1]
    tp=target.cumsum()[ends];fp=ends+1-tp
    return float(np.trapezoid(np.r_[0,tp],np.r_[0,fp])/(tp[-1]*fp[-1]))

def average_precision_score(y,p):
    order=np.argsort(-p,kind='stable');score=p[order];target=y[order]
    ends=np.r_[np.flatnonzero(score[1:]!=score[:-1]),len(score)-1]
    tp=target.cumsum()[ends]
    return float(np.diff(tp,prepend=0)@(tp/(ends+1))/tp[-1])

def log_loss(y,p,labels=None):
    p=np.clip(p,np.finfo(np.float64).eps,1-np.finfo(np.float64).eps)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))

def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def metrics(y,p):return dict(AUROC=float(roc_auc_score(y,p)),AP=float(average_precision_score(y,p)),BCE=float(log_loss(y,p.astype(np.float64),labels=[0,1])))
def rank(r):
    m=r['metrics'];return m['mean_AUROC'],m['mean_AP'],-m['mean_BCE']

def verify(p,domain,teacher,gold):
    done=read(p/'TEST_COMPLETE.json');lock=read(p/'VAL_LOCK.json')
    assert done['selection']==lock==read(p/'TEST_STARTED.json')
    assert done['domain']==domain and done['teacher']==teacher
    rows=[read(p/f'layer_{i:02d}.json') for i in range(1,65)]
    assert max(rows,key=rank)=={k:v for k,v in lock.items() if k not in ('threshold','step_threshold')}
    assert all({k:v for k,v in r['binding'].items() if k!='layer'}=={k:v for k,v in lock['binding'].items() if k!='layer'} and r['layer']==i for i,r in enumerate(rows,1))
    for name,digest in ((lock['checkpoint'],lock['checkpoint_sha256']),(lock['prediction'],lock['prediction_sha256']),('test.npz',done['prediction_sha256'])):assert sha(p/name)==digest
    with np.load(p/lock['prediction'],allow_pickle=False) as b:
        for key,value in metrics(b['c'],b['probability']).items():assert np.isclose(value,lock['metrics'][key],rtol=0,atol=1e-10)
        prefix=metrics(b['c'],b['probability']);step=metrics(b['y'],b['q'])
        for metric in ('AUROC','AP','BCE'):
            assert np.isclose(step[metric],lock['metrics']['step_'+metric],rtol=0,atol=1e-10)
            assert np.isclose((prefix[metric]+step[metric])/2,lock['metrics']['mean_'+metric],rtol=0,atol=1e-10)
    with np.load(gold,allow_pickle=False) as b:
        truth={k:(int(y),int(c)) for k,y,c in zip(b['position_id'],b['y'],b['c'],strict=True)}
    with np.load(p/'test.npz',allow_pickle=False) as b:
        actual={k:(int(y),int(c)) for k,y,c in zip(b['position_id'],b['y'],b['c'],strict=True)}
        assert len(actual)==len(b['y'])==len(truth) and actual==truth
        assert np.isfinite(b['probability']).all()
        assert np.allclose(b['probability'],b['q']+(1-b['q'])*b['kappa'],rtol=0,atol=2e-7)
        m=metrics(b['c'],b['probability'])
        active=b['probability']>lock['threshold'];m['F1']=float(2*np.sum(active*b['c'])/(active.sum()+b['c'].sum()))
        for key,value in m.items():
            if key in done['metrics']:assert np.isclose(value,done['metrics'][key],rtol=0,atol=1e-10)
        step=metrics(b['y'],b['q'])
        for key,value in step.items():assert np.isclose(value,done['metrics']['step_'+key],rtol=0,atol=1e-10)
        cohort=(b['y']==0)&((b['c']==1)|b['prior'])
        rec=float(roc_auc_score(b['c'][cohort],b['probability'][cohort]))
        assert np.isclose(rec,done['inherited_vs_recovered_AUROC'],rtol=0,atol=1e-10)
    return dict(status='CPU_RECOMPUTED_FROM_SAVED_TEST',**done)

def main(domain,teacher):
    assert domain in ('Logic','Factuality') and teacher in ('H1','H2')
    s3=boto3.Session(profile_name='runpod-qwen').client('s3',endpoint_url='https://s3api-eu-ro-1.runpod.io',region_name='eu-ro-1')
    prefix=f'qwen_relation/results/{NAME}/{domain}/{teacher}/'
    parent=RUNS/NAME/domain/teacher;parent.mkdir(parents=True,exist_ok=True)
    dest=parent/'runpod_verified'
    def get(key,path):s3.download_file('fcsu28qkr3',key,str(path))
    with tempfile.TemporaryDirectory(prefix='.receiving-',dir=parent) as temp:
        p=Path(temp)/'verified';p.mkdir()
        get(prefix+'VAL_LOCK.json',p/'VAL_LOCK.json');lock=read(p/'VAL_LOCK.json')
        names=[f'layer_{i:02d}.json' for i in range(1,65)]+['TEST_STARTED.json','TEST_COMPLETE.json','test.npz',lock['checkpoint'],lock['prediction']]
        if teacher in ('F5','F6'):names += ['train.frozen_step_all_history.npz','val.frozen_step_all_history.npz','test.frozen_step_all_history.npz']
        for name in names:
            assert Path(name).name==name
            get(prefix+name,p/name)
        if teacher in ('F5','F6'):
            for part,key in (('train','q_train_sha256'),('val','q_val_sha256')):assert sha(p/f'{part}.frozen_step_all_history.npz')==lock['binding'][key]
        # Preserve execution provenance for equivalent optimized fits without
        # changing the frozen scientific binding or repeating model inference.
        for page in s3.get_paginator('list_objects_v2').paginate(Bucket='fcsu28qkr3',Prefix=prefix):
            for row in page.get('Contents',[]):
                name=row['Key'][len(prefix):]
                if name.startswith('EXECUTION_') and name.endswith('.json') and Path(name).name==name:
                    get(row['Key'],p/name)
                    proof=read(p/name);gate=proof['gate']
                    assert gate['status']=='PASS' and gate['fast_sha256']==proof['implementation_sha256']
                    assert gate['contract_sha256']==proof['scientific_contract_sha256']
        baseline='logic_ordered_dual/Logic/ordered_dual/linear/ce' if domain=='Logic' else 'factuality_c01_c20_20261112/dual/Factuality/ordered_dual/linear/ce'
        gold=Path(temp)/'gold.npz';meta=Path(temp)/'gold.json'
        get('qwen_relation/results/'+baseline+'/TEST_RESULT.json',meta)
        get('qwen_relation/results/'+baseline+'/test_predictions.npz',gold)
        assert sha(gold)==read(meta)['predictions_SHA256']
        accepted=verify(p,domain,teacher,gold)
        (p/'ACCEPTANCE.json').write_text(json.dumps(accepted,indent=2)+'\n')
        if dest.exists():
            assert read(dest/'ACCEPTANCE.json')==accepted
        else:os.rename(p,dest)
    print(json.dumps(accepted))

if __name__=='__main__':main(*sys.argv[1:])
