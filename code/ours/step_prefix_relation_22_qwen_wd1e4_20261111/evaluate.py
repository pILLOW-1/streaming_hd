"""Selected TEST once, followed by CPU acceptance of persisted predictions only."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from relation_common import *
from model import make_model, predict


def metrics(y, s, threshold):
    predicted=s>threshold
    tp=int(((y==1)&predicted).sum());fp=int(((y==0)&predicted).sum())
    tn=int(((y==0)&~predicted).sum());fn=int(((y==1)&~predicted).sum())
    fpr,tpr,_=roc_curve(y,s,drop_intermediate=False)
    return dict(AUROC=float(roc_auc_score(y,s)),AP=float(average_precision_score(y,s)),
                TPR_at_1pct_FPR=float(tpr[fpr<=.01].max()),ACC=(tp+tn)/len(y),
                F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn),FPR=fp/(fp+tn),
                TP=tp,FP=fp,TN=tn,FN=fn,positions=len(y),positive=int(y.sum()),
                negative=int((y==0).sum()),missing=0,threshold=float(threshold))


def main(domain, cpu_only=False):
    cfg=config();data=dataset(domain,('test',))['test']
    receipt=json.loads((RUN/'reports/test_features'/domain/'FEATURE_MANIFEST.json').read_text())
    assert receipt['status']=='COMPLETE'
    assert json.loads((STATE/(domain+'.json')).read_text())['data_hashes']==data_hashes(domain)
    root=Path(receipt['feature_root'])
    assert receipt['identity']['position_hash']==hashlib.sha256(json.dumps(data['keys']).encode()).hexdigest()
    for arm in cfg['arms']:
        if not arm.startswith(domain+'.'):continue
        _,readout,head,loss=arm.split('.');out,checkpoints=arm_paths(arm)
        result=out/'TEST_RESULT.json';freeze=out/'FROZEN_SELECTION.json';frozen=json.loads(freeze.read_text())
        if result.exists():
            assert json.loads(result.read_text())['frozen_SHA256']==sha(freeze)
            continue
        pred=out/'test_predictions.npz'
        if not cpu_only:
            assert torch.cuda.is_available()
            if not pred.exists():
                # An interrupted selected inference remains visible; never silently replay it.
                assert not (out/'TEST_STARTED.json').exists(),'TEST interrupted; saved-output recovery required, no replay'
                layer=frozen['selected']['layer'];feature=root/f'layer_{layer}'/(readout+'.npy')
                assert sha(feature)==receipt['files'][f'layer_{layer}/{readout}.npy']['SHA256']
                checkpoint=checkpoints/f'best_layer_{layer}.pt'
                assert sha(checkpoint)==frozen['selected']['checkpoint_SHA256']
                state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                net=make_model(5120,head).cuda();net.load_state_dict(state['model'])
                x=torch.from_numpy(np.load(feature,allow_pickle=False)).cuda()
                save(out/'TEST_STARTED.json',dict(job=os.environ['PBS_JOBID'],frozen_SHA256=sha(freeze),time=__import__('time').time()))
                p=predict(net,x)
                tmp=pred.with_suffix('.tmp')
                with tmp.open('wb') as f:np.savez(f,probability=p,position_id=np.asarray(data['keys']),y=data['y'],c=data['c'])
                tmp.replace(pred)
                del net,x;torch.cuda.empty_cache()
            continue
        # CPU-only separate process: do not instantiate a model or perform inference.
        z=np.load(pred,allow_pickle=False);p=z['probability']
        assert list(z['position_id'])==data['keys']
        np.testing.assert_array_equal(z['y'],data['y']);np.testing.assert_array_equal(z['c'],data['c'])
        assert p.shape==(len(data['keys']),3) and np.isfinite(p).all()
        assert (p>=0).all() and (p<=1).all();np.testing.assert_allclose(p.sum(1),1,rtol=1e-6,atol=1e-7)
        sy=p[:,0].astype(np.float64);sc=p[:,:2].sum(1).astype(np.float64)
        assert (sy<=sc).all()
        save(result,dict(status='PASS_SELECTED_TEST_ONCE_CPU_VERIFIED',arm=arm,model=MODEL,domain=domain,
                         readout=readout,head=head,loss=loss,selected_layer=frozen['selected']['layer'],
                         selected_epoch=frozen['selected']['epoch'],VAL=frozen['selected']['metrics'],
                         step=metrics(data['y'],sy,frozen['threshold_step']),
                         prefix=metrics(data['c'],sc,frozen['threshold_prefix']),
                         state_accuracy=float((p.argmax(1)==data['k']).mean()),
                         state_counts={str(i):int((data['k']==i).sum()) for i in range(3)},
                         frozen_SHA256=sha(freeze),predictions_SHA256=sha(pred),config_SHA256=sha(WORK/'config.json'),
                         data_hashes=data_hashes(domain),checkpoint_SHA256=frozen['selected']['checkpoint_SHA256'],
                         source_feature_manifest_SHA256=sha(RUN/'reports/test_features'/domain/'FEATURE_MANIFEST.json'),
                         TEST_model_replay='WAIVED_BY_DIRECT_HUMAN_INSTRUCTION; saved-prediction CPU checks only'))
        import subprocess
        subprocess.run(['/usr/bin/python3.12',str(WORK/'publish.py')],check=True,timeout=600)
    return 0


if __name__=='__main__':raise SystemExit(main(sys.argv[1],len(sys.argv)>2 and sys.argv[2]=='--cpu-only'))
