"""Compare pooled TEST recovery from existing frozen Prefix predictions only."""
import csv,hashlib,json,os
from pathlib import Path
import numpy as np

PROJECT=Path(os.environ['STREAMING_HD_PRIVATE_PROJECT']).expanduser().resolve()
RUNS=PROJECT/'experiments/streaming_hd/runs'
CANON=Path(os.environ['STREAMING_HD_REPOSITORY']).expanduser().resolve()
RUN=RUNS/'qwen_c18_error_memory_20261114'
OUT=RUN/'diagnostics/pooled_prefix_recovery_20261119'
DISPLAY={'streaming':'Streaming','global_mean':'Global Mean','saplma':'SAPLMA','icr_probe':'ICR Probe','llm_check':'LLM-Check'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def auc(y,p):
    pos=p[y==1];neg=np.sort(p[y==0]);assert len(pos) and len(neg)
    value=float((np.searchsorted(neg,pos,'left')+np.searchsorted(neg,pos,'right')).sum()/2/(len(pos)*len(neg)))
    order=np.argsort(-p,kind='stable');target=y[order];scores=p[order]
    ends=np.r_[np.flatnonzero(scores[1:]!=scores[:-1]),len(scores)-1]
    tp=target.cumsum()[ends];fp=ends+1-tp
    independent=float(np.trapezoid(np.r_[0,tp],np.r_[0,fp])/(tp[-1]*fp[-1]))
    assert abs(value-independent)<1e-12
    return value

def load_score(path,keys,y,c):
    with np.load(path,allow_pickle=False) as b:
        ids=b['position_id'] if 'position_id' in b else b['sample_id']
        assert len(set(ids))==len(ids)==len(keys) and set(ids)==set(keys)
        lookup={k:i for i,k in enumerate(ids)};order=np.asarray([lookup[k] for k in keys])
        if 'c' in b:
            assert np.array_equal(b['y'][order],y) and np.array_equal(b['c'][order],c)
        else:assert np.array_equal(b['labels' if 'labels' in b else 'y'][order],c)
        if 'scores' in b:score=b['scores'][order].astype(np.float64)
        else:
            probabilities=b['probability'][order].astype(np.float64)
            assert probabilities.shape==(len(keys),3) and np.allclose(probabilities.sum(1),1)
            score=probabilities[:,:2].sum(1)
        assert score.shape==c.shape and np.isfinite(score).all()
    return score

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    official=json.loads((CANON/'reports/main_result/sources.json').read_text())
    diagnostic=json.loads((CANON/'experiments/prefix_recovery_diagnostic_20261112/result/recovery.json').read_text())
    result=dict(status='CPU_VERIFIED_FROM_SAVED_TEST',split='TEST',model='qwen3_8_27b',metric='Prefix recovery AUROC, pooled inherited vs recovered',positive='all TEST positions with Step=0 and Prefix=1',negative='TEST positions with Step=0 and Prefix=0 after an earlier retained Prefix=1 in their trajectory',aggregation='pool across trajectories; no same-trajectory pair restriction or per-trajectory averaging',score='frozen selected detector final Prefix probability',selection='reuse original VAL-frozen configurations; no new selection, training, model inference, bootstrap or threshold tuning',best_baseline_rule='highest published overall Prefix TEST AUROC per domain, not recovery AUROC',domains={},sources=[])
    for domain in ('Logic','Factuality'):
        gold=RUN/'runpod'/domain/'A2/test.npz'
        with np.load(gold,allow_pickle=False) as z:
            keys,y,c,prior,groups=(z[k] for k in ('position_id','y','c','prior','group'))
            assert len(set(keys))==len(keys) and np.all(y<=c)
            seen={};last={};expected=[]
            for key,group,label in zip(keys,groups,c):
                step=int(key.rsplit(':',1)[1]);assert step>last.get(group,-1);last[group]=step
                expected.append(seen.get(group,False));seen[group]=seen.get(group,False) or bool(label)
            assert np.array_equal(prior,np.asarray(expected))
            positive=(y==0)&(c==1);negative=(y==0)&(c==0)&prior;mask=positive|negative
            cohorts=dict(inherited=int(positive.sum()),recovered_clean=int(negative.sum()),fresh_clean=int(((y==0)&(c==0)&~prior).sum()),inherited_without_earlier_observed_positive=int((positive&~prior).sum()),total_TEST_positions=len(keys),cohort_positions=int(mask.sum()))
        choices=[r for r in official if r['level']=='prefix' and r['model']=='qwen3_8_27b' and r['domain']==domain and r.get('AUROC') is not None]
        best=max(choices,key=lambda r:r['AUROC']);best_method=best['method']
        verified=json.loads((RUN/'runpod'/domain/'verified.json').read_text())
        source_lookup={(r['method'],r['domain']):r for r in diagnostic['sources'] if r['model']=='qwen3_8_27b'}
        methods={}
        for label,method in [('Streaming','streaming'),('Best published baseline',best_method),('A1','A1'),('A2','A2')]:
            if method in ('A1','A2'):
                if method=='A2':path=gold
                else:
                    path=RUNS/'step_prefix_ordered_dual_qwen_wd1e4_20261111/runpod'/domain
                    if domain=='Factuality':path=path/domain
                    path=path/'ordered_dual/linear/ce/test_predictions.npz'
                    pin=json.loads((path.parent/'TEST_RESULT.json').read_text());assert sha(path)==pin['predictions_SHA256']
                expected_overall=verified['test'][method]['prefix'];source_method=method
            else:
                pin=source_lookup[(method,domain)];path=RUNS/pin['file'];assert sha(path)==pin['SHA256']
                original=next(r for r in diagnostic['baselines'] if r['model']=='qwen3_8_27b' and r['domain']==domain and r['method']==method)
                assert all(cohorts[k]==original[k] for k in ('inherited','recovered_clean','fresh_clean'))
                published=next(r for r in choices if r['method']==method)
                assert sha(CANON/published['source'])==published['SHA256']
                expected_overall=published['AUROC'];source_method=DISPLAY.get(method,method)
            score=load_score(path,keys,y,c);overall=auc(c,score);recovery=auc(c[mask],score[mask])
            assert abs(overall-expected_overall)<2e-6
            if method not in ('A1','A2'):assert abs(recovery-original['AUROC_rec'])<1e-12
            methods[label]=dict(method=source_method,overall_Prefix_TEST_AUROC=overall,pooled_Prefix_recovery_AUROC=recovery,predictions_SHA256=sha(path),source=str(path.relative_to(RUNS)))
        result['domains'][domain]=dict(cohort=cohorts,best_baseline_method=DISPLAY.get(best_method,best_method),methods=methods)
    result['sources']=[dict(path=f,sha256=sha(CANON/f)) for f in ('reports/main_result/sources.json','experiments/prefix_recovery_diagnostic_20261112/result/recovery.json','experiments/prefix_recovery_diagnostic_20261112/config.json')]
    result['analysis_source_SHA256']=sha(Path(__file__))
    (OUT/'pooled_prefix_recovery.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'pooled_prefix_recovery.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['configuration','domain','method','Prefix_TEST_AUROC','pooled_Prefix_recovery_AUROC','inherited','recovered_clean'])
        for domain,d in result['domains'].items():
            for label,m in d['methods'].items():writer.writerow([label,domain,m['method'],m['overall_Prefix_TEST_AUROC'],m['pooled_Prefix_recovery_AUROC'],d['cohort']['inherited'],d['cohort']['recovered_clean']])
    print(json.dumps(result))

if __name__=='__main__':main()
