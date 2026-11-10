#!/usr/bin/env python3
"""Publish only verified completed baseline results and admitted code corrections."""
import fcntl,hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
P=Path('/shared/homes/u26487468/agent_uncertainty_quantification');S=P/'state/streaming_hd/baselines_622_p70_20261108';RUN=P/'experiments/streaming_hd/runs/baselines_622_p70_20261108';R=Path('/shared/homes/u26487468/streaming_hd')
GH='/shared/homes/u26487468/.local/bin/gh';REPO='pILLOW-1/streaming_hd';PREFIX='repos/'+REPO

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def sync_checkout():
    subprocess.run(['/usr/bin/python3.12',str(P/'workspaces/streaming_hd/common/sync_published_checkout.py')],check=True,timeout=240)


def main():
    with (S/'RESULT_PUBLICATION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        files={};groups={}
        for path in sorted((RUN/'fits').glob('*/*/*/*/reports/TEST_RESULT.json')):
            result=json.loads(path.read_text())
            if result.get('status')=='INVALID_SCORE_DIRECTION':
                model,domain,method,level=path.parts[-6:-2]
                relative=f'experiments/{method}/622_p70_20261108/result/{model}/{domain}/{level}.json'
                files[relative]=(json.dumps(dict(model=model,domain=domain,level=level,method=method,status='WITHDRAWN_IMPLEMENTATION_ERROR',reason='The official Attention Score requires a negative sign before VAL layer selection. The previous result used the opposite sign. Corrected own-VAL selection and TEST evaluation are pending.',TEST=None),indent=2)+'\n').encode()
                files[f'experiments/{method}/622_p70_20261108/README.md']=b'# LLM-Check\n\nThe initial result has been withdrawn: raw Attention Scores were not negated as required by the official evaluation notebook. Corrected own-VAL layer/threshold selection and selected-TEST evaluation are pending. No valid result is currently reported.\n'
                continue
            if result.get('status')!='PASS_SELECTED_TEST_ONCE_CPU_VERIFIED':continue
            model,domain,method,level=path.parts[-6:-2]
            frozen_file=path.parent/'FROZEN_SELECTION.json';assert sha(frozen_file)==result['frozen_SHA256']
            frozen=json.loads(frozen_file.read_text());pred=path.parent/'test_predictions.npz'
            expected=result.get('predictions_SHA256',result.get('TEST_predictions_SHA256'));assert expected==sha(pred)
            public=dict(model=model,domain=domain,level=level,method=method,split='6:2:2, TEST target p >= 70% (frozen per-cell relaxations retained)',selected_layer=result.get('selected_layer'),selection='own VAL AUROC, tie by VAL AP; threshold by own VAL F1max; strict score > threshold',threshold=result['threshold'],VAL=frozen['VAL'],TEST=result['metrics'],validation_status='independently checked from saved TEST predictions',result_SHA256=sha(path))
            public['selected_configuration']=frozen.get('selected',{})
            if result.get('method_adaptations'):public['method_configuration']=result['method_adaptations']
            relative=f'experiments/{method}/622_p70_20261108/result/{model}/{domain}/{level}.json'
            files[relative]=(json.dumps(public,indent=2,allow_nan=False)+'\n').encode();groups.setdefault(method,[]).append(public)
        names={'saplma':'SAPLMA','global_mean':'Global Mean','hami':'HaMI (ori)','ars':'ARS','rede':'ReDe','llm_check':'LLM-Check','icr_probe':'ICR Probe','rhd':'RHD','race':'RACE'}
        for method,rows in groups.items():
            table=['| Model | Domain | Level | Layer | AUROC | AP | TPR@1%FPR | ACC | F1 |','|---|---|---|---:|---:|---:|---:|---:|---:|']
            for row in rows:
                m=row['TEST'];table.append('| '+' | '.join([row['model'],row['domain'],row['level'],str(row['selected_layer'])]+[f'{100*m[k]:.2f}' for k in ['AUROC','AP','TPR_at_1pct_FPR','ACC','F1']])+' |')
            text=f"# {names.get(method,method)}\n\nBaseline evaluation on the selected 6:2:2 assignment. TEST targets trajectories with repeated Prefix 1→0 transitions (target p ≥ 70%, with the frozen domain-specific relaxations).\n\nEach model/domain/level is independent. We select layers and detector weights on complete VAL AUROC, breaking ties by VAL AP, then freeze a VAL F1max threshold and evaluate TEST once. TPR@1%FPR is the maximum empirical TEST ROC TPR at FPR ≤ 1%, without interpolation. Higher scores indicate hallucination.\n\n## Completed results\n\nMetrics below are percentages. JSON preserves unrounded values, thresholds and confusion counts. New rows are added only after saved-score checks pass.\n\n"+'\n'.join(table)+'\n\nDetector weights are not included in this incremental release.\n'
            files[f'experiments/{method}/622_p70_20261108/README.md']=text.encode()
            files[f'experiments/{method}/README.md']=(f'# {names.get(method,method)}\n\n- [6:2:2, TEST target p ≥ 70%](622_p70_20261108/README.md)\n').encode()
        admission=S/'CODE_PUBLICATION_ADMISSION.json'
        if admission.exists():
            accepted=json.loads(admission.read_text());assert accepted['status']=='PASS_CURRENT_CODE_VALIDATED'
            current=all((R/path).is_file() and sha(R/path)==expected for path,expected in accepted['files'].items())
            if current:
                for path,expected in accepted['files'].items():
                    assert path.startswith(('code/','configs/')) and not any(k in path.lower() for k in ['checkpoint','credential','secret'])
                    files[path]=(R/path).read_bytes()
            else:print('CODE_PENDING_REVALIDATION; completed results remain publishable',flush=True)
        if not files:return
        ledger_file=S/'PUBLICATION_LEDGER.json';ledger=json.loads(ledger_file.read_text()) if ledger_file.exists() else {'files':{},'commits':[]}
        changed={k:v for k,v in files.items() if ledger['files'].get(k)!=hashlib.sha256(v).hexdigest()}
        if not changed:
            sync_checkout();return
        for relative,data in changed.items():
            assert relative.startswith(('experiments/','code/','configs/'))
            assert not relative.endswith(('.pt','.pth','.safetensors','.bin','.npy','.npz'))
            assert b'/shared/homes/' not in data and b'/scratch/' not in data
        # Publish through standard Git commands in an isolated temporary clone.
        # The shared checkout can contain collaborators' unfinished changes;
        # only the explicitly admitted paths enter this publication commit.
        with tempfile.TemporaryDirectory(prefix='streaming_hd_publish_') as directory:
            checkout=Path(directory)/'repository'
            def git(*args):
                return subprocess.check_output(['git','-c','credential.helper=',
                    '-c','credential.helper=!'+GH+' auth git-credential',
                    '-c','user.name=pILLOW-1','-c','user.email=pILLOW-1@users.noreply.github.com',
                    *args],cwd=checkout if checkout.exists() else directory,
                    text=True,stderr=subprocess.PIPE,timeout=180).strip()
            git('clone','--quiet','--branch','main','https://github.com/'+REPO+'.git',str(checkout))
            paths=sorted(changed)
            for relative,data in changed.items():
                dest=checkout/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
            # Direct human authorization: refresh the paper comparison in the
            # same Git commit as every newly accepted experiment result.
            # Build against this exact result set, not the stale local copy.
            subprocess.run(['/usr/bin/python3.12',str(R/'code/tools/build_comparison_report.py'),
                            '--repo-root',str(checkout)],check=True,timeout=120)
            for relative in ['reports/main_result/comparison.md',
                             'reports/main_result/comparison.csv',
                             'reports/main_result/sources.json']:
                data=(checkout/relative).read_bytes()
                assert b'/shared/homes/' not in data and b'/scratch/' not in data
                changed[relative]=data
            paths=sorted(changed)
            git('add','--',*paths)
            if git('diff','--cached','--name-only'):
                result_paths=[p for p in paths if p.startswith('experiments/') and '/result/' in p and p.endswith('.json')]
                details=[]
                for relative in result_paths:
                    row=json.loads(changed[relative])
                    description='{model} / {domain} / {method} / {level}'.format(**row)
                    metrics=row.get('TEST') or {}
                    details.append(description+'; '+', '.join(k+'='+str(metrics[k]) for k in ['AUROC','AP','F1'] if k in metrics))
                if result_paths:
                    methods=sorted({json.loads(changed[p])['method'] for p in result_paths})
                    subject='results: publish '+str(len(result_paths))+' verified '+', '.join(methods)+' cells'
                else:
                    subject='reports: refresh '+', '.join(paths[:3])
                body='Verified result cells:\n'+'\n'.join('- '+d for d in details) if details else 'Updated files:\n'+'\n'.join('- '+p for p in paths)
                if any(p.startswith('reports/') for p in paths):
                    body+='\n\nRefresh AUROC comparisons and bold column maxima, including exact ties.'
                body+='\n\nChanged files:\n'+'\n'.join(paths)
                git('commit','--quiet','-m',subject,'-m',body)
                git('push','origin','HEAD:main')
            new_sha=git('rev-parse','HEAD')
            remote_sha=git('ls-remote','origin','refs/heads/main').split()[0]
            assert new_sha==remote_sha,'Remote advanced; retain local evidence and reconcile before ledger acceptance'
            for relative,data in changed.items():
                assert (checkout/relative).read_bytes()==data
                assert git('rev-parse','HEAD:'+relative)==git('hash-object',str(checkout/relative))
        new={'sha':new_sha}
        for relative,data in changed.items():
            ledger['files'][relative]=hashlib.sha256(data).hexdigest();dest=R/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            if relative.startswith(('experiments/','reports/')):dest.write_bytes(data)
        ledger['commits'].append(dict(commit=new['sha'],files=sorted(changed),completed_cells=sum(map(len,groups.values())),timestamp=time.time()))
        save(ledger_file,ledger);sync_checkout();print('PUBLISHED',new['sha'],'completed_cells',sum(map(len,groups.values())),'files',len(changed),flush=True)
if __name__=='__main__':main()
