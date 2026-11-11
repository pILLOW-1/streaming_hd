"""Normal Git publication of this experiment only, with concrete commit descriptions."""
import fcntl
import json
import subprocess
import tempfile
import time
import sys
import csv
import io
from pathlib import Path
from relation_common import *

PUBLIC=Path('experiments')/IDENTITY
PUBLIC_CODE='code/ours/'+IDENTITY+'/'
GH='/shared/homes/u26487468/.local/bin/gh'
RULE='''\n## Method-design experiment records\n\nFor every new method-design experiment, create a dedicated `experiments/<experiment_id>/` directory before execution. Use concise English: motivation, a configuration table with one configuration per row, and a separate training-parameter table. Result tables use configurations as rows and domains as columns. Step and Prefix have separate tables with identical rows/columns; unfinished cells stay empty. Put each independently verified completed configuration in `result/` immediately and refresh the result index in the same concrete Git commit/push. Never report registered, queued, trained-only or frozen-only configurations as completed results. Keep code/configuration versioned, raw data/features and trained weights outside the publication. Commit messages identify the actual model, domain, configuration and result or fix.\n'''


def sync_checkout():
    subprocess.run(['/usr/bin/python3.12',str(PROJECT/'workspaces/streaming_hd/common/sync_published_checkout.py')],check=True,timeout=240)


def main():
    cfg=config();files={};results=[];index=[]
    for arm_index,arm in enumerate(cfg['arms']):
        out,_=arm_paths(arm);p=out/'TEST_RESULT.json'
        domain,readout,head,loss=arm.split('.')
        row=dict(ID='C'+str(arm_index%16+1).zfill(2),domain=domain,readout=readout,head=head,loss=loss,
                 configuration=arm,status='PENDING',step=None,prefix=None)
        if p.exists():
            x=json.loads(p.read_text());assert x['status']=='PASS_SELECTED_TEST_ONCE_CPU_VERIFIED'
            assert x['predictions_SHA256']==sha(out/'test_predictions.npz')
            assert x['frozen_SHA256']==sha(out/'FROZEN_SELECTION.json')
            assert x['config_SHA256']==sha(WORK/'config.json')
            results.append(x)
            row.update(status='CPU_VERIFIED',step=x['step'],prefix=x['prefix'],selected_layer=x['selected_layer'],selected_epoch=x['selected_epoch'])
            files[str(PUBLIC/'result'/Path(*arm.split('.')).with_suffix('.json'))]=(json.dumps(x,indent=2)+'\n').encode()
        index.append(row)
    files[str(PUBLIC/'config.json')]=(WORK/'config.json').read_bytes()
    files[str(PUBLIC/'result/index.json')]=(json.dumps(dict(completed=len(results),total=80,configurations=index),indent=2)+'\n').encode()
    domains=['Math','Logic','Science','Multi-hop','Factuality']
    ids=['C'+str(i).zfill(2) for i in range(1,17)]
    lookup={(item['ID'],item['domain']):item for item in index}
    assert len(lookup)==80
    def matrix_csv(level,metric):
        stream=io.StringIO();writer=csv.writer(stream,lineterminator='\n');writer.writerow(['Configuration']+domains)
        for cid in ids:
            writer.writerow([cid]+[(lookup[cid,d][level] or {}).get(metric,'') for d in domains])
        return stream.getvalue().encode()
    for level in ('step','prefix'):
        for metric in ('AUROC','AP'):
            files[str(PUBLIC/'result'/(level+'_'+metric.lower()+'.csv'))]=matrix_csv(level,metric)
    summary=io.StringIO();writer=csv.writer(summary,lineterminator='\n');writer.writerow(['Configuration']+domains)
    for cid in ids:
        writer.writerow([cid]+[(str(lookup[cid,d]['step']['AUROC'])+' / '+str(lookup[cid,d]['prefix']['AUROC'])) if lookup[cid,d]['step'] else '' for d in domains])
    files[str(PUBLIC/'result/summary.csv')]=summary.getvalue().encode()
    readme=(WORK/'README.md').read_text().replace('{{COMPLETED}}',str(len(results)))
    table=[]
    for level in ('step','prefix'):
        table+=['\n## '+level.title()+' AUROC\n','| Configuration | '+' | '.join(domains)+' |','|---|'+'---:|'*len(domains)]
        for cid in ids:
            values=[f"{lookup[cid,d][level]['AUROC']:.4f}" if lookup[cid,d][level] else '' for d in domains]
            table.append('| '+' | '.join([cid]+values)+' |')
    if not results:table.append('\nNo accepted results yet. Pending configurations have no result values.')
    files[str(PUBLIC/'README.md')]=(readme+'\n'.join(table)+'\n').encode()
    for p in sorted(WORK.glob('*.py')):
        compile(p.read_text(),str(p),'exec')
        files[PUBLIC_CODE+p.name]=p.read_bytes()
    files[PUBLIC_CODE+'SOURCE_MANIFEST.json']=(WORK/'SOURCE_MANIFEST.json').read_bytes()
    files[PUBLIC_CODE+'FEATURE_COMPATIBILITY.json']=(WORK/'FEATURE_COMPATIBILITY.json').read_bytes()
    with (STATE/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Serialize with the existing baseline publisher without changing its state.
        global_lock=(PROJECT/'state/streaming_hd/baselines_622_p70_20261108/RESULT_PUBLICATION.lock').open('a')
        fcntl.flock(global_lock,fcntl.LOCK_EX)
        ledger_path=STATE/'PUBLICATION_LEDGER.json'
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(files={},commits=[])
        changed={p:b for p,b in files.items() if ledger['files'].get(p)!=hashlib.sha256(b).hexdigest()}
        if not changed:
            sync_checkout();return
        with tempfile.TemporaryDirectory(prefix='sp22_git_') as directory:
            repo=Path(directory)/'repo'
            def git(*args):
                return subprocess.check_output(['git','-c','credential.helper=',
                    '-c','credential.helper=!'+GH+' auth git-credential','-c','user.name=pILLOW-1',
                    '-c','user.email=pILLOW-1@users.noreply.github.com',*args],cwd=repo if repo.exists() else directory,
                    text=True,stderr=subprocess.PIPE,timeout=180).strip()
            git('clone','--quiet','--branch','main','https://github.com/pILLOW-1/streaming_hd.git',str(repo))
            for name,b in changed.items():
                assert name.startswith((str(PUBLIC)+'/', PUBLIC_CODE))
                p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
            policy=repo/'AGENTS.md';policy_text=policy.read_text() if policy.exists() else '# Repository instructions\n'
            marker='## Method-design experiment records'
            if marker not in policy_text:updated=policy_text.rstrip()+'\n'+RULE
            else:
                start=policy_text.index(marker);end=policy_text.find('\n## ',start+len(marker))
                updated=policy_text[:start].rstrip()+'\n'+RULE+(policy_text[end:] if end>=0 else '')
            if updated!=policy_text:
                policy.write_text(updated);changed['AGENTS.md']=policy.read_bytes()
            git('add','--',*sorted(changed))
            if git('diff','--cached','--name-only'):
                fresh=[x for x in results if str(PUBLIC/'result'/Path(*x['arm'].split('.')).with_suffix('.json')) in changed]
                if fresh:
                    subject='results: add '+str(len(fresh))+' Qwen Step-Prefix configurations'
                    body='\n'.join(x['arm']+f"; layer={x['selected_layer']}; Step AUROC={x['step']['AUROC']:.6f}; Prefix AUROC={x['prefix']['AUROC']:.6f}" for x in fresh)
                else:
                    subject='experiment: configure Qwen five-domain Step-Prefix readout/head/loss study'
                    body='80 independent configurations:4 raw Streaming readouts x2 heads x2 objectives x5 domains; VAL selects64 layers and checkpoint; selected TEST once.\nAdd experiment README/config/result index and versioned execution/validation source; preserve pending results as null.'
                if '--subject' in sys.argv:
                    subject=sys.argv[sys.argv.index('--subject')+1]
                    body=sys.argv[sys.argv.index('--body')+1]
                git('commit','--quiet','-m',subject,'-m',body+'\n\nFiles:\n'+'\n'.join(sorted(changed)))
                git('push','origin','HEAD:main')
            commit=git('rev-parse','HEAD');assert git('ls-remote','origin','refs/heads/main').split()[0]==commit
            for p,b in changed.items():assert (repo/p).read_bytes()==b
        for p,b in changed.items():ledger['files'][p]=hashlib.sha256(b).hexdigest()
        ledger['commits'].append(dict(commit=commit,completed=len(results),files=sorted(changed),time=time.time()))
        save(ledger_path,ledger)
        sync_checkout()
        print(json.dumps(ledger['commits'][-1]))


if __name__=='__main__':main()
