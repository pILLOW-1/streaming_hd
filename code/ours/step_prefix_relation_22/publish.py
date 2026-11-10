"""Normal Git publication of this experiment only, with concrete commit descriptions."""
import fcntl
import json
import subprocess
import tempfile
import time
import sys
from pathlib import Path
from relation_common import *

PUBLIC=Path('experiments')/IDENTITY
GH='/shared/homes/u26487468/.local/bin/gh'
RULE='''\n## Method-design experiment records\n\nFor every new method-design experiment, create a dedicated `experiments/<experiment_id>/` directory before execution. Its concise README states motivation, frozen configuration, selection/evaluation protocol and current completion status. Put each independently verified completed configuration in `result/` immediately and refresh the result index in the same concrete Git commit/push. Never report registered, queued, trained-only or frozen-only configurations as completed results. Keep code/configuration versioned, raw data/features and trained weights outside the publication. Commit messages identify the actual model, domain, configuration and result or fix.\n'''


def main():
    cfg=config();files={};results=[];index=[]
    for arm in cfg['arms']:
        out,_=arm_paths(arm);p=out/'TEST_RESULT.json'
        row=dict(configuration=arm,status='PENDING',step=None,prefix=None)
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
    readme=(WORK/'README.md').read_text().replace('{{COMPLETED}}',str(len(results)))
    table=['\n## 已完成配置\n','| Domain | Readout | Head | Loss | Layer | Step AUROC/AP | Prefix AUROC/AP |',
           '|---|---|---|---|---:|---:|---:|']
    for x in results:
        table.append('| '+' | '.join([x['domain'],x['readout'],x['head'],x['loss'],str(x['selected_layer']),
             f"{x['step']['AUROC']:.4f} / {x['step']['AP']:.4f}",f"{x['prefix']['AUROC']:.4f} / {x['prefix']['AP']:.4f}"])+' |')
    if not results:table.append('\n尚无已验收结果；未完成配置不填入数值。')
    files[str(PUBLIC/'README.md')]=(readme+'\n'.join(table)+'\n').encode()
    for p in sorted(WORK.glob('*.py')):
        compile(p.read_text(),str(p),'exec')
        files['code/ours/step_prefix_relation_22/'+p.name]=p.read_bytes()
    files['code/ours/step_prefix_relation_22/SOURCE_MANIFEST.json']=(WORK/'SOURCE_MANIFEST.json').read_bytes()
    with (STATE/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Serialize with the existing baseline publisher without changing its state.
        global_lock=(PROJECT/'state/streaming_hd/baselines_622_p70_20261108/RESULT_PUBLICATION.lock').open('a')
        fcntl.flock(global_lock,fcntl.LOCK_EX)
        ledger_path=STATE/'PUBLICATION_LEDGER.json'
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(files={},commits=[])
        changed={p:b for p,b in files.items() if ledger['files'].get(p)!=hashlib.sha256(b).hexdigest()}
        if not changed:return
        with tempfile.TemporaryDirectory(prefix='sp22_git_') as directory:
            repo=Path(directory)/'repo'
            def git(*args):
                return subprocess.check_output(['git','-c','credential.helper=',
                    '-c','credential.helper=!'+GH+' auth git-credential','-c','user.name=pILLOW-1',
                    '-c','user.email=pILLOW-1@users.noreply.github.com',*args],cwd=repo if repo.exists() else directory,
                    text=True,stderr=subprocess.PIPE,timeout=180).strip()
            git('clone','--quiet','--branch','main','https://github.com/pILLOW-1/streaming_hd.git',str(repo))
            for name,b in changed.items():
                assert name.startswith((str(PUBLIC)+'/', 'code/ours/step_prefix_relation_22/'))
                p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
            policy=repo/'AGENTS.md';policy_text=policy.read_text() if policy.exists() else '# Repository instructions\n'
            if '## Method-design experiment records' not in policy_text:
                policy.write_text(policy_text.rstrip()+'\n'+RULE);changed['AGENTS.md']=policy.read_bytes()
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
        print(json.dumps(ledger['commits'][-1]))


if __name__=='__main__':main()
