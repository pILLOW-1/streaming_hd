"""Bounded audit/unique-owner PBS admission; callbacks and existing heartbeat only."""
import concurrent.futures
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from relation_common import *


def command(args, timeout=25):
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout)
    return p.returncode,p.stdout,p.stderr


def audit():
    queues=json.loads(command(['/opt/pbs/bin/qstat','-Qf','-F','json'])[1])['Queue']
    nodes=json.loads(command(['/opt/pbs/bin/pbsnodes','-av','-F','json'])[1])['nodes']
    # qstat -f covers current jobs; filters immediately to this account.
    jobs=json.loads(command(['/opt/pbs/bin/qstat','-f','-F','json'],60)[1]).get('Jobs',{})
    own={j:v for j,v in jobs.items() if v.get('Job_Owner','').split('@')[0]=='u26487468'}
    active_hosts=sorted({piece.split('/')[0] for v in own.values() if v.get('job_state')=='R'
                         for piece in v.get('exec_host','').split('+') if piece})
    def physical(host):
        try:
            rc,o,e=command(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=4',host,
                            'nvidia-smi --query-gpu=uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits'],8)
            return host,dict(returncode=rc,inventory=o.strip(),error=e[:200])
        except Exception as e:return host,dict(status='UNKNOWN',error=str(e)[:200])
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:physical_rows=dict(pool.map(physical,active_hosts))
    candidates={}
    for host,n in nodes.items():
        a=n.get('resources_available',{});used=n.get('resources_assigned',{})
        if int(a.get('ngpus',0))<1:continue
        candidates[host]=dict(state=n['state'],queue=a.get('Qlist'),available=a,assigned=used,
                              reservations=n.get('resv',[]),physical=physical_rows.get(host,{'status':'UNKNOWN_NO_OWN_ALLOCATION'}))
    rc,reservations,error=command(['/opt/pbs/bin/pbs_rstat','-f'])
    snapshot=dict(time=time.time(),queues=queues,candidates=candidates,reservations=reservations[:30000],
                  reservation_query_exit=rc,own_jobs={j:{k:v.get(k) for k in ['job_state','queue','exec_host','Resource_List','resources_used','comment']} for j,v in own.items()},
                  policy='Smallest sufficient GPU for fit; CPU-only exact source chunk; no free physical GPU inference from Q/R counts. Native Qwen BF16 missing-pool phase requires87GiB based on existing measured producer. No unrelated mutation.')
    save(STATE/'RESOURCE_AUDIT.json',snapshot)
    return queues,nodes,own


def main():
    STATE.mkdir(parents=True,exist_ok=True);(RUN/'logs').mkdir(parents=True,exist_ok=True)
    with (STATE/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=STATE/'OWNERS.json';owners=json.loads(path.read_text()) if path.exists() else {}
        if '--completed' in sys.argv:
            i=sys.argv.index('--completed');token=sys.argv[i+1];code=int(sys.argv[i+2])
            if owners.get(token,{}).get('job')==os.environ.get('PBS_JOBID'):
                owners[token].update(status='DONE' if code==0 else ('CONTINUE' if code==75 else 'FAILED'),exit=code)
        cfg=config();queues,nodes,own=audit();blocked_queues=set()
        tokens=[('prepare','all')]
        if (RUN/'reports/PREPARATION.json').exists():
            assert json.loads((RUN/'reports/PREPARATION.json').read_text())['status']=='PASS'
            tokens += [('features',d) for d in DOMAINS]
            tokens += [('fit',a) for a in cfg['arms']]
            tokens += [('test',d) for d in DOMAINS]
        for kind,key in tokens:
            token=kind+':'+key;old=owners.get(token,{})
            if old.get('status') in ('DONE','FAILED'):continue
            job=old.get('job')
            if job and old.get('status')!='CONTINUE':
                if job in own and own[job].get('job_state') in ('Q','H','R','E','W','B','T'):continue
                # An unobserved owner is UNKNOWN; never duplicate a submission.
                rc,o,e=command(['/opt/pbs/bin/qstat','-x','-f','-F','json',job])
                if rc==0:
                    history=json.loads(o).get('Jobs',{}).get(job,{})
                    if history.get('job_state')=='F':
                        code=int(history.get('Exit_status',-999))
                        old.update(status='CONTINUE' if code==75 else ('DONE' if code==0 else 'FAILED'),exit=code)
                        owners[token]=old
                        if old['status']!='CONTINUE':continue
                    else:continue
                else:continue
            if kind=='fit':
                if (arm_paths(key)[0]/'FROZEN_SELECTION.json').exists():
                    owners[token]=dict(status='DONE',reason='frozen accepted fit skipped');continue
                domain,readout,head,loss=key.split('.')
                try:receipt,_,_=source(domain,readout)
                except (FileNotFoundError,AssertionError):continue
                source_host=receipt['host'].split('.')[0]
                # All candidate layers stay in this continuous job entry.
                options=['small_gpuq','med_gpuq','large_gpuq']
                need='1:ncpus=4:mem=32gb:ngpus=1:mpiprocs=1:Qlist={queue}+1:ncpus=1:mem=4gb:mpiprocs=1:host='+source_host
            elif kind in ('features','test'):
                domain=key
                if kind=='features' and (RUN/'reports/features'/domain/'FEATURE_MANIFEST.json').exists():
                    owners[token]=dict(status='DONE',reason='exact manifest exists');continue
                if kind=='test':
                    arms=[a for a in cfg['arms'] if a.startswith(domain+'.')]
                    if not all((arm_paths(a)[0]/'FROZEN_SELECTION.json').exists() for a in arms):continue
                    if all((arm_paths(a)[0]/'TEST_RESULT.json').exists() for a in arms):
                        owners[token]=dict(status='DONE');continue
                options=['large_gpuq'];need='1:ncpus=4:mem=96gb:ngpus=1:mpiprocs=1:Qlist={queue}'
                progress=RUN/'reports'/('features' if kind=='features' else 'test_features')/domain/'PROGRESS.json'
                if progress.exists():
                    previous=json.loads(progress.read_text());host=previous['host'].split('.')[0]
                    need+=':host='+host
            else:options=['smallq','medq'];need='1:ncpus=2:mem=12gb:mpiprocs=1'
            for queue in options:
                if queue in blocked_queues:continue
                q=queues.get(queue,{})
                if q.get('enabled')!='True' or q.get('started')!='True':continue
                if q.get('acl_user_enable')=='True' and 'u26487468' not in str(q.get('acl_users','')):continue
                script=WORK/'pbs'/((kind+'_'+key)+'.pbs');script.parent.mkdir(exist_ok=True)
                script.write_text('#!/bin/bash\nset -euo pipefail\nexec /usr/bin/python3.12 -u '+str(WORK/'launch.py')+' '+kind+' '+key+'\n')
                args=['/opt/pbs/bin/qsub','-q',queue,'-N',('sp22_'+kind+'_'+key)[:60],
                      '-l','select='+need.format(queue=queue),'-l','walltime=06:00:00' if kind!='prepare' else 'walltime=01:00:00',
                      '-l','place=free','-j','oe','-o',str(RUN/'logs'/(kind+'_'+key+'.pbs.log')),str(script)]
                rc,o,e=command(args)
                if rc==0:
                    owners[token]=dict(status='SUBMITTED',job=o.strip(),queue=queue,time=time.time(),
                                       previous_job=job,resource_request=args,expected_input_ready=True)
                    save(path,owners);break
                owners.setdefault(token,{}).update(status='ADMISSION_PENDING',last_error=e.strip(),last_queue=queue,time=time.time())
                if any(s in e.lower() for s in ['limit','exceed','maximum','max_queued']):blocked_queues.add(queue)
        save(path,owners)
        print(json.dumps({'submitted':{k:v['job'] for k,v in owners.items() if v.get('status')=='SUBMITTED'},
                          'pending':sum(v.get('status')=='ADMISSION_PENDING' for v in owners.values()),
                          'failed':{k:v for k,v in owners.items() if v.get('status')=='FAILED'}}))


if __name__=='__main__':main()
