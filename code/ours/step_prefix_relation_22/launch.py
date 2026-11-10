"""Finite PBS stage runner; no resident polling, service, or remote Codex session."""
import fcntl
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from relation_common import *


def gpu(minimum):
    assert os.environ.get('PBS_JOBID')
    nodefile=Path(os.environ['PBS_NODEFILE']);nodes=nodefile.read_text().splitlines()
    assert os.uname().nodename.split('.')[0]==nodes[0].split('.')[0]
    root=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True))
    candidates=[]
    allowed=os.environ.get('CUDA_VISIBLE_DEVICES')
    allowed=set(allowed.split(',')) if allowed else None
    for g in root.findall('gpu'):
        uuid=g.findtext('uuid');index=g.findtext('minor_number')
        if allowed and uuid not in allowed and index not in allowed:continue
        processes=g.find('processes')
        total=float(g.findtext('fb_memory_usage/total').split()[0])
        used=float(g.findtext('fb_memory_usage/used').split()[0])
        utilization=float(g.findtext('utilization/gpu_util').split()[0])
        if total<minimum or utilization or (processes is not None and processes.findall('process_info')):continue
        candidates.append((used,index,uuid,total,g.findtext('product_name')))
    locks=Path('/scratch/u26487468/agent_uncertainty_quantification/gpu_ownership_locks');locks.mkdir(parents=True,exist_ok=True)
    for used,index,uuid,total,name in sorted(candidates):
        handle=(locks/('gpu'+index+'.lock')).open('a')
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:handle.close();continue
        os.environ['CUDA_VISIBLE_DEVICES']=uuid
        return handle,dict(job=os.environ['PBS_JOBID'],host=os.uname().nodename,GPU_UUID=uuid,
                           memory_mib=total,name=name,initial_used_mib=used,PBS_nodes=nodes)
    raise RuntimeError('PREMODEL_PHYSICAL_IDLE_ADMISSION: no idle compatible PBS-owned card')


def main(kind,key):
    token=kind+':'+key
    root=SCRATCH/'execution'/kind/key;root.mkdir(parents=True,exist_ok=True)
    (RUN/'logs').mkdir(parents=True,exist_ok=True)
    owner=(STATE/'locks');owner.mkdir(parents=True,exist_ok=True)
    handle=(owner/(kind+'_'+key+'.lock')).open('a')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_handle=None
    if kind!='prepare':
        gpu_handle,proof=gpu(87000 if kind in ('features','test') else 16000)
        save(RUN/'reports/allocations'/(os.environ['PBS_JOBID']+'.json'),dict(stage=token,**proof))
    env=os.environ.copy();env.update(PYTHONPATH=str(PROJECT/'assets/runtime/streaming_step_622_20261104/pydeps')+':'+str(WORK),
                                    OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',
                                    CUBLAS_WORKSPACE_CONFIG=':4096:8',EXECUTION_SECONDS='19800')
    image=PROJECT/'assets/runtime/streaming_native/vllm-openai-v0.30.0.sif'
    def execute(script,*args,cpu=False):
        cmd=['apptainer','exec']+([] if cpu else ['--nv'])+['--bind',
             '/shared/homes/u26487468:/shared/homes/u26487468,/scratch:/scratch,/opt/pbs:/opt/pbs,/etc/pbs.conf:/etc/pbs.conf',
             str(image),'python3','-u',str(WORK/script),*args]
        with (RUN/'logs'/(kind+'_'+key+'.runtime.log')).open('a') as log:
            return subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    if kind=='prepare':code=execute('prepare.py',cpu=True)
    elif kind=='fit':code=execute('train.py',key)
    elif kind=='features':code=execute('extract.py',key,'train')
    elif kind=='test':
        code=execute('extract.py',key,'test')
        if code==0:code=execute('evaluate.py',key)
        if code==0:code=execute('evaluate.py',key,'--cpu-only',cpu=True)
    else:raise ValueError(kind)
    save(RUN/'reports/exits'/(os.environ['PBS_JOBID']+'.json'),dict(stage=token,job=os.environ['PBS_JOBID'],exit=code,time=time.time()))
    # Bounded stage-completion callback only; no polling loop or extra monitor.
    subprocess.run(['/usr/bin/python3.12',str(WORK/'dispatch.py'),'--completed',token,str(code)],timeout=180)
    subprocess.run(['/usr/bin/python3.12',str(WORK/'publish.py')],timeout=600)
    return code


if __name__=='__main__':raise SystemExit(main(*sys.argv[1:]))
