"""One bounded shared layer queue per domain; two readout arms, one TEST each."""
import os,sys,time,fcntl,subprocess
from pathlib import Path

def child(domain,arm,layer):
 import torch,worker as m,feature_cache
 feature_cache.attach(m.w,m.matched)
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 device=torch.device('cuda');out=m.w.BASE/m.NAME/domain/arm;out.mkdir(parents=True,exist_ok=True)
 train,val=(m.w.split(domain,s) for s in ('train','val'));assert not set(train.group)&set(val.group)
 with (out.parent/'.staging.lock').open('a') as stage:
  fcntl.flock(stage,fcntl.LOCK_EX)
  bt,initial,pin=m.setup(domain,arm,train,out,device);bv,_,vpin=m.setup(domain,arm,val,out,device);assert pin==vpin
  binding=dict(experiment=m.NAME,arm=arm,worker_sha256=m.w.sha(Path(m.__file__)),train=train.pins,val=val.pins,Step=pin,selection_protocol='joint mean Step/Prefix VAL AUROC, AP, lower BCE')
  with (out/f'.layer_{layer:02d}.lock').open('a') as guard:
   fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
   if (out/f'layer_{layer:02d}.json').exists():return
   x=m.layer_inputs(train,layer,arm,bt,device);v=m.layer_inputs(val,layer,arm,bv,device)
   fcntl.flock(stage,fcntl.LOCK_UN)
   print('FIT_START',domain,arm,layer,flush=True)
   m.fit(domain,arm,layer,out,train,val,x,v,initial,binding,device)
   print('FIT_COMPLETE',domain,arm,layer,flush=True)

def parent(domain,maximum):
 import worker as m
 root=m.w.BASE/m.NAME/domain;root.mkdir(parents=True,exist_ok=True)
 with (root/'.scheduler.lock').open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  queue=[(a,i) for i in range(1,65) for a in m.ARMS if not (root/a/f'layer_{i:02d}.json').exists()]
  children={};failed=[];print('PARALLEL_PLAN',len(queue),'MAX',maximum,flush=True)
  while queue or children:
   for key,(p,log) in list(children.items()):
    rc=p.poll()
    if rc is not None:
     log.close();del children[key]
     if rc or not (root/key[0]/f'layer_{key[1]:02d}.json').exists():failed.append((key,rc))
   if failed:
    if children:time.sleep(1);continue
    raise RuntimeError(('failure; state preserved, no blind retry',failed))
   while queue and len(children)<maximum:
    stat=dict(x.split() for x in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
    if int(stat.get('anon',0))+int(stat.get('kernel',0))+6_000_000_000>limit:break
    arm,layer=queue.pop(0);folder=root/arm;folder.mkdir(exist_ok=True)
    log=(folder/f'parallel.layer_{layer:02d}.log').open('a')
    proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'child',domain,arm,str(layer)],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
    children[(arm,layer)]=(proc,log);print('START',arm,layer,proc.pid,flush=True)
   time.sleep(1)
  for arm in m.ARMS:
   assert all((root/arm/f'layer_{i:02d}.json').exists() for i in range(1,65))
   if not (root/arm/'TEST_COMPLETE.json').exists():
    subprocess.run([sys.executable,'-u','run_cached.py','worker',domain,arm],check=True)
  print('DOMAIN_COMPLETE',domain,flush=True)

if __name__=='__main__':
 if sys.argv[1]=='child':child(sys.argv[2],sys.argv[3],int(sys.argv[4]))
 else:parent(sys.argv[2],int(sys.argv[3]))
