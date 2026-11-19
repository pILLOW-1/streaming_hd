"""Two disjoint pod slots, with measured per-GPU layer concurrency.

Ordinary training controller only; does not create/stop cloud resources or run
independent scientific inference. Run slot0 and slot1 on the two requested GPUs.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/workspace/qwen_relation/results/qwen_a2_hami_topk_memory_20261119')
GIB = 1024**3


def memory_room():
    v2 = Path('/sys/fs/cgroup/memory.stat')
    if v2.exists():
        stat = dict(line.split() for line in v2.read_text().splitlines())
        lim = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        used = int(stat.get('anon', 0)) + int(stat.get('kernel', 0))
        return lim == 'max' or used + 5*GIB < int(lim)
    v1 = Path('/sys/fs/cgroup/memory/memory.stat')
    if v1.exists():
        stat = dict(line.split() for line in v1.read_text().splitlines())
        lim = int(v1.with_name('memory.limit_in_bytes').read_text())
        used = int(stat.get('total_rss', stat.get('rss', 0)))
        return used + 5*GIB < lim
    # Explicit pod system-RAM cap if the provider hides cgroup files.
    rss = 0
    for path in Path('/proc').glob('[0-9]*/status'):
        try:
            rss += next((int(x.split()[1])*1024 for x in path.read_text().splitlines()
                         if x.startswith('VmRSS:')), 0)
        except (OSError, ValueError):
            pass
    return rss + 5*GIB < int(os.environ.get('HAMI_HOST_RAM_GB', '32'))*GIB


def cpu_capacity():
    v2 = Path('/sys/fs/cgroup/cpu.max')
    if v2.exists():
        a, b = v2.read_text().split()
        return int(a)/int(b) if a != 'max' else os.cpu_count()
    root = Path('/sys/fs/cgroup/cpu')
    if (root/'cpu.cfs_quota_us').exists():
        quota = int((root/'cpu.cfs_quota_us').read_text())
        if quota > 0:
            return quota/int((root/'cpu.cfs_period_us').read_text())
    return min(os.cpu_count(), int(os.environ.get('HAMI_VCPU', '4')))


def room(children):
    policy = ROOT/'RESOURCE_POLICY.json'
    conf = json.loads(policy.read_text()) if policy.exists() else {}
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,memory.total',
                                    '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    assert len(rows) == 1, 'one GPU per pod required'
    free, total = (int(v.strip()) * 1024**2 for v in rows[0].split(','))
    if children:
        resources = []
        provisional = False
        for domain, arm, layer in children:
            p = ROOT / domain / arm / f'resource_{layer:02d}.json'
            if not p.exists():
                proc = children[(domain, arm, layer)][0]
                if time.time() - proc.started < float(conf.get('warmup_seconds', 30)):
                    return False
                provisional = True
            else:
                resources.append(json.loads(p.read_text()))
        # Resource policy can be tuned without touching any live training job.
        # The provisional reservation is not reported as a measured CUDA peak.
        provisional_bytes = float(conf.get('provisional_GiB', 1.5))*GIB
        budget = max([r['peak_cuda_bytes'] for r in resources] + ([provisional_bytes] if provisional else []))
        # Current free memory already accounts for every live CUDA context and
        # resident tensor. Reserve a measured new-worker peak plus context and
        #2GiB headroom instead of multiplying a cold-start reservation forever.
        required = budget * 1.35 + .7*GIB
        if free < required + 2*GIB:
            return False
    if not memory_room():
        return False
    return free > 3*GIB


class AttachedProcess:
    def __init__(self, pid):
        self.pid = pid
        self.started = time.time() - 120  # Existing GPU fit has already warmed up.

    def poll(self):
        path = Path('/proc')/str(self.pid)
        if not path.exists():
            return 0  # Completion still requires the accepted layer report.
        try:
            state = next(x for x in (path/'status').read_text().splitlines() if x.startswith('State:'))
            return 0 if 'Z (zombie)' in state else None
        except OSError:
            return 0


def cleanup_finished_staging(domain, layer, running):
    if any(d == domain and l == layer for d,a,l in running):
        return
    if not all((ROOT/domain/a/f'layer_{layer:02d}.json').exists() for a in ('H1','H2')):
        return
    # These exact files are job-private copies; formal network-volume sources,
    # reports, checkpoints and shared derived readouts are never removed here.
    staging = Path('/tmp/a2_hami_tokens').resolve()
    for part in ('train', 'val'):
        path = staging/domain/part/f'layer_{layer:02d}'/'reasoning_answer.bf16.npy'
        assert path.resolve().is_relative_to(staging)
        path.unlink(missing_ok=True)


def main(slot):
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / f'.slot_{slot}.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        policy = ROOT/'RESOURCE_POLICY.json'
        conf = json.loads(policy.read_text()) if policy.exists() else {}
        domains = tuple(conf.get('domains', ('Logic', 'Factuality')))
        assert domains and set(domains) <= {'Logic', 'Factuality'}
        # Logic and Factuality are each split across the two cards. Each layer
        # owns independent optimizer, BN, dropout/shuffle RNG and output paths.
        queue = [(domain, arm, layer) for layer in range(1+slot, 65, 2)
                 for domain in domains for arm in ('H1', 'H2')
                 if not (ROOT/domain/arm/f'layer_{layer:02d}.json').exists()]
        running = {}; failures = []
        # Reattach healthy fits when only the scheduling controller is replaced.
        # Per-layer locks remain the final protection against duplicate fits.
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                argv = path.read_bytes().decode().split('\0')
                if not any(n in argv for n in ('worker.py','worker_fast.py')) or 'fit' not in argv:
                    continue
                at = argv.index('fit'); d,a,l = argv[at+1:at+4]; key=(d,a,int(l))
                if key not in queue or Path(os.readlink(path.parent/'cwd')) != Path.cwd():
                    continue
                log=(ROOT/d/a/f'layer_{int(l):02d}.log').open('a')
                running[key]=(AttachedProcess(int(path.parent.name)),log);queue.remove(key)
            except (OSError,ValueError,UnicodeError):
                continue
        queue.sort(key=lambda k: ((k[2]-1)//4, ('Logic','Factuality').index(k[0]), k[1], k[2]))
        cores = cpu_capacity()
        policy = ROOT/'RESOURCE_POLICY.json'
        conf = json.loads(policy.read_text()) if policy.exists() else {}
        maximum = max(1, min(int(conf.get('max_workers', 12)), int(cores)))
        print('PLAN', dict(slot=slot, domains=domains, jobs=len(queue), maximum=maximum), flush=True)
        while queue or running:
            for key, (proc, log) in list(running.items()):
                code = proc.poll()
                if code is not None:
                    log.close(); del running[key]
                    d, a, layer = key
                    if code or not (ROOT/d/a/f'layer_{layer:02d}.json').exists():
                        failures.append(dict(unit=key, exit=code))
                    else:
                        cleanup_finished_staging(d, layer, running)
            if failures:
                if running:
                    time.sleep(2); continue
                raise RuntimeError(('training failed; full state retained, no blind retry', failures))
            if queue and len(running) < maximum and room(running):
                key = queue.pop(0); d, a, layer = key
                out = ROOT/d/a; out.mkdir(parents=True, exist_ok=True)
                log = (out/f'layer_{layer:02d}.log').open('a')
                entry = conf.get('worker_entry', 'worker.py')
                assert entry in ('worker.py','worker_fast.py')
                proc = subprocess.Popen([sys.executable, '-u', entry, 'fit', d, a, str(layer)],
                    stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
                proc.started = time.time()
                running[key] = proc, log
                print('START', key, proc.pid, flush=True)
            time.sleep(2)
        (ROOT/f'SLOT_{slot}_FIT_COMPLETE.json').write_text(json.dumps(dict(slot=slot, time=time.time())))
        # Each domain/arm finish is protected by a global shared-volume lock.
        for d in domains:
            for a in ('H1', 'H2'):
                if all((ROOT/d/a/f'layer_{i:02d}.json').exists() for i in range(1,65)):
                    if (ROOT/d/a/'TEST_COMPLETE.json').exists():
                        continue
                    try:
                        with (ROOT/d/a/'.test_dispatch.lock').open('a') as f:
                            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            subprocess.run([sys.executable, '-u', 'worker.py', 'finish', d, a], check=True)
                    except BlockingIOError:
                        pass
        print('SLOT_FINISHED', slot, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('slot', type=int, choices=(0, 1))
    main(p.parse_args().slot)
