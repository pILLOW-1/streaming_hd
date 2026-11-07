"""Frozen scientific identity and atomic durable records for the 192 search."""
from pathlib import Path
import hashlib
import json
import os
import fcntl
import subprocess

N = Path(os.environ.get('LHP_SHARED', Path(__file__).resolve().parents[1]))
ASSET = N.parents[3] / 'assets/streaming_hd_new'
MODEL = 'deepseek_r1_distill_llama_8b'
REV = '6a6f4aa4197940add57724a7707d069478df56b1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    assert path.suffix == '.json', f'Refuse JSON write to non-JSON artifact: {path}'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def identity():
    contract = json.loads((N / 'configs/CONTRACT.json').read_text())
    assert contract['candidate_count'] == 192 and contract['layers'] == list(range(1, 33))
    override = N / 'configs/DIRECT_HUMAN_RESOURCE_OVERRIDE_20261028.json'
    assert json.loads(override.read_text())['resource_caps_removed']
    for relative, digest in json.loads((N / 'audits/EXECUTION_CODE_FREEZE.json').read_text())['files'].items():
        assert sha(N / relative) == digest, relative
    frozen = N / 'audits/FORMAL_CODE_FREEZE.json'
    if frozen.exists():
        for relative, digest in json.loads(frozen.read_text())['files'].items():
            assert sha(N / relative) == digest, relative
    return contract


def lease_gpu(minimum_bytes=0):
    """One strict physical-idle lease, only inside our accepted PBS allocation."""
    assert os.environ.get('PBS_JOBID') and Path(os.environ['PBS_NODEFILE']).exists()
    command = ['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu,memory.total',
               '--format=csv,noheader,nounits']
    def snapshot():
        return subprocess.check_output(command, text=True)
    def active():
        text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader,nounits'], text=True)
        return {r.split(',')[0].strip() for r in text.splitlines() if r.startswith('GPU-')}
    initial = snapshot()
    save(N / 'audits' / f'PHYSICAL_INPUT_{os.environ["PBS_JOBID"]}.json',
         dict(rows=initial, active=sorted(active())))
    for line in initial.splitlines():
        index, uuid, used, util, total = [v.strip() for v in line.split(',')]
        if int(used) > 64 or int(util) or uuid in active() or int(total) * 1024**2 < minimum_bytes:
            continue
        directory = Path(os.environ['STREAMING_HD_GPU_LOCK_DIR'])
        directory.mkdir(parents=True, exist_ok=True)
        lock = (directory / f'gpu{index}.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue
        current = {r.split(',')[0].strip(): [v.strip() for v in r.split(',')]
                   for r in snapshot().splitlines()}[index]
        if current[1] != uuid or int(current[2]) > 64 or int(current[3]) or uuid in active():
            lock.close()
            continue
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        save(N / 'audits' / f'PHYSICAL_LEASE_{os.environ["PBS_JOBID"]}.json',
             dict(index=index, uuid=uuid, idle_MiB=int(current[2]), utilization=0,
                  total_MiB=int(total), job=os.environ['PBS_JOBID']))
        return lock
    raise RuntimeError('NO_ALLOCATED_PHYSICALLY_IDLE_GPU_NO_BLIND_RETRY')
