from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import fcntl, hashlib, json, os, random, signal, socket, sys, time
ROOT = Path(__file__).resolve().parent
PROTOCOL = "method4-graph-v1"

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+".partial")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def atomic_torch(path, value):
    import torch
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_name(path.name+".partial")
    with temp.open("wb") as f:
        torch.save(value,f); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)


def verify_package():
    expected=read_json(ROOT/"PACKAGE_SHA256.json")
    actual={str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py")}
    if actual != {n for n in expected if n.endswith(".py")}:
        raise ValueError("Package Python inventory changed")
    for name, checksum in expected.items():
        if sha(ROOT/name) != checksum:
            raise ValueError("Package file changed: "+name)
    return digest(expected)


def artifact(directory, relative):
    return {"path":relative,"sha256":sha(Path(directory)/relative)}


def verify_artifacts(directory, items):
    directory=Path(directory).resolve()
    for item in items.values():
        path=(directory/item["path"]).resolve()
        if not path.is_relative_to(directory) or sha(path)!=item["sha256"]:
            raise ValueError("Artifact checksum/path mismatch: "+str(path))


@contextmanager
def lock(path, wait_seconds=0):
    if isinstance(wait_seconds, bool) or not isinstance(wait_seconds, (int, float)) or not 0 <= wait_seconds < float("inf"):
        raise ValueError("Lock wait_seconds must be finite and nonnegative")
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+wait_seconds
    with path.open("a+") as f:
        while True:
            try:
                fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                remaining=deadline-time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Another process is using "+str(path)) from exc
                time.sleep(min(0.05, remaining))
        try:
            f.seek(0); f.truncate()
            json.dump(dict(pid=os.getpid(),host=socket.gethostname()),f)
            f.flush(); yield
        finally:
            fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def cpu(value):
    import torch
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu(v) for v in value)
    return value


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def rng_state():
    import numpy as np
    import torch
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=dict(kind=n[0],keys=n[1].tolist(),
                position=int(n[2]),has_gauss=int(n[3]),cached_gaussian=float(n[4])),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(r):
    import numpy as np
    import torch
    random.setstate(r["python"]); n=r["numpy"]
    np.random.set_state((n["kind"],np.asarray(n["keys"],dtype=np.uint32),n["position"],n["has_gauss"],n["cached_gaussian"]))
    torch.set_rng_state(r["torch"])
    if r["cuda"]: torch.cuda.set_rng_state_all(r["cuda"])


def runtime(device,threads):
    import torch,numpy,sklearn
    return dict(python=sys.version.split()[0],torch=str(torch.__version__),
                numpy=str(numpy.__version__),sklearn=str(sklearn.__version__),
                device_type=str(device).split(":")[0],threads=threads,cuda_version=torch.version.cuda)


class Paused(Exception): pass


class Control:
    def __init__(self,max_seconds=19500,max_updates=None):
        self.start=time.monotonic(); self.max_seconds=max_seconds
        self.max_updates=max_updates; self.updates=0; self.reason=None; self.previous={}
    def install(self):
        for number in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
            self.previous[number]=signal.signal(number,self.on_signal)
    def on_signal(self,number,_frame): self.reason=f"signal_{number}"
    def check(self):
        if time.monotonic()-self.start >= self.max_seconds: self.reason=self.reason or "time_budget"
        if self.max_updates is not None and self.updates>=self.max_updates: self.reason=self.reason or "update_budget"
        if self.reason: raise Paused(self.reason)
    def close(self):
        for number,handler in self.previous.items(): signal.signal(number,handler)
