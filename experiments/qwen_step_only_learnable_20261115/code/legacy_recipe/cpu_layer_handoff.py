"""Release CPU shard locks after the current layer commits its checkpoint."""
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/qwen_relation/results/qwen_step_only_learnable_20261115/legacy_recipe")
DOMAIN = sys.argv[1]
WORK = ROOT / DOMAIN
SOURCES = {
    "aggr": WORK / "domain.log",
    "aggr_residual": WORK / "aggr_residual" / "residual.log",
}


def start_id(pid):
    return Path("/proc", str(pid), "stat").read_text().rsplit(")", 1)[1].split()[19]


def latest_layer(arm):
    prefix = f"BOUND {DOMAIN} {arm} "
    records = [line for line in SOURCES[arm].read_text().splitlines() if line.startswith(prefix)]
    return int(records[-1].split()[3]) if records else None


targets = {}
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        argv = (proc / "cmdline").read_bytes().decode().rstrip("\x00").split("\x00")
        if len(argv) < 8 or argv[0] != "python3.12" or argv[1:4] != ["-u", "train.py", DOMAIN]:
            continue
        arm, output = argv[4], Path(argv[7])
        if arm in SOURCES and output == WORK / arm / "lower":
            targets[int(proc.name)] = (arm, output, start_id(proc.name))
    except (FileNotFoundError, ProcessLookupError, UnicodeDecodeError):
        continue

assert targets, "No owned CPU training process matched"
pending = dict(targets)
print("DRAIN_TARGETS", {pid: arm for pid, (arm, _, _) in targets.items()}, flush=True)
deadline = time.monotonic() + 1500
while pending:
    if time.monotonic() > deadline:
        raise TimeoutError(f"CPU handoff still pending: {list(pending)}")
    for pid, (arm, output, identity) in list(pending.items()):
        try:
            if start_id(pid) != identity:
                raise ProcessLookupError(pid)
            layer = latest_layer(arm)
            if layer is None:
                continue
            result = output / f"layer_{layer:02d}.json"
            if not result.is_file() or (output / f"resume_{layer:02d}.pt").exists():
                continue
            data = json.loads(result.read_text())
            assert data["layer"] == layer and data["arm"] == arm and data["device"] == "cpu"
            assert (output / data["checkpoint"]).is_file()
            os.kill(pid, signal.SIGSTOP)
            try:
                # Stop only after the last reported layer finished. The next layer
                # may be staging a cache, but it must not have begun fitting.
                stop_deadline = time.monotonic() + 3
                while True:
                    fields = Path("/proc", str(pid), "stat").read_text().rsplit(")", 1)[1].split()
                    if fields[19] != identity or fields[0] == "Z":
                        raise ProcessLookupError(pid)
                    if fields[0] == "T" or time.monotonic() >= stop_deadline:
                        break
                    time.sleep(0.01)
                if fields[0] != "T":
                    continue
                if latest_layer(arm) != layer or any(output.glob("resume_*.pt")):
                    continue
                os.kill(pid, signal.SIGTERM)
                print(f"RELEASED {DOMAIN} {arm} {layer} {pid}", flush=True)
                del pending[pid]
            finally:
                try:
                    if start_id(pid) == identity:
                        os.kill(pid, signal.SIGCONT)
                except (FileNotFoundError, ProcessLookupError):
                    pass
        except (FileNotFoundError, ProcessLookupError):
            if pid in pending:
                print(f"EXITED {DOMAIN} {arm} {pid}", flush=True)
                del pending[pid]
    if pending:
        time.sleep(0.25)
print("CPU_HANDOFF_READY", DOMAIN, flush=True)
