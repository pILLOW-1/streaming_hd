"""Merge the disjoint layer grids, then evaluate the single global winner."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from run import digest, save_json
from train_memory import ARMS, rank


def read(path):
    return json.loads(path.read_text())


def main(root, domain, index):
    low = root / domain
    high = root / "shards" / domain / "upper"
    claim = read(root / "LAYER_WORKERS.json")[domain]
    cut = claim["lower_last"]
    ranges = ((low, 1, cut), (high, cut + 1, 64))
    if any(not (path / "SHARD_COMPLETE.json").is_file() for path, _, _ in ranges):
        return
    lock = root / f".merge_{domain}"
    try:
        lock.mkdir()
    except FileExistsError:
        return
    try:
        if not (low / "TEST_COMPLETE.json").exists():
            for path, first, last in ranges:
                assert read(path / "SHARD_COMPLETE.json") == dict(
                    first=first, last=last, arms=list(ARMS), domain=domain)
            for arm in ARMS:
                for layer in range(cut + 1, 65):
                    name = f"layer_{layer:02d}.json"
                    src, dst = high / arm / name, low / arm / name
                    report = read(src)
                    assert (report["domain"], report["arm"], report["layer"]) == (domain, arm, layer)
                    if dst.exists():
                        assert dst.read_bytes() == src.read_bytes()
                    else:
                        tmp = dst.with_suffix(".pending")
                        shutil.copyfile(src, tmp)
                        tmp.replace(dst)
            arm = "A3 normalized"
            plans = [(read(path / arm / "selection.json"), path / arm) for path, _, _ in ranges]
            for pin, path in plans:
                assert digest((path / pin["checkpoint"]).read_bytes()) == pin["checkpoint_sha256"]
                assert digest((path / pin["prediction"]).read_bytes()) == pin["prediction_sha256"]
            chosen, source = max(plans, key=lambda plan: rank(plan[0]["metrics"]))
            for key in ("checkpoint", "prediction"):
                if source != low / arm:
                    shutil.copyfile(source / chosen[key], low / arm / chosen[key])
            save_json(low / arm / "selection.json", chosen)
            save_json(low / "TRAIN_COMPLETE.json", dict(layers=64, arms=list(ARMS), domain=domain))
            subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("test_memory.py")),
                            domain, str(low), str(index / f"{domain}.test.json")], check=True)
            assert (low / "TEST_COMPLETE.json").is_file()
    finally:
        lock.rmdir()
    if all((root / d / "TEST_COMPLETE.json").is_file() for d in ("Logic", "Factuality")):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("compare_layers.py"))])
        if result.returncode == 0:
            from datetime import datetime, timezone
            (root / "COMPLETED_UTC").write_text(datetime.now(timezone.utc).isoformat() + "\n")
        elif result.returncode != 75:
            result.check_returncode()


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
