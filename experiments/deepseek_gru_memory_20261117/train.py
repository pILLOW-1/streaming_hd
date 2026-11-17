"""Run the fixed GRU+memory recipe using the project's verified feature loader."""
import argparse
import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from gru_memory.api import execute
from gru_memory.common import Paused
from gru_memory.config import SEEDS

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("train", "evaluate"))
    p.add_argument("--dataset-factory", required=True, help="Importable module:function taking the private data-config dict")
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-seconds", type=float, default=19800)
    a = p.parse_args()
    module, separator, name = a.dataset_factory.partition(":")
    if not separator or not module or not name:
        p.error("dataset-factory must be module:function")
    factory = getattr(importlib.import_module(module), name)
    dataset = factory(json.loads(a.data_config.read_text()))
    try:
        result = execute(dataset, a.output_root, a.seed, a.phase, a.device, a.threads, a.max_seconds)
    except Paused as exc:
        print("PAUSED:", exc)
        return 75
    print(json.dumps({k: result[k] for k in ("status", "seed", "evaluation_split")}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
