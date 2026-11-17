"""Small entry point around original training/evaluation functions.

The caller supplies an already verified native-cache dataset adapter.
This module does not generate, relabel, move, or silently substitute features.
"""
from pathlib import Path
from types import SimpleNamespace
import torch
from .common import Control, atomic_json, read_json, sha, verify_package
from .config import SEEDS, config_for
from .train import train_run
from .evaluate import prepare_protocol, test_run
from .recovery import summarize as recovery_summary

def execute(dataset, output_root, seed, phase, device="cuda", threads=4, max_seconds=19800):
    if seed not in SEEDS or phase not in ("train", "evaluate"):
        raise ValueError("Use a fixed seed and phase train/evaluate")
    if threads < 1 or max_seconds <= 0:
        raise ValueError("Invalid runtime limits")
    if not isinstance(dataset.identity, dict) or not dataset.identity:
        raise ValueError("Dataset must carry its verified source/split identity")
    if dataset.identity.get("kind") != "synthetic_test_fixture" and dataset.width != 4096:
        raise ValueError("This DeepSeek recipe requires width 4096")
    for name in ("split", "load"):
        if not callable(getattr(dataset, name, None)):
            raise ValueError("Dataset adapter must implement " + name)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No visible CUDA device; choose cpu explicitly for CPU checks")
    root = Path(output_root).expanduser().resolve()
    experiment = Path(__file__).resolve().parents[2]
    if root.is_relative_to(experiment):
        raise ValueError("Keep new run outputs outside the published experiment folder")
    torch.set_num_threads(threads)
    package_sha = verify_package()
    args = SimpleNamespace(dataset="expanded", variant="gru_memory", seed=seed, smoke=False,
        device=device, threads=threads, run_dir=root/"runs/gru_memory"/f"seed_{seed}")
    control = Control(max_seconds)
    control.install()
    try:
        if phase == "train":
            return train_run(args, dataset, config_for("gru_memory"), package_sha, control)
        # The original gate requires every fixed seed to finish before any test run.
        protocol = prepare_protocol(root, dataset.identity, package_sha, "expanded")
        result = test_run(args, dataset, args.run_dir, protocol, control)
        prediction_path = args.run_dir / result["artifacts"]["test_predictions"]["path"]
        records = read_json(prediction_path)
        normalized = [dict(trajectory_id=r["trajectory_id"], original_step_ids=r["step_ids"],
            prefix_labels=r["prefix_labels"], prefix_probabilities=r["prefix_probabilities"]) for r in records]
        recovery = recovery_summary(normalized, include_events=False)
        recovery.update(prediction_sha256=sha(prediction_path), seed=seed,
            note="Adapter must preserve original integer step IDs and exact test coverage.")
        atomic_json(args.run_dir / "recovery.json", recovery)
        return result
    finally:
        control.close()
