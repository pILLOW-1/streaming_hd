"""Freeze all five validation runs before evaluating the test split."""
from __future__ import annotations
from pathlib import Path
import time
import torch
from .common import (PROTOCOL, Paused, atomic_json, artifact, digest, lock,
                     read_json, runtime, sha, verify_artifacts)
from .config import SEEDS, VARIANTS, config_for
from .metrics import evaluate_model
from .model import build_model

def validation_inventory(root, source, package_sha, dataset):
    inventory = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            relative = f"runs/{variant}/seed_{seed}/result.json"
            path = root / relative
            if not path.exists():
                raise ValueError("Finish every fixed variant/seed validation run before test: " + str(path))
            result = read_json(path)
            expected = dict(source=source, package_sha256=package_sha,
                            config=config_for(variant), dataset=dataset, seed=seed, smoke=False)
            if result.get("status") != "VALIDATION_COMPLETE" or result.get("identity") != expected:
                raise ValueError("Validation identity/status mismatch: " + str(path))
            verify_artifacts(path.parent, result["artifacts"])
            inventory[relative] = sha(path)
    return inventory


def prepare_protocol(root, source, package_sha, dataset):
    root = Path(root)
    with lock(root / ".test_protocol.lock", wait_seconds=300):
        content = dict(status="FIXED_PROTOCOL_READY", format_version=PROTOCOL,
                       dataset=dataset, source=source, package_sha256=package_sha,
                       variants={v: config_for(v) for v in VARIANTS}, seeds=list(SEEDS),
                       validation_results=validation_inventory(root, source, package_sha, dataset),
                       checkpoint_selection="mean_validation_step_and_prefix_AUROC",
                       threshold=0.5, threshold_rule="strictly_greater",
                       interpretation="existing_test_split_development_comparison",
                       fresh_confirmatory_holdout=False)
        content["protocol_id"] = digest(content)
        path = root / "test_protocol.json"
        if path.exists():
            if read_json(path) != content:
                raise ValueError("Frozen test protocol differs; do not overwrite previous experiments")
        else:
            atomic_json(path, content)
    return content


def verify_protocol(root, package_sha=None):
    root = Path(root)
    content = read_json(root / "test_protocol.json")
    unsigned = {k: v for k, v in content.items() if k != "protocol_id"}
    if content.get("status") != "FIXED_PROTOCOL_READY" or digest(unsigned) != content.get("protocol_id"):
        raise ValueError("Invalid test protocol digest/status")
    if package_sha is not None and content["package_sha256"] != package_sha:
        raise ValueError("Test protocol uses another source package")
    actual = validation_inventory(root, content["source"], content["package_sha256"], content["dataset"])
    if actual != content["validation_results"]:
        raise ValueError("Validation result changed after test protocol was frozen")
    return content


def test_run(args, dataset, directory, protocol, control):
    directory = Path(directory)
    training = read_json(directory / "result.json")
    identity = training["identity"]
    training_sha = sha(directory / "result.json")
    verify_artifacts(directory, training["artifacts"])
    with lock(directory / ".test.lock"):
        path = directory / "test_result.json"
        if path.exists():
            result = read_json(path)
            if (result.get("status") != "TEST_COMPLETE" or result.get("identity") != identity or
                    result.get("test_protocol_id") != protocol["protocol_id"] or
                    result.get("training_result_sha256") != training_sha):
                raise ValueError("Existing test result belongs to a different experiment")
            verify_artifacts(directory, result["artifacts"])
            print("Verified completed test:", directory, flush=True)
            return result
        status_path = directory / "test_status.json"
        atomic_json(status_path, dict(status="running", variant=training["variant"], seed=training["seed"]))
        start = time.monotonic()
        try:
            checkpoint = torch.load(directory / training["artifacts"]["checkpoint"]["path"],
                                    map_location="cpu", weights_only=True)
            if checkpoint.get("identity") != identity or checkpoint.get("hidden_dim") != dataset.width:
                raise ValueError("Checkpoint identity/feature dimension mismatch")
            model = build_model(dataset.width, identity["config"]).to(args.device)
            model.load_state_dict(checkpoint["state_dict"])
            scores, records = evaluate_model(model, dataset, dataset.split("test"), args.device, control)
            atomic_json(directory / "test_predictions.json", records)
            result = dict(status="TEST_COMPLETE", format_version=PROTOCOL,
                          dataset=args.dataset, variant=training["variant"], seed=training["seed"],
                          evaluation_split="test", identity=identity,
                          runtime=runtime(args.device, args.threads), scores=scores,
                          selection=training["selection"], training_result_sha256=training_sha,
                          trainable_parameters=training["trainable_parameters"],
                          elapsed_evaluation_seconds=time.monotonic() - start,
                          benchmark_eligible=training["benchmark_eligible"],
                          test_protocol_id=protocol["protocol_id"],
                          fresh_confirmatory_holdout=False,
                          prefix_transfer_is_unchanged_prefix_score=True,
                          graph_edges_are_causal_attributions=False,
                          artifacts={"checkpoint": training["artifacts"]["checkpoint"],
                                     "test_predictions": artifact(directory, "test_predictions.json")})
            atomic_json(path, result)
            atomic_json(status_path, dict(status="completed", variant=training["variant"], seed=training["seed"]))
            print("TEST_COMPLETE", args.dataset, training["variant"], training["seed"], flush=True)
            return result
        except Paused as exc:
            atomic_json(status_path, dict(status="paused", reason=str(exc),
                                         restart_evaluation_from_beginning=True))
            raise
        except Exception as exc:
            atomic_json(status_path, dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
            raise
