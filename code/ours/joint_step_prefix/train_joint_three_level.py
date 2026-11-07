#!/usr/bin/env python3
"""Joint learned-token Step and scalar-transition Prefix training on three-level data.

This is the reconstructed-data counterpart of the audited J-Unified ablation.
The canonical Core loader remains the sole authority for ordered reasoning Steps,
unified Prefix targets, token states, selected layers, and exclusion of Answers.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score


PROJECT = Path(__import__("os").environ.get("STREAMING_HD_LEGACY_ROOT", ""))
RUN = PROJECT / "experiments/streaming_hd/runs/core_three_level_methods_20260902/extensions/joint_step_prefix"
CORE_RUN = PROJECT / "experiments/streaming_hd/runs/core_three_level_methods_20260902"
CORE_SOURCE = Path(os.environ.get("JOINT_CORE_SOURCE", str(CORE_RUN / "code/run_core_model.py")))
DATA = PROJECT / "assets/streaming_hd/features_v2_three_level_20260831/canonical"
SEED = 42
EPS = 1e-12
EPOCHS = 20
TRAJECTORY_BATCH_SIZE = 64
LAMBDA_PREFIX = 1.0
LAMBDA_FINAL = 30.0
LAMBDA_SYNC = 120.0
STEP_LR = 1e-3
STEP_WEIGHT_DECAY = 5e-4
PREFIX_LR = 1e-3
PREFIX_WEIGHT_DECAY = 1e-2
THRESHOLD = 0.5


def load_core() -> Any:
    spec = importlib.util.spec_from_file_location("streaming_hd_core_runtime", CORE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Core source: {CORE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = load_core()
MODELS = core.MODELS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class JointModel(nn.Module):
    """Unchanged Core learned-Step and scalar Prefix modules in one graph."""

    def __init__(self, dimension: int):
        super().__init__()
        # Match the archived audited Joint initialization: each component starts
        # from the same independent seed-42 initialization stream.
        set_seed(SEED)
        self.step = core.LearnedStep(dimension)
        set_seed(SEED)
        self.prefix = core.ScalarPrefixTransition(dimension)


def trajectory_batches(slices: list[slice], seed: int, shuffle: bool) -> list[list[slice]]:
    order = list(range(len(slices)))
    if shuffle:
        random.Random(seed).shuffle(order)
    return [
        [slices[index] for index in order[begin : begin + TRAJECTORY_BATCH_SIZE]]
        for begin in range(0, len(order), TRAJECTORY_BATCH_SIZE)
    ]


def row_indices(parts: list[slice]) -> torch.Tensor:
    return torch.cat([torch.arange(part.start, part.stop, dtype=torch.int64) for part in parts])


def compute_losses(
    model: JointModel,
    data: Any,
    parts: list[slice],
    device: torch.device,
) -> dict[str, Any]:
    indices = row_indices(parts)
    step_logits, z = model.step.forward_rows(data, indices, device)
    lengths = [part.stop - part.start for part in parts]
    prefix_logits: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    reset_errors: list[torch.Tensor] = []
    cursor = 0
    for length in lengths:
        states, row_gates = model.prefix.states(z[cursor : cursor + length])
        prefix_logits.append(model.prefix.probe(states).flatten())
        gates.append(row_gates)
        reset_errors.append(torch.max(torch.abs(states[0] - z[cursor])))
        cursor += length
    prefix_logits_t = torch.cat(prefix_logits)
    step_labels = data.step_labels[indices].to(device)
    prefix_labels = data.prefix_labels[indices].to(device)

    # Both losses are normalized by the number of valid reasoning rows.
    step_loss = nn.functional.binary_cross_entropy_with_logits(step_logits, step_labels)
    prefix_rows = nn.functional.binary_cross_entropy_with_logits(
        prefix_logits_t, prefix_labels, reduction="none"
    )
    weights = torch.ones_like(prefix_rows)
    weights[torch.tensor(np.cumsum(lengths) - 1, device=device)] = LAMBDA_FINAL
    anchor = (prefix_rows * weights).sum() / len(indices)
    # The one-way teacher remains stop-gradient. The supervised Prefix path to
    # z is intentionally live, so the token scorer receives Prefix gradients.
    teacher = torch.sigmoid(step_logits).detach()
    probabilities = torch.sigmoid(prefix_logits_t)
    delta = torch.relu(teacher - probabilities)
    sync = (delta.square() * teacher.square()).sum() / len(indices)
    prefix_loss = anchor + LAMBDA_SYNC * sync
    return {
        "joint": step_loss + LAMBDA_PREFIX * prefix_loss,
        "step": step_loss,
        "prefix": prefix_loss,
        "anchor": anchor,
        "sync": sync,
        "z": z,
        "step_logits": step_logits,
        "prefix_logits": prefix_logits_t,
        "teacher": teacher,
        "gates": torch.cat(gates),
        "indices": indices,
        "valid_step_mask_count": len(indices),
        "valid_prefix_mask_count": len(indices),
        "trajectory_reset_max_abs_error": torch.stack(reset_errors).max(),
    }


def parameter_groups(model: JointModel) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    step_parameters = list(model.step.scorer.parameters()) + list(model.step.probe.parameters())
    prefix_parameters = list(model.prefix.gate.parameters()) + list(model.prefix.probe.parameters())
    step_ids = {id(parameter) for parameter in step_parameters}
    prefix_ids = {id(parameter) for parameter in prefix_parameters}
    all_ids = {id(parameter) for parameter in model.parameters()}
    if step_ids & prefix_ids:
        raise AssertionError("AdamW parameter groups overlap")
    if step_ids | prefix_ids != all_ids:
        raise AssertionError("AdamW groups do not cover every parameter exactly once")
    return step_parameters, prefix_parameters


def make_optimizer(model: JointModel) -> torch.optim.Optimizer:
    step_parameters, prefix_parameters = parameter_groups(model)
    return torch.optim.AdamW(
        [
            {
                "params": step_parameters,
                "lr": STEP_LR,
                "weight_decay": STEP_WEIGHT_DECAY,
                "name": "token_scorer_and_step_probe",
            },
            {
                "params": prefix_parameters,
                "lr": PREFIX_LR,
                "weight_decay": PREFIX_WEIGHT_DECAY,
                "name": "scalar_gate_and_prefix_probe",
            },
        ]
    )


def grad_norm(parameters: list[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().cpu())
    return math.sqrt(total)


def gradient_flow_audit(model: JointModel, data: Any, device: torch.device) -> dict[str, Any]:
    parts = data.slices[: min(2, len(data.slices))]
    scorer = list(model.step.scorer.parameters())
    step_probe = list(model.step.probe.parameters())
    gate = list(model.prefix.gate.parameters())
    prefix_probe = list(model.prefix.probe.parameters())

    model.train()
    model.zero_grad(set_to_none=True)
    values = compute_losses(model, data, parts, device)
    prefix_to_z = torch.autograd.grad(values["prefix"], values["z"], retain_graph=True)[0]
    sync_to_step_logits = torch.autograd.grad(
        values["sync"], values["step_logits"], retain_graph=True, allow_unused=True
    )[0]
    values["step"].backward()
    step_only = {
        "token_scorer": grad_norm(scorer),
        "step_probe": grad_norm(step_probe),
        "scalar_gate": grad_norm(gate),
        "prefix_probe": grad_norm(prefix_probe),
    }

    model.zero_grad(set_to_none=True)
    values = compute_losses(model, data, parts, device)
    values["prefix"].backward()
    prefix_only = {
        "token_scorer": grad_norm(scorer),
        "step_probe": grad_norm(step_probe),
        "scalar_gate": grad_norm(gate),
        "prefix_probe": grad_norm(prefix_probe),
    }
    if not (step_only["token_scorer"] > 0 and step_only["step_probe"] > 0):
        raise AssertionError("Step loss did not reach scorer and Step probe")
    if step_only["scalar_gate"] != 0 or step_only["prefix_probe"] != 0:
        raise AssertionError("Step loss leaked to Prefix-only parameters")
    if not (
        prefix_only["token_scorer"] > 0
        and prefix_only["scalar_gate"] > 0
        and prefix_only["prefix_probe"] > 0
    ):
        raise AssertionError("Prefix loss did not reach scorer, gate, and Prefix probe")
    if prefix_only["step_probe"] != 0:
        raise AssertionError("Prefix loss leaked to Step probe")
    if float(prefix_to_z.detach().abs().max().cpu()) <= 0:
        raise AssertionError("Prefix-to-z gradient is zero")
    if sync_to_step_logits is not None and float(sync_to_step_logits.abs().max().cpu()) != 0:
        raise AssertionError("sync teacher is not stop-gradient")
    if float(values["trajectory_reset_max_abs_error"].detach().cpu()) != 0:
        raise AssertionError("Prefix state did not reset exactly at trajectory boundary")
    model.zero_grad(set_to_none=True)
    return {
        "status": "PASS",
        "diagnostic_trajectories": len(parts),
        "step_loss_only": step_only,
        "prefix_loss_only": prefix_only,
        "prefix_to_z_gradient_max_abs": float(prefix_to_z.detach().abs().max().cpu()),
        "sync_teacher_stop_gradient": True,
        "prefix_z_not_detached": True,
        "trajectory_reset_max_abs_error": 0.0,
        "answer_tokens_loaded": False,
        "parameter_groups_disjoint_and_complete": True,
    }


def train_epoch(
    model: JointModel,
    data: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.train()
    totals = {key: 0.0 for key in ("joint", "step", "prefix", "anchor", "sync")}
    rows = valid_step = valid_prefix = 0
    reset_error = 0.0
    for parts in trajectory_batches(data.slices, seed, shuffle=True):
        values = compute_losses(model, data, parts, device)
        optimizer.zero_grad(set_to_none=True)
        values["joint"].backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.all(torch.isfinite(parameter.grad)):
                raise FloatingPointError("non-finite joint gradient")
        optimizer.step()
        count = len(values["indices"])
        for key in totals:
            totals[key] += float(values[key].detach().cpu()) * count
        rows += count
        valid_step += int(values["valid_step_mask_count"])
        valid_prefix += int(values["valid_prefix_mask_count"])
        reset_error = max(reset_error, float(values["trajectory_reset_max_abs_error"].detach().cpu()))
    if rows != len(data.steps) or valid_step != rows or valid_prefix != rows:
        raise AssertionError("trajectory batching lost or duplicated a valid Step/Prefix row")
    if reset_error != 0:
        raise AssertionError("trajectory reset audit failed")
    result = {key: value / rows for key, value in totals.items()}
    result.update({
        "valid_step_mask_count": valid_step,
        "valid_prefix_mask_count": valid_prefix,
        "trajectories": len(data.slices),
        "trajectory_reset_max_abs_error": reset_error,
    })
    return result


def score_split(model: JointModel, data: Any, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    count = len(data.steps)
    step_scores = np.empty(count, dtype=np.float64)
    prefix_scores = np.empty(count, dtype=np.float64)
    gates = np.full(count, np.nan, dtype=np.float64)
    with torch.inference_mode():
        for parts in trajectory_batches(data.slices, 0, shuffle=False):
            indices = row_indices(parts)
            step_logits, z = model.step.forward_rows(data, indices, device)
            step_scores[indices.numpy()] = torch.sigmoid(step_logits).cpu().numpy()
            cursor = 0
            for part in parts:
                length = part.stop - part.start
                states, row_gates = model.prefix.states(z[cursor : cursor + length])
                if float(torch.max(torch.abs(states[0] - z[cursor])).cpu()) != 0:
                    raise AssertionError("trajectory reset failed during evaluation")
                prefix_scores[part] = torch.sigmoid(model.prefix.probe(states).flatten()).cpu().numpy()
                gates[part] = row_gates.cpu().numpy()
                cursor += length
    if not (np.isfinite(step_scores).all() and np.isfinite(prefix_scores).all()):
        raise FloatingPointError("non-finite test score")
    return {"step": step_scores, "prefix": prefix_scores, "gate": gates}


def classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(np.int64)
    predictions = (scores > THRESHOLD).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "threshold": THRESHOLD,
        "threshold_operator": ">",
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
    }


def evaluate(data: Any, scores: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "step": classification_metrics(data.step_labels.numpy(), scores["step"]),
        "prefix": classification_metrics(data.prefix_labels.numpy(), scores["prefix"]),
    }


def data_audit(data: Any) -> dict[str, Any]:
    audit = dict(data.audit)
    audit.update({
        "valid_step_mask_count": len(data.steps),
        "valid_prefix_mask_count": len(data.prefixes),
        "trajectory_reset_count": len(data.slices),
        "answer_tokens_loaded": False,
        "canonical_step_prefix_one_to_one": len(data.steps) == len(data.prefixes),
    })
    return audit


def smoke(model_key: str, output_dir: Path, device: torch.device) -> None:
    gate = core.require_gates(smoke=True)
    train = core.load_split(model_key, "train", smoke_trajectories=3)
    val = core.load_split(model_key, "val", smoke_trajectories=3)
    model = JointModel(MODELS[model_key]["dimension"]).to(device)
    gradient = gradient_flow_audit(model, train, device)
    optimizer = make_optimizer(model)
    trace = train_epoch(model, train, optimizer, device, seed=SEED * 1000 + 1)
    val_metrics = evaluate(val, score_split(model, val, device))
    report = {
        "status": "JOINT_THREE_LEVEL_SMOKE_PASS",
        "model": model_key,
        "seed": SEED,
        "device": str(device),
        "data_gate": gate,
        "train_audit": data_audit(train),
        "val_audit": data_audit(val),
        "gradient_flow": gradient,
        "one_joint_epoch": trace,
        "val_metrics_schema": val_metrics,
        "test_touched": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"smoke_{model_key}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "JOINT_THREE_LEVEL_SMOKE_PASS", "model": model_key}), flush=True)


def full(model_key: str, output_dir: Path, device: torch.device) -> None:
    started = time.time()
    gate = core.require_gates(smoke=False)
    splits = {split: core.load_split(model_key, split) for split in ("train", "val", "test")}
    set_seed(SEED)
    model = JointModel(MODELS[model_key]["dimension"]).to(device)
    gradient = gradient_flow_audit(model, splits["train"], device)
    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=20, min_lr=1e-6
    )
    best_auc = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    trace: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        row = train_epoch(model, splits["train"], optimizer, device, seed=SEED * 1000 + epoch)
        scheduler.step(row["step"])
        val_metrics = evaluate(splits["val"], score_split(model, splits["val"], device))
        row.update({
            "epoch": epoch,
            "val_step_auroc": val_metrics["step"]["auroc"],
            "val_prefix_auroc": val_metrics["prefix"]["auroc"],
            "step_lr": optimizer.param_groups[0]["lr"],
            "prefix_lr": optimizer.param_groups[1]["lr"],
        })
        trace.append(row)
        print(json.dumps({"event": "JOINT_THREE_LEVEL_EPOCH", "model": model_key, **row}), flush=True)
        if val_metrics["prefix"]["auroc"] > best_auc:
            best_auc = val_metrics["prefix"]["auroc"]
            best_epoch = epoch
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
    if best_state is None:
        raise RuntimeError("no joint validation checkpoint selected")
    model.load_state_dict(best_state)
    validation = evaluate(splits["val"], score_split(model, splits["val"], device))
    # Single target-test touch after checkpoint selection.
    test_scores = score_split(model, splits["test"], device)
    test_metrics = evaluate(splits["test"], test_scores)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "model": model_key,
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_val_prefix_auroc": best_auc,
        "dimension": MODELS[model_key]["dimension"],
        "layer": MODELS[model_key]["layer"],
        "dataset_manifest_sha256": sha256(core.manifest_path()),
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        trajectory_order=np.asarray([row["trajectory_order"] for row in splits["test"].steps]),
        step_order=np.asarray([row["step_order"] for row in splits["test"].steps]),
        step_sample_id=np.asarray([row["sample_id"] for row in splits["test"].steps], dtype=str),
        prefix_sample_id=np.asarray([row["sample_id"] for row in splits["test"].prefixes], dtype=str),
        step_label=splits["test"].step_labels.numpy().astype(np.int64),
        prefix_label=splits["test"].prefix_labels.numpy().astype(np.int64),
        step_score=test_scores["step"],
        prefix_score=test_scores["prefix"],
        gate_value=test_scores["gate"],
    )
    result = {
        "status": "JOINT_THREE_LEVEL_MODEL_COMPLETE",
        "method": "joint_step_prefix_unified_adamw",
        "model": model_key,
        "seed": SEED,
        "dataset": str(core.DATA),
        "dataset_manifest_sha256": sha256(core.manifest_path()),
        "best_epoch": best_epoch,
        "best_val_prefix_auroc": best_auc,
        "gradient_flow": gradient,
        "optimizer": {
            "family": "AdamW",
            "single_optimizer": True,
            "parameter_groups": [
                {"name": "token_scorer_and_step_probe", "lr": STEP_LR, "weight_decay": STEP_WEIGHT_DECAY},
                {"name": "scalar_gate_and_prefix_probe", "lr": PREFIX_LR, "weight_decay": PREFIX_WEIGHT_DECAY},
            ],
            "parameter_groups_disjoint_and_complete": True,
        },
        "loss": {
            "formula": "mean_step_bce + 1.0 * (mean_weighted_prefix_bce + 120 * mean_one_way_sync)",
            "lambda_prefix": LAMBDA_PREFIX,
            "prefix_final_weight": LAMBDA_FINAL,
            "prefix_sync_coefficient": LAMBDA_SYNC,
            "step_and_prefix_each_normalized_by_valid_rows": True,
        },
        "scheduler": {"name": "ReduceLROnPlateau", "signal": "train Step loss", "patience": 20},
        "checkpoint_selection": "maximum validation unified Prefix AUROC over 20 epochs",
        "validation": validation,
        "test": test_metrics,
        "input_audit": {split: data_audit(data) for split, data in splits.items()},
        "data_gate": gate,
        "trace": trace,
        "test_touches": 1,
        "answer_companion_loaded": False,
        "walltime_s": time.time() - started,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "JOINT_THREE_LEVEL_TEST_COMPLETE", "model": model_key, "test": test_metrics}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    os.environ["STREAMING_HD_REBUILD_ROOT"] = str(DATA)
    device = torch.device(args.device)
    print(json.dumps({
        "event": "JOINT_THREE_LEVEL_START",
        "mode": args.mode,
        "model": args.model,
        "device": str(device),
        "core_source": str(CORE_SOURCE),
        "core_source_sha256": sha256(CORE_SOURCE),
        "torch": torch.__version__,
    }), flush=True)
    if args.mode == "smoke":
        smoke(args.model, args.output_dir, device)
    else:
        full(args.model, args.output_dir, device)


if __name__ == "__main__":
    main()
