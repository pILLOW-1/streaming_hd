#!/usr/bin/env python3
"""Joint learned-token Step and dynamic scalar Prefix training.

The LRM and canonical token-state assets stay frozen.  This trainer changes
only the optimization schedule: the current-Step token scorer receives the
sum of the existing Step BCE gradient and the existing supervised Prefix
gradient.  The Step probability used by the one-way synchronization term is
explicitly detached.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score


PROJECT = Path(__import__("os").environ.get("STREAMING_HD_LEGACY_ROOT", ""))
FEATURE_ROOT = PROJECT / "assets/streaming_hd/features"
EPSILON = 1e-12
TRAJECTORY_BATCH_SIZE = 64
EPOCHS = 20
STEP_LR = 1e-3
STEP_WEIGHT_DECAY = 5e-4
PREFIX_LR = 1e-3
PREFIX_WEIGHT_DECAY = 1e-2
LAMBDA_FINAL = 30.0
LAMBDA_SYNC = 120.0
THRESHOLD = 0.5

MODEL_SPECS = {
    "qwen": {
        "dimension": 3584,
        "paper_layer": 20,
        "original": {split: f"Qwen_{split}.pt" for split in ("train", "val", "test")},
        "expected": {"train": (1422, 26044), "val": (854, 15446), "test": (565, 10110)},
    },
    "llama": {
        "dimension": 4096,
        "paper_layer": 16,
        "original": {split: f"Llama_{split}.pt" for split in ("train", "val", "test")},
        "expected": {"train": (1319, 30491), "val": (659, 15573), "test": (536, 12546)},
        "released_rows": {"train": 30499, "val": 15574, "test": 12546},
    },
    "deepseek": {
        "dimension": 4096,
        "paper_layer": 18,
        "original": {split: f"deepseek_{split}.pt" for split in ("train", "val", "test")},
        "expected": {"train": (1404, 43931), "val": (842, 26918), "test": (562, 19069)},
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class LearnedTokenStep(nn.Module):
    """Exact architecture from the completed learned-token Step run."""

    def __init__(self, dimension: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dimension, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        self.probe = nn.Linear(dimension, 1)

    def pool_indices(
        self,
        cache: dict[str, Any],
        indices: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        offsets = cache["offsets"]
        pieces: list[torch.Tensor] = []
        lengths: list[int] = []
        for index in indices.tolist():
            left, right = int(offsets[index]), int(offsets[index + 1])
            pieces.append(cache["tokens"][left:right])
            lengths.append(right - left)
        flat = torch.cat(pieces).to(device=device, dtype=torch.float32, non_blocking=True)
        importance = self.scorer(flat).flatten()
        pooled: list[torch.Tensor] = []
        cursor = 0
        for length in lengths:
            states = flat[cursor : cursor + length]
            weights = torch.softmax(importance[cursor : cursor + length], dim=0)
            vector = torch.sum(states * weights.unsqueeze(1), dim=0)
            pooled.append(vector / (torch.linalg.vector_norm(vector) + EPSILON))
            cursor += length
        z = torch.stack(pooled)
        return z, self.probe(z).flatten(), lengths


class DynamicPrefixGate(nn.Module):
    """Exact scalar transition and linear Prefix probe from Two-stage."""

    def __init__(self, dimension: int):
        super().__init__()
        self.gate = nn.Linear(2 * dimension, 1)
        self.probe = nn.Linear(dimension, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward_sequence(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefixes = [z[0]]
        gates = [torch.full((), float("nan"), device=z.device)]
        norms = [torch.linalg.vector_norm(z[0])]
        previous = z[0]
        for current in z[1:]:
            gate = torch.sigmoid(self.gate(torch.cat([previous, current])).squeeze())
            mixed = (1.0 - gate) * previous + gate * current
            norm = torch.linalg.vector_norm(mixed)
            previous = mixed / (norm + EPSILON)
            prefixes.append(previous)
            gates.append(gate)
            norms.append(norm)
        states = torch.stack(prefixes)
        return self.probe(states).flatten(), torch.stack(gates), torch.stack(norms)


class JointModel(nn.Module):
    def __init__(self, dimension: int, seed: int):
        super().__init__()
        # Each component receives its standalone seed-42 initialization stream.
        set_seed(seed)
        self.step = LearnedTokenStep(dimension)
        set_seed(seed)
        self.prefix = DynamicPrefixGate(dimension)


def input_paths(model_key: str, split: str) -> tuple[Path, Path]:
    learned = FEATURE_ROOT / "learned_token_weighting" / model_key / f"{split}.pt"
    original = FEATURE_ROOT / "original_streaming" / model_key / MODEL_SPECS[model_key]["original"][split]
    return learned, original


def sequence_slices(trajectory_indices: torch.Tensor) -> list[slice]:
    boundaries = torch.nonzero(trajectory_indices[1:] != trajectory_indices[:-1], as_tuple=False).flatten() + 1
    starts = [0] + boundaries.tolist()
    ends = boundaries.tolist() + [len(trajectory_indices)]
    return [slice(start, end) for start, end in zip(starts, ends)]


def load_split(model_key: str, split: str, limit_trajectories: int | None = None) -> dict[str, Any]:
    learned_path, original_path = input_paths(model_key, split)
    learned = torch.load(learned_path, map_location="cpu", mmap=True, weights_only=False)
    original = torch.load(original_path, map_location="cpu", mmap=True, weights_only=False)
    required = {"tokens", "offsets", "labels", "trajectory_indices", "positions", "metadata"}
    missing = required.difference(learned)
    if missing:
        raise AssertionError(f"{model_key}/{split}: missing {sorted(missing)}")
    trajectories_expected, rows_expected = MODEL_SPECS[model_key]["expected"][split]
    if int(learned["labels"].numel()) != rows_expected:
        raise AssertionError(f"{model_key}/{split}: row mismatch")
    if int(torch.unique(learned["trajectory_indices"]).numel()) != trajectories_expected:
        raise AssertionError(f"{model_key}/{split}: trajectory mismatch")
    if int(learned["tokens"].shape[1]) != MODEL_SPECS[model_key]["dimension"]:
        raise AssertionError(f"{model_key}/{split}: dimension mismatch")
    if int(learned["offsets"][-1]) != int(learned["tokens"].shape[0]):
        raise AssertionError(f"{model_key}/{split}: offset mismatch")
    flat_indices = learned.get("flat_step_indices", torch.arange(rows_expected)).to(torch.int64)
    if not torch.equal(original["labels"][flat_indices].to(torch.uint8), learned["labels"].to(torch.uint8)):
        raise AssertionError(f"{model_key}/{split}: Step labels differ from official rows")
    row_count = rows_expected
    if limit_trajectories is not None:
        slices = sequence_slices(learned["trajectory_indices"].to(torch.int64))
        row_count = slices[min(limit_trajectories, len(slices)) - 1].stop
    flat_indices = flat_indices[:row_count]
    trajectory_ids = learned.get("trajectory_ids", [])
    if len(trajectory_ids) == rows_expected:
        ids = [str(value) for value in trajectory_ids[:row_count]]
    else:
        ids = [str(int(value)) for value in learned["trajectory_indices"][:row_count]]
    sidecar_path = learned_path.with_suffix(learned_path.suffix + ".manifest.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else {}
    metadata = learned.get("metadata", {})
    data = {
        "cache": learned,
        "rows": row_count,
        "y_step": learned["labels"][:row_count].to(torch.float32),
        "y_prefix": original["cumulative_labels"][flat_indices].to(torch.float32),
        "trajectory_indices": learned["trajectory_indices"][:row_count].to(torch.int64),
        "positions": learned["positions"][:row_count].to(torch.int64),
        "flat_step_indices": flat_indices,
        "trajectory_ids": ids,
    }
    data["slices"] = sequence_slices(data["trajectory_indices"])
    data["audit"] = {
        "model": model_key,
        "split": split,
        "learned_path": str(learned_path),
        "original_path": str(original_path),
        "learned_sha256": sha256(learned_path),
        "original_sha256": sha256(original_path),
        "rows": row_count,
        "full_rows": rows_expected,
        "trajectories": len(data["slices"]),
        "step_positive": int(data["y_step"].sum()),
        "prefix_positive": int(data["y_prefix"].sum()),
        "tokens": int(learned["offsets"][row_count]),
        "dimension": int(learned["tokens"].shape[1]),
        "dtype": str(learned["tokens"].dtype),
        "paper_layer": MODEL_SPECS[model_key]["paper_layer"],
        "model_revision": sidecar.get("model_revision", metadata.get("model_revision")),
        "source_split_sha256": sidecar.get("source_split_sha256", metadata.get("source_split_sha256")),
        "answer_companion_loaded": False,
        "released_unlabelled_rows_excluded": int(MODEL_SPECS[model_key].get("released_rows", {}).get(split, rows_expected) - rows_expected),
    }
    return data


def batches(slices: list[slice], seed: int, shuffle: bool) -> list[list[slice]]:
    order = list(range(len(slices)))
    if shuffle:
        random.Random(seed).shuffle(order)
    return [[slices[j] for j in order[i : i + TRAJECTORY_BATCH_SIZE]] for i in range(0, len(order), TRAJECTORY_BATCH_SIZE)]


def row_indices(parts: list[slice]) -> torch.Tensor:
    return torch.cat([torch.arange(part.start, part.stop, dtype=torch.int64) for part in parts])


def compute_losses(
    model: JointModel,
    data: dict[str, Any],
    parts: list[slice],
    device: torch.device,
) -> dict[str, Any]:
    indices = row_indices(parts)
    z, step_logits, _ = model.step.pool_indices(data["cache"], indices, device)
    lengths = [part.stop - part.start for part in parts]
    prefix_logits: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    norms: list[torch.Tensor] = []
    cursor = 0
    for length in lengths:
        logits, row_gates, row_norms = model.prefix.forward_sequence(z[cursor : cursor + length])
        prefix_logits.append(logits)
        gates.append(row_gates)
        norms.append(row_norms)
        cursor += length
    prefix_logits_tensor = torch.cat(prefix_logits)
    step_labels = data["y_step"][indices].to(device)
    prefix_labels = data["y_prefix"][indices].to(device)
    step_loss = nn.functional.binary_cross_entropy_with_logits(step_logits, step_labels)
    prefix_rows = nn.functional.binary_cross_entropy_with_logits(prefix_logits_tensor, prefix_labels, reduction="none")
    weights = torch.ones_like(prefix_rows)
    weights[torch.tensor(np.cumsum(lengths) - 1, device=device)] = LAMBDA_FINAL
    anchor = (prefix_rows * weights).mean()
    teacher = torch.sigmoid(step_logits).detach()
    probabilities = torch.sigmoid(prefix_logits_tensor)
    delta = torch.relu(teacher - probabilities)
    sync = (delta.pow(2) * teacher.pow(2)).sum() / len(indices)
    prefix_loss = anchor + LAMBDA_SYNC * sync
    return {
        "joint": step_loss + prefix_loss,
        "step": step_loss,
        "prefix": prefix_loss,
        "anchor": anchor,
        "sync": sync,
        "z": z,
        "step_logits": step_logits,
        "prefix_logits": prefix_logits_tensor,
        "teacher": teacher,
        "gates": torch.cat(gates),
        "norms": torch.cat(norms),
        "indices": indices,
    }


def grad_norm(parameters: list[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(torch.sum(parameter.grad.detach().float().pow(2)).cpu())
    return math.sqrt(total)


def gradient_flow_audit(model: JointModel, data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    parts = data["slices"][: min(2, len(data["slices"]))]
    scorer_params = list(model.step.scorer.parameters())
    step_probe_params = list(model.step.probe.parameters())
    gate_params = list(model.prefix.gate.parameters())
    prefix_probe_params = list(model.prefix.probe.parameters())

    model.train()
    model.zero_grad(set_to_none=True)
    values = compute_losses(model, data, parts, device)
    sync_to_step_logits = torch.autograd.grad(
        values["sync"], values["step_logits"], retain_graph=True, allow_unused=True
    )[0]
    values["step"].backward()
    step_only = {
        "scorer": grad_norm(scorer_params),
        "step_probe": grad_norm(step_probe_params),
        "gate": grad_norm(gate_params),
        "prefix_probe": grad_norm(prefix_probe_params),
    }

    model.zero_grad(set_to_none=True)
    values = compute_losses(model, data, parts, device)
    values["prefix"].backward()
    prefix_only = {
        "scorer": grad_norm(scorer_params),
        "step_probe": grad_norm(step_probe_params),
        "gate": grad_norm(gate_params),
        "prefix_probe": grad_norm(prefix_probe_params),
    }
    if not (step_only["scorer"] > 0 and step_only["step_probe"] > 0):
        raise AssertionError("Step loss did not update scorer and Step probe")
    if step_only["gate"] != 0 or step_only["prefix_probe"] != 0:
        raise AssertionError("Step loss leaked into Prefix-only parameters")
    if not (prefix_only["scorer"] > 0 and prefix_only["gate"] > 0 and prefix_only["prefix_probe"] > 0):
        raise AssertionError("Prefix loss did not update shared scorer and Prefix parameters")
    if prefix_only["step_probe"] != 0:
        raise AssertionError("Prefix loss leaked into Step probe")
    if sync_to_step_logits is not None and float(sync_to_step_logits.abs().max()) != 0.0:
        raise AssertionError("synchronization teacher is not stop-gradient")
    model.zero_grad(set_to_none=True)
    return {
        "status": "PASS",
        "diagnostic_split": "train",
        "diagnostic_trajectories": len(parts),
        "step_loss_only_shared_scorer_grad_norm": step_only["scorer"],
        "prefix_loss_only_shared_scorer_grad_norm": prefix_only["scorer"],
        "step_loss_only": step_only,
        "prefix_loss_only": prefix_only,
        "sync_teacher_stop_gradient": True,
        "prefix_z_not_detached": True,
    }


def make_optimizers(model: JointModel, regime: str) -> list[torch.optim.Optimizer]:
    step_parameters = list(model.step.scorer.parameters()) + list(model.step.probe.parameters())
    prefix_parameters = list(model.prefix.gate.parameters()) + list(model.prefix.probe.parameters())
    step_ids = {id(parameter) for parameter in step_parameters}
    prefix_ids = {id(parameter) for parameter in prefix_parameters}
    all_ids = {id(parameter) for parameter in model.parameters()}
    if step_ids.intersection(prefix_ids):
        raise AssertionError("Step and Prefix optimizer parameter sets overlap")
    if step_ids.union(prefix_ids) != all_ids:
        raise AssertionError("optimizer parameter sets do not cover the model exactly once")
    if regime == "j_dual":
        return [
            torch.optim.Adam(step_parameters, lr=STEP_LR, weight_decay=STEP_WEIGHT_DECAY),
            torch.optim.AdamW(prefix_parameters, lr=PREFIX_LR, weight_decay=PREFIX_WEIGHT_DECAY),
        ]
    if regime == "j_unified":
        return [
            torch.optim.AdamW(
                [
                    {"params": step_parameters, "lr": STEP_LR, "weight_decay": STEP_WEIGHT_DECAY, "name": "step"},
                    {"params": prefix_parameters, "lr": PREFIX_LR, "weight_decay": PREFIX_WEIGHT_DECAY, "name": "prefix"},
                ]
            )
        ]
    raise ValueError(f"unknown optimizer regime: {regime}")


def train_epoch(
    model: JointModel,
    data: dict[str, Any],
    optimizers: list[torch.optim.Optimizer],
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.train()
    totals = {key: 0.0 for key in ("joint", "step", "prefix", "anchor", "sync")}
    rows = 0
    min_norm = math.inf
    gate_low = gate_high = gate_count = 0
    for parts in batches(data["slices"], seed, shuffle=True):
        values = compute_losses(model, data, parts, device)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        values["joint"].backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.all(torch.isfinite(parameter.grad)):
                raise FloatingPointError("non-finite gradient")
        for optimizer in optimizers:
            optimizer.step()
        count = len(values["indices"])
        for key in totals:
            totals[key] += float(values[key].detach().cpu()) * count
        rows += count
        min_norm = min(min_norm, float(values["norms"].detach().min().cpu()))
        finite = values["gates"][torch.isfinite(values["gates"])].detach()
        gate_low += int((finite < 0.01).sum().cpu())
        gate_high += int((finite > 0.99).sum().cpu())
        gate_count += int(finite.numel())
    result = {key: value / rows for key, value in totals.items()}
    result.update({
        "min_pre_normalization_norm": min_norm,
        "gate_below_001_fraction": gate_low / max(gate_count, 1),
        "gate_above_099_fraction": gate_high / max(gate_count, 1),
    })
    return result


def score_split(model: JointModel, data: dict[str, Any], device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    n = data["rows"]
    step_scores = np.empty(n, dtype=np.float64)
    prefix_scores = np.empty(n, dtype=np.float64)
    gates = np.full(n, np.nan, dtype=np.float64)
    norms = np.empty(n, dtype=np.float64)
    with torch.inference_mode():
        for parts in batches(data["slices"], 0, shuffle=False):
            indices = row_indices(parts)
            z, step_logits, _ = model.step.pool_indices(data["cache"], indices, device)
            step_scores[indices.numpy()] = torch.sigmoid(step_logits).cpu().numpy()
            cursor = 0
            for part in parts:
                length = part.stop - part.start
                prefix_logits, row_gates, row_norms = model.prefix.forward_sequence(z[cursor : cursor + length])
                prefix_scores[part] = torch.sigmoid(prefix_logits).cpu().numpy()
                gates[part] = row_gates.cpu().numpy()
                norms[part] = row_norms.cpu().numpy()
                cursor += length
    return {"step": step_scores, "prefix": prefix_scores, "gates": gates, "norms": norms}


def classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
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


def evaluate(data: dict[str, Any], scores: dict[str, np.ndarray]) -> dict[str, Any]:
    step_labels = data["y_step"].numpy().astype(np.int64)
    prefix_labels = data["y_prefix"].numpy().astype(np.int64)
    final_indices = np.asarray([part.stop - 1 for part in data["slices"]], dtype=np.int64)
    return {
        "step": classification_metrics(step_labels, scores["step"]),
        "prefix_local": classification_metrics(prefix_labels, scores["prefix"]),
        "prefix_final": classification_metrics(prefix_labels[final_indices], scores["prefix"][final_indices]),
    }


def smoke(model_key: str, output_dir: Path, device: torch.device, regime: str) -> None:
    train = load_split(model_key, "train", limit_trajectories=3)
    val = load_split(model_key, "val", limit_trajectories=3)
    model = JointModel(MODEL_SPECS[model_key]["dimension"], 42).to(device)
    if not torch.equal(model.prefix.gate.weight, torch.zeros_like(model.prefix.gate.weight)):
        raise AssertionError("gate weight initialization changed")
    gradient = gradient_flow_audit(model, train, device)
    optimizers = make_optimizers(model, regime)
    trace = train_epoch(model, train, optimizers, device, seed=42001)
    val_metrics = evaluate(val, score_split(model, val, device))
    report = {
        "status": "PASS",
        "model": model_key,
        "regime": regime,
        "device": str(device),
        "train_audit": train["audit"],
        "val_audit": val["audit"],
        "gradient_flow": gradient,
        "one_joint_epoch": trace,
        "val_metrics": val_metrics,
        "test_touched": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"smoke_{model_key}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "JOINT_SMOKE_PASS", "model": model_key, "gradient": gradient}), flush=True)


def full(model_key: str, output_dir: Path, device: torch.device, regime: str) -> None:
    started = time.time()
    set_seed(42)
    splits = {split: load_split(model_key, split) for split in ("train", "val", "test")}
    model = JointModel(MODEL_SPECS[model_key]["dimension"], 42).to(device)
    gradient = gradient_flow_audit(model, splits["train"], device)
    optimizers = make_optimizers(model, regime)
    step_optimizer = optimizers[0]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        step_optimizer, mode="min", factor=0.1, patience=20, min_lr=1e-6
    )
    best_auc, best_epoch, best_state = -math.inf, 0, None
    best_val_step: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        row = train_epoch(
            model,
            splits["train"],
            optimizers,
            device,
            seed=42000 + epoch,
        )
        scheduler.step(row["step"])
        val_scores = score_split(model, splits["val"], device)
        val_metrics = evaluate(splits["val"], val_scores)
        row.update({
            "epoch": epoch,
            "val_step_auroc": val_metrics["step"]["auroc"],
            "val_prefix_local_auroc": val_metrics["prefix_local"]["auroc"],
            "step_lr": optimizers[0].param_groups[0]["lr"],
            "prefix_lr": (optimizers[1].param_groups[0]["lr"] if regime == "j_dual" else optimizers[0].param_groups[1]["lr"]),
        })
        trace.append(row)
        print(json.dumps({"event": "JOINT_EPOCH", "model": model_key, **row}), flush=True)
        if val_metrics["prefix_local"]["auroc"] > best_auc:
            best_auc = val_metrics["prefix_local"]["auroc"]
            best_epoch = epoch
            best_val_step = val_metrics["step"]
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is None or best_val_step is None:
        raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best_state)
    val_scores = score_split(model, splits["val"], device)
    val_metrics = evaluate(splits["val"], val_scores)
    # This is the single test touch after checkpoint selection.
    test_scores = score_split(model, splits["test"], device)
    test_metrics = evaluate(splits["test"], test_scores)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "model": model_key,
        "seed": 42,
        "best_epoch": best_epoch,
        "best_val_prefix_local_auroc": best_auc,
        "best_val_step_metrics": best_val_step,
        "dimension": MODEL_SPECS[model_key]["dimension"],
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        trajectory_id=np.asarray(splits["test"]["trajectory_ids"], dtype=str),
        trajectory_index=splits["test"]["trajectory_indices"].numpy(),
        step_id=splits["test"]["positions"].numpy(),
        flat_step_index=splits["test"]["flat_step_indices"].numpy(),
        step_label=splits["test"]["y_step"].numpy().astype(np.int64),
        prefix_label=splits["test"]["y_prefix"].numpy().astype(np.int64),
        step_score=test_scores["step"],
        prefix_score=test_scores["prefix"],
        gate_value=test_scores["gates"],
        pre_normalization_norm=test_scores["norms"],
        model=np.asarray([model_key] * splits["test"]["rows"]),
        method=np.asarray([regime] * splits["test"]["rows"]),
    )
    result = {
        "status": "PASS",
        "method": "joint_step_prefix",
        "regime": regime,
        "model": model_key,
        "seed": 42,
        "best_epoch": best_epoch,
        "best_val_prefix_local_auroc": best_auc,
        "gradient_flow": gradient,
        "optimizers": {
            "regime": regime,
            "step": {"family": ("Adam" if regime == "j_dual" else "AdamW"), "lr": STEP_LR, "weight_decay": STEP_WEIGHT_DECAY, "parameters": "token scorer + Step probe"},
            "prefix": {"family": "AdamW", "lr": PREFIX_LR, "weight_decay": PREFIX_WEIGHT_DECAY, "parameters": "scalar gate + Prefix probe"},
            "parameter_sets_disjoint_and_complete": True,
            "update_order": ("both zero_grad; one backward on Step+Prefix; Adam.step then AdamW.step" if regime == "j_dual" else "one AdamW zero_grad; one backward on Step+Prefix; one AdamW step with two groups"),
        },
        "scheduler": {"name": "ReduceLROnPlateau", "signal": "train Step loss", "factor": 0.1, "patience": 20, "min_lr": 1e-6},
        "checkpoint_selection": "maximum validation Prefix-Local AUROC after each of 20 completed epochs",
        "validation": val_metrics,
        "test": test_metrics,
        "input_audit": {split: data["audit"] for split, data in splits.items()},
        "trace": trace,
        "walltime_s": time.time() - started,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "test_touches": 1,
        "answer_companion_loaded": False,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "JOINT_TEST_COMPLETE", "model": model_key, "test": test_metrics}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    # Default preserves already-submitted J-Dual PBS scripts that predate the
    # authorized J-Unified extension but stage this canonical source at start.
    parser.add_argument("--regime", choices=("j_dual", "j_unified"), default="j_dual")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    print(json.dumps({"event": "JOINT_START", "mode": args.mode, "model": args.model, "device": str(device), "torch": torch.__version__}), flush=True)
    if args.mode == "smoke":
        smoke(args.model, args.output_dir, device, args.regime)
    else:
        full(args.model, args.output_dir, device, args.regime)


if __name__ == "__main__":
    main()
