#!/usr/bin/env python3
"""Frozen two-stage Direct-MLP Prefix transition experiment.

Only the Prefix state transition changes.  Loading, learned-Step pooling,
Prefix loss, batching, metrics, and checkpoint selection are imported from the
audited scalar-gate implementation staged beside this file at runtime.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score


HIDDEN_WIDTH = 256
METHOD = "direct_mlp_prefix"


def load_reference(path: Path):
    spec = importlib.util.spec_from_file_location("scalar_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scalar reference: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DirectMLPPrefix(nn.Module):
    def __init__(self, dimension: int, epsilon: float):
        super().__init__()
        self.dimension = dimension
        self.epsilon = epsilon
        self.transition = nn.Sequential(
            nn.Linear(2 * dimension, HIDDEN_WIDTH),
            nn.GELU(),
            nn.Linear(HIDDEN_WIDTH, dimension),
        )
        self.probe = nn.Linear(dimension, 1)

    def states(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z.ndim != 2 or z.shape[1] != self.dimension or len(z) == 0:
            raise AssertionError(f"invalid sequence shape {tuple(z.shape)}")
        prefixes = [z[0]]
        norms = [torch.linalg.vector_norm(z[0])]
        previous = z[0]
        for current in z[1:]:
            raw = self.transition(torch.cat([previous, current]))
            norm = torch.linalg.vector_norm(raw)
            if not bool(torch.isfinite(norm)) or float(norm.detach()) <= self.epsilon:
                raise FloatingPointError(f"invalid pre-normalization norm {float(norm.detach())}")
            previous = raw / (norm + self.epsilon)
            if not bool(torch.all(torch.isfinite(previous))):
                raise FloatingPointError("non-finite normalized Prefix state")
            prefixes.append(previous)
            norms.append(norm)
        return torch.stack(prefixes), torch.stack(norms)

    def forward_sequence(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        states, norms = self.states(z)
        return self.probe(states).flatten(), norms


def parameter_counts(dimension: int) -> dict[str, int]:
    scalar_gate = 2 * dimension + 1
    direct_transition = (2 * dimension * HIDDEN_WIDTH + HIDDEN_WIDTH) + (
        HIDDEN_WIDTH * dimension + dimension
    )
    prefix_probe = dimension + 1
    return {
        "scalar_gate_transition": scalar_gate,
        "direct_mlp_transition": direct_transition,
        "prefix_probe": prefix_probe,
        "direct_mlp_trainable_total": direct_transition + prefix_probe,
    }


def forward_batch(model: DirectMLPPrefix, data: dict[str, Any], parts: list[slice], device: torch.device):
    logits, teachers, labels, lengths, norm_rows = [], [], [], [], []
    for part in parts:
        value, norms = model.forward_sequence(data["z"][part].to(device))
        logits.append(value)
        teachers.append(data["teacher"][part].to(device))
        labels.append(data["y_cum"][part].to(device))
        lengths.append(part.stop - part.start)
        norm_rows.append(norms)
    return torch.cat(logits), torch.cat(teachers), torch.cat(labels), lengths, norm_rows


def train_epoch(model: DirectMLPPrefix, data: dict[str, Any], slices: list[slice], optimizer, device: torch.device, seed: int, ref) -> dict[str, float]:
    model.train()
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    total = anchor_total = sync_total = 0.0
    rows = 0
    min_norm = math.inf
    max_norm = 0.0
    for parts in ref.batch_slices(slices, seed):
        logits, teacher, labels, lengths, norm_rows = forward_batch(model, data, parts, device)
        weights = torch.ones_like(labels)
        weights[torch.tensor(np.cumsum(lengths) - 1, device=device)] = ref.LAMBDA_FINAL
        anchor = (criterion(logits, labels) * weights).mean()
        probabilities = torch.sigmoid(logits)
        delta = torch.relu(teacher - probabilities)
        sync = (delta.pow(2) * teacher.pow(2)).sum() / sum(lengths)
        loss = anchor + ref.LAMBDA_SYNC * sync
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(parameter.grad is None or bool(torch.all(torch.isfinite(parameter.grad))) for parameter in model.parameters()):
            raise FloatingPointError("non-finite trainable gradient")
        optimizer.step()
        count = sum(lengths)
        total += float(loss.detach().cpu()) * count
        anchor_total += float(anchor.detach().cpu()) * count
        sync_total += float(sync.detach().cpu()) * count
        rows += count
        min_norm = min(min_norm, min(float(row.min().detach().cpu()) for row in norm_rows))
        max_norm = max(max_norm, max(float(row.max().detach().cpu()) for row in norm_rows))
    return {
        "loss": total / rows,
        "anchor": anchor_total / rows,
        "sync": sync_total / rows,
        "min_pre_normalization_norm": min_norm,
        "max_pre_normalization_norm": max_norm,
    }


def score_model(model: DirectMLPPrefix, data: dict[str, Any], slices: list[slice], device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    scores = torch.empty(len(data["y_cum"]), dtype=torch.float32)
    norms = torch.empty(len(scores), dtype=torch.float32)
    with torch.inference_mode():
        for part in slices:
            logits, row_norms = model.forward_sequence(data["z"][part].to(device))
            scores[part] = torch.sigmoid(logits).cpu()
            norms[part] = row_norms.cpu()
    if not bool(torch.all(torch.isfinite(scores))) or float(norms.min()) <= model.epsilon:
        raise FloatingPointError("invalid full-split scores or states")
    return {
        "scores": scores.numpy().astype(np.float64),
        "norms": norms.numpy().astype(np.float64),
    }


def evaluate(data: dict[str, Any], slices: list[slice], scored: dict[str, np.ndarray], ref) -> dict[str, Any]:
    labels = data["y_cum"].numpy().astype(np.int64)
    final_indices = np.asarray([part.stop - 1 for part in slices], dtype=np.int64)
    return {
        "local": ref.classification_metrics(labels, scored["scores"]),
        "final": ref.classification_metrics(labels[final_indices], scored["scores"][final_indices]),
        "dynamic": ref.dynamic_metrics(data, slices, scored["scores"]),
    }


def metrics_when_defined(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (scores > threshold).astype(np.int64)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    return {
        "n": int(len(labels)),
        "positive": positives,
        "negative": negatives,
        "auroc": float(roc_auc_score(labels, scores)) if positives and negatives else None,
        "auprc": float(average_precision_score(labels, scores)) if positives else None,
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "score_mean": float(scores.mean()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "threshold": threshold,
        "threshold_operator": ">",
    }


def transition_diagnostics(data: dict[str, Any], slices: list[slice], scored: dict[str, np.ndarray], threshold: float) -> dict[str, Any]:
    labels = data["y_cum"].numpy().astype(np.int64)
    groups: dict[str, list[int]] = {name: [] for name in ("0->0", "0->1", "1->1", "1->0")}
    for part in slices:
        for index in range(part.start + 1, part.stop):
            groups[f"{labels[index - 1]}->{labels[index]}"].append(index)
    result = {}
    for name, indices in groups.items():
        idx = np.asarray(indices, dtype=np.int64)
        result[name] = metrics_when_defined(labels[idx], scored["scores"][idx], threshold) if len(idx) else {"n": 0}
    result["pre_normalization_norm"] = {
        "min": float(scored["norms"].min()),
        "mean": float(scored["norms"].mean()),
        "max": float(scored["norms"].max()),
    }
    return result


def audit_smoke(model_key: str, output_dir: Path, device: torch.device, batch_size: int, ref) -> None:
    ref.set_seed(42)
    frozen = ref.checkpoint_model(model_key, device)
    train = ref.build_split(model_key, "train", frozen, device, batch_size, smoke=True)
    val = ref.build_split(model_key, "val", frozen, device, batch_size, smoke=True)
    train_slices = ref.sequence_slices(train["trajectory_indices"])
    val_slices = ref.sequence_slices(val["trajectory_indices"])
    dimension = ref.MODEL_SPECS[model_key]["dimension"]
    baseline_features = ref.initial_prefix_features(train["z"], train_slices, device)
    baseline = ref.train_baseline(baseline_features, train["y_cum"], dimension, device)
    model = DirectMLPPrefix(dimension, ref.EPSILON).to(device)
    model.probe.load_state_dict(baseline.state_dict())

    first = train["z"][train_slices[0]].to(device)
    logits, norms = model.forward_sequence(first)
    if logits.shape != (len(first),) or not bool(torch.all(torch.isfinite(logits))):
        raise AssertionError("shape/finite smoke failed")
    states, _ = model.states(first)
    if not torch.equal(states[0], first[0]):
        raise AssertionError("p1 is not exactly z1")
    modified = first.clone()
    modified[-1] = torch.roll(modified[-1], 1)
    modified_states, _ = model.states(modified)
    if len(first) > 1 and not torch.equal(states[:-1], modified_states[:-1]):
        raise AssertionError("future Step changed an earlier Prefix state")
    second = train["z"][train_slices[1]].to(device)
    second_states, _ = model.states(second)
    if not torch.equal(second_states[0], second[0]):
        raise AssertionError("trajectory reset failed")

    optimizer = torch.optim.AdamW(model.parameters(), lr=ref.LR)
    trace = train_epoch(model, train, train_slices, optimizer, device, 42, ref)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in frozen.parameters()):
        raise AssertionError("frozen Step path received gradients")
    scored = score_model(model, val, val_slices, device)
    val_metrics = evaluate(val, val_slices, scored, ref)
    report = {
        "status": "PASS",
        "model": model_key,
        "device": str(device),
        "train_audit": train["audit"],
        "val_audit": val["audit"],
        "train_smoke_rows": len(train["y_cum"]),
        "val_smoke_rows": len(val["y_cum"]),
        "shape_causality_reset_pass": True,
        "frozen_step_gradient_pass": True,
        "finite_and_positive_norm_pass": bool(np.isfinite(scored["norms"]).all() and scored["norms"].min() > ref.EPSILON),
        "parameter_counts": parameter_counts(dimension),
        "optimizer": {"name": "AdamW", "lr": ref.LR, "weight_decay": 0.01},
        "epoch_trace": trace,
        "val_metrics": val_metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"smoke_{model_key}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "DIRECT_MLP_SMOKE_PASS", "model": model_key, "rows": len(train["y_cum"])}), flush=True)


def full_one(model_key: str, output_dir: Path, device: torch.device, batch_size: int, ref) -> None:
    started = time.time()
    seed = 42
    ref.set_seed(seed)
    frozen = ref.checkpoint_model(model_key, device)
    splits = {name: ref.build_split(model_key, name, frozen, device, batch_size, smoke=False) for name in ("train", "val", "test")}
    slices = {name: ref.sequence_slices(value["trajectory_indices"]) for name, value in splits.items()}
    audit = {name: value["audit"] for name, value in splits.items()}
    dimension = ref.MODEL_SPECS[model_key]["dimension"]
    initial = ref.initial_prefix_features(splits["train"]["z"], slices["train"], device)
    baseline = ref.train_baseline(initial, splits["train"]["y_cum"], dimension, device)
    model = DirectMLPPrefix(dimension, ref.EPSILON).to(device)
    model.probe.load_state_dict(baseline.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=ref.LR)
    best_auc, best_epoch, best_state = -math.inf, 0, None
    trace = []
    for epoch in range(1, ref.EPOCHS + 1):
        row = train_epoch(model, splits["train"], slices["train"], optimizer, device, seed * 1000 + epoch, ref)
        val_scored = score_model(model, splits["val"], slices["val"], device)
        val_auc = ref.classification_metrics(splits["val"]["y_cum"].numpy().astype(np.int64), val_scored["scores"])["auroc"]
        row.update({"epoch": epoch, "val_local_auroc": val_auc})
        trace.append(row)
        print(json.dumps({"event": "DIRECT_MLP_EPOCH", "model": model_key, **row}), flush=True)
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best_state)
    val_scored = score_model(model, splits["val"], slices["val"], device)
    val_metrics = evaluate(splits["val"], slices["val"], val_scored, ref)
    test_scored = score_model(model, splits["test"], slices["test"], device)
    test_metrics = evaluate(splits["test"], slices["test"], test_scored, ref)
    diagnostics = transition_diagnostics(splits["test"], slices["test"], test_scored, ref.THRESHOLD)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_local_auroc": best_auc,
        "dimension": dimension,
        "hidden_width": HIDDEN_WIDTH,
        "method": METHOD,
    }, output_dir / "checkpoint.pt")
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        trajectory_id=np.asarray(splits["test"]["trajectory_ids"], dtype=str),
        trajectory_index=splits["test"]["trajectory_indices"].numpy(),
        step_id=splits["test"]["positions"].numpy(),
        flat_step_index=splits["test"]["flat_step_indices"].numpy(),
        prefix_label=splits["test"]["y_cum"].numpy().astype(np.int64),
        step_label=splits["test"]["y_step"].numpy().astype(np.int64),
        score=test_scored["scores"],
        pre_normalization_norm=test_scored["norms"],
        split=np.asarray(["test"] * len(test_scored["scores"])),
        model=np.asarray([model_key] * len(test_scored["scores"])),
        method=np.asarray([METHOD] * len(test_scored["scores"])),
    )
    record = {
        "status": "PASS",
        "model": model_key,
        "method": METHOD,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_local_auroc": best_auc,
        "validation": val_metrics,
        "test": test_metrics,
        "transition_diagnostics": diagnostics,
        "parameter_counts": parameter_counts(dimension),
        "input_audit": audit,
        "optimizer": {"name": "AdamW", "lr": ref.LR, "weight_decay": 0.01},
        "trace": trace,
        "walltime_s": time.time() - started,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
    }
    (output_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "DIRECT_MLP_FULL_COMPLETE", "model": model_key, "test": test_metrics, "best_epoch": best_epoch}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--model", choices=("qwen", "llama", "deepseek"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scalar-reference", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    ref = load_reference(args.scalar_reference)
    device = torch.device(args.device)
    print(json.dumps({"event": "DIRECT_MLP_START", "mode": args.mode, "model": args.model, "device": str(device), "torch": torch.__version__}), flush=True)
    if args.mode == "smoke":
        audit_smoke(args.model, args.output_dir, device, args.feature_batch_size, ref)
    else:
        full_one(args.model, args.output_dir, device, args.feature_batch_size, ref)


if __name__ == "__main__":
    main()
