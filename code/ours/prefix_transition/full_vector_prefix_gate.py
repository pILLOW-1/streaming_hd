#!/usr/bin/env python3
"""Full d-dimensional Prefix gate on the frozen learned-Token Step path.

This is the strict vector-width variant of the archived scalar-gate experiment.
It imports the audited scalar implementation as the common Prefix code path and
changes only the recurrent gate from Linear(2d, 1) to Linear(2d, d).  The
released split handling, frozen Step pooling, Prefix loss, optimizer, batch
order, checkpoint rule and metrics are reused unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


PROJECT = Path(__import__("os").environ.get("STREAMING_HD_LEGACY_ROOT", ""))
SCALAR_ROOT = PROJECT / "experiments/streaming_hd/runs/prefix_dynamic_gate_within_three_models"
SCALAR_SOURCE = SCALAR_ROOT / "code/dynamic_prefix_gate.py"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scalar_module():
    specification = importlib.util.spec_from_file_location("audited_scalar_prefix_gate", SCALAR_SOURCE)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import audited scalar source: {SCALAR_SOURCE}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


base = load_scalar_module()


class FullVectorPrefixGate(nn.Module):
    """Exact full gate W_r:[2d]->d followed by the unchanged linear probe."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.gate = nn.Linear(2 * dimension, dimension)
        self.probe = nn.Linear(dimension, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def states_sequence(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = [z[0]]
        gates = [torch.full((self.dimension,), float("nan"), device=z.device)]
        pre_norms = [torch.linalg.vector_norm(z[0])]
        previous = z[0]
        for current in z[1:]:
            gate = torch.sigmoid(self.gate(torch.cat([previous, current], dim=-1)))
            mixed = (1.0 - gate) * previous + gate * current
            norm = torch.linalg.vector_norm(mixed)
            previous = mixed / (norm + base.EPSILON)
            states.append(previous)
            gates.append(gate)
            pre_norms.append(norm)
        return torch.stack(states), torch.stack(gates), torch.stack(pre_norms)

    def forward_sequence(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states, vector_gates, norms = self.states_sequence(z)
        return self.probe(states).flatten(), vector_gates.mean(dim=-1), norms

    def forward_padded(
        self, sequences: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorize independent trajectories at each recurrent time index."""
        lengths = torch.tensor([len(value) for value in sequences], device=sequences[0].device)
        padded = pad_sequence(sequences, batch_first=True)
        previous = padded[:, 0]
        logits = [self.probe(previous).flatten()]
        gate_means = [torch.full((len(sequences),), float("nan"), device=previous.device)]
        norms = [torch.linalg.vector_norm(previous, dim=-1)]
        for position in range(1, padded.shape[1]):
            current = padded[:, position]
            vector_gate = torch.sigmoid(self.gate(torch.cat([previous, current], dim=-1)))
            mixed = (1.0 - vector_gate) * previous + vector_gate * current
            row_norm = torch.linalg.vector_norm(mixed, dim=-1)
            candidate = mixed / (row_norm.unsqueeze(-1) + base.EPSILON)
            active = lengths > position
            previous = torch.where(active.unsqueeze(-1), candidate, previous)
            logits.append(self.probe(previous).flatten())
            gate_means.append(torch.where(active, vector_gate.mean(dim=-1), torch.full_like(row_norm, float("nan"))))
            norms.append(torch.where(active, row_norm, torch.full_like(row_norm, float("nan"))))
        return torch.stack(logits, dim=1), torch.stack(gate_means, dim=1), torch.stack(norms, dim=1), lengths


def vectorized_forward_batch(
    model: FullVectorPrefixGate,
    data: dict[str, Any],
    parts: list[slice],
    device: torch.device,
):
    sequences = [data["z"][part].to(device) for part in parts]
    padded_logits, padded_gate_means, padded_norms, length_tensor = model.forward_padded(sequences)
    lengths = [int(value) for value in length_tensor.tolist()]
    logits = torch.cat([padded_logits[row, :length] for row, length in enumerate(lengths)])
    teachers = torch.cat([data["teacher"][part].to(device) for part in parts])
    labels = torch.cat([data["y_cum"][part].to(device) for part in parts])
    gate_rows = [padded_gate_means[row, 1:length] for row, length in enumerate(lengths) if length > 1]
    norm_rows = [padded_norms[row, :length] for row, length in enumerate(lengths)]
    return logits, teachers, labels, lengths, gate_rows, norm_rows


def score_model(
    model: FullVectorPrefixGate,
    data: dict[str, Any],
    slices: list[slice],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    scores = torch.empty(len(data["y_cum"]), dtype=torch.float32)
    gate_means = torch.full((len(scores),), float("nan"), dtype=torch.float32)
    norms = torch.empty(len(scores), dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, len(slices), base.BATCH_SIZE_COT):
            parts = slices[start : start + base.BATCH_SIZE_COT]
            sequences = [data["z"][part].to(device) for part in parts]
            padded_logits, padded_gates, padded_norms, length_tensor = model.forward_padded(sequences)
            for row, (part, length_value) in enumerate(zip(parts, length_tensor.tolist())):
                length = int(length_value)
                scores[part] = torch.sigmoid(padded_logits[row, :length]).cpu()
                gate_means[part] = padded_gates[row, :length].cpu()
                norms[part] = padded_norms[row, :length].cpu()
    return {
        "scores": scores.numpy().astype(np.float64),
        "gates": gate_means.numpy().astype(np.float64),
        "norms": norms.numpy().astype(np.float64),
    }


# Make the audited training/evaluation helpers dispatch through this variant.
base.DynamicPrefixGate = FullVectorPrefixGate
base.forward_batch = vectorized_forward_batch
base.score_model = score_model


def gate_parameter_count(dimension: int) -> int:
    return dimension * (2 * dimension) + dimension


def smoke_one(model_key: str, output_dir: Path, device: torch.device, feature_batch_size: int) -> None:
    base.set_seed(42)
    frozen = base.checkpoint_model(model_key, device)
    frozen_before = {key: value.detach().cpu().clone() for key, value in frozen.state_dict().items()}
    train = base.build_split(model_key, "train", frozen, device, feature_batch_size, smoke=True)
    validation = base.build_split(model_key, "val", frozen, device, feature_batch_size, smoke=True)
    slices = base.sequence_slices(train["trajectory_indices"])
    dimension = base.MODEL_SPECS[model_key]["dimension"]
    model = FullVectorPrefixGate(dimension).to(device)
    if list(model.gate.weight.shape) != [dimension, 2 * dimension] or list(model.gate.bias.shape) != [dimension]:
        raise AssertionError("full vector gate parameter shape mismatch")

    reset_checks = []
    for part in slices:
        z = train["z"][part].to(device)
        states, vector_gates, norms = model.states_sequence(z)
        reset_checks.append(bool(torch.equal(states[0], z[0])))
        if len(z) > 1 and not torch.equal(vector_gates[1:], torch.full_like(vector_gates[1:], 0.5)):
            raise AssertionError("zero-initialized vector gate is not exactly 0.5")
        if not torch.all(torch.isfinite(states)) or not torch.all(torch.isfinite(norms)):
            raise AssertionError("non-finite recurrence")
        state_norms = torch.linalg.vector_norm(states, dim=-1)
        if not torch.allclose(state_norms, torch.ones_like(state_norms), atol=2e-5, rtol=2e-5):
            raise AssertionError("prefix states are not L2 normalized")
    if not all(reset_checks):
        raise AssertionError("trajectory reset failed")

    initial = base.initial_prefix_features(train["z"], slices, device)
    baseline = base.train_baseline(initial, train["y_cum"], dimension, device)
    model.probe.load_state_dict(baseline.state_dict())
    del baseline, initial
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LR)
    parts = slices[: min(len(slices), base.BATCH_SIZE_COT)]
    logits, teacher, labels, lengths, _, _ = vectorized_forward_batch(model, train, parts, device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    weights = torch.ones_like(labels)
    weights[torch.tensor(np.cumsum(lengths) - 1, device=device)] = base.LAMBDA_FINAL
    anchor = (criterion(logits, labels) * weights).mean()
    probabilities = torch.sigmoid(logits)
    sync = (torch.relu(teacher - probabilities).pow(2) * teacher.pow(2)).sum() / sum(lengths)
    loss = anchor + base.LAMBDA_SYNC * sync
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = {
        "W_r": float(torch.linalg.vector_norm(model.gate.weight.grad).detach().cpu()),
        "b_r": float(torch.linalg.vector_norm(model.gate.bias.grad).detach().cpu()),
        "prefix_probe": float(torch.linalg.vector_norm(model.probe.weight.grad).detach().cpu()),
    }
    if any(not math.isfinite(value) or value <= 0 for value in gradients.values()):
        raise AssertionError(f"missing full-gate/probe gradient: {gradients}")
    if any(parameter.grad is not None for parameter in frozen.parameters()):
        raise AssertionError("frozen Step branch received gradients")
    optimizer.step()
    frozen_after = {key: value.detach().cpu() for key, value in frozen.state_dict().items()}
    if any(not torch.equal(frozen_before[key], frozen_after[key]) for key in frozen_before):
        raise AssertionError("frozen Step branch was updated")

    validation_slices = base.sequence_slices(validation["trajectory_indices"])
    validation_metrics = base.evaluate(validation, validation_slices, score_model(model, validation, validation_slices, device))
    report = {
        "status": "PASS",
        "model": model_key,
        "device": str(device),
        "gate_type": "full_d_dimensional_vector",
        "gate_weight_shape": list(model.gate.weight.shape),
        "gate_bias_shape": list(model.gate.bias.shape),
        "gate_parameters": gate_parameter_count(dimension),
        "probe_parameters": sum(parameter.numel() for parameter in model.probe.parameters()),
        "gradients": gradients,
        "trajectory_reset_all": all(reset_checks),
        "frozen_step_unchanged": True,
        "frozen_step_gradient_free": True,
        "train_smoke_rows": len(train["y_cum"]),
        "validation_smoke_rows": len(validation["y_cum"]),
        "validation_local": validation_metrics["local"],
        "train_audit": train["audit"],
        "validation_audit": validation["audit"],
        "scalar_source": str(SCALAR_SOURCE),
        "scalar_source_sha256": file_sha256(SCALAR_SOURCE),
        "answer_companion_loaded": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"smoke_{model_key}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "FULL_VECTOR_GATE_SMOKE_PASS", "model": model_key, "gradients": gradients}), flush=True)


def full_one(model_key: str, output_dir: Path, device: torch.device, feature_batch_size: int, seed: int) -> None:
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    frozen = base.checkpoint_model(model_key, device)
    splits = {
        name: base.build_split(model_key, name, frozen, device, feature_batch_size, smoke=False)
        for name in ("train", "val", "test")
    }
    slices = {name: base.sequence_slices(value["trajectory_indices"]) for name, value in splits.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    input_audit = {name: value["audit"] for name, value in splits.items()}
    (output_dir / "input_audit.json").write_text(
        json.dumps({"status": "PASS", "splits": input_audit, "answer_companion_loaded": False}, indent=2) + "\n",
        encoding="utf-8",
    )

    dimension = base.MODEL_SPECS[model_key]["dimension"]
    base.set_seed(seed)
    initial = base.initial_prefix_features(splits["train"]["z"], slices["train"], device)
    baseline = base.train_baseline(initial, splits["train"]["y_cum"], dimension, device)
    model = FullVectorPrefixGate(dimension).to(device)
    model.probe.load_state_dict(baseline.state_dict())
    del baseline, initial
    if not torch.equal(model.gate.weight, torch.zeros_like(model.gate.weight)):
        raise AssertionError("W_r initialization changed")
    if not torch.equal(model.gate.bias, torch.zeros_like(model.gate.bias)):
        raise AssertionError("b_r initialization changed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LR)
    best_auc, best_epoch, best_state = -math.inf, 0, None
    trace = []
    for epoch in range(1, base.EPOCHS + 1):
        row = base.train_epoch(model, splits["train"], slices["train"], optimizer, device, seed=seed * 1000 + epoch)
        validation_scores = score_model(model, splits["val"], slices["val"], device)
        validation_auc = base.classification_metrics(
            splits["val"]["y_cum"].numpy().astype(np.int64), validation_scores["scores"]
        )["auroc"]
        row.update({"epoch": epoch, "val_local_auroc": validation_auc})
        trace.append(row)
        print(json.dumps({"event": "EPOCH", "model": model_key, "seed": seed, **row}), flush=True)
        if validation_auc > best_auc:
            best_auc, best_epoch = validation_auc, epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best_state)
    validation_scored = score_model(model, splits["val"], slices["val"], device)
    validation_metrics = base.evaluate(splits["val"], slices["val"], validation_scored)
    test_scored = score_model(model, splits["test"], slices["test"], device)
    test_metrics = base.evaluate(splits["test"], slices["test"], test_scored)
    diagnostics = base.gate_diagnostics(splits["test"], slices["test"], test_scored)

    checkpoint = {
        "state_dict": best_state,
        "method": "prefix_full_vector_gate",
        "gate_type": "full_d_dimensional_vector",
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_local_auroc": best_auc,
        "dimension": dimension,
        "gate_weight_shape": [dimension, 2 * dimension],
        "gate_bias_shape": [dimension],
        "gate_parameters": gate_parameter_count(dimension),
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        trajectory_id=np.asarray(splits["test"]["trajectory_ids"], dtype=str),
        trajectory_index=splits["test"]["trajectory_indices"].numpy(),
        step_id=splits["test"]["positions"].numpy(),
        flat_step_index=splits["test"]["flat_step_indices"].numpy(),
        prefix_label=splits["test"]["y_cum"].numpy().astype(np.int64),
        step_label=splits["test"]["y_step"].numpy().astype(np.int64),
        score=test_scored["scores"],
        vector_gate_mean=test_scored["gates"],
        pre_normalization_norm=test_scored["norms"],
        split=np.asarray(["test"] * len(test_scored["scores"])),
        model=np.asarray([model_key] * len(test_scored["scores"])),
        method=np.asarray(["prefix_full_vector_gate"] * len(test_scored["scores"])),
    )
    record = {
        "status": "PASS",
        "model": model_key,
        "method": "prefix_full_vector_gate",
        "seed": seed,
        "gate_type": "full_d_dimensional_vector",
        "gate_weight_shape": [dimension, 2 * dimension],
        "gate_bias_shape": [dimension],
        "gate_parameters": gate_parameter_count(dimension),
        "prefix_probe_parameters": dimension + 1,
        "best_epoch": best_epoch,
        "best_val_local_auroc": best_auc,
        "validation": validation_metrics,
        "test": test_metrics,
        "gate_mean_diagnostics": diagnostics,
        "trace": trace,
        "input_audit": input_audit,
        "scalar_source_sha256": file_sha256(SCALAR_SOURCE),
        "classification_rule": "score > 0.5",
        "answer_companion_loaded": False,
        "walltime_s": time.time() - started,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
    }
    (output_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "FULL_VECTOR_MODEL_COMPLETE", "model": model_key, "seed": seed, "test_local": test_metrics["local"], "peak_memory_bytes": record["cuda_peak_memory_bytes"]}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--model", choices=tuple(base.MODEL_SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(arguments.device)
    print(json.dumps({"event": "FULL_VECTOR_START", "mode": arguments.mode, "model": arguments.model, "seed": arguments.seed, "device": str(device), "torch": torch.__version__, "scalar_source_sha256": file_sha256(SCALAR_SOURCE)}), flush=True)
    if arguments.mode == "smoke":
        smoke_one(arguments.model, arguments.output_dir, device, arguments.feature_batch_size)
    else:
        full_one(arguments.model, arguments.output_dir, device, arguments.feature_batch_size, arguments.seed)


if __name__ == "__main__":
    main()
