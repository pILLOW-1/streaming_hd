#!/usr/bin/env python3
"""Frozen two-stage Prefix gate with one shared content-relation projection."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

try:
    import dynamic_prefix_gate as predecessor
except ModuleNotFoundError:
    sys.path.insert(
        0,
        str(Path(__import__("os").environ["STREAMING_HD_LEGACY_ROOT"]) / "experiments/streaming_hd/runs/prefix_dynamic_gate_within_three_models/code"),
    )
    import dynamic_prefix_gate as predecessor  # type: ignore[no-redef]


RELATION_WIDTH = 256
EPSILON = predecessor.EPSILON


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, default=str), flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RelationConditionedPrefixGate(nn.Module):
    """Scalar predecessor gate plus the frozen shared relation descriptor."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

        # Base path must be first and identical to archived DynamicPrefixGate.
        self.gate = nn.Linear(2 * dimension, 1)
        self.probe = nn.Linear(dimension, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        # Relation parameters are instantiated only after all base parameters.
        self.relation_projection = nn.Linear(
            dimension, RELATION_WIDTH, bias=False
        )
        self.relation_mlp = nn.Sequential(
            nn.Linear(2 * RELATION_WIDTH, RELATION_WIDTH),
            nn.GELU(),
            nn.Linear(RELATION_WIDTH, RELATION_WIDTH),
        )
        self.relation_layer_norm = nn.LayerNorm(RELATION_WIDTH)
        self.relation_readout = nn.Linear(RELATION_WIDTH, 1)

    def relation(
        self, previous: torch.Tensor, current: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # The same parameter object is deliberately called twice so product and
        # absolute difference compare aligned coordinates.
        query = self.relation_projection(previous)
        key = self.relation_projection(current)
        product = query * key
        absolute_difference = torch.abs(query - key)
        match = torch.cat((product, absolute_difference), dim=-1)
        edge = self.relation_mlp(match)
        # No sqrt(width) divisor: this is a normalized learned linear readout.
        relation_delta = self.relation_readout(
            self.relation_layer_norm(edge)
        ).squeeze(-1)
        return relation_delta, {
            "query": query,
            "key": key,
            "product": product,
            "absolute_difference": absolute_difference,
            "match": match,
            "edge": edge,
        }

    def forward_sequence(
        self,
        z: torch.Tensor,
        relation_enabled: bool = True,
        return_debug: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prefixes = [z[0]]
        nan = torch.full((), float("nan"), device=z.device)
        gates = [nan]
        base_gates = [nan]
        base_logits = [nan]
        relation_deltas = [nan]
        pre_norms = [torch.linalg.vector_norm(z[0])]
        product_norms = [nan]
        difference_norms = [nan]
        previous = z[0]
        for current in z[1:]:
            base_logit = self.gate(torch.cat((previous, current))).squeeze()
            relation_delta, relation = self.relation(previous, current)
            effective_delta = (
                relation_delta
                if relation_enabled
                else torch.zeros_like(relation_delta)
            )
            gate = torch.sigmoid(base_logit + effective_delta)
            mixed = (1.0 - gate) * previous + gate * current
            norm = torch.linalg.vector_norm(mixed)
            previous = mixed / (norm + EPSILON)
            prefixes.append(previous)
            gates.append(gate)
            base_gates.append(torch.sigmoid(base_logit))
            base_logits.append(base_logit)
            relation_deltas.append(relation_delta)
            pre_norms.append(norm)
            product_norms.append(torch.linalg.vector_norm(relation["product"]))
            difference_norms.append(
                torch.linalg.vector_norm(relation["absolute_difference"])
            )
        states = torch.stack(prefixes)
        debug = {
            "states": states,
            "gates": torch.stack(gates),
            "base_gates": torch.stack(base_gates),
            "base_logits": torch.stack(base_logits),
            "relation_deltas": torch.stack(relation_deltas),
            "pre_normalization_norms": torch.stack(pre_norms),
            "product_norms": torch.stack(product_norms),
            "absolute_difference_norms": torch.stack(difference_norms),
        }
        logits = self.probe(states).flatten()
        return logits, debug if return_debug else {}


def base_initialization_audit(
    model: RelationConditionedPrefixGate,
    rng_state: torch.Tensor,
    post_model_rng_state: torch.Tensor,
    dimension: int,
) -> dict[str, Any]:
    torch.random.set_rng_state(rng_state)
    archived = predecessor.DynamicPrefixGate(dimension)
    torch.random.set_rng_state(post_model_rng_state)
    checks = {}
    for component in ("gate", "probe"):
        new_state = getattr(model, component).state_dict()
        old_state = getattr(archived, component).state_dict()
        for name, value in old_state.items():
            checks[f"{component}.{name}"] = bool(
                torch.equal(value, new_state[name].detach().cpu())
            )
    if not all(checks.values()):
        raise AssertionError("base scalar-gate initialization changed")
    return {"status": "PASS", "tensors": checks}


def forward_batch(
    model: RelationConditionedPrefixGate,
    data: dict[str, Any],
    parts: list[slice],
    device: torch.device,
    relation_enabled: bool = True,
    return_debug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[dict[str, torch.Tensor]]]:
    logits, teachers, labels, lengths, debug_rows = [], [], [], [], []
    for part in parts:
        value, debug = model.forward_sequence(
            data["z"][part].to(device),
            relation_enabled=relation_enabled,
            return_debug=return_debug,
        )
        logits.append(value)
        teachers.append(data["teacher"][part].to(device))
        labels.append(data["y_cum"][part].to(device))
        lengths.append(part.stop - part.start)
        if return_debug:
            debug_rows.append(debug)
    return (
        torch.cat(logits),
        torch.cat(teachers),
        torch.cat(labels),
        lengths,
        debug_rows,
    )


def prefix_loss(
    logits: torch.Tensor,
    teacher: torch.Tensor,
    labels: torch.Tensor,
    lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    weights = torch.ones_like(labels)
    weights[
        torch.tensor(np.cumsum(lengths) - 1, device=labels.device)
    ] = predecessor.LAMBDA_FINAL
    anchor = (criterion(logits, labels) * weights).mean()
    probabilities = torch.sigmoid(logits)
    synchronization_delta = torch.relu(teacher - probabilities)
    synchronization = (
        synchronization_delta.pow(2) * teacher.pow(2)
    ).sum() / sum(lengths)
    loss = anchor + predecessor.LAMBDA_SYNC * synchronization
    return loss, anchor, synchronization


def train_epoch(
    model: RelationConditionedPrefixGate,
    data: dict[str, Any],
    slices: list[slice],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.train()
    total = anchor_total = sync_total = 0.0
    rows = 0
    min_norm = math.inf
    for parts in predecessor.batch_slices(slices, seed):
        logits, teacher, labels, lengths, debug_rows = forward_batch(
            model, data, parts, device, return_debug=True
        )
        loss, anchor, synchronization = prefix_loss(
            logits, teacher, labels, lengths
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        count = sum(lengths)
        total += float(loss.detach().cpu()) * count
        anchor_total += float(anchor.detach().cpu()) * count
        sync_total += float(synchronization.detach().cpu()) * count
        rows += count
        min_norm = min(
            min_norm,
            min(
                float(row["pre_normalization_norms"].min().detach().cpu())
                for row in debug_rows
            ),
        )
    return {
        "loss": total / rows,
        "anchor": anchor_total / rows,
        "sync": sync_total / rows,
        "min_pre_normalization_norm": min_norm,
    }


def score_model(
    model: RelationConditionedPrefixGate,
    data: dict[str, Any],
    slices: list[slice],
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    size = len(data["y_cum"])
    fields = (
        "gates",
        "base_gates",
        "base_logits",
        "relation_deltas",
        "pre_normalization_norms",
        "product_norms",
        "absolute_difference_norms",
    )
    stored = {
        name: torch.full((size,), float("nan"), dtype=torch.float32)
        for name in fields
    }
    scores = torch.empty(size, dtype=torch.float32)
    with torch.inference_mode():
        for part in slices:
            logits, debug = model.forward_sequence(
                data["z"][part].to(device), return_debug=True
            )
            scores[part] = torch.sigmoid(logits).cpu()
            for name in fields:
                stored[name][part] = debug[name].cpu()
    return {
        "scores": scores.numpy().astype(np.float64),
        **{
            name: value.numpy().astype(np.float64)
            for name, value in stored.items()
        },
    }


def evaluate(
    data: dict[str, Any], slices: list[slice], scored: dict[str, np.ndarray]
) -> dict[str, Any]:
    labels = data["y_cum"].numpy().astype(np.int64)
    final_indices = np.asarray([part.stop - 1 for part in slices], dtype=np.int64)
    return {
        "local": predecessor.classification_metrics(labels, scored["scores"]),
        "final": predecessor.classification_metrics(
            labels[final_indices], scored["scores"][final_indices]
        ),
        "dynamic": predecessor.dynamic_metrics(data, slices, scored["scores"]),
    }


def finite_summary(values: np.ndarray) -> dict[str, Any]:
    values = values[np.isfinite(values)]
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def relation_diagnostics(
    data: dict[str, Any],
    slices: list[slice],
    scored: dict[str, np.ndarray],
) -> dict[str, Any]:
    labels = data["y_cum"].numpy().astype(np.int64)
    valid = np.isfinite(scored["gates"])
    mean_abs_delta = float(np.abs(scored["relation_deltas"][valid]).mean())
    mean_abs_base = float(np.abs(scored["base_logits"][valid]).mean())
    gate_change = np.abs(
        scored["gates"][valid] - scored["base_gates"][valid]
    )
    prefix_lengths = np.empty(len(labels), dtype=np.int64)
    for part in slices:
        prefix_lengths[part] = np.arange(1, part.stop - part.start + 1)
    length_bins = []
    for name, lower, upper in (
        ("1", 1, 1),
        ("2-4", 2, 4),
        ("5-8", 5, 8),
        ("9-16", 9, 16),
        (">=17", 17, None),
    ):
        selected = prefix_lengths >= lower
        if upper is not None:
            selected &= prefix_lengths <= upper
        y = labels[selected]
        row: dict[str, Any] = {
            "bin": name,
            "n": int(selected.sum()),
            "positive": int(y.sum()),
            "negative": int(y.size - y.sum()),
        }
        if len(np.unique(y)) == 2:
            row["metrics"] = predecessor.classification_metrics(
                y, scored["scores"][selected]
            )
        length_bins.append(row)
    transition_groups: dict[str, list[float]] = defaultdict(list)
    for part in slices:
        for index in range(part.start + 1, part.stop):
            transition_groups[
                f"{labels[index - 1]}->{labels[index]}"
            ].append(float(scored["gates"][index]))
    return {
        "gate": finite_summary(scored["gates"]),
        "base_gate_disabled_relation": finite_summary(scored["base_gates"]),
        "relation_delta": finite_summary(scored["relation_deltas"]),
        "mean_abs_delta": mean_abs_delta,
        "mean_abs_base_logit": mean_abs_base,
        "delta_to_base_magnitude_ratio": mean_abs_delta
        / max(mean_abs_base, EPSILON),
        "gate_change_vs_disabled_relation": finite_summary(gate_change),
        "product_norm": finite_summary(scored["product_norms"]),
        "absolute_difference_norm": finite_summary(
            scored["absolute_difference_norms"]
        ),
        "pre_normalization_norm": finite_summary(
            scored["pre_normalization_norms"]
        ),
        "transition_gate_mean": {
            name: {
                "n": len(values),
                "mean": float(np.mean(values)) if values else None,
            }
            for name, values in sorted(transition_groups.items())
        },
        "prefix_length_bins": length_bins,
    }


def assert_full_predecessor_alignment(
    model_key: str,
    scalar_root: Path,
) -> dict[str, Any]:
    scalar_summary = json.loads(
        (scalar_root / "results" / model_key / "metrics.json").read_text()
    )
    split_audits = {}
    for split in ("train", "val", "test"):
        learned, original, meta = predecessor.load_raw_pair(model_key, split)
        current = meta["audit"]
        archived = scalar_summary["input_audit"][split]
        for key in (
            "learned_path",
            "original_path",
            "learned_rows",
            "original_rows",
            "trajectories",
            "step_positive",
            "prefix_positive",
            "tokens",
            "dimension",
            "dtype",
            "paper_layer",
            "model_revision",
            "source_split_sha256",
        ):
            if current[key] != archived[key]:
                raise AssertionError(
                    f"{model_key}/{split}: predecessor mismatch {key}"
                )
        trajectory_indices = learned["trajectory_indices"].to(torch.int64)
        positions = learned["positions"].to(torch.int64)
        for part in predecessor.sequence_slices(trajectory_indices):
            if int(positions[part.start]) != 0:
                raise AssertionError(
                    f"{model_key}/{split}: trajectory does not reset position"
                )
            if len(positions[part]) > 1 and not bool(
                torch.all(positions[part][1:] > positions[part][:-1])
            ):
                raise AssertionError(
                    f"{model_key}/{split}: positions not increasing"
                )
        if split == "test":
            with np.load(
                scalar_root
                / "results"
                / model_key
                / "seed_42"
                / "test_predictions.npz"
            ) as archived_predictions:
                expected = {
                    "trajectory_index": trajectory_indices.numpy(),
                    "step_id": positions.numpy(),
                    "flat_step_index": meta["flat_indices"].numpy(),
                    "step_label": learned["labels"].numpy().astype(np.int64),
                    "prefix_label": original["cumulative_labels"][
                        meta["flat_indices"]
                    ]
                    .numpy()
                    .astype(np.int64),
                }
                for key, value in expected.items():
                    if not np.array_equal(archived_predictions[key], value):
                        raise AssertionError(
                            f"{model_key}/test archived {key} mismatch"
                        )
        split_audits[split] = current
    return {
        "status": "PASS",
        "scalar_seed42_metrics": str(
            scalar_root / "results" / model_key / "seed_42" / "metrics.json"
        ),
        "splits": split_audits,
    }


def smoke_one(
    model_key: str,
    output_dir: Path,
    device: torch.device,
    feature_batch_size: int,
    scalar_root: Path,
) -> None:
    predecessor_audit = assert_full_predecessor_alignment(model_key, scalar_root)
    predecessor.set_seed(42)
    frozen = predecessor.checkpoint_model(model_key, device)
    train = predecessor.build_split(
        model_key, "train", frozen, device, feature_batch_size, smoke=True
    )
    val = predecessor.build_split(
        model_key, "val", frozen, device, feature_batch_size, smoke=True
    )
    train_slices = predecessor.sequence_slices(train["trajectory_indices"])
    dimension = predecessor.MODEL_SPECS[model_key]["dimension"]

    initial_features = predecessor.initial_prefix_features(
        train["z"], train_slices, device
    )
    baseline = predecessor.train_baseline(
        initial_features, train["y_cum"], dimension, device
    )
    rng_state = torch.random.get_rng_state()
    model = RelationConditionedPrefixGate(dimension)
    post_model_rng_state = torch.random.get_rng_state()
    base_init = base_initialization_audit(
        model, rng_state, post_model_rng_state, dimension
    )
    # Recreate the archived object from the same RNG state for disabled-path
    # output comparison, then restore the already-recorded post-new-model RNG.
    torch.random.set_rng_state(rng_state)
    archived = predecessor.DynamicPrefixGate(dimension)
    torch.random.set_rng_state(post_model_rng_state)
    model.probe.load_state_dict(baseline.state_dict())
    archived.probe.load_state_dict(baseline.state_dict())
    model = model.to(device)
    archived = archived.to(device)

    first = train_slices[0]
    z = train["z"][first].to(device)
    disabled_logits, disabled_debug = model.forward_sequence(
        z, relation_enabled=False, return_debug=True
    )
    archived_logits, archived_gates, archived_norms = archived.forward_sequence(z)
    if not torch.equal(disabled_logits, archived_logits):
        raise AssertionError("disabled relation does not reproduce scalar logits")
    if not torch.allclose(
        disabled_debug["gates"],
        archived_gates,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    ):
        raise AssertionError("disabled relation does not reproduce scalar gates")
    if not torch.equal(
        disabled_debug["pre_normalization_norms"], archived_norms
    ):
        raise AssertionError("disabled relation does not reproduce scalar norms")
    if not torch.equal(disabled_debug["states"][0], z[0]):
        raise AssertionError("p_1 != z_1")

    identical_delta, identical = model.relation(z[0], z[0])
    perturbed_delta, perturbed = model.relation(z[0], z[-1] + 0.125)
    if identical["match"].shape[-1] != 512:
        raise AssertionError("match width is not 512")
    if torch.count_nonzero(identical["absolute_difference"]).item() != 0:
        raise AssertionError("identical pair has nonzero abs-difference block")
    if torch.count_nonzero(perturbed["absolute_difference"]).item() == 0:
        raise AssertionError("perturbed pair has zero abs-difference block")
    if not torch.isfinite(identical_delta) or not torch.isfinite(perturbed_delta):
        raise AssertionError("non-finite relation delta")
    projection_modules = [
        name
        for name, _ in model.named_modules()
        if "relation_projection" in name
    ]
    if projection_modules != ["relation_projection"]:
        raise AssertionError("relation projection is not uniquely shared")

    model.train()
    parts = train_slices
    logits, teacher, labels, lengths, debug_rows = forward_batch(
        model, train, parts, device, return_debug=True
    )
    loss, anchor, synchronization = prefix_loss(
        logits, teacher, labels, lengths
    )
    model.zero_grad(set_to_none=True)
    loss.backward()
    relation_gradient_norms = {}
    for name, parameter in model.named_parameters():
        if name.startswith("relation_"):
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError(f"missing/non-finite gradient: {name}")
            value = float(parameter.grad.norm().detach().cpu())
            if value <= 0:
                raise AssertionError(f"zero relation gradient: {name}")
            relation_gradient_norms[name] = value
    if any(parameter.grad is not None for parameter in frozen.parameters()):
        raise AssertionError("frozen Step parameters received gradients")
    gates = torch.cat([row["gates"][1:] for row in debug_rows])
    states = torch.cat([row["states"] for row in debug_rows])
    if not bool(torch.all((gates > 0) & (gates < 1))):
        raise AssertionError("gate outside (0,1)")
    if not torch.allclose(
        torch.linalg.vector_norm(states, dim=1),
        torch.ones(len(states), device=device),
        atol=2e-5,
        rtol=2e-5,
    ):
        raise AssertionError("Prefix states are not unit norm")
    if not all(
        torch.isfinite(value)
        for value in (loss, anchor, synchronization)
    ):
        raise AssertionError("non-finite smoke loss")

    val_slices = predecessor.sequence_slices(val["trajectory_indices"])
    model.eval()
    scored = score_model(model, val, val_slices, device)
    report = {
        "status": "PASS",
        "model": model_key,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0),
        "predecessor_audit": predecessor_audit,
        "base_initialization": base_init,
        "disabled_relation_exact_scalar_recovery": True,
        "trajectory_reset_and_p1_equals_z1": True,
        "match_width": 512,
        "identical_abs_difference_exact_zero": True,
        "perturbed_abs_difference_nonzero": True,
        "shared_projection_parameter": "relation_projection.weight",
        "separate_history_or_current_projection": False,
        "delta_destination": "scalar gate logit only",
        "sqrt_width_scaling": False,
        "cosine_or_global_distance": False,
        "relation_gradient_norms": relation_gradient_norms,
        "frozen_step_gradient_free": True,
        "standalone_answer_companion_loaded": False,
        "gates_strictly_inside_unit_interval": True,
        "prefix_states_unit_norm": True,
        "finite_outputs_and_loss": True,
        "val_metrics": evaluate(val, val_slices, scored),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"smoke_{model_key}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    emit("CONTENT_PREFIX_SMOKE_PASS", model=model_key)


def full_one(
    model_key: str,
    output_dir: Path,
    device: torch.device,
    feature_batch_size: int,
    scalar_root: Path,
) -> None:
    started = time.time()
    predecessor_audit = assert_full_predecessor_alignment(model_key, scalar_root)
    predecessor.set_seed(42)
    frozen = predecessor.checkpoint_model(model_key, device)
    splits = {
        name: predecessor.build_split(
            model_key,
            name,
            frozen,
            device,
            feature_batch_size,
            smoke=False,
        )
        for name in ("train", "val", "test")
    }
    slices = {
        name: predecessor.sequence_slices(data["trajectory_indices"])
        for name, data in splits.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "input_audit.json").write_text(
        json.dumps(predecessor_audit, indent=2) + "\n"
    )
    dimension = predecessor.MODEL_SPECS[model_key]["dimension"]

    # Exact predecessor seed/order: seed, half-update features, LBFGS baseline,
    # scalar base parameters first, then new relation parameters.
    predecessor.set_seed(42)
    initial = predecessor.initial_prefix_features(
        splits["train"]["z"], slices["train"], device
    )
    baseline = predecessor.train_baseline(
        initial, splits["train"]["y_cum"], dimension, device
    )
    model = RelationConditionedPrefixGate(dimension).to(device)
    model.probe.load_state_dict(baseline.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=predecessor.LR)
    best_auc, best_epoch, best_state = -math.inf, 0, None
    trace = []
    for epoch in range(1, predecessor.EPOCHS + 1):
        row = train_epoch(
            model,
            splits["train"],
            slices["train"],
            optimizer,
            device,
            seed=42 * 1000 + epoch,
        )
        val_scored = score_model(model, splits["val"], slices["val"], device)
        val_auc = predecessor.classification_metrics(
            splits["val"]["y_cum"].numpy().astype(np.int64),
            val_scored["scores"],
        )["auroc"]
        row.update({"epoch": epoch, "val_local_auroc": val_auc})
        trace.append(row)
        emit("EPOCH", model=model_key, seed=42, **row)
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = copy.deepcopy(
                {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                }
            )
    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best_state)
    val_scored = score_model(model, splits["val"], slices["val"], device)
    test_scored = score_model(model, splits["test"], slices["test"], device)
    validation = evaluate(splits["val"], slices["val"], val_scored)
    test = evaluate(splits["test"], slices["test"], test_scored)
    diagnostics = relation_diagnostics(
        splits["test"], slices["test"], test_scored
    )
    checkpoint = {
        "state_dict": best_state,
        "seed": 42,
        "best_epoch": best_epoch,
        "best_val_local_auroc": best_auc,
        "dimension": dimension,
        "relation_width": RELATION_WIDTH,
        "architecture": "shared W_R; [q*k, abs(q-k)]; MLP 512-256-256; LN; unscaled scalar delta",
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
        relation_gate_score=test_scored["scores"],
        gate_value=test_scored["gates"],
        disabled_relation_gate=test_scored["base_gates"],
        base_gate_logit=test_scored["base_logits"],
        relation_delta=test_scored["relation_deltas"],
        product_norm=test_scored["product_norms"],
        absolute_difference_norm=test_scored["absolute_difference_norms"],
        pre_normalization_norm=test_scored["pre_normalization_norms"],
        split=np.asarray(["test"] * len(test_scored["scores"])),
        model=np.asarray([model_key] * len(test_scored["scores"])),
        method=np.asarray(
            ["relation_conditioned_scalar_gate"] * len(test_scored["scores"])
        ),
    )
    relation_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("relation_")
    )
    record = {
        "model": model_key,
        "method": "relation_conditioned_scalar_gate",
        "seed": 42,
        "best_epoch": best_epoch,
        "best_val_local_auroc": best_auc,
        "validation": validation,
        "test": test,
        "relation_diagnostics": diagnostics,
        "trace": trace,
        "input_audit": predecessor_audit,
        "relation_width": RELATION_WIDTH,
        "relation_parameters": relation_parameters,
        "total_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "optimizer": {
            "name": "AdamW",
            "lr": predecessor.LR,
            "weight_decay": 0.01,
        },
        "frozen_step_parameters": sum(
            parameter.numel() for parameter in frozen.parameters()
        ),
        "decision_rule": "score > 0.5",
        "answer_companion": False,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "walltime_s": time.time() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    emit(
        "CONTENT_PREFIX_TEST_COMPLETE",
        model=model_key,
        best_epoch=best_epoch,
        test=test,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--model", choices=tuple(predecessor.MODEL_SPECS), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scalar-root", type=Path, required=True)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("scheduler-allocated CUDA is required")
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    emit(
        "START",
        mode=args.mode,
        model=args.model,
        device=str(device),
        torch=torch.__version__,
    )
    if args.mode == "smoke":
        smoke_one(
            args.model,
            args.output_dir,
            device,
            args.feature_batch_size,
            args.scalar_root,
        )
    else:
        full_one(
            args.model,
            args.output_dir,
            device,
            args.feature_batch_size,
            args.scalar_root,
        )


if __name__ == "__main__":
    main()
