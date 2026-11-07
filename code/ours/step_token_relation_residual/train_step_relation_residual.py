#!/usr/bin/env python3
"""Within-model Step detector with a zero-initialized Shaw relation residual.

The archived independent token scorer remains the primary scoring path.  The
only scientific addition is a one-head relative-position attention branch whose
zero-initialized scalar head corrects the independent token logits.  The
``--value-side-relation`` switch adds Shaw's Value-side embedding while keeping
the archived Key-only parameter initialization unchanged.
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
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, default=str), flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cache(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        value = torch.load(path, map_location="cpu", weights_only=False)
    required = {"tokens", "offsets", "labels", "trajectory_indices", "positions", "metadata"}
    missing = required.difference(value)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    if value["tokens"].ndim != 2:
        raise ValueError(f"tokens must be rank 2: {path}")
    if value["offsets"].numel() != value["labels"].numel() + 1:
        raise ValueError(f"offset/label mismatch: {path}")
    if int(value["offsets"][-1]) != int(value["tokens"].shape[0]):
        raise ValueError(f"offset/token mismatch: {path}")
    if torch.any(value["offsets"][1:] <= value["offsets"][:-1]):
        raise ValueError(f"empty or decreasing Step token span: {path}")
    return value


def array_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def cache_summary(cache: dict[str, Any], split: str) -> dict[str, Any]:
    labels = cache["labels"].to(torch.int64)
    offsets = cache["offsets"].to(torch.int64)
    lengths = offsets[1:] - offsets[:-1]
    trajectories = cache["trajectory_indices"].to(torch.int64)
    result = {
        "split": split,
        "steps": int(labels.numel()),
        "tokens": int(offsets[-1]),
        "positive": int(labels.sum()),
        "negative": int(labels.numel() - labels.sum()),
        "trajectories": int(torch.unique(trajectories).numel()),
        "dimension": int(cache["tokens"].shape[1]),
        "dtype": str(cache["tokens"].dtype),
        "min_step_tokens": int(lengths.min()),
        "max_step_tokens": int(lengths.max()),
        "mean_step_tokens": float(lengths.to(torch.float64).mean()),
        "label_sha256": array_hash(labels),
        "trajectory_sha256": array_hash(trajectories),
        "position_sha256": array_hash(cache["positions"].to(torch.int64)),
        "metadata": cache["metadata"],
    }
    for key in ("trajectory_ids", "step_ids"):
        if key in cache:
            value = cache[key]
            result[f"{key}_count"] = len(value)
    return result


def chunks(indices: torch.Tensor, size: int) -> Iterable[torch.Tensor]:
    for start in range(0, len(indices), size):
        yield indices[start : start + size]


class TokenRelationResidualStep(nn.Module):
    """Independent scorer plus zero-initialized Shaw Key-side correction."""

    def __init__(
        self,
        dimension: int,
        attention_dimension: int = 64,
        clipping_distance: int = 4,
        tau: float = 1.0,
        epsilon: float = 1e-12,
        value_side_relation: bool = False,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.attention_dimension = attention_dimension
        self.clipping_distance = clipping_distance
        self.tau = tau
        self.epsilon = epsilon
        self.value_side_relation = value_side_relation

        # This order exactly preserves seed-42 initialization of the archived
        # scorer and probe before any new module consumes RNG state.
        self.scorer = nn.Sequential(
            nn.Linear(dimension, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        self.probe = nn.Linear(dimension, 1)

        self.token_layer_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.q_proj = nn.Linear(dimension, attention_dimension, bias=False)
        self.k_proj = nn.Linear(dimension, attention_dimension, bias=False)
        self.v_proj = nn.Linear(dimension, attention_dimension, bias=False)
        self.rel_key = nn.Embedding(2 * clipping_distance + 1, attention_dimension)
        self.context_layer_norm = nn.LayerNorm(attention_dimension)
        self.rel_head = nn.Linear(attention_dimension, 1)
        nn.init.zeros_(self.rel_head.weight)
        nn.init.zeros_(self.rel_head.bias)
        # Instantiate the only new scientific component strictly after every
        # predecessor parameter so seed-42 initialization of the Key-only path
        # remains bit-exact.  E_V uses nn.Embedding's same scheme as E_K.
        self.rel_value = (
            nn.Embedding(2 * clipping_distance + 1, attention_dimension)
            if value_side_relation else None
        )

    def relation_indices(self, length: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(length, device=device)
        relative = positions[None, :] - positions[:, None]  # j - i
        return relative.clamp(-self.clipping_distance, self.clipping_distance) + self.clipping_distance

    def forward_padded(
        self,
        states: torch.Tensor,
        valid_mask: torch.Tensor,
        independent_logits: torch.Tensor,
        return_debug: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if states.ndim != 3 or valid_mask.ndim != 2 or independent_logits.ndim != 2:
            raise ValueError("expected states [B,L,D], mask [B,L], independent logits [B,L]")
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("every Step must contain at least one token")
        hbar = self.token_layer_norm(states)
        q = self.q_proj(hbar)
        k = self.k_proj(hbar)
        v = self.v_proj(hbar)
        length = states.shape[1]
        relation_indices = self.relation_indices(length, states.device)
        relative_keys = self.rel_key(relation_indices)
        content_logits = torch.matmul(q, k.transpose(-1, -2))
        relation_logits = torch.einsum("bid,ijd->bij", q, relative_keys)
        attention_logits = (content_logits + relation_logits) / math.sqrt(self.attention_dimension)
        attention_logits = attention_logits.masked_fill(~valid_mask[:, None, :], -torch.inf)
        attention = torch.softmax(attention_logits, dim=-1)
        content_context = torch.matmul(attention, v)
        if self.rel_value is not None:
            relative_values = self.rel_value(relation_indices)
            relative_value_context = torch.einsum("bij,ijd->bid", attention, relative_values)
            context = content_context + relative_value_context
        else:
            relative_values = torch.zeros((), dtype=states.dtype, device=states.device)
            relative_value_context = torch.zeros_like(content_context)
            context = content_context
        correction = self.rel_head(self.context_layer_norm(context)).squeeze(-1)
        new_logits = independent_logits + correction
        new_logits = new_logits.masked_fill(~valid_mask, -torch.inf)
        final_weights = torch.softmax(new_logits / self.tau, dim=-1)
        pooled = torch.sum(states * final_weights.unsqueeze(-1), dim=1)
        pooled = pooled / torch.linalg.vector_norm(pooled, dim=1, keepdim=True).clamp_min(self.epsilon)
        detector_logits = self.probe(pooled).flatten()
        debug: dict[str, torch.Tensor] = {}
        if return_debug:
            independent_masked = independent_logits.masked_fill(~valid_mask, -torch.inf)
            independent_weights = torch.softmax(independent_masked / self.tau, dim=-1)
            independent_pooled = torch.sum(states * independent_weights.unsqueeze(-1), dim=1)
            independent_pooled = independent_pooled / torch.linalg.vector_norm(
                independent_pooled, dim=1, keepdim=True
            ).clamp_min(self.epsilon)
            debug = {
                "attention": attention,
                "correction": correction,
                "new_logits": new_logits,
                "independent_logits": independent_masked,
                "final_weights": final_weights,
                "independent_weights": independent_weights,
                "pooled": pooled,
                "independent_pooled": independent_pooled,
                "independent_detector_logits": self.probe(independent_pooled).flatten(),
                "relation_indices": relation_indices,
                "relative_values": relative_values,
                "content_context": content_context,
                "relative_value_context": relative_value_context,
                "context": context,
            }
        return detector_logits, debug

    def forward_cache(
        self,
        cache: dict[str, Any],
        indices: torch.Tensor,
        device: torch.device,
        relation_microbatch_size: int,
        return_debug: bool = False,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], list[int]]:
        offsets = cache["offsets"]
        cpu_pieces: list[torch.Tensor] = []
        lengths: list[int] = []
        for index in indices.tolist():
            start, end = int(offsets[index]), int(offsets[index + 1])
            cpu_pieces.append(cache["tokens"][start:end])
            lengths.append(end - start)
        flat = torch.cat(cpu_pieces, dim=0).to(device=device, dtype=torch.float32, non_blocking=True)
        # BatchNorm sees exactly the same logical batch and token order as the
        # archived independent scorer; relation microbatching is downstream.
        independent_flat = self.scorer(flat).flatten()
        pieces = list(flat.split(lengths))
        independent_pieces = list(independent_flat.split(lengths))
        outputs: list[torch.Tensor] = []
        debug_rows: list[dict[str, torch.Tensor]] = []
        for start in range(0, len(pieces), relation_microbatch_size):
            group_states = pieces[start : start + relation_microbatch_size]
            group_independent = independent_pieces[start : start + relation_microbatch_size]
            group_lengths = lengths[start : start + relation_microbatch_size]
            max_length = max(group_lengths)
            states = torch.zeros(
                (len(group_states), max_length, self.dimension), dtype=torch.float32, device=device
            )
            mask = torch.zeros((len(group_states), max_length), dtype=torch.bool, device=device)
            independent = torch.zeros((len(group_states), max_length), dtype=torch.float32, device=device)
            for row, (piece, scores, piece_length) in enumerate(
                zip(group_states, group_independent, group_lengths, strict=True)
            ):
                states[row, :piece_length] = piece
                independent[row, :piece_length] = scores
                mask[row, :piece_length] = True
            logits, debug = self.forward_padded(states, mask, independent, return_debug=return_debug)
            outputs.append(logits)
            if return_debug:
                debug["valid_mask"] = mask
                debug_rows.append(debug)
        return torch.cat(outputs), debug_rows, lengths


def score_split(
    model: TokenRelationResidualStep,
    cache: dict[str, Any],
    device: torch.device,
    logical_batch_size: int,
    relation_microbatch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_scores: list[torch.Tensor] = []
    all_entropy: list[torch.Tensor] = []
    all_lengths: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(cache["labels"]), logical_batch_size):
            indices = torch.arange(start, min(start + logical_batch_size, len(cache["labels"])))
            logits, debug_rows, lengths = model.forward_cache(
                cache, indices, device, relation_microbatch_size, return_debug=True
            )
            all_scores.append(torch.sigmoid(logits).cpu())
            for debug in debug_rows:
                weights = debug["final_weights"]
                entropy = -torch.sum(weights * torch.log(weights.clamp_min(model.epsilon)), dim=1)
                all_entropy.append(entropy.cpu())
            all_lengths.extend(lengths)
    return (
        torch.cat(all_scores).numpy().astype(np.float64),
        torch.cat(all_entropy).numpy().astype(np.float64),
        np.asarray(all_lengths, dtype=np.int64),
    )


def calculate_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    prediction = (scores > 0.5).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "threshold": 0.5,
        "decision_rule": "score > 0.5",
        "n": int(labels.size),
        "positive": int(labels.sum()),
        "negative": int(labels.size - labels.sum()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
    }


def parameter_gradient(model: nn.Module, name: str) -> float:
    parameter = dict(model.named_parameters())[name]
    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
        raise AssertionError(f"missing/non-finite gradient: {name}")
    return float(parameter.grad.norm().detach().cpu())


def verify_archived_initialization(model: TokenRelationResidualStep, seed: int, dimension: int) -> dict[str, Any]:
    """Confirm new modules did not perturb archived scorer/probe initialization."""
    set_seed(seed)
    scorer = nn.Sequential(
        nn.Linear(dimension, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 1),
    ).to(next(model.parameters()).device)
    probe = nn.Linear(dimension, 1).to(next(model.parameters()).device)
    checks: dict[str, bool] = {}
    for key, value in scorer.state_dict().items():
        checks[f"scorer.{key}"] = bool(torch.equal(value, model.scorer.state_dict()[key]))
    for key, value in probe.state_dict().items():
        checks[f"probe.{key}"] = bool(torch.equal(value, model.probe.state_dict()[key]))
    if not all(checks.values()):
        raise AssertionError("new relation modules changed archived seed initialization")
    return {"status": "PASS", "seed": seed, "all_scorer_probe_tensors_exact": True}


def verify_key_only_parameter_initialization(
    model: TokenRelationResidualStep, seed: int, dimension: int
) -> dict[str, Any]:
    """Verify E_V was added after, and did not perturb, every Key-only tensor."""
    set_seed(seed)
    reference = TokenRelationResidualStep(
        dimension=dimension,
        attention_dimension=model.attention_dimension,
        clipping_distance=model.clipping_distance,
        tau=model.tau,
        epsilon=model.epsilon,
        value_side_relation=False,
    ).to(next(model.parameters()).device)
    shared = reference.state_dict()
    current = model.state_dict()
    checks = {name: bool(torch.equal(value, current[name])) for name, value in shared.items()}
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"E_V perturbed Key-only initialization: {failed}")
    if model.rel_value is None or tuple(model.rel_value.weight.shape) != (9, 64):
        raise AssertionError("Value-side embedding must have shape [9,64]")
    return {
        "status": "PASS",
        "all_predecessor_tensors_bit_exact": True,
        "added_parameter_count": int(model.rel_value.weight.numel()),
        "added_shape": list(model.rel_value.weight.shape),
        "initialization_scheme": "torch.nn.Embedding default, identical to E_K",
    }


def run_smoke(
    model: TokenRelationResidualStep,
    train: dict[str, Any],
    val: dict[str, Any],
    test: dict[str, Any],
    baseline_scores: Path,
    key_only_metrics: Path | None,
    device: torch.device,
    expected: dict[str, int],
    relation_microbatch_size: int,
) -> dict[str, Any]:
    initialization_audit = verify_archived_initialization(model, 42, expected["dimension"])
    key_only_initialization_audit = verify_key_only_parameter_initialization(
        model, 42, expected["dimension"]
    ) if model.value_side_relation else None
    summaries = {name: cache_summary(cache, name) for name, cache in (("train", train), ("val", val), ("test", test))}
    predecessor_summary_exact = None
    if key_only_metrics is not None:
        predecessor = json.loads(key_only_metrics.read_text())
        predecessor_summaries = predecessor["summaries"]
        for split in ("train", "val", "test"):
            if json.dumps(summaries[split], sort_keys=True, default=str) != json.dumps(
                predecessor_summaries[split], sort_keys=True, default=str
            ):
                raise AssertionError(f"{split} cache summary/hash differs from Key-only predecessor")
            metadata = summaries[split]["metadata"]
            if "bag_boundary" in metadata and metadata["bag_boundary"] != "current Step only":
                raise AssertionError(f"{split} is not a current-Step-only cache")
            if "future_offset_violations" in metadata and int(metadata["future_offset_violations"]) != 0:
                raise AssertionError(f"{split} contains a future-offset violation")
            if int(metadata.get("eligible_steps", summaries[split]["steps"])) != summaries[split]["steps"]:
                raise AssertionError(f"{split} eligible-Step metadata mismatch")
        predecessor_summary_exact = True
    for key in ("steps", "trajectories", "positive", "dimension"):
        actual = summaries["test"][key]
        if int(actual) != int(expected[key]):
            raise AssertionError(f"test {key}: {actual} != {expected[key]}")
    with np.load(baseline_scores) as archived:
        if not np.array_equal(test["labels"].numpy(), archived["labels"]):
            raise AssertionError("test labels differ from archived independent scorer")
        for key in ("trajectory_indices", "positions"):
            if key not in archived or not np.array_equal(test[key].numpy(), archived[key]):
                raise AssertionError(f"test {key} differs from archived independent scorer")

    lengths = train["offsets"][1:] - train["offsets"][:-1]
    shortest = int(torch.argmin(lengths))
    longest = int(torch.argmax(lengths))
    different_candidates = torch.nonzero(lengths != lengths[shortest], as_tuple=False)
    different = int(different_candidates[0]) if len(different_candidates) else longest
    indices = torch.tensor([shortest, different, longest], dtype=torch.int64)

    # At initialization the relation residual is exactly zero, and the entire
    # detector output exactly recovers this model's independent path.
    model.eval()
    init_logits, init_debug_rows, selected_lengths = model.forward_cache(
        train, indices, device, relation_microbatch_size, return_debug=True
    )
    independent_logits = torch.cat([row["independent_detector_logits"] for row in init_debug_rows])
    max_correction = max(
        float(row["correction"][row["valid_mask"]].abs().max().detach().cpu())
        for row in init_debug_rows
    )
    max_logit_error = float((init_logits - independent_logits).abs().max().detach().cpu())
    if max_correction != 0.0 or max_logit_error > 1e-7:
        raise AssertionError("zero-initialized relation branch does not recover independent path")

    for row in init_debug_rows:
        mask = row["valid_mask"]
        attention = row["attention"]
        weights = row["final_weights"]
        if not torch.allclose(attention.sum(-1)[mask], torch.ones_like(attention.sum(-1)[mask]), atol=1e-5):
            raise AssertionError("valid attention rows do not sum to one")
        if torch.count_nonzero(attention.masked_select((~mask)[:, None, :])).item() != 0:
            raise AssertionError("padded keys receive nonzero attention")
        if model.value_side_relation:
            # Explicitly reconstruct the Shaw Value message.  The only valid
            # formulation is sum_j A_ij * (V_j + E_V[r_ij]).
            reconstructed = row["content_context"] + torch.einsum(
                "bij,ijd->bid", attention, row["relative_values"]
            )
            if not torch.allclose(row["context"], reconstructed, atol=1e-7, rtol=1e-6):
                raise AssertionError("E_V is not inside the attention-weighted value message")
            padded_value_contribution = torch.einsum(
                "bij,ijd->bid",
                attention.masked_fill(mask[:, None, :], 0.0),
                row["relative_values"],
            )
            if torch.count_nonzero(padded_value_contribution).item() != 0:
                raise AssertionError("padded keys contribute to E_V message")
        if torch.count_nonzero(weights[~mask]).item() != 0:
            raise AssertionError("padded tokens receive nonzero final pooling weight")
        if not torch.allclose(weights.sum(-1), torch.ones(weights.shape[0], device=device), atol=1e-6):
            raise AssertionError("final token weights do not sum to one")
        relation = row["relation_indices"] - model.clipping_distance
        raw = torch.arange(relation.shape[0], device=device)[None, :] - torch.arange(relation.shape[0], device=device)[:, None]
        if not torch.equal(relation, raw.clamp(-model.clipping_distance, model.clipping_distance)):
            raise AssertionError("relative labels are not clip(j-i,-k,k)")

    # First backward: zero w_rel mathematically blocks Q/K/V/E gradients.  Take
    # one ordinary optimizer step so w_rel becomes nonzero, then verify the next
    # backward reaches every relation parameter.  Production reinitializes.
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    optimizer.zero_grad(set_to_none=True)
    logits, _, _ = model.forward_cache(train, indices, device, relation_microbatch_size)
    labels = train["labels"][indices].to(device=device, dtype=torch.float32)
    loss_first = nn.BCEWithLogitsLoss()(logits, labels)
    loss_first.backward()
    relation_gradient_names = [
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "rel_key.weight"
    ]
    if model.rel_value is not None:
        relation_gradient_names.append("rel_value.weight")
    first_gradient_names = ["scorer.0.weight", "probe.weight", *relation_gradient_names,
                            "rel_head.weight", "rel_head.bias"]
    first_gradients = {
        name: parameter_gradient(model, name)
        for name in first_gradient_names
    }
    if first_gradients["rel_head.weight"] <= 0 or first_gradients["scorer.0.weight"] <= 0:
        raise AssertionError("first backward did not reach main path and relation head")
    for name in relation_gradient_names:
        if first_gradients[name] > 1e-12:
            raise AssertionError(f"{name} should be blocked by zero relation head on first backward")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    logits_second, _, _ = model.forward_cache(train, indices, device, relation_microbatch_size)
    loss_second = nn.BCEWithLogitsLoss()(logits_second, labels)
    loss_second.backward()
    second_gradient_names = [*relation_gradient_names, "rel_head.weight", "rel_head.bias"]
    second_gradients = {
        name: parameter_gradient(model, name)
        for name in second_gradient_names
    }
    for name in [*relation_gradient_names, "rel_head.weight"]:
        if second_gradients[name] <= 0:
            raise AssertionError(f"second backward has zero relation gradient: {name}")
    if abs(second_gradients["rel_head.bias"]) > 1e-5:
        raise AssertionError("shared relation bias should be unidentifiable through token softmax")
    if not torch.isfinite(loss_first) or not torch.isfinite(loss_second):
        raise AssertionError("non-finite smoke loss")

    result = {
        "status": "PASS",
        "archived_initialization_audit": initialization_audit,
        "key_only_initialization_audit": key_only_initialization_audit,
        "predecessor_cache_summaries_and_hashes_exact": predecessor_summary_exact,
        "summaries": summaries,
        "selected_lengths": selected_lengths,
        "initial_max_relation_correction": max_correction,
        "initial_max_detector_logit_error_vs_independent_path": max_logit_error,
        "first_backward_gradient_norms": first_gradients,
        "second_backward_after_one_ordinary_update_gradient_norms": second_gradients,
        "gradient_gate_explanation": "At exact zero initialization, w_rel blocks gradients to Q/K/V/E_K/E_V. After the first ordinary optimizer update makes w_rel nonzero, the next backward reaches all relation parameters.",
        "value_message_rule": "C_i=sum_j A_ij*(V_j+E_V[clip(j-i,-4,4)]); padded keys have exactly zero A_ij contribution",
        "relation_bias_identifiability": "b_rel is shared across tokens and cancels under within-Step softmax; its gradient is expected to be numerical zero.",
        "hidden_states_requires_grad": False,
        "relative_label_rule": "clip(j-i,-4,4), directional, generated from unpadded local positions",
        "answer_boundary": "canonical released current-Step cache only; no answer companion or answer-level label is loaded",
        "all_losses_finite": True,
    }
    model.zero_grad(set_to_none=True)
    emit("TOKEN_RELATION_RESIDUAL_SMOKE_PASS", **result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=("qwen", "llama", "deepseek"))
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--baseline-scores", type=Path, required=True)
    parser.add_argument("--key-only-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--paper-layer", type=int, required=True)
    parser.add_argument("--hidden-states-index", type=int, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--expected-test-steps", type=int, required=True)
    parser.add_argument("--expected-test-trajectories", type=int, required=True)
    parser.add_argument("--expected-test-positive", type=int, required=True)
    parser.add_argument("--attention-dimension", type=int, default=64)
    parser.add_argument("--clipping-distance", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--relation-microbatch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--early-patience", type=int, default=5)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--value-side-relation", action="store_true")
    args = parser.parse_args()

    started = time.time()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("semantic smoke and production both require scheduler-allocated CUDA")
    train, val, test = load_cache(args.train), load_cache(args.val), load_cache(args.test)
    summaries: dict[str, Any] = {}
    for split, cache in (("train", train), ("val", val), ("test", test)):
        summary = cache_summary(cache, split)
        if summary["dimension"] != args.dimension:
            raise ValueError(f"{split} dimension {summary['dimension']} != {args.dimension}")
        summaries[split] = summary
        emit("CACHE_READY", **summary)

    model = TokenRelationResidualStep(
        args.dimension, args.attention_dimension, args.clipping_distance, args.tau, args.epsilon,
        args.value_side_relation,
    ).to(device)
    torch.cuda.reset_peak_memory_stats()
    scorer_parameters = sum(p.numel() for p in model.scorer.parameters())
    probe_parameters = sum(p.numel() for p in model.probe.parameters())
    relation_parameters = sum(
        p.numel()
        for name, p in model.named_parameters()
        if not name.startswith("scorer.") and not name.startswith("probe.")
    )
    smoke = run_smoke(
        model, train, val, test, args.baseline_scores, args.key_only_metrics, device,
        {"steps": args.expected_test_steps, "trajectories": args.expected_test_trajectories,
         "positive": args.expected_test_positive, "dimension": args.dimension},
        args.relation_microbatch_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "status": "PASS", "model": args.model_key, "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0), "smoke": smoke,
        "scorer_parameters": scorer_parameters, "probe_parameters": probe_parameters,
        "relation_parameters": relation_parameters,
        "architecture": "archived independent MLP logits + zero-init Linear(LN(sum_j A_ij*(V_j+E_V[r_ij])))); Shaw Key+Value E_K/E_V[clip(j-i,-4,4)], d_a=64; weighted original h; L2; Linear(D,1)",
        "added_value_parameters": 576 if args.value_side_relation else 0,
        "mask": "bidirectional within completed current Step; padding only; no triangular mask",
        "decision_rule": "score > 0.5",
    }
    (args.output_dir / "smoke_audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")
    if args.smoke_only:
        emit("SMOKE_JOB_COMPLETE", output=str(args.output_dir / "smoke_audit.json"))
        return

    del model
    torch.cuda.empty_cache()
    set_seed(args.seed)
    model = TokenRelationResidualStep(
        args.dimension, args.attention_dimension, args.clipping_distance, args.tau, args.epsilon,
        args.value_side_relation,
    ).to(device)
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=20, min_lr=1e-6
    )
    criterion = nn.BCEWithLogitsLoss()
    train_generator = torch.Generator().manual_seed(args.seed)
    val_labels = val["labels"].numpy().astype(np.int64)
    best_auc, best_epoch, best_state = -math.inf, 0, None
    epochs_without_gain = 0
    trace: list[dict[str, Any]] = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(train["labels"]), generator=train_generator)
        loss_sum, seen = 0.0, 0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            logits, _, _ = model.forward_cache(
                train, indices, device, args.relation_microbatch_size
            )
            labels = train["labels"][indices].to(device=device, dtype=torch.float32)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)
        train_loss = loss_sum / seen
        scheduler.step(train_loss)
        val_scores, _, _ = score_split(
            model, val, device, args.eval_batch_size, args.relation_microbatch_size
        )
        val_auc = float(roc_auc_score(val_labels, val_scores))
        row = {"epoch": epoch, "train_loss": train_loss, "val_auroc": val_auc,
               "lr": optimizer.param_groups[0]["lr"], "elapsed_s": time.time() - started}
        trace.append(row)
        emit("EPOCH", **row)
        if val_auc > best_auc + 1e-7:
            best_auc, best_epoch = val_auc, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
        if epochs_without_gain >= args.early_patience:
            emit("EARLY_STOP", epoch=epoch, patience=args.early_patience)
            break
    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best_state)
    test_scores, test_entropy, test_lengths = score_split(
        model, test, device, args.eval_batch_size, args.relation_microbatch_size
    )
    test_labels = test["labels"].numpy().astype(np.int64)
    test_metrics = calculate_metrics(test_labels, test_scores)
    checkpoint = args.output_dir / "checkpoint.pt"
    torch.save({"state_dict": best_state, "config": vars(args), "best_epoch": best_epoch,
                "best_val_auroc": best_auc}, checkpoint)
    np.savez_compressed(
        args.output_dir / "test_scores.npz", scores=test_scores, labels=test_labels,
        token_weight_entropy=test_entropy, token_lengths=test_lengths,
        trajectory_indices=test["trajectory_indices"].numpy(), positions=test["positions"].numpy(),
    )
    result = {
        "method": "step_token_relation_residual_key_value_da64_k4" if args.value_side_relation else "step_token_relation_residual_key_only_da64_k4", "model": args.model_key,
        "model_id": args.model_id, "model_revision": args.model_revision, "seed": args.seed,
        "paper_layer": args.paper_layer, "hf_hidden_states_index": args.hidden_states_index,
        "hidden_dimension": args.dimension, "attention_dimension": args.attention_dimension,
        "clipping_distance": args.clipping_distance, "key_side_only": not args.value_side_relation,
        "causal_triangular_mask": False, "value_side_relation": args.value_side_relation,
        "added_value_parameters": 576 if args.value_side_relation else 0,
        "scorer_parameters": scorer_parameters, "probe_parameters": probe_parameters,
        "relation_parameters": relation_parameters,
        "optimizer": {"name": "Adam", "lr": 1e-3, "weight_decay": 5e-4},
        "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.1, "patience": 20,
                      "min_lr": 1e-6, "signal": "train_epoch_loss"},
        "loss": "unweighted BCEWithLogitsLoss", "logical_batch_size": args.batch_size,
        "eval_logical_batch_size": args.eval_batch_size,
        "relation_microbatch_size_implementation_only": args.relation_microbatch_size,
        "best_epoch": best_epoch, "best_val_auroc": best_auc, "test": test_metrics,
        "summaries": summaries, "smoke": smoke, "trace": trace,
        "walltime_s": time.time() - started,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint": str(checkpoint),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    emit("TEST_COMPLETE", test=test_metrics, best_epoch=best_epoch,
         peak_memory_bytes=result["cuda_peak_memory_bytes"])


if __name__ == "__main__":
    main()
