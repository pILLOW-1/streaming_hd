#!/usr/bin/env python3
"""CHORD-style content-relation residual for within-model Step detection."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

try:
    from train_step_relation_residual import (
        array_hash,
        cache_summary,
        calculate_metrics,
        chunks,
        load_cache,
        set_seed,
    )
except ModuleNotFoundError:
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "step_token_relation_residual")
    )
    from train_step_relation_residual import (  # type: ignore[no-redef]
        array_hash,
        cache_summary,
        calculate_metrics,
        chunks,
        load_cache,
        set_seed,
    )


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, default=str), flush=True)


class ContentRelationResidualStep(nn.Module):
    """Independent token scorer plus a gamma-gated content-relation branch."""

    def __init__(
        self,
        dimension: int,
        attention_dimension: int = 64,
        clipping_distance: int = 4,
        position_dimension: int = 16,
        query_chunk_size: int = 64,
        tau: float = 1.0,
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.attention_dimension = attention_dimension
        self.clipping_distance = clipping_distance
        self.position_dimension = position_dimension
        self.query_chunk_size = query_chunk_size
        self.tau = tau
        self.epsilon = epsilon

        # Base path is instantiated first and exactly matches the archived
        # Independent seed-42 scorer/probe order.
        self.scorer = nn.Sequential(
            nn.Linear(dimension, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        self.probe = nn.Linear(dimension, 1)

        # All content-relation parameters are instantiated only after the base.
        self.token_layer_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.relation_proj = nn.Linear(dimension, attention_dimension, bias=False)
        self.q_proj = nn.Linear(dimension, attention_dimension, bias=False)
        self.k_proj = nn.Linear(dimension, attention_dimension, bias=False)
        self.v_proj = nn.Linear(dimension, attention_dimension, bias=False)
        self.position_embedding = nn.Embedding(
            2 * clipping_distance + 1, position_dimension
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(position_dimension + 2, attention_dimension),
            nn.GELU(),
            nn.Linear(attention_dimension, attention_dimension),
        )
        self.edge_key_proj = nn.Linear(
            attention_dimension, attention_dimension, bias=False
        )
        self.edge_value_proj = nn.Linear(
            attention_dimension, attention_dimension, bias=False
        )
        self.context_layer_norm = nn.LayerNorm(attention_dimension)
        self.rel_head = nn.Linear(attention_dimension, 1)
        self.gamma = nn.Parameter(torch.zeros(()))

    def relation_indices(
        self, query_start: int, query_end: int, length: int, device: torch.device
    ) -> torch.Tensor:
        query = torch.arange(query_start, query_end, device=device)[:, None]
        key = torch.arange(length, device=device)[None, :]
        return (key - query).clamp(
            -self.clipping_distance, self.clipping_distance
        ) + self.clipping_distance

    def _content_context(
        self,
        hbar: torch.Tensor,
        valid_mask: torch.Tensor,
        query_chunk_size: int,
        pair_debug: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.relation_proj(hbar)
        q = self.q_proj(hbar)
        k = self.k_proj(hbar)
        v = self.v_proj(hbar)
        batch, length, _ = hbar.shape
        contexts: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []
        debug_edges: dict[str, torch.Tensor] = {}

        # Unit vectors are used only inside cosine. Euclidean distance retains
        # the raw projected x, as frozen by the contract.
        x_norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(
            self.epsilon
        )
        x_cosine = x / x_norm

        for query_start in range(0, length, query_chunk_size):
            query_end = min(query_start + query_chunk_size, length)
            relation_indices = self.relation_indices(
                query_start, query_end, length, hbar.device
            )
            position = self.position_embedding(relation_indices)
            cosine = torch.matmul(
                x_cosine[:, query_start:query_end], x_cosine.transpose(-1, -2)
            )
            difference = (
                x[:, query_start:query_end, None, :] - x[:, None, :, :]
            )
            distance = torch.log1p(torch.linalg.vector_norm(difference, dim=-1))
            position_batched = position.unsqueeze(0).expand(batch, -1, -1, -1)
            relation_features = torch.cat(
                (position_batched, cosine.unsqueeze(-1), distance.unsqueeze(-1)),
                dim=-1,
            )
            edge = self.edge_mlp(relation_features)
            edge_key = self.edge_key_proj(edge)
            edge_value = self.edge_value_proj(edge)
            content_logits = torch.einsum(
                "bqd,bkd->bqk", q[:, query_start:query_end], k
            )
            edge_logits = torch.einsum(
                "bqd,bqkd->bqk", q[:, query_start:query_end], edge_key
            )
            logits = (content_logits + edge_logits) / math.sqrt(
                self.attention_dimension
            )
            logits = logits.masked_fill(~valid_mask[:, None, :], -torch.inf)
            attention = torch.softmax(logits, dim=-1)
            content_message = torch.matmul(attention, v)
            edge_message = torch.einsum("bqk,bqkd->bqd", attention, edge_value)
            contexts.append(content_message + edge_message)
            if pair_debug:
                attentions.append(attention)
                if not debug_edges:
                    debug_edges = {
                        "relation_indices": relation_indices,
                        "position": position,
                        "cosine": cosine,
                        "distance": distance,
                        "relation_features": relation_features,
                        "edge": edge,
                        "edge_key": edge_key,
                        "edge_value": edge_value,
                        "content_message": content_message,
                        "edge_message": edge_message,
                    }
        context = torch.cat(contexts, dim=1)
        debug: dict[str, torch.Tensor] = {}
        if pair_debug:
            debug = {**debug_edges, "attention": torch.cat(attentions, dim=1), "x": x}
        return context, debug

    def forward_padded(
        self,
        states: torch.Tensor,
        valid_mask: torch.Tensor,
        independent_logits: torch.Tensor,
        return_debug: bool = False,
        pair_debug: bool = False,
        query_chunk_size_override: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("every Step must contain at least one token")
        chunk_size = query_chunk_size_override or self.query_chunk_size
        hbar = self.token_layer_norm(states)
        context, pair = self._content_context(
            hbar, valid_mask, chunk_size, pair_debug
        )
        delta = self.rel_head(self.context_layer_norm(context)).squeeze(-1)
        delta = delta.masked_fill(~valid_mask, 0.0)
        correction = self.gamma * delta
        new_logits = independent_logits + correction
        new_logits = new_logits.masked_fill(~valid_mask, -torch.inf)
        final_weights = torch.softmax(new_logits / self.tau, dim=-1)
        pooled = torch.sum(states * final_weights.unsqueeze(-1), dim=1)
        pooled = pooled / torch.linalg.vector_norm(
            pooled, dim=1, keepdim=True
        ).clamp_min(self.epsilon)
        detector_logits = self.probe(pooled).flatten()
        debug: dict[str, torch.Tensor] = {}
        if return_debug:
            independent_masked = independent_logits.masked_fill(
                ~valid_mask, -torch.inf
            )
            independent_weights = torch.softmax(independent_masked / self.tau, dim=-1)
            independent_pooled = torch.sum(
                states * independent_weights.unsqueeze(-1), dim=1
            )
            independent_pooled = independent_pooled / torch.linalg.vector_norm(
                independent_pooled, dim=1, keepdim=True
            ).clamp_min(self.epsilon)
            debug = {
                "valid_mask": valid_mask,
                "delta": delta,
                "correction": correction,
                "new_logits": new_logits,
                "independent_logits": independent_masked,
                "final_weights": final_weights,
                "independent_weights": independent_weights,
                "pooled": pooled,
                "independent_pooled": independent_pooled,
                "independent_detector_logits": self.probe(independent_pooled).flatten(),
                "context": context,
                **pair,
            }
        return detector_logits, debug

    def forward_cache(
        self,
        cache: dict[str, Any],
        indices: torch.Tensor,
        device: torch.device,
        relation_microbatch_size: int,
        return_debug: bool = False,
        pair_debug: bool = False,
        query_chunk_size_override: int | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], list[int]]:
        offsets = cache["offsets"]
        pieces_cpu: list[torch.Tensor] = []
        lengths: list[int] = []
        for index in indices.tolist():
            start, end = int(offsets[index]), int(offsets[index + 1])
            pieces_cpu.append(cache["tokens"][start:end])
            lengths.append(end - start)
        flat = torch.cat(pieces_cpu, dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        # Preserve the predecessor's exact logical BatchNorm population/order.
        independent_flat = self.scorer(flat).flatten()
        pieces = list(flat.split(lengths))
        independent_pieces = list(independent_flat.split(lengths))
        outputs: list[torch.Tensor] = []
        debug_rows: list[dict[str, torch.Tensor]] = []
        for start in range(0, len(pieces), relation_microbatch_size):
            group_states = pieces[start : start + relation_microbatch_size]
            group_independent = independent_pieces[
                start : start + relation_microbatch_size
            ]
            group_lengths = lengths[start : start + relation_microbatch_size]
            max_length = max(group_lengths)
            states = torch.zeros(
                (len(group_states), max_length, self.dimension),
                dtype=torch.float32,
                device=device,
            )
            mask = torch.zeros(
                (len(group_states), max_length), dtype=torch.bool, device=device
            )
            independent = torch.zeros(
                (len(group_states), max_length), dtype=torch.float32, device=device
            )
            for row, (piece, logits, length) in enumerate(
                zip(group_states, group_independent, group_lengths, strict=True)
            ):
                states[row, :length] = piece
                independent[row, :length] = logits
                mask[row, :length] = True
            logits, debug = self.forward_padded(
                states,
                mask,
                independent,
                return_debug=return_debug,
                pair_debug=pair_debug,
                query_chunk_size_override=query_chunk_size_override,
            )
            outputs.append(logits)
            if return_debug:
                debug_rows.append(debug)
        return torch.cat(outputs), debug_rows, lengths


def score_split(
    model: ContentRelationResidualStep,
    cache: dict[str, Any],
    device: torch.device,
    logical_batch_size: int,
    relation_microbatch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    scores: list[torch.Tensor] = []
    entropy: list[torch.Tensor] = []
    correction: list[torch.Tensor] = []
    independent_magnitude: list[torch.Tensor] = []
    lengths_all: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(cache["labels"]), logical_batch_size):
            indices = torch.arange(
                start, min(start + logical_batch_size, len(cache["labels"]))
            )
            logits, rows, lengths = model.forward_cache(
                cache,
                indices,
                device,
                relation_microbatch_size,
                return_debug=True,
            )
            scores.append(torch.sigmoid(logits).cpu())
            for row in rows:
                mask = row["valid_mask"]
                weights = row["final_weights"]
                entropy.append(
                    (-torch.sum(weights * torch.log(weights.clamp_min(model.epsilon)), dim=1)).cpu()
                )
                correction.append(
                    (row["correction"].abs().sum(dim=1) / mask.sum(dim=1)).cpu()
                )
                independent_magnitude.append(
                    (
                        row["independent_logits"].masked_fill(~mask, 0.0).abs().sum(dim=1)
                        / mask.sum(dim=1)
                    ).cpu()
                )
            lengths_all.extend(lengths)
    return torch.cat(scores).numpy().astype(np.float64), {
        "token_weight_entropy": torch.cat(entropy).numpy().astype(np.float64),
        "mean_abs_gamma_delta": torch.cat(correction).numpy().astype(np.float64),
        "mean_abs_independent_logits": torch.cat(independent_magnitude).numpy().astype(np.float64),
        "token_lengths": np.asarray(lengths_all, dtype=np.int64),
    }


def gradient_norm(model: nn.Module, name: str) -> float:
    parameter = dict(model.named_parameters())[name]
    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
        raise AssertionError(f"missing/non-finite gradient: {name}")
    return float(parameter.grad.norm().detach().cpu())


def verify_base_initialization(
    model: ContentRelationResidualStep, seed: int, dimension: int
) -> dict[str, Any]:
    set_seed(seed)
    scorer = nn.Sequential(
        nn.Linear(dimension, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    ).to(next(model.parameters()).device)
    probe = nn.Linear(dimension, 1).to(next(model.parameters()).device)
    checks = {
        f"scorer.{key}": bool(torch.equal(value, model.scorer.state_dict()[key]))
        for key, value in scorer.state_dict().items()
    }
    checks.update(
        {
            f"probe.{key}": bool(torch.equal(value, model.probe.state_dict()[key]))
            for key, value in probe.state_dict().items()
        }
    )
    if not all(checks.values()):
        raise AssertionError("content branch perturbed Independent initialization")
    return {"status": "PASS", "all_base_tensors_bit_exact": True, "seed": seed}


def run_smoke(
    model: ContentRelationResidualStep,
    train: dict[str, Any],
    val: dict[str, Any],
    test: dict[str, Any],
    baseline_scores: Path,
    position_scores: Path,
    position_metrics: Path,
    device: torch.device,
    expected: dict[str, int],
    relation_microbatch_size: int,
) -> dict[str, Any]:
    base_initialization = verify_base_initialization(model, 42, expected["dimension"])
    summaries = {
        split: cache_summary(cache, split)
        for split, cache in (("train", train), ("val", val), ("test", test))
    }
    predecessor = json.loads(position_metrics.read_text())
    for split in ("train", "val", "test"):
        if json.dumps(summaries[split], sort_keys=True, default=str) != json.dumps(
            predecessor["summaries"][split], sort_keys=True, default=str
        ):
            raise AssertionError(f"{split} differs from position Key+Value predecessor")
        metadata = summaries[split]["metadata"]
        if int(metadata.get("eligible_steps", summaries[split]["steps"])) != summaries[split]["steps"]:
            raise AssertionError(f"{split} eligible-Step mismatch")
    for key in ("steps", "trajectories", "positive", "dimension"):
        if int(summaries["test"][key]) != int(expected[key]):
            raise AssertionError(f"test {key} mismatch")
    with np.load(baseline_scores) as independent, np.load(position_scores) as position:
        for name, payload in (("Independent", independent), ("Position", position)):
            for key in ("labels", "trajectory_indices", "positions"):
                if not np.array_equal(test[key].numpy(), payload[key]):
                    raise AssertionError(f"{name} {key} ordering mismatch")

    lengths = train["offsets"][1:] - train["offsets"][:-1]
    small_candidates = torch.nonzero(lengths <= 16, as_tuple=False).flatten()
    if len(small_candidates) < 3:
        raise AssertionError("need three small Steps for semantic smoke")
    indices = small_candidates[:3].to(torch.int64)
    model.eval()
    chunked, chunk_rows, selected_lengths = model.forward_cache(
        train,
        indices,
        device,
        relation_microbatch_size=3,
        return_debug=True,
        pair_debug=True,
        query_chunk_size_override=2,
    )
    dense, dense_rows, _ = model.forward_cache(
        train,
        indices,
        device,
        relation_microbatch_size=3,
        return_debug=True,
        pair_debug=True,
        query_chunk_size_override=max(selected_lengths),
    )
    if not torch.allclose(chunked, dense, atol=1e-6, rtol=1e-6):
        raise AssertionError("chunked/dense detector mismatch")
    chunk_debug, dense_debug = chunk_rows[0], dense_rows[0]
    for key in ("attention", "context", "correction", "final_weights"):
        if not torch.allclose(chunk_debug[key], dense_debug[key], atol=1e-6, rtol=1e-6):
            raise AssertionError(f"chunked/dense {key} mismatch")
    if chunk_debug["relation_features"].shape[-1] != 18:
        raise AssertionError("relation feature width is not 18")
    relation = chunk_debug["relation_indices"] - model.clipping_distance
    query_count, key_count = relation.shape
    raw = torch.arange(key_count, device=device)[None, :] - torch.arange(
        query_count, device=device
    )[:, None]
    if not torch.equal(
        relation, raw.clamp(-model.clipping_distance, model.clipping_distance)
    ):
        raise AssertionError("offset is not directed clip(j-i,-4,4)")
    mask = chunk_debug["valid_mask"]
    diagonal = torch.arange(min(query_count, key_count), device=device)
    valid_diagonal = mask[:, diagonal]
    cosine_diagonal = chunk_debug["cosine"][:, diagonal, diagonal][valid_diagonal]
    distance_diagonal = chunk_debug["distance"][:, diagonal, diagonal][valid_diagonal]
    if not torch.allclose(cosine_diagonal, torch.ones_like(cosine_diagonal), atol=1e-5):
        raise AssertionError("self cosine is not one")
    if not torch.allclose(distance_diagonal, torch.zeros_like(distance_diagonal), atol=1e-7):
        raise AssertionError("self distance is not zero")
    edge = chunk_debug["edge"]
    if not torch.allclose(chunk_debug["edge_key"], model.edge_key_proj(edge)):
        raise AssertionError("Key projection does not consume shared edge")
    if not torch.allclose(chunk_debug["edge_value"], model.edge_value_proj(edge)):
        raise AssertionError("Value projection does not consume shared edge")
    attention = chunk_debug["attention"]
    if torch.count_nonzero(attention.masked_select((~mask)[:, None, :])).item() != 0:
        raise AssertionError("padded key contributes to attention/message")
    if not torch.allclose(
        attention.sum(-1)[mask], torch.ones_like(attention.sum(-1)[mask]), atol=1e-6
    ):
        raise AssertionError("attention row does not sum to one")
    if not torch.allclose(
        chunk_debug["final_weights"].sum(-1),
        torch.ones(mask.shape[0], device=device),
        atol=1e-6,
    ):
        raise AssertionError("final token weights do not sum to one")
    max_correction = float(chunk_debug["correction"].abs().max().cpu())
    independent_logits = torch.cat(
        [row["independent_detector_logits"] for row in chunk_rows]
    )
    max_recovery_error = float((chunked - independent_logits).abs().max().cpu())
    if float(model.gamma.detach().cpu()) != 0.0 or max_correction != 0.0 or max_recovery_error > 1e-7:
        raise AssertionError("gamma-zero path does not exactly recover Independent")

    relation_names = (
        "relation_proj.weight",
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "position_embedding.weight",
        "edge_mlp.0.weight",
        "edge_mlp.2.weight",
        "edge_key_proj.weight",
        "edge_value_proj.weight",
        "context_layer_norm.weight",
        "rel_head.weight",
    )
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    labels = train["labels"][indices].to(device=device, dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    first_logits, _, _ = model.forward_cache(train, indices, device, 3)
    first_loss = nn.BCEWithLogitsLoss()(first_logits, labels)
    first_loss.backward()
    first_gradients = {name: gradient_norm(model, name) for name in (*relation_names, "gamma")}
    if first_gradients["gamma"] <= 0:
        raise AssertionError("first backward did not reach gamma")
    for name in relation_names:
        if first_gradients[name] > 1e-12:
            raise AssertionError(f"first backward should be gamma-gated: {name}")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second_logits, _, _ = model.forward_cache(train, indices, device, 3)
    second_loss = nn.BCEWithLogitsLoss()(second_logits, labels)
    second_loss.backward()
    second_gradients = {name: gradient_norm(model, name) for name in (*relation_names, "gamma")}
    for name in relation_names:
        if second_gradients[name] <= 0:
            raise AssertionError(f"second backward has zero gradient: {name}")
    if train["tokens"].requires_grad:
        raise AssertionError("cached hidden states unexpectedly require gradients")
    if not torch.isfinite(first_loss) or not torch.isfinite(second_loss):
        raise AssertionError("non-finite smoke loss")

    result = {
        "status": "PASS",
        "base_initialization": base_initialization,
        "summaries": summaries,
        "predecessor_cache_summaries_and_hashes_exact": True,
        "selected_lengths": selected_lengths,
        "maximum_step_length": max(s["max_step_tokens"] for s in summaries.values()),
        "query_chunk_size": model.query_chunk_size,
        "dense_chunk_match": True,
        "relation_width": 18,
        "offset_rule": "clip(j-i,-4,4)",
        "self_pair_cosine_one_distance_zero": True,
        "shared_edge_separate_key_value_projections": True,
        "padded_keys_zero_contribution": True,
        "initial_gamma": 0.0,
        "initial_max_correction": max_correction,
        "initial_max_score_error_vs_independent": max_recovery_error,
        "first_backward_gradient_norms": first_gradients,
        "second_backward_after_one_update_gradient_norms": second_gradients,
        "hidden_states_requires_grad": False,
        "answer_boundary": "released current-Step population only; no Answer companion loaded",
    }
    model.zero_grad(set_to_none=True)
    emit("CONTENT_RELATION_SMOKE_PASS", **result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=("qwen", "llama", "deepseek"), required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--baseline-scores", type=Path, required=True)
    parser.add_argument("--position-scores", type=Path, required=True)
    parser.add_argument("--position-metrics", type=Path, required=True)
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
    parser.add_argument("--position-dimension", type=int, default=16)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--relation-microbatch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--early-patience", type=int, default=5)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    started = time.time()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("smoke and training require scheduler-allocated CUDA")
    train, val, test = load_cache(args.train), load_cache(args.val), load_cache(args.test)
    summaries = {
        split: cache_summary(cache, split)
        for split, cache in (("train", train), ("val", val), ("test", test))
    }
    for split, summary in summaries.items():
        if summary["dimension"] != args.dimension:
            raise ValueError(f"{split} dimension mismatch")
        emit("CACHE_READY", **summary)

    model = ContentRelationResidualStep(
        args.dimension,
        args.attention_dimension,
        args.clipping_distance,
        args.position_dimension,
        args.query_chunk_size,
        args.tau,
        args.epsilon,
    ).to(device)
    torch.cuda.reset_peak_memory_stats()
    scorer_parameters = sum(parameter.numel() for parameter in model.scorer.parameters())
    probe_parameters = sum(parameter.numel() for parameter in model.probe.parameters())
    relation_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("scorer.") and not name.startswith("probe.")
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    smoke = run_smoke(
        model,
        train,
        val,
        test,
        args.baseline_scores,
        args.position_scores,
        args.position_metrics,
        device,
        {
            "steps": args.expected_test_steps,
            "trajectories": args.expected_test_trajectories,
            "positive": args.expected_test_positive,
            "dimension": args.dimension,
        },
        args.relation_microbatch_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    smoke_audit = {
        "status": "PASS",
        "model": args.model_key,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0),
        "smoke": smoke,
        "scorer_parameters": scorer_parameters,
        "probe_parameters": probe_parameters,
        "relation_parameters": relation_parameters,
        "total_parameters": total_parameters,
        "optimizer_assignment": "one Adam group containing scorer, probe, and all content-relation parameters; lr=1e-3, weight_decay=5e-4",
        "architecture": "gamma-gated CHORD-style directed content edge: position16 + cosine + log1p(distance) -> MLP64 -> separate EK/EV; d_a=64; original-H pooling",
        "decision_rule": "score > 0.5",
    }
    (args.output_dir / "smoke_audit.json").write_text(
        json.dumps(smoke_audit, indent=2, default=str) + "\n"
    )
    if args.smoke_only:
        emit("SMOKE_JOB_COMPLETE", output=str(args.output_dir / "smoke_audit.json"))
        return

    del model
    torch.cuda.empty_cache()
    set_seed(args.seed)
    model = ContentRelationResidualStep(
        args.dimension,
        args.attention_dimension,
        args.clipping_distance,
        args.position_dimension,
        args.query_chunk_size,
        args.tau,
        args.epsilon,
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
        for batch_indices in chunks(permutation, args.batch_size):
            logits, _, _ = model.forward_cache(
                train, batch_indices, device, args.relation_microbatch_size
            )
            labels = train["labels"][batch_indices].to(device=device, dtype=torch.float32)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(batch_indices)
            seen += len(batch_indices)
        train_loss = loss_sum / seen
        scheduler.step(train_loss)
        val_scores, _ = score_split(
            model, val, device, args.eval_batch_size, args.relation_microbatch_size
        )
        val_auc = float(roc_auc_score(val_labels, val_scores))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auroc": val_auc,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.time() - started,
        }
        trace.append(row)
        emit("EPOCH", **row)
        if val_auc > best_auc + 1e-7:
            best_auc, best_epoch = val_auc, epoch
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
        if epochs_without_gain >= args.early_patience:
            emit("EARLY_STOP", epoch=epoch, patience=args.early_patience)
            break
    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best_state)
    test_scores, diagnostics = score_split(
        model, test, device, args.eval_batch_size, args.relation_microbatch_size
    )
    test_labels = test["labels"].numpy().astype(np.int64)
    test_metrics = calculate_metrics(test_labels, test_scores)
    checkpoint = args.output_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": best_state,
            "config": vars(args),
            "best_epoch": best_epoch,
            "best_val_auroc": best_auc,
        },
        checkpoint,
    )
    np.savez_compressed(
        args.output_dir / "test_scores.npz",
        scores=test_scores,
        labels=test_labels,
        trajectory_indices=test["trajectory_indices"].numpy(),
        positions=test["positions"].numpy(),
        **diagnostics,
    )
    correction_mean = float(diagnostics["mean_abs_gamma_delta"].mean())
    independent_mean = float(diagnostics["mean_abs_independent_logits"].mean())
    result = {
        "method": "step_content_relation_residual_chord_da64_k4",
        "model": args.model_key,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "seed": args.seed,
        "paper_layer": args.paper_layer,
        "hf_hidden_states_index": args.hidden_states_index,
        "hidden_dimension": args.dimension,
        "attention_dimension": args.attention_dimension,
        "position_dimension": args.position_dimension,
        "clipping_distance": args.clipping_distance,
        "query_chunk_size": args.query_chunk_size,
        "causal_triangular_mask": False,
        "scorer_parameters": scorer_parameters,
        "probe_parameters": probe_parameters,
        "relation_parameters": relation_parameters,
        "total_parameters": total_parameters,
        "optimizer": {"name": "Adam", "lr": 1e-3, "weight_decay": 5e-4},
        "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.1, "patience": 20, "min_lr": 1e-6, "signal": "train_epoch_loss"},
        "loss": "unweighted BCEWithLogitsLoss",
        "logical_batch_size": args.batch_size,
        "eval_logical_batch_size": args.eval_batch_size,
        "relation_microbatch_size": args.relation_microbatch_size,
        "best_epoch": best_epoch,
        "best_val_auroc": best_auc,
        "test": test_metrics,
        "learned_gamma": float(model.gamma.detach().cpu()),
        "mean_abs_gamma_delta": correction_mean,
        "mean_abs_independent_token_logit": independent_mean,
        "correction_to_independent_ratio": correction_mean / max(independent_mean, args.epsilon),
        "token_weight_entropy_mean": float(diagnostics["token_weight_entropy"].mean()),
        "token_weight_entropy_std": float(diagnostics["token_weight_entropy"].std()),
        "maximum_step_length": max(summary["max_step_tokens"] for summary in summaries.values()),
        "summaries": summaries,
        "smoke": smoke,
        "trace": trace,
        "walltime_s": time.time() - started,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint": str(checkpoint),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    emit("TEST_COMPLETE", test=test_metrics, best_epoch=best_epoch, learned_gamma=result["learned_gamma"], peak_memory_bytes=result["cuda_peak_memory_bytes"])


if __name__ == "__main__":
    main()
