#!/usr/bin/env python3
"""Within-model Step detector with one standard self-attention token scorer.

The data, split, label, pooling target, probe, loss, optimizer, scheduler and
checkpoint rule intentionally match the archived learned-token-weighting run.
Only the token scoring mechanism changes to single-head scaled dot-product
self-attention with d_a=64 and a scalar value projection.
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
    return {
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


class StandardSelfAttentionStep(nn.Module):
    """Single-head scaled dot-product attention followed by scalar re-pooling."""

    def __init__(self, dimension: int, attention_dimension: int = 64, epsilon: float = 1e-12):
        super().__init__()
        self.dimension = dimension
        self.attention_dimension = attention_dimension
        self.epsilon = epsilon
        # No learned affine or projection biases: exactly Hbar WQ, Hbar WK, Hbar wV.
        self.layer_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.q_proj = nn.Linear(dimension, attention_dimension, bias=False)
        self.k_proj = nn.Linear(dimension, attention_dimension, bias=False)
        self.v_proj = nn.Linear(dimension, 1, bias=False)
        self.probe = nn.Linear(dimension, 1)

    def forward_padded(
        self, states: torch.Tensor, valid_mask: torch.Tensor, return_debug: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if states.ndim != 3 or valid_mask.ndim != 2:
            raise ValueError("expected states [B,L,D] and valid_mask [B,L]")
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("every Step must contain at least one token")
        hbar = self.layer_norm(states)
        q = self.q_proj(hbar)
        k = self.k_proj(hbar)
        values = self.v_proj(hbar)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_dimension)
        logits = logits.masked_fill(~valid_mask[:, None, :], -torch.inf)
        attention = torch.softmax(logits, dim=-1)
        scalar_context = torch.matmul(attention, values).squeeze(-1)
        scalar_context = scalar_context.masked_fill(~valid_mask, -torch.inf)
        final_weights = torch.softmax(scalar_context, dim=-1)
        pooled = torch.sum(states * final_weights.unsqueeze(-1), dim=1)
        pooled = pooled / torch.linalg.vector_norm(pooled, dim=1, keepdim=True).clamp_min(self.epsilon)
        detector_logits = self.probe(pooled).flatten()
        debug: dict[str, torch.Tensor] = {}
        if return_debug:
            debug = {
                "attention": attention,
                "final_weights": final_weights,
                "scalar_context": scalar_context,
                "pooled": pooled,
            }
        return detector_logits, debug


def padded_batch(
    cache: dict[str, Any], indices: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    offsets = cache["offsets"]
    pieces: list[torch.Tensor] = []
    lengths: list[int] = []
    for index in indices.tolist():
        start, end = int(offsets[index]), int(offsets[index + 1])
        pieces.append(cache["tokens"][start:end])
        lengths.append(end - start)
    max_length = max(lengths)
    states = torch.zeros(
        (len(pieces), max_length, cache["tokens"].shape[1]),
        dtype=torch.float32,
        device=device,
    )
    mask = torch.zeros((len(pieces), max_length), dtype=torch.bool, device=device)
    for row, (piece, length) in enumerate(zip(pieces, lengths, strict=True)):
        states[row, :length].copy_(piece.to(device=device, dtype=torch.float32, non_blocking=True))
        mask[row, :length] = True
    return states, mask, lengths


def chunks(indices: torch.Tensor, size: int) -> Iterable[torch.Tensor]:
    for start in range(0, len(indices), size):
        yield indices[start : start + size]


def score_split(
    model: StandardSelfAttentionStep,
    cache: dict[str, Any],
    device: torch.device,
    logical_batch_size: int,
    microbatch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_scores: list[torch.Tensor] = []
    all_entropy: list[torch.Tensor] = []
    all_lengths: list[int] = []
    with torch.inference_mode():
        for logical_start in range(0, len(cache["labels"]), logical_batch_size):
            logical = torch.arange(logical_start, min(logical_start + logical_batch_size, len(cache["labels"])))
            for indices in chunks(logical, microbatch_size):
                states, mask, lengths = padded_batch(cache, indices, device)
                logits, debug = model.forward_padded(states, mask, return_debug=True)
                weights = debug["final_weights"]
                entropy = -torch.sum(weights * torch.log(weights.clamp_min(model.epsilon)), dim=1)
                all_scores.append(torch.sigmoid(logits).cpu())
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


def run_smoke(
    model: StandardSelfAttentionStep,
    train: dict[str, Any],
    test: dict[str, Any],
    baseline_scores: Path,
    device: torch.device,
    expected: dict[str, int],
) -> dict[str, Any]:
    # Full-array alignment checks are light relative to the hidden-state payload.
    for key in ("steps", "trajectories", "positive", "dimension"):
        actual = cache_summary(test, "test")[key]
        if int(actual) != int(expected[key]):
            raise AssertionError(f"test {key}: {actual} != {expected[key]}")
    archived = np.load(baseline_scores)
    if not np.array_equal(test["labels"].numpy(), archived["labels"]):
        raise AssertionError("test labels differ from archived independent scorer")
    for key in ("trajectory_indices", "positions"):
        if key in archived and not np.array_equal(test[key].numpy(), archived[key]):
            raise AssertionError(f"test {key} differs from archived independent scorer")

    offsets = train["offsets"]
    lengths = offsets[1:] - offsets[:-1]
    shortest = int(torch.argmin(lengths))
    longest = int(torch.argmax(lengths))
    different = int(torch.nonzero(lengths != lengths[shortest], as_tuple=False)[0])
    indices = torch.tensor([shortest, different, longest], dtype=torch.int64)
    states, mask, selected_lengths = padded_batch(train, indices, device)
    if states.requires_grad:
        raise AssertionError("cached hidden states unexpectedly require gradients")
    model.train()
    logits, debug = model.forward_padded(states, mask, return_debug=True)
    labels = train["labels"][indices].to(device=device, dtype=torch.float32)
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()
    if not torch.isfinite(loss):
        raise AssertionError("non-finite smoke loss")
    attention = debug["attention"]
    final_weights = debug["final_weights"]
    valid_attention_sums = attention.sum(dim=-1)[mask]
    if not torch.allclose(valid_attention_sums, torch.ones_like(valid_attention_sums), atol=1e-5):
        raise AssertionError("valid attention rows do not sum to one")
    if torch.count_nonzero(final_weights[~mask]).item() != 0:
        raise AssertionError("padded final token weights are not exactly zero")
    if not torch.allclose(final_weights.sum(dim=-1), torch.ones(len(indices), device=device), atol=1e-6):
        raise AssertionError("valid final token weights do not sum to one")
    gradient_norms = {}
    for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "probe.weight"):
        parameter = dict(model.named_parameters())[name]
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"missing/non-finite gradient: {name}")
        norm = float(parameter.grad.norm().detach().cpu())
        if norm <= 0.0:
            raise AssertionError(f"zero gradient: {name}")
        gradient_norms[name] = norm
    result = {
        "status": "PASS",
        "loss": float(loss.detach().cpu()),
        "selected_lengths": selected_lengths,
        "gradient_norms": gradient_norms,
        "hidden_states_requires_grad": bool(states.requires_grad),
        "valid_attention_row_sum_max_error": float((valid_attention_sums - 1).abs().max().cpu()),
        "padded_final_weight_nonzero": int(torch.count_nonzero(final_weights[~mask]).item()),
        "final_weight_sum_max_error": float((final_weights.sum(dim=-1) - 1).abs().max().cpu()),
    }
    model.zero_grad(set_to_none=True)
    emit("SEMANTIC_MASK_GRADIENT_SMOKE_PASS", **result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=("qwen", "llama", "deepseek"))
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--baseline-scores", type=Path, required=True)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--early-patience", type=int, default=5)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    started = time.time()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.smoke_only and device.type != "cuda":
        raise RuntimeError("production training requires CUDA")
    train, val, test = load_cache(args.train), load_cache(args.val), load_cache(args.test)
    summaries = {}
    for split, cache in (("train", train), ("val", val), ("test", test)):
        summary = cache_summary(cache, split)
        if summary["dimension"] != args.dimension:
            raise ValueError(f"{split} dimension {summary['dimension']} != {args.dimension}")
        summaries[split] = summary
        emit("CACHE_READY", **summary)

    model = StandardSelfAttentionStep(args.dimension, args.attention_dimension, args.epsilon).to(device)
    scorer_parameters = sum(
        p.numel() for name, p in model.named_parameters() if not name.startswith("probe.")
    )
    probe_parameters = sum(p.numel() for p in model.probe.parameters())
    smoke_result = run_smoke(
        model,
        train,
        test,
        args.baseline_scores,
        device,
        {
            "steps": args.expected_test_steps,
            "trajectories": args.expected_test_trajectories,
            "positive": args.expected_test_positive,
            "dimension": args.dimension,
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "status": "PASS",
        "model": args.model_key,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "summaries": summaries,
        "smoke": smoke_result,
        "attention_parameters": scorer_parameters,
        "probe_parameters": probe_parameters,
        "architecture": "LN(no affine); Q=HbarWQ,K=HbarWK,V=HbarwV; A=softmax(QK^T/sqrt(64)+key_padding); s=AV; alpha=softmax(s+token_padding); z=sum alpha*h_original; L2(z); Linear(D,1)",
        "mask": "bidirectional within current parsed Step; padding mask only; no triangular mask",
        "decision_rule": "score > 0.5",
    }
    (args.output_dir / "smoke_audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")
    if args.smoke_only:
        emit("SMOKE_JOB_COMPLETE", output=str(args.output_dir / "smoke_audit.json"))
        return

    del model
    torch.cuda.empty_cache()
    set_seed(args.seed)
    model = StandardSelfAttentionStep(args.dimension, args.attention_dimension, args.epsilon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=20, min_lr=1e-6
    )
    criterion_sum = nn.BCEWithLogitsLoss(reduction="sum")
    train_generator = torch.Generator().manual_seed(args.seed)
    val_labels = val["labels"].numpy().astype(np.int64)
    best_auc, best_epoch, best_state = -math.inf, 0, None
    epochs_without_gain = 0
    trace: list[dict[str, Any]] = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(train["labels"]), generator=train_generator)
        loss_sum = 0.0
        seen = 0
        for logical_start in range(0, len(permutation), args.batch_size):
            logical = permutation[logical_start : logical_start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logical_loss_sum = 0.0
            for indices in chunks(logical, args.microbatch_size):
                states, mask, _ = padded_batch(train, indices, device)
                logits, _ = model.forward_padded(states, mask)
                labels = train["labels"][indices].to(device=device, dtype=torch.float32)
                micro_loss_sum = criterion_sum(logits, labels)
                (micro_loss_sum / len(logical)).backward()
                logical_loss_sum += float(micro_loss_sum.detach().cpu())
            optimizer.step()
            loss_sum += logical_loss_sum
            seen += len(logical)
        train_loss = loss_sum / seen
        scheduler.step(train_loss)
        val_scores, _, _ = score_split(
            model, val, device, args.eval_batch_size, args.microbatch_size
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
        model, test, device, args.eval_batch_size, args.microbatch_size
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
        attention_entropy=test_entropy,
        token_lengths=test_lengths,
        trajectory_indices=test["trajectory_indices"].numpy(),
        positions=test["positions"].numpy(),
    )
    result = {
        "method": "step_standard_self_attention_da64",
        "model": args.model_key,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "seed": args.seed,
        "paper_layer": args.paper_layer,
        "hf_hidden_states_index": args.hidden_states_index,
        "hidden_dimension": args.dimension,
        "attention_dimension": args.attention_dimension,
        "attention_parameters": scorer_parameters,
        "probe_parameters": probe_parameters,
        "optimizer": {"name": "Adam", "lr": 1e-3, "weight_decay": 5e-4},
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": 0.1,
            "patience": 20,
            "min_lr": 1e-6,
            "signal": "train_epoch_loss",
        },
        "loss": "unweighted BCEWithLogitsLoss",
        "logical_batch_size": args.batch_size,
        "eval_logical_batch_size": args.eval_batch_size,
        "microbatch_size_implementation_only": args.microbatch_size,
        "best_epoch": best_epoch,
        "best_val_auroc": best_auc,
        "test": test_metrics,
        "summaries": summaries,
        "trace": trace,
        "walltime_s": time.time() - started,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint": str(checkpoint),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    emit(
        "TEST_COMPLETE",
        test=test_metrics,
        best_epoch=best_epoch,
        peak_memory_bytes=result["cuda_peak_memory_bytes"],
    )


if __name__ == "__main__":
    main()
