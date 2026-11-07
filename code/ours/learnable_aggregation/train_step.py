#!/usr/bin/env python3
"""Learn current-Step token weights and a linear Streaming HD probe."""

from __future__ import annotations

import argparse
import copy
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
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, roc_auc_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cache(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    required = {"tokens", "offsets", "labels", "trajectory_indices", "positions", "metadata"}
    missing = required.difference(value)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    if value["offsets"].numel() != value["labels"].numel() + 1:
        raise ValueError(f"offset/label mismatch: {path}")
    if int(value["offsets"][-1]) != value["tokens"].shape[0]:
        raise ValueError(f"offset/token mismatch: {path}")
    return value


class LearnedTokenStep(nn.Module):
    """Official-HaMI-width scorer adapted to logits, softmax pooling, linear probe."""

    def __init__(self, dimension: int, tau: float, epsilon: float):
        super().__init__()
        self.tau = tau
        self.epsilon = epsilon
        self.scorer = nn.Sequential(
            nn.Linear(dimension, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        self.probe = nn.Linear(dimension, 1)

    def forward_bags(self, tokens: torch.Tensor, offsets: torch.Tensor, indices: torch.Tensor, device: torch.device):
        pieces = []
        lengths = []
        for index in indices.tolist():
            start = int(offsets[index])
            end = int(offsets[index + 1])
            pieces.append(tokens[start:end])
            lengths.append(end - start)
        flat = torch.cat(pieces, dim=0).to(device=device, dtype=torch.float32, non_blocking=True)
        importance_logits = self.scorer(flat).flatten()
        pooled = []
        entropies = []
        effective = []
        cursor = 0
        for length in lengths:
            states = flat[cursor : cursor + length]
            weights = torch.softmax(importance_logits[cursor : cursor + length] / self.tau, dim=0)
            vector = torch.sum(states * weights.unsqueeze(1), dim=0)
            vector = vector / (torch.linalg.vector_norm(vector) + self.epsilon)
            entropy = -torch.sum(weights * torch.log(weights.clamp_min(self.epsilon)))
            pooled.append(vector)
            entropies.append(entropy)
            effective.append(torch.exp(entropy))
            cursor += length
        z = torch.stack(pooled)
        return self.probe(z).flatten(), torch.stack(entropies), torch.stack(effective), lengths


def score_split(model, cache, device, batch_size):
    model.eval()
    n = len(cache["labels"])
    scores, entropies, effective, lengths = [], [], [], []
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            indices = torch.arange(start, min(start + batch_size, n))
            logits, entropy, eff, lens = model.forward_bags(
                cache["tokens"], cache["offsets"], indices, device
            )
            scores.append(torch.sigmoid(logits).cpu())
            entropies.append(entropy.cpu())
            effective.append(eff.cpu())
            lengths.extend(lens)
    return (
        torch.cat(scores).numpy().astype(np.float64),
        torch.cat(entropies).numpy().astype(np.float64),
        torch.cat(effective).numpy().astype(np.float64),
        np.asarray(lengths, dtype=np.int64),
    )


def select_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def metrics(labels, scores, threshold):
    pred = (scores >= threshold).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, pred)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "threshold": float(threshold),
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
    }


def evaluate_val_loss(model, cache, device, batch_size):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(cache["labels"]), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(cache["labels"])))
            logits, _, _, _ = model.forward_bags(cache["tokens"], cache["offsets"], indices, device)
            labels = cache["labels"][indices].to(device=device, dtype=torch.float32)
            total += float(criterion(logits, labels).cpu())
    return total / len(cache["labels"])


def smoke(model, train, val, device, batch_size):
    model.train()
    indices = torch.arange(min(128, len(train["labels"])))
    logits, _, _, _ = model.forward_bags(train["tokens"], train["offsets"], indices, device)
    labels = train["labels"][indices].to(device=device, dtype=torch.float32)
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite smoke loss")
    model.zero_grad(set_to_none=True)
    small = {key: value for key, value in val.items()}
    small["labels"] = val["labels"][: min(128, len(val["labels"]))]
    small["offsets"] = val["offsets"][: len(small["labels"]) + 1]
    small["tokens"] = val["tokens"][: int(small["offsets"][-1])]
    score_split(model, small, device, batch_size)
    print(json.dumps({"event": "SMOKE_COMPLETE", "loss": float(loss.detach().cpu())}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--paper-layer", type=int, required=True)
    parser.add_argument("--hidden-states-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--early-patience", type=int, default=5)
    args = parser.parse_args()

    started = time.time()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this production job")
    device = torch.device("cuda")
    train, val, test = load_cache(args.train), load_cache(args.val), load_cache(args.test)
    for split, cache in (("train", train), ("val", val), ("test", test)):
        dimension = int(cache["tokens"].shape[1])
        if dimension != args.dimension:
            raise ValueError(f"{split} dimension {dimension} != {args.dimension}")
        print(json.dumps({"event": "CACHE_READY", "split": split, "steps": len(cache["labels"]), "tokens": int(cache["offsets"][-1]), "positive": int(cache["labels"].sum()), "negative": int(len(cache["labels"]) - cache["labels"].sum()), "dimension": dimension}), flush=True)

    smoke_model = LearnedTokenStep(args.dimension, args.tau, args.epsilon).to(device)
    smoke(smoke_model, train, val, device, args.eval_batch_size)
    del smoke_model
    torch.cuda.empty_cache()
    set_seed(args.seed)
    model = LearnedTokenStep(args.dimension, args.tau, args.epsilon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=20, min_lr=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_epoch, best_state = -math.inf, 0, None
    epochs_without_gain = 0
    trace = []
    train_generator = torch.Generator().manual_seed(args.seed)
    val_labels = val["labels"].numpy().astype(np.int64)

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(train["labels"]), generator=train_generator)
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            logits, _, _, _ = model.forward_bags(train["tokens"], train["offsets"], indices, device)
            labels = train["labels"][indices].to(device=device, dtype=torch.float32)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)
        train_loss = loss_sum / seen
        scheduler.step(train_loss)
        val_scores, _, _, _ = score_split(model, val, device, args.eval_batch_size)
        val_auc = float(roc_auc_score(val_labels, val_scores))
        val_loss = evaluate_val_loss(model, val, device, args.eval_batch_size)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auroc": val_auc, "lr": optimizer.param_groups[0]["lr"], "elapsed_s": time.time() - started}
        trace.append(row)
        print(json.dumps({"event": "EPOCH", **row}), flush=True)
        if val_auc > best_auc + 1e-7:
            best_auc, best_epoch = val_auc, epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
        if epochs_without_gain >= args.early_patience:
            break
    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best_state)
    val_scores, val_entropy, val_effective, val_lengths = score_split(model, val, device, args.eval_batch_size)
    threshold = select_threshold(val_labels, val_scores)
    test_scores, test_entropy, test_effective, test_lengths = score_split(model, test, device, args.eval_batch_size)
    test_labels = test["labels"].numpy().astype(np.int64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint.pt"
    torch.save({"state_dict": best_state, "config": vars(args), "best_epoch": best_epoch, "best_val_auroc": best_auc}, checkpoint)
    np.savez_compressed(args.output_dir / "test_scores.npz", scores=test_scores, labels=test_labels, entropy=test_entropy, effective_tokens=test_effective, token_lengths=test_lengths, trajectory_indices=test["trajectory_indices"].numpy(), positions=test["positions"].numpy())
    result = {
        "method": "learned_token_weighted_step",
        "model": args.model_key,
        "seed": args.seed,
        "paper_layer": args.paper_layer,
        "hf_hidden_states_index": args.hidden_states_index,
        "hidden_dimension": args.dimension,
        "temperature": args.tau,
        "scorer": "Linear(D,256)-BatchNorm-ReLU-Dropout(0.3)-Linear(256,1); official HaMI Sigmoid removed to expose logits",
        "scorer_parameters": sum(p.numel() for p in model.scorer.parameters()),
        "probe_parameters": sum(p.numel() for p in model.probe.parameters()),
        "optimizer": {"name": "Adam", "lr": 0.001, "weight_decay": 5e-4},
        "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.1, "patience": 20, "min_lr": 1e-6, "signal": "train_epoch_loss"},
        "loss": "unweighted BCEWithLogitsLoss",
        "best_epoch": best_epoch,
        "best_val_auroc": best_auc,
        "validation_threshold": threshold,
        "val": metrics(val_labels, val_scores, threshold),
        "test": metrics(test_labels, test_scores, threshold),
        "token_weight_diagnostics": {"test_entropy_mean": float(test_entropy.mean()), "test_effective_tokens_mean": float(test_effective.mean()), "test_token_count_mean": float(test_lengths.mean()), "test_effective_fraction_mean": float(np.mean(test_effective / test_lengths))},
        "trace": trace,
        "walltime_s": time.time() - started,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint": str(checkpoint),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"event": "TEST_COMPLETE", "test": result["test"], "best_epoch": best_epoch, "peak_memory_bytes": result["cuda_peak_memory_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
