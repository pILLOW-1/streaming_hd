#!/usr/bin/env python3
"""Independent validator for the frozen Prefix shared-relation experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODELS = ("qwen", "llama", "deepseek")
DISPLAY = {"qwen": "Qwen", "llama": "Llama", "deepseek": "DeepSeek"}
DIMENSIONS = {"qwen": 3584, "llama": 4096, "deepseek": 4096}
PAPER = {
    "qwen": {
        "local": {"auroc": 0.8802, "accuracy": 0.7890, "f1": 0.8048},
        "final": {"auroc": 0.8105, "accuracy": 0.7612, "f1": 0.8430},
    },
    "llama": {
        "local": {"auroc": 0.8730, "accuracy": 0.7938, "f1": 0.8245},
        "final": {"auroc": 0.7269, "accuracy": 0.7803, "f1": 0.8688},
    },
    "deepseek": {
        "local": {"auroc": 0.8798, "accuracy": 0.8714, "f1": 0.6943},
        "final": {"auroc": 0.9218, "accuracy": 0.8752, "f1": 0.8658},
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def score_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    predictions = (scores > 0.5).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "threshold": 0.5,
        "decision_rule": "score > 0.5",
        "n": int(labels.size),
        "positive": int(labels.sum()),
        "negative": int(labels.size - labels.sum()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
    }


def validate_checkpoint(path: Path, dimension: int) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    expected = {
        "gate.weight": (1, 2 * dimension),
        "gate.bias": (1,),
        "probe.weight": (1, dimension),
        "probe.bias": (1,),
        "relation_projection.weight": (256, dimension),
        "relation_mlp.0.weight": (256, 512),
        "relation_mlp.0.bias": (256,),
        "relation_mlp.2.weight": (256, 256),
        "relation_mlp.2.bias": (256,),
        "relation_layer_norm.weight": (256,),
        "relation_layer_norm.bias": (256,),
        "relation_readout.weight": (1, 256),
        "relation_readout.bias": (1,),
    }
    actual = {name: tuple(state[name].shape) for name in expected}
    if actual != expected or set(state) != set(expected):
        raise AssertionError(f"checkpoint architecture mismatch: {actual}")
    if checkpoint["relation_width"] != 256:
        raise AssertionError("wrong relation width")
    return {"status": "PASS", "shapes": {k: list(v) for k, v in actual.items()}}


def exact_alignment(reference: dict[str, np.ndarray], new: dict[str, np.ndarray]) -> dict[str, bool]:
    result = {}
    for key in (
        "trajectory_id",
        "trajectory_index",
        "step_id",
        "flat_step_index",
        "prefix_label",
        "step_label",
    ):
        result[key] = bool(np.array_equal(reference[key], new[key]))
        if not result[key]:
            raise AssertionError(f"test alignment mismatch: {key}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scalar-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()

    combined: dict[str, Any] = {}
    validation: dict[str, Any] = {
        "status": "PASS",
        "source_sha256": sha256(args.source),
        "models": {},
    }
    for model in MODELS:
        new_path = args.run_root / "results" / model / "test_predictions.npz"
        scalar_path = (
            args.scalar_root / "results" / model / "seed_42" / "test_predictions.npz"
        )
        new = load_npz(new_path)
        scalar = load_npz(scalar_path)
        alignment = exact_alignment(scalar, new)
        labels = new["prefix_label"].astype(np.int64)
        trajectory = new["trajectory_index"].astype(np.int64)
        final = np.r_[trajectory[1:] != trajectory[:-1], True]
        new_metrics = {
            "local": score_metrics(labels, new["relation_gate_score"]),
            "final": score_metrics(labels[final], new["relation_gate_score"][final]),
        }
        scalar_metrics = {
            "local": score_metrics(labels, scalar["dynamic_gate_score"]),
            "final": score_metrics(labels[final], scalar["dynamic_gate_score"][final]),
        }
        saved = json.loads(
            (args.run_root / "results" / model / "metrics.json").read_text()
        )
        for scope in ("local", "final"):
            for metric in ("auroc", "auprc", "accuracy", "f1", "precision", "recall"):
                if abs(saved["test"][scope][metric] - new_metrics[scope][metric]) > 1e-12:
                    raise AssertionError(f"{model}: saved {scope}/{metric} mismatch")
        smoke = json.loads(
            (args.run_root / "audits" / "smoke" / f"smoke_{model}.json").read_text()
        )
        if smoke.get("status") != "PASS":
            raise AssertionError(f"{model}: mandatory smoke missing")
        if not all(value > 0 for value in smoke["relation_gradient_norms"].values()):
            raise AssertionError(f"{model}: relation gradient audit failed")
        checkpoint = validate_checkpoint(
            args.run_root / "results" / model / "checkpoint.pt", DIMENSIONS[model]
        )
        delta = {
            scope: {
                metric: new_metrics[scope][metric] - scalar_metrics[scope][metric]
                for metric in ("auroc", "auprc", "accuracy", "f1", "precision", "recall")
            }
            for scope in ("local", "final")
        }
        combined[model] = {
            "paper_streaming": PAPER[model],
            "scalar_gate_seed42_archived": scalar_metrics,
            "shared_relation_gate_seed42": new_metrics,
            "delta_new_minus_scalar": delta,
            "best_epoch": saved["best_epoch"],
            "best_val_local_auroc": saved["best_val_local_auroc"],
            "relation_diagnostics": saved["relation_diagnostics"],
            "relation_gradient_norms_first_backward": smoke["relation_gradient_norms"],
            "relation_parameters": saved["relation_parameters"],
            "total_trainable_parameters": saved["total_trainable_parameters"],
            "cuda_peak_memory_bytes": saved["cuda_peak_memory_bytes"],
            "walltime_s": saved["walltime_s"],
        }
        validation["models"][model] = {
            "test_alignment": alignment,
            "saved_metrics_exact_recompute": True,
            "checkpoint": checkpoint,
            "new_prediction_sha256": sha256(new_path),
            "scalar_prediction_sha256": sha256(scalar_path),
            "n_local": new_metrics["local"]["n"],
            "n_final": new_metrics["final"]["n"],
        }

    primary_pass = all(
        combined[model]["delta_new_minus_scalar"]["local"]["auroc"] > 0
        for model in MODELS
    )
    verdict = (
        "GO: shared-relation Prefix Local AUROC exceeds the archived scalar gate in all three models."
        if primary_pass
        else "NO-GO: shared-relation Prefix Local AUROC does not exceed the archived scalar gate in all three models; abandon this relation route under the frozen gate."
    )
    result = {
        "experiment": "prefix_content_relation_gate_within_three_models_20260831",
        "seed": 42,
        "training": "independent within-model two-stage; learned-token Step frozen",
        "decision_rule": "score > 0.5",
        "models": combined,
        "primary_gate": "new Prefix Local AUROC > scalar Prefix Local AUROC for all three models",
        "primary_gate_pass": primary_pass,
        "verdict": verdict,
        "scientific_change": "shared bias-free W_R d-to-256; [q elementwise-product k, absolute q-minus-k]; MLP 512-256-256; LayerNorm; unscaled scalar correction to predecessor gate logit",
    }

    for relative in ("results", "audits", "reports", "plots"):
        (args.run_root / relative).mkdir(parents=True, exist_ok=True)
    (args.run_root / "results" / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (args.run_root / "audits" / "FINAL_VALIDATION.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )

    lines = [
        "# Prefix Shared-Relation Gate — Final Results",
        "",
        "Three independent within-model seed-42 runs. The learned-token Step representation is frozen. Metrics below were independently recomputed from aligned full-test predictions; classification uses fixed `score > 0.5`.",
        "",
        "| Model | Method | Local AUROC | Local AUPRC | Local Acc | Local F1 | Final AUROC | Final AUPRC | Final Acc | Final F1 | Δ Local AUROC vs scalar |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        paper = combined[model]["paper_streaming"]
        scalar = combined[model]["scalar_gate_seed42_archived"]
        new = combined[model]["shared_relation_gate_seed42"]
        delta = combined[model]["delta_new_minus_scalar"]["local"]["auroc"]
        lines.append(
            f"| {DISPLAY[model]} | Streaming paper | {paper['local']['auroc']:.4f} | — | {paper['local']['accuracy']:.4f} | {paper['local']['f1']:.4f} | {paper['final']['auroc']:.4f} | — | {paper['final']['accuracy']:.4f} | {paper['final']['f1']:.4f} | — |"
        )
        lines.append(
            f"| {DISPLAY[model]} | Scalar gate seed 42 (archived) | {scalar['local']['auroc']:.4f} | {scalar['local']['auprc']:.4f} | {scalar['local']['accuracy']:.4f} | {scalar['local']['f1']:.4f} | {scalar['final']['auroc']:.4f} | {scalar['final']['auprc']:.4f} | {scalar['final']['accuracy']:.4f} | {scalar['final']['f1']:.4f} | — |"
        )
        lines.append(
            f"| {DISPLAY[model]} | Shared relation gate (new) | {new['local']['auroc']:.4f} | {new['local']['auprc']:.4f} | {new['local']['accuracy']:.4f} | {new['local']['f1']:.4f} | {new['final']['auroc']:.4f} | {new['final']['auprc']:.4f} | {new['final']['accuracy']:.4f} | {new['final']['f1']:.4f} | {delta * 100:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## Terminal verdict",
            "",
            verdict,
            "",
            "The paper rows are published references; the scalar and shared-relation rows are the comparable seed-42 implementations. No Answer companion, target calibration, or threshold selection was used.",
        ]
    )
    (args.run_root / "reports" / "RESULTS.md").write_text("\n".join(lines) + "\n")

    x = np.arange(len(MODELS))
    scalar_auc = [combined[m]["scalar_gate_seed42_archived"]["local"]["auroc"] for m in MODELS]
    new_auc = [combined[m]["shared_relation_gate_seed42"]["local"]["auroc"] for m in MODELS]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - 0.18, scalar_auc, width=0.36, label="Scalar gate")
    ax.bar(x + 0.18, new_auc, width=0.36, label="Shared relation gate")
    ax.set_xticks(x, [DISPLAY[m] for m in MODELS])
    ax.set_ylabel("Prefix Local AUROC")
    ax.set_ylim(min(scalar_auc + new_auc) - 0.03, max(scalar_auc + new_auc) + 0.02)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.run_root / "plots" / "prefix_local_auroc_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"event": "INDEPENDENT_VALIDATION_PASS", "verdict": verdict}), flush=True)


if __name__ == "__main__":
    main()
