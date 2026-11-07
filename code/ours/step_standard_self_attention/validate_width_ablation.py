#!/usr/bin/env python3
"""Independent three-way validator for the d_a=256 width ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score


MODELS = ("qwen", "llama", "deepseek")
DISPLAY = {"qwen": "Qwen", "llama": "Llama", "deepseek": "DeepSeek"}
METHODS = ("independent", "self_attention_da64", "self_attention_da256")
METHOD_LABELS = {
    "independent": "Independent token scorer",
    "self_attention_da64": "Self-attention (d_a=64)",
    "self_attention_da256": "Self-attention (d_a=256)",
}
PRIMARY = ("auroc", "accuracy", "f1")


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def calculate(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
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
        "score_sha256": digest(scores),
        "label_sha256": digest(labels),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as value:
        return {key: value[key] for key in value.files}


def format_delta(values: dict[str, float]) -> str:
    return "/".join(f"{values[key] * 100:+.2f}" for key in PRIMARY) + " pp"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--independent-root", type=Path, required=True)
    parser.add_argument("--da64-root", type=Path, required=True)
    parser.add_argument("--training-source", type=Path, required=True)
    args = parser.parse_args()

    training_sha = hashlib.sha256(args.training_source.read_bytes()).hexdigest()
    if training_sha != "d8d0e160582cafe37ae32db494c69eaee2b0f5c1668571adf0dee2a9b499fd1d":
        raise AssertionError("training source differs from completed d_a=64 run")

    roots = {
        "independent": args.independent_root,
        "self_attention_da64": args.da64_root / "results",
        "self_attention_da256": args.run_root / "results",
    }
    comparison: dict[str, Any] = {}
    validation: dict[str, Any] = {
        "status": "PASS",
        "training_source_sha256": training_sha,
        "same_training_source_as_da64": True,
        "models": {},
    }
    for model in MODELS:
        arrays = {
            method: load_npz(root / model / "test_scores.npz")
            for method, root in roots.items()
        }
        checks: dict[str, bool] = {}
        for method in ("self_attention_da64", "self_attention_da256"):
            for key in ("labels", "trajectory_indices", "positions"):
                checks[f"{method}.{key}"] = bool(
                    np.array_equal(arrays["independent"][key], arrays[method][key])
                )
                if not checks[f"{method}.{key}"]:
                    raise AssertionError(f"{model}: {method} {key} mismatch")
        method_metrics = {
            method: calculate(value["labels"], value["scores"])
            for method, value in arrays.items()
        }
        current_saved = json.loads((args.run_root / "results" / model / "metrics.json").read_text())
        if int(current_saved["attention_dimension"]) != 256:
            raise AssertionError(f"{model}: saved attention dimension is not 256")
        for key in ("auroc", "auprc", "accuracy", "f1", "precision", "recall"):
            if abs(float(current_saved["test"][key]) - method_metrics["self_attention_da256"][key]) > 1e-12:
                raise AssertionError(f"{model}: saved/recomputed {key} mismatch")
        delta_independent = {
            key: method_metrics["self_attention_da256"][key] - method_metrics["independent"][key]
            for key in ("auroc", "auprc", "accuracy", "f1", "precision", "recall")
        }
        delta_da64 = {
            key: method_metrics["self_attention_da256"][key] - method_metrics["self_attention_da64"][key]
            for key in ("auroc", "auprc", "accuracy", "f1", "precision", "recall")
        }
        delta_da64_independent = {
            key: method_metrics["self_attention_da64"][key] - method_metrics["independent"][key]
            for key in ("auroc", "auprc", "accuracy", "f1", "precision", "recall")
        }
        comparison[model] = {
            "methods": method_metrics,
            "delta_da64_minus_independent": delta_da64_independent,
            "delta_da256_minus_independent": delta_independent,
            "delta_da256_minus_da64": delta_da64,
            "da256_best_epoch": current_saved["best_epoch"],
            "da256_best_val_auroc": current_saved["best_val_auroc"],
            "da256_attention_parameters": current_saved["attention_parameters"],
            "da256_probe_parameters": current_saved["probe_parameters"],
            "da256_cuda_peak_memory_bytes": current_saved["cuda_peak_memory_bytes"],
            "da256_walltime_s": current_saved["walltime_s"],
        }
        validation["models"][model] = {
            **checks,
            "saved_metrics_exact_recompute": True,
            "attention_dimension": current_saved["attention_dimension"],
            "n": method_metrics["self_attention_da256"]["n"],
            "positive": method_metrics["self_attention_da256"]["positive"],
            "negative": method_metrics["self_attention_da256"]["negative"],
            "score_sha256": method_metrics["self_attention_da256"]["score_sha256"],
        }

    repaired = all(
        comparison[model]["delta_da256_minus_independent"][metric] >= 0
        for model in MODELS for metric in PRIMARY
    )
    consistent_width_gain = all(
        comparison[model]["delta_da256_minus_da64"][metric] > 0
        for model in MODELS for metric in PRIMARY
    )
    if repaired:
        verdict = "GO: increasing d_a to 256 repairs the three-model regression on AUROC, Accuracy and F1."
    elif consistent_width_gain:
        verdict = "PARTIAL: d_a=256 consistently improves all three primary metrics over d_a=64, but does not fully recover the independent-token baseline across all models."
    else:
        verdict = "NO-GO: increasing d_a to 256 does not consistently improve d_a=64 on all primary metrics and does not repair the three-model regression."

    result = {
        "experiment": "step_standard_self_attention_da256_width_ablation",
        "seed": 42,
        "sole_scientific_delta": {"attention_dimension": {"from": 64, "to": 256}},
        "decision_rule": "score > 0.5",
        "comparison": comparison,
        "repaired_all_models_all_primary_metrics": repaired,
        "consistent_da256_gain_over_da64_all_models_all_primary_metrics": consistent_width_gain,
        "verdict": verdict,
        "baseline_reuse": {
            "independent": str(args.independent_root),
            "self_attention_da64": str(args.da64_root),
            "retrained": False,
        },
    }
    for directory in ("results", "audits", "reports", "plots"):
        (args.run_root / directory).mkdir(parents=True, exist_ok=True)
    (args.run_root / "results" / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.run_root / "audits" / "FINAL_VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n")

    lines = [
        "# Step Self-Attention d_a=256 — Width Ablation Results",
        "",
        "The training source is byte-identical to the completed d_a=64 run. The sole scientific change is `attention_dimension: 64 -> 256`; independent and d_a=64 predictions are reused without retraining. All metrics below were independently recomputed with strict `score > 0.5`.",
        "",
        "| Model | Method | AUROC | Accuracy | F1 | Delta vs Independent (AUC/ACC/F1) | Delta vs d_a=64 (AUC/ACC/F1) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        values = comparison[model]
        for method in METHODS:
            metric = values["methods"][method]
            if method == "independent":
                delta_ind, delta_64 = "—", "—"
            elif method == "self_attention_da64":
                delta_ind = format_delta(values["delta_da64_minus_independent"])
                delta_64 = "—"
            else:
                delta_ind = format_delta(values["delta_da256_minus_independent"])
                delta_64 = format_delta(values["delta_da256_minus_da64"])
            lines.append(
                f"| {DISPLAY[model]} | {METHOD_LABELS[method]} | {metric['auroc']:.4f} | "
                f"{metric['accuracy']:.4f} | {metric['f1']:.4f} | {delta_ind} | {delta_64} |"
            )
    lines.extend([
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "All three test arrays match both archived baselines exactly by labels, trajectory indices and Step positions. No target calibration, Answer companion, feature duplication or cross-model sharing was used.",
        "",
    ])
    (args.run_root / "reports" / "RESULTS.md").write_text("\n".join(lines))

    x = np.arange(len(MODELS))
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for axis, metric, title in zip(axes, PRIMARY, ("AUROC", "Accuracy", "F1"), strict=True):
        all_values: list[float] = []
        for offset, method in zip((-width, 0.0, width), METHODS, strict=True):
            values = [comparison[model]["methods"][method][metric] for model in MODELS]
            all_values.extend(values)
            axis.bar(x + offset, values, width, label=METHOD_LABELS[method])
        axis.set_xticks(x, [DISPLAY[model] for model in MODELS])
        axis.set_title(title)
        axis.set_ylim(max(0.0, min(all_values) - 0.07), min(1.0, max(all_values) + 0.035))
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Step self-attention width ablation (seed 42)")
    fig.savefig(args.run_root / "plots" / "three_way_width_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"event": "FINAL_WIDTH_VALIDATION_PASS", "verdict": verdict}), flush=True)


if __name__ == "__main__":
    main()
