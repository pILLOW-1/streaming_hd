#!/usr/bin/env python3
"""Independent validator and two-method report for Step self-attention d_a=64."""

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


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(np.int64)
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
        "score_sha256": digest(scores.astype(np.float64)),
        "label_sha256": digest(labels),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as value:
        return {key: value[key] for key in value.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()

    comparison: dict[str, Any] = {}
    validation: dict[str, Any] = {"status": "PASS", "models": {}, "source_sha256": None}
    validation["source_sha256"] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        baseline_path = args.baseline_root / model / "test_scores.npz"
        current_path = args.run_root / "results" / model / "test_scores.npz"
        baseline = load_npz(baseline_path)
        current = load_npz(current_path)
        checks: dict[str, bool] = {}
        for key in ("labels", "trajectory_indices", "positions"):
            if key not in baseline or key not in current:
                raise AssertionError(f"{model}: missing comparison array {key}")
            checks[f"exact_{key}"] = bool(np.array_equal(baseline[key], current[key]))
            if not checks[f"exact_{key}"]:
                raise AssertionError(f"{model}: {key} mismatch")
        baseline_metrics = metrics(baseline["labels"], baseline["scores"].astype(np.float64))
        current_metrics = metrics(current["labels"], current["scores"].astype(np.float64))
        saved = json.loads((args.run_root / "results" / model / "metrics.json").read_text())
        for key in ("auroc", "auprc", "accuracy", "f1", "precision", "recall"):
            if abs(float(saved["test"][key]) - float(current_metrics[key])) > 1e-12:
                raise AssertionError(f"{model}: saved/recomputed {key} mismatch")
        deltas = {
            key: float(current_metrics[key] - baseline_metrics[key])
            for key in ("auroc", "auprc", "accuracy", "f1", "precision", "recall")
        }
        comparison[model] = {
            "archived_independent_token_scorer": baseline_metrics,
            "standard_self_attention_da64": current_metrics,
            "delta_self_attention_minus_independent": deltas,
            "best_epoch": saved["best_epoch"],
            "best_val_auroc": saved["best_val_auroc"],
            "attention_parameters": saved["attention_parameters"],
            "probe_parameters": saved["probe_parameters"],
            "cuda_peak_memory_bytes": saved["cuda_peak_memory_bytes"],
            "walltime_s": saved["walltime_s"],
        }
        validation["models"][model] = {
            **checks,
            "n": current_metrics["n"],
            "positive": current_metrics["positive"],
            "negative": current_metrics["negative"],
            "score_sha256": current_metrics["score_sha256"],
            "saved_metrics_exact_recompute": True,
        }
        rows.extend(
            [
                {"model": model, "method": "independent", **baseline_metrics},
                {"model": model, "method": "self_attention_da64", **current_metrics},
            ]
        )

    auroc_wins = [comparison[m]["delta_self_attention_minus_independent"]["auroc"] > 0 for m in MODELS]
    f1_wins = [comparison[m]["delta_self_attention_minus_independent"]["f1"] > 0 for m in MODELS]
    if all(auroc_wins) and all(f1_wins):
        verdict = "GO: self-attention improves both AUROC and F1 consistently across all three within-model runs."
    else:
        verdict = (
            "NO-GO for a consistency claim: the self-attention replacement does not improve both AUROC "
            "and F1 in every one of the three within-model runs."
        )
    result = {
        "experiment": "step_standard_self_attention_da64_within_three_models",
        "seed": 42,
        "decision_rule": "score > 0.5",
        "comparison": comparison,
        "verdict": verdict,
        "baseline_provenance": {
            "run": str(args.baseline_root),
            "source": "${STREAMING_HD_LEGACY_ROOT}/workspaces/streaming_hd/ours/learnable_aggregation/train_step.py",
            "method": "independent per-token Linear(D,256)-BN-ReLU-Dropout(0.3)-Linear(256,1) scorer; within-Step softmax; weighted original hidden states; L2; Linear(D,1)",
            "note": "baseline metrics independently recomputed from archived full-test scores with the same strict score > 0.5 decision rule",
        },
    }
    (args.run_root / "results").mkdir(parents=True, exist_ok=True)
    (args.run_root / "audits").mkdir(parents=True, exist_ok=True)
    (args.run_root / "reports").mkdir(parents=True, exist_ok=True)
    (args.run_root / "plots").mkdir(parents=True, exist_ok=True)
    (args.run_root / "results" / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.run_root / "audits" / "FINAL_VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n")

    lines = [
        "# Step Standard Self-Attention d_a=64 — Results",
        "",
        "Three independent within-model detectors, seed 42. The canonical current-Step bf16 token states, released splits/labels, post-pooling L2 normalization, linear probe, loss, Adam optimizer, scheduler, validation-AUROC checkpointing and strict fixed `score > 0.5` decision rule are unchanged. The sole scientific replacement is the token scorer.",
        "",
        "## Primary two-method comparison",
        "",
        "| Model | Method | AUROC | Accuracy | F1 | ΔAUROC | ΔF1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        base = comparison[model]["archived_independent_token_scorer"]
        cur = comparison[model]["standard_self_attention_da64"]
        delta = comparison[model]["delta_self_attention_minus_independent"]
        lines.append(f"| {DISPLAY[model]} | Archived independent scorer | {base['auroc']:.4f} | {base['accuracy']:.4f} | {base['f1']:.4f} | — | — |")
        lines.append(f"| {DISPLAY[model]} | Standard self-attention d_a=64 | {cur['auroc']:.4f} | {cur['accuracy']:.4f} | {cur['f1']:.4f} | {delta['auroc']*100:+.2f} pp | {delta['f1']*100:+.2f} pp |")
    lines.extend(["", "## Strict verdict", "", verdict, "", "## Audit boundary", "", "All target test arrays were matched exactly to the archived scorer by labels, trajectory indices and Step positions before comparison. No target calibration, Answer companion, hidden-state duplication or cross-model sharing was used.", ""])
    (args.run_root / "reports" / "RESULTS.md").write_text("\n".join(lines))

    x = np.arange(len(MODELS))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for axis, metric, title in zip(axes, ("auroc", "f1"), ("AUROC", "F1"), strict=True):
        base = [comparison[m]["archived_independent_token_scorer"][metric] for m in MODELS]
        cur = [comparison[m]["standard_self_attention_da64"][metric] for m in MODELS]
        axis.bar(x - width / 2, base, width, label="Independent scorer")
        axis.bar(x + width / 2, cur, width, label="Self-attention d_a=64")
        axis.set_xticks(x, [DISPLAY[m] for m in MODELS])
        axis.set_title(title)
        axis.set_ylim(max(0.0, min(base + cur) - 0.08), min(1.0, max(base + cur) + 0.04))
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Within-model Step detection (seed 42)")
    fig.savefig(args.run_root / "plots" / "step_self_attention_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"event": "FINAL_VALIDATION_PASS", "verdict": verdict}), flush=True)


if __name__ == "__main__":
    main()
