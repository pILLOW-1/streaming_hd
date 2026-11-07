#!/usr/bin/env python3
"""Independent fixed-threshold validator and three-model comparison report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score


MODELS = ("qwen", "llama", "deepseek")
PAPER_LOCAL = {
    "qwen": {"auroc": 0.8802, "accuracy": 0.7890, "f1": 0.8048},
    "llama": {"auroc": 0.8730, "accuracy": 0.7938, "f1": 0.8245},
    "deepseek": {"auroc": 0.8798, "accuracy": 0.8714, "f1": 0.6943},
}
DIMENSIONS = {"qwen": 3584, "llama": 4096, "deepseek": 4096}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    predictions = (scores > 0.5).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "threshold": 0.5,
        "threshold_operator": ">",
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
    }


def compare_one(run_root: Path, scalar_root: Path, model: str) -> dict[str, Any]:
    vector_dir = run_root / "results" / model
    vector_metrics_path = vector_dir / "metrics.json"
    vector_predictions_path = vector_dir / "test_predictions.npz"
    vector_checkpoint_path = vector_dir / "checkpoint.pt"
    scalar_metrics_path = scalar_root / "results" / model / "seed_42" / "metrics.json"
    scalar_predictions_path = scalar_root / "results" / model / "seed_42" / "test_predictions.npz"
    for path in (vector_metrics_path, vector_predictions_path, vector_checkpoint_path, scalar_metrics_path, scalar_predictions_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    saved = json.loads(vector_metrics_path.read_text(encoding="utf-8"))
    scalar = json.loads(scalar_metrics_path.read_text(encoding="utf-8"))
    with np.load(vector_predictions_path, allow_pickle=False) as vector_npz, np.load(scalar_predictions_path, allow_pickle=False) as scalar_npz:
        vector_labels = vector_npz["prefix_label"].astype(np.int64)
        vector_scores = vector_npz["score"].astype(np.float64)
        array_checks = {}
        for vector_key, scalar_key in (
            ("prefix_label", "prefix_label"),
            ("step_label", "step_label"),
            ("trajectory_index", "trajectory_index"),
            ("step_id", "step_id"),
            ("flat_step_index", "flat_step_index"),
        ):
            array_checks[vector_key] = bool(np.array_equal(vector_npz[vector_key], scalar_npz[scalar_key]))
        if not all(array_checks.values()):
            raise AssertionError(f"{model}: vector/scalar test arrays differ: {array_checks}")
        recomputed = metrics(vector_labels, vector_scores)

    for key in ("auroc", "auprc", "accuracy", "f1", "precision", "recall"):
        if not np.isclose(recomputed[key], saved["test"]["local"][key], rtol=0, atol=1e-12):
            raise AssertionError(f"{model}: saved {key} differs from independent recomputation")
    if saved.get("classification_rule") != "score > 0.5" or saved.get("answer_companion_loaded") is not False:
        raise AssertionError(f"{model}: evaluation boundary mismatch")

    checkpoint = torch.load(vector_checkpoint_path, map_location="cpu", weights_only=False)
    dimension = DIMENSIONS[model]
    expected_weight = [dimension, 2 * dimension]
    expected_bias = [dimension]
    if checkpoint.get("gate_weight_shape") != expected_weight or checkpoint.get("gate_bias_shape") != expected_bias:
        raise AssertionError(f"{model}: checkpoint is not a full vector gate")
    state = checkpoint["state_dict"]
    if list(state["gate.weight"].shape) != expected_weight or list(state["gate.bias"].shape) != expected_bias:
        raise AssertionError(f"{model}: state_dict full-gate shape mismatch")
    expected_parameters = dimension * (2 * dimension) + dimension
    if checkpoint.get("gate_parameters") != expected_parameters:
        raise AssertionError(f"{model}: gate parameter count mismatch")

    scalar_local = scalar["test"]["local"]
    deltas = {
        key: 100.0 * (recomputed[key] - scalar_local[key])
        for key in ("auroc", "accuracy", "f1")
    }
    return {
        "status": "PASS",
        "model": model,
        "array_checks": array_checks,
        "fixed_threshold_metrics_exact": True,
        "full_vector_gate_shape": {"weight": expected_weight, "bias": expected_bias},
        "gate_parameters": expected_parameters,
        "vector": recomputed,
        "scalar_seed42": {key: scalar_local[key] for key in ("auroc", "auprc", "accuracy", "f1", "precision", "recall")},
        "streaming_paper_local": PAPER_LOCAL[model],
        "vector_minus_scalar_percentage_points": deltas,
        "files": {
            "metrics": str(vector_metrics_path),
            "predictions": str(vector_predictions_path),
            "checkpoint": str(vector_checkpoint_path),
        },
        "sha256": {
            "metrics": sha256(vector_metrics_path),
            "predictions": sha256(vector_predictions_path),
            "checkpoint": sha256(vector_checkpoint_path),
        },
    }


def fmt(values: dict[str, float]) -> str:
    return f"AUC {100*values['auroc']:.2f} / ACC {100*values['accuracy']:.2f} / F1 {100*values['f1']:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scalar-root", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    arguments = parser.parse_args()

    model_results = {model: compare_one(arguments.run_root, arguments.scalar_root, model) for model in MODELS}
    consistently_improved = all(
        model_results[model]["vector"][metric] > model_results[model]["scalar_seed42"][metric]
        for model in MODELS for metric in ("auroc", "accuracy", "f1")
    )
    verdict = (
        "GO: the full vector gate consistently improves AUROC, Accuracy and F1 over the archived scalar gate for all three models."
        if consistently_improved
        else "NO-GO: the full vector gate does not consistently improve AUROC, Accuracy and F1 over the archived scalar gate across all three models."
    )
    validation = {
        "status": "PASS",
        "seed": 42,
        "method": "prefix_full_vector_gate",
        "implementation": str(arguments.implementation),
        "implementation_sha256": sha256(arguments.implementation),
        "scalar_reference_root": str(arguments.scalar_root),
        "models": model_results,
        "consistent_improvement": consistently_improved,
        "verdict": verdict,
    }
    metrics_path = arguments.run_root / "metrics" / "metrics.json"
    audit_path = arguments.run_root / "audits" / "FINAL_VALIDATION.json"
    report_path = arguments.run_root / "reports" / "RESULTS.md"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Streaming HD full vector-gated Prefix transition", "",
        "Status: **PASS**", "",
        "Three independent within-model runs used seed 42. The learned-token Step branch and canonical features were frozen; the sole method change from the archived scalar run was `Linear(2d,1) -> Linear(2d,d)`. Test classification used the fixed rule `score > 0.5`.", "",
        "| Model | Streaming paper | Scalar gate (archived, seed 42) | Full vector gate (seed 42) | Vector − scalar (AUC/ACC/F1 pp) | Gate parameters |",
        "|---|---|---|---|---:|---:|",
    ]
    for model in MODELS:
        row = model_results[model]
        delta = row["vector_minus_scalar_percentage_points"]
        lines.append(
            f"| {model.title()} | {fmt(row['streaming_paper_local'])} | {fmt(row['scalar_seed42'])} | {fmt(row['vector'])} | "
            f"{delta['auroc']:+.2f}/{delta['accuracy']:+.2f}/{delta['f1']:+.2f} | {row['gate_parameters']:,} |"
        )
    lines.extend([
        "", "## Verdict", "", verdict, "",
        "Prefix-Local is the frozen primary endpoint. AUROC is threshold-free; Accuracy/F1 use `score > 0.5`. No Answer companion, virtual Answer Step, target calibration, cross-model sharing, or Step retraining was used.", "",
        "The full gate has materially greater capacity than the scalar gate; any gain is an effectiveness result, not evidence of parameter efficiency.", "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    run_state_path = arguments.run_root / "RUN_STATE.md"
    with run_state_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Automated completion\n\n"
            "- Status: COMPLETE_PASS\n"
            "- Independent validator: PASS\n"
            f"- Verdict: {verdict}\n"
            "- Machine-readable metrics: `metrics/metrics.json`\n"
            "- Validation audit: `audits/FINAL_VALIDATION.json`\n"
            "- Final report: `reports/RESULTS.md`\n"
        )
    print(json.dumps({"event": "FULL_VECTOR_FINAL_VALIDATION_PASS", "verdict": verdict}), flush=True)


if __name__ == "__main__":
    main()
