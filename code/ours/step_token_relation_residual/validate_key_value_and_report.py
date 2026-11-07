#!/usr/bin/env python3
"""Independent validator for the Shaw Key+Value relation-residual experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score


MODELS = ("qwen", "llama", "deepseek")
DISPLAY = {"qwen": "Qwen", "llama": "Llama", "deepseek": "DeepSeek"}
DIMENSION = {"qwen": 3584, "llama": 4096, "deepseek": 4096}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def recompute(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
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
        "score_sha256": array_sha256(scores),
        "label_sha256": array_sha256(labels),
    }


def verify_checkpoint(path: Path, dimension: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    expected = {
        "scorer.0.weight": (256, dimension),
        "scorer.4.weight": (1, 256),
        "probe.weight": (1, dimension),
        "q_proj.weight": (64, dimension),
        "k_proj.weight": (64, dimension),
        "v_proj.weight": (64, dimension),
        "rel_key.weight": (9, 64),
        "rel_value.weight": (9, 64),
        "context_layer_norm.weight": (64,),
        "rel_head.weight": (1, 64),
        "rel_head.bias": (1,),
    }
    actual = {name: tuple(state[name].shape) for name in expected}
    if actual != expected:
        raise AssertionError(f"checkpoint architecture mismatch: {actual} != {expected}")
    if int(np.prod(actual["rel_value.weight"])) != 576:
        raise AssertionError("E_V must add exactly 576 parameters")
    return {
        "status": "PASS",
        "shapes": {name: list(shape) for name, shape in actual.items()},
        "added_value_parameters": 576,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--independent-root", type=Path, required=True)
    parser.add_argument("--key-only-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()

    comparison: dict[str, Any] = {}
    validation: dict[str, Any] = {
        "status": "PASS",
        "source_sha256": sha256(args.source),
        "models": {},
    }
    for model in MODELS:
        paths = {
            "independent": args.independent_root / model / "test_scores.npz",
            "key_only": args.key_only_root / "results" / model / "test_scores.npz",
            "key_value": args.run_root / "results" / model / "test_scores.npz",
        }
        payloads = {name: load_npz(path) for name, path in paths.items()}
        reference = payloads["independent"]
        alignment: dict[str, bool] = {}
        for method, payload in payloads.items():
            for field in ("labels", "trajectory_indices", "positions"):
                key = f"{method}_exact_{field}"
                alignment[key] = bool(np.array_equal(reference[field], payload[field]))
                if not alignment[key]:
                    raise AssertionError(f"{model}: {method} {field} misaligned")

        method_metrics = {
            method: recompute(payload["labels"], payload["scores"])
            for method, payload in payloads.items()
        }
        saved = json.loads(
            (args.run_root / "results" / model / "metrics.json").read_text()
        )
        if saved.get("value_side_relation") is not True or saved.get("added_value_parameters") != 576:
            raise AssertionError(f"{model}: saved run is not the frozen Key+Value method")
        for metric in ("auroc", "auprc", "accuracy", "f1", "precision", "recall"):
            if abs(saved["test"][metric] - method_metrics["key_value"][metric]) > 1e-12:
                raise AssertionError(f"{model}: saved/recomputed {metric} mismatch")

        smoke = json.loads(
            (args.run_root / "results" / model / "smoke_audit.json").read_text()
        )
        smoke_body = smoke["smoke"]
        if smoke.get("status") != "PASS" or smoke_body.get("status") != "PASS":
            raise AssertionError(f"{model}: smoke did not pass")
        if smoke_body["key_only_initialization_audit"]["added_parameter_count"] != 576:
            raise AssertionError(f"{model}: E_V parameter audit missing")
        if smoke_body["initial_max_relation_correction"] != 0.0:
            raise AssertionError(f"{model}: zero-init recovery failed")
        for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "rel_key.weight", "rel_value.weight"):
            if smoke_body["second_backward_after_one_ordinary_update_gradient_norms"][name] <= 0:
                raise AssertionError(f"{model}: no second-backward gradient for {name}")

        checkpoint = verify_checkpoint(
            args.run_root / "results" / model / "checkpoint.pt", DIMENSION[model]
        )
        delta_independent = {
            metric: method_metrics["key_value"][metric] - method_metrics["independent"][metric]
            for metric in ("auroc", "auprc", "accuracy", "f1", "precision", "recall")
        }
        delta_key_only = {
            metric: method_metrics["key_value"][metric] - method_metrics["key_only"][metric]
            for metric in ("auroc", "auprc", "accuracy", "f1", "precision", "recall")
        }
        comparison[model] = {
            "archived_independent_token_scorer": method_metrics["independent"],
            "archived_relation_residual_key_only_da64_k4": method_metrics["key_only"],
            "relation_residual_key_value_da64_k4": method_metrics["key_value"],
            "delta_key_value_minus_independent": delta_independent,
            "delta_key_value_minus_key_only": delta_key_only,
            "best_epoch": saved["best_epoch"],
            "best_val_auroc": saved["best_val_auroc"],
            "walltime_s": saved["walltime_s"],
            "cuda_peak_memory_bytes": saved["cuda_peak_memory_bytes"],
        }
        validation["models"][model] = {
            **alignment,
            "saved_metrics_exact_recompute": True,
            "checkpoint": checkpoint,
            "prediction_sha256": {name: sha256(path) for name, path in paths.items()},
            "n": method_metrics["key_value"]["n"],
            "positive": method_metrics["key_value"]["positive"],
            "negative": method_metrics["key_value"]["negative"],
            "mask_value_placement_zero_init_gradient_audit": "PASS",
        }

    consistent = all(
        comparison[model]["delta_key_value_minus_independent"]["auroc"] > 0
        and comparison[model]["delta_key_value_minus_key_only"]["auroc"] > 0
        for model in MODELS
    )
    verdict = (
        "GO: Key+Value improves AUROC over both Independent and Key-only in all three models."
        if consistent
        else "NO-GO for the consistency claim: Key+Value does not improve AUROC over both Independent and Key-only in every model."
    )
    result = {
        "experiment": "step_token_relation_key_value_within_three_models_20260831",
        "seed": 42,
        "decision_rule": "score > 0.5",
        "comparison": comparison,
        "verdict": verdict,
        "provenance": {
            "independent_root": str(args.independent_root),
            "key_only_root": str(args.key_only_root),
            "source": str(args.source),
            "only_scientific_change": "E_V[clip(j-i,-4,4)] added inside the attention-weighted value message",
        },
    }
    for name in ("results", "audits", "reports", "plots"):
        (args.run_root / name).mkdir(parents=True, exist_ok=True)
    (args.run_root / "results" / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.run_root / "audits" / "FINAL_VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n")

    lines = [
        "# Step Shaw Key+Value Relation Residual — Results", "",
        "Three independent within-model seed-42 runs. The only change from archived Key-only is `E_V[clip(j-i,-4,4)]` inside the attention-weighted value message. All metrics are independently recomputed from aligned full-test predictions; Accuracy/F1 use strict `score > 0.5`.", "",
        "| Model | Method | AUROC | Accuracy | F1 | ΔAUROC vs Independent | ΔAUROC vs Key-only | ΔF1 vs Independent | ΔF1 vs Key-only |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        rows = comparison[model]
        independent = rows["archived_independent_token_scorer"]
        key_only = rows["archived_relation_residual_key_only_da64_k4"]
        key_value = rows["relation_residual_key_value_da64_k4"]
        di = rows["delta_key_value_minus_independent"]
        dk = rows["delta_key_value_minus_key_only"]
        lines.append(f"| {DISPLAY[model]} | Independent (archived) | {independent['auroc']:.4f} | {independent['accuracy']:.4f} | {independent['f1']:.4f} | — | — | — | — |")
        lines.append(f"| {DISPLAY[model]} | Key-only (archived) | {key_only['auroc']:.4f} | {key_only['accuracy']:.4f} | {key_only['f1']:.4f} | {(key_only['auroc']-independent['auroc'])*100:+.2f} pp | — | {(key_only['f1']-independent['f1'])*100:+.2f} pp | — |")
        lines.append(f"| {DISPLAY[model]} | Key+Value (new) | {key_value['auroc']:.4f} | {key_value['accuracy']:.4f} | {key_value['f1']:.4f} | {di['auroc']*100:+.2f} pp | {dk['auroc']*100:+.2f} pp | {di['f1']*100:+.2f} pp | {dk['f1']*100:+.2f} pp |")
    lines.extend(["", "## Verdict", "", verdict, "", "## Validation boundary", "",
                  "The validator required exact equality of labels, trajectory indices, and Step positions across Independent, Key-only, and Key+Value predictions; recomputed all metrics; verified the 576-parameter E_V tensor and checkpoint shapes; and consumed no Answer companion.", ""])
    (args.run_root / "reports" / "RESULTS.md").write_text("\n".join(lines))

    x = np.arange(len(MODELS))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for axis, metric, title in zip(axes, ("auroc", "f1"), ("AUROC", "F1"), strict=True):
        values = {
            "Independent": [comparison[m]["archived_independent_token_scorer"][metric] for m in MODELS],
            "Key-only": [comparison[m]["archived_relation_residual_key_only_da64_k4"][metric] for m in MODELS],
            "Key+Value": [comparison[m]["relation_residual_key_value_da64_k4"][metric] for m in MODELS],
        }
        for index, (label, scores) in enumerate(values.items()):
            axis.bar(x + (index - 1) * width, scores, width, label=label)
        axis.set_xticks(x, [DISPLAY[m] for m in MODELS])
        axis.set_title(title)
        axis.set_ylim(max(0.0, min(min(v) for v in values.values()) - 0.08), 1.0)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Step relation residual, seed 42")
    fig.savefig(args.run_root / "plots" / "key_value_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"event": "KEY_VALUE_FINAL_VALIDATION_PASS", "verdict": verdict}), flush=True)


if __name__ == "__main__":
    main()
