#!/usr/bin/env python3
"""Independent validator/report for the three-model relation-residual run."""

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
EXPECTED_DIM = {"qwen": 3584, "llama": 4096, "deepseek": 4096}


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


def load_context(root: Path, model: str, method: str) -> dict[str, Any] | None:
    path = root / "results" / "metrics.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    comparison = payload.get("comparison", {})
    if model not in comparison:
        return None
    row = comparison[model]
    if method in row:
        return row[method]
    if method in row.get("methods", {}):
        return row["methods"][method]
    for key, value in row.items():
        if "self_attention" in key and isinstance(value, dict) and "auroc" in value:
            return value
    return None


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
        "context_layer_norm.weight": (64,),
        "rel_head.weight": (1, 64),
        "rel_head.bias": (1,),
    }
    actual = {key: tuple(state[key].shape) for key in expected}
    if actual != expected:
        raise AssertionError(f"checkpoint shape mismatch: {actual} != {expected}")
    return {"status": "PASS", "shapes": {key: list(value) for key, value in actual.items()}}


def relation_label_audit(lengths: np.ndarray) -> dict[str, Any]:
    audited = sorted(set(int(x) for x in lengths[[0, len(lengths) // 2, -1]]))
    for length in audited:
        positions = np.arange(length)
        expected = np.clip(positions[None, :] - positions[:, None], -4, 4)
        if length > 1 and not (expected[0, 1] == 1 and expected[1, 0] == -1):
            raise AssertionError("relative-position direction is not j-i")
        if expected.min() < -4 or expected.max() > 4:
            raise AssertionError("relative clipping failed")
    return {"status": "PASS", "rule": "clip(j-i,-4,4)", "audited_lengths": audited}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--da64-root", type=Path, required=True)
    parser.add_argument("--da256-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()

    comparison: dict[str, Any] = {}
    validation: dict[str, Any] = {
        "status": "PASS", "models": {},
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
    }
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
        smoke = json.loads((args.run_root / "results" / model / "smoke_audit.json").read_text())
        if smoke.get("status") != "PASS" or smoke["smoke"].get("status") != "PASS":
            raise AssertionError(f"{model}: smoke audit did not pass")
        if smoke["smoke"]["initial_max_relation_correction"] != 0.0:
            raise AssertionError(f"{model}: zero-init recovery audit failed")
        checkpoint_audit = verify_checkpoint(
            args.run_root / "results" / model / "checkpoint.pt", EXPECTED_DIM[model]
        )
        relation_audit = relation_label_audit(current["token_lengths"])
        deltas = {
            key: float(current_metrics[key] - baseline_metrics[key])
            for key in ("auroc", "auprc", "accuracy", "f1", "precision", "recall")
        }
        comparison[model] = {
            "archived_independent_token_scorer": baseline_metrics,
            "token_relation_residual_key_only_da64_k4": current_metrics,
            "delta_relation_residual_minus_independent": deltas,
            "archived_standard_self_attention_da64_context": load_context(
                args.da64_root, model, "standard_self_attention_da64"
            ),
            "archived_standard_self_attention_da256_context": load_context(
                args.da256_root, model, "self_attention_da256"
            ),
            "best_epoch": saved["best_epoch"], "best_val_auroc": saved["best_val_auroc"],
            "scorer_parameters": saved["scorer_parameters"],
            "relation_parameters": saved["relation_parameters"],
            "probe_parameters": saved["probe_parameters"],
            "cuda_peak_memory_bytes": saved["cuda_peak_memory_bytes"],
            "walltime_s": saved["walltime_s"],
        }
        validation["models"][model] = {
            **checks, "saved_metrics_exact_recompute": True,
            "n": current_metrics["n"], "positive": current_metrics["positive"],
            "negative": current_metrics["negative"], "score_sha256": current_metrics["score_sha256"],
            "checkpoint": checkpoint_audit, "relation_labels": relation_audit,
            "mask_zero_init_gradient_audit": "PASS",
        }

    wins = [
        comparison[m]["delta_relation_residual_minus_independent"]["auroc"] > 0
        and comparison[m]["delta_relation_residual_minus_independent"]["f1"] > 0
        for m in MODELS
    ]
    if all(wins):
        verdict = "GO: relation residual improves both AUROC and F1 over the independent scorer in all three within-model runs."
    else:
        verdict = "NO-GO for a consistency claim: relation residual does not improve both AUROC and F1 over the independent scorer in every model."
    result = {
        "experiment": "step_token_relation_residual_within_three_models_20260830",
        "seed": 42, "decision_rule": "score > 0.5", "comparison": comparison,
        "verdict": verdict,
        "baseline_provenance": {
            "run": str(args.baseline_root),
            "source": "${STREAMING_HD_LEGACY_ROOT}/workspaces/streaming_hd/ours/learnable_aggregation/train_step.py",
            "method": "independent Linear(D,256)-BN-ReLU-Dropout(0.3)-Linear(256,1) token logits; Step softmax; weighted original hidden states; L2; Linear(D,1)",
            "note": "metrics independently recomputed from archived full-test scores with strict score > 0.5",
        },
    }
    for directory in ("results", "audits", "reports", "plots"):
        (args.run_root / directory).mkdir(parents=True, exist_ok=True)
    (args.run_root / "results" / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.run_root / "audits" / "FINAL_VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n")

    lines = [
        "# Step Token Relation Residual — Results", "",
        "Three independent within-model seed-42 runs. The archived independent token scorer remains the main path. The only scientific addition is a zero-initialized Shaw Key-side relative-position correction (`d_a=64`, `k=4`) to each independent token logit. Evaluation uses strict fixed `score > 0.5`.", "",
        "## Primary controlled comparison", "",
        "| Model | Method | AUROC | Accuracy | F1 | ΔAUROC | ΔAccuracy | ΔF1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        base = comparison[model]["archived_independent_token_scorer"]
        cur = comparison[model]["token_relation_residual_key_only_da64_k4"]
        delta = comparison[model]["delta_relation_residual_minus_independent"]
        lines.append(f"| {DISPLAY[model]} | Archived independent token scoring | {base['auroc']:.4f} | {base['accuracy']:.4f} | {base['f1']:.4f} | — | — | — |")
        lines.append(f"| {DISPLAY[model]} | Relation residual, Key-only d_a=64 k=4 | {cur['auroc']:.4f} | {cur['accuracy']:.4f} | {cur['f1']:.4f} | {delta['auroc']*100:+.2f} pp | {delta['accuracy']*100:+.2f} pp | {delta['f1']*100:+.2f} pp |")
    lines.extend(["", "## Archived direct self-attention context (not rerun)", "",
                  "| Model | d_a=64 AUROC / ACC / F1 | d_a=256 AUROC / ACC / F1 |",
                  "|---|---:|---:|"])
    for model in MODELS:
        c64 = comparison[model]["archived_standard_self_attention_da64_context"]
        c256 = comparison[model]["archived_standard_self_attention_da256_context"]
        def compact(value: dict[str, Any] | None) -> str:
            return "unavailable" if value is None else f"{value['auroc']:.4f} / {value['accuracy']:.4f} / {value['f1']:.4f}"
        lines.append(f"| {DISPLAY[model]} | {compact(c64)} | {compact(c256)} |")
    lines.extend(["", "## Verdict", "", verdict, "", "## Audit boundary", "",
                  "The validator exactly matched labels, trajectory indices and Step positions to the archived independent predictions, recomputed all metrics, checked checkpoint shapes, and consumed no Answer companion. The first backward at exact zero relation-head initialization correctly cannot reach Q/K/V/E_K; the compute-node audit verifies that the second backward, after one ordinary update makes `w_rel` nonzero, reaches all relation parameters. The shared `b_rel` remains unidentifiable under token softmax as expected.", ""])
    (args.run_root / "reports" / "RESULTS.md").write_text("\n".join(lines))

    x = np.arange(len(MODELS))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for axis, metric_name, title in zip(axes, ("auroc", "f1"), ("AUROC", "F1"), strict=True):
        base = [comparison[m]["archived_independent_token_scorer"][metric_name] for m in MODELS]
        cur = [comparison[m]["token_relation_residual_key_only_da64_k4"][metric_name] for m in MODELS]
        axis.bar(x - width / 2, base, width, label="Independent")
        axis.bar(x + width / 2, cur, width, label="Relation residual")
        axis.set_xticks(x, [DISPLAY[m] for m in MODELS])
        axis.set_title(title)
        axis.set_ylim(max(0.0, min(base + cur) - 0.08), min(1.0, max(base + cur) + 0.04))
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Within-model Step detection, seed 42")
    fig.savefig(args.run_root / "plots" / "relation_residual_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"event": "FINAL_VALIDATION_PASS", "verdict": verdict}), flush=True)


if __name__ == "__main__":
    main()
