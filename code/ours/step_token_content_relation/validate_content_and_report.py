#!/usr/bin/env python3
"""Independent validation and report for the Step content-relation residual."""

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


def file_sha256(path: Path) -> str:
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


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
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
        "gamma": (),
        "scorer.0.weight": (256, dimension),
        "scorer.4.weight": (1, 256),
        "probe.weight": (1, dimension),
        "relation_proj.weight": (64, dimension),
        "q_proj.weight": (64, dimension),
        "k_proj.weight": (64, dimension),
        "v_proj.weight": (64, dimension),
        "position_embedding.weight": (9, 16),
        "edge_mlp.0.weight": (64, 18),
        "edge_mlp.0.bias": (64,),
        "edge_mlp.2.weight": (64, 64),
        "edge_mlp.2.bias": (64,),
        "edge_key_proj.weight": (64, 64),
        "edge_value_proj.weight": (64, 64),
        "context_layer_norm.weight": (64,),
        "rel_head.weight": (1, 64),
        "rel_head.bias": (1,),
    }
    actual = {name: tuple(state[name].shape) for name in expected}
    if actual != expected:
        raise AssertionError(f"checkpoint architecture mismatch: {actual} != {expected}")
    return {
        "status": "PASS",
        "shapes": {name: list(shape) for name, shape in actual.items()},
        "learned_gamma": float(state["gamma"]),
    }


def length_diagnostics(
    labels: np.ndarray,
    lengths: np.ndarray,
    independent: np.ndarray,
    position: np.ndarray,
    content: np.ndarray,
) -> list[dict[str, Any]]:
    bins = ((1, 8), (9, 16), (17, 32), (33, 64), (65, None))
    result: list[dict[str, Any]] = []
    for lower, upper in bins:
        selected = lengths >= lower
        if upper is not None:
            selected &= lengths <= upper
        y = labels[selected]
        row: dict[str, Any] = {
            "range": f"{lower}-{upper}" if upper is not None else f">={lower}",
            "n": int(selected.sum()),
            "positive": int(y.sum()),
            "negative": int(y.size - y.sum()),
        }
        if len(np.unique(y)) == 2:
            auc = {
                "independent": float(roc_auc_score(y, independent[selected])),
                "position_key_value": float(roc_auc_score(y, position[selected])),
                "content_relation": float(roc_auc_score(y, content[selected])),
            }
            row["auroc"] = auc
            row["delta_content_minus_independent"] = (
                auc["content_relation"] - auc["independent"]
            )
            row["delta_content_minus_position"] = (
                auc["content_relation"] - auc["position_key_value"]
            )
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--independent-root", type=Path, required=True)
    parser.add_argument("--position-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()

    comparison: dict[str, Any] = {}
    validation: dict[str, Any] = {
        "status": "PASS",
        "source_sha256": file_sha256(args.source),
        "models": {},
    }
    for model in MODELS:
        paths = {
            "independent": args.independent_root / model / "test_scores.npz",
            "position": args.position_root / "results" / model / "test_scores.npz",
            "content": args.run_root / "results" / model / "test_scores.npz",
        }
        payloads = {name: load_npz(path) for name, path in paths.items()}
        reference = payloads["independent"]
        alignment: dict[str, bool] = {}
        for method, payload in payloads.items():
            for key in ("labels", "trajectory_indices", "positions"):
                name = f"{method}_exact_{key}"
                alignment[name] = bool(np.array_equal(reference[key], payload[key]))
                if not alignment[name]:
                    raise AssertionError(f"{model}: {method} {key} mismatch")
        recomputed = {
            method: metrics(payload["labels"], payload["scores"])
            for method, payload in payloads.items()
        }
        saved = json.loads(
            (args.run_root / "results" / model / "metrics.json").read_text()
        )
        if saved.get("method") != "step_content_relation_residual_chord_da64_k4":
            raise AssertionError(f"{model}: wrong method")
        for key in ("auroc", "auprc", "accuracy", "f1", "precision", "recall"):
            if abs(saved["test"][key] - recomputed["content"][key]) > 1e-12:
                raise AssertionError(f"{model}: saved/recomputed {key} mismatch")
        smoke = json.loads(
            (args.run_root / "results" / model / "smoke_audit.json").read_text()
        )
        if smoke.get("status") != "PASS" or smoke["smoke"].get("status") != "PASS":
            raise AssertionError(f"{model}: smoke did not pass")
        required_smoke = (
            "dense_chunk_match",
            "self_pair_cosine_one_distance_zero",
            "shared_edge_separate_key_value_projections",
            "padded_keys_zero_contribution",
        )
        if not all(smoke["smoke"].get(key) is True for key in required_smoke):
            raise AssertionError(f"{model}: mandatory relation smoke check missing")
        for name in (
            "relation_proj.weight",
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "edge_mlp.0.weight",
            "edge_mlp.2.weight",
            "edge_key_proj.weight",
            "edge_value_proj.weight",
            "rel_head.weight",
        ):
            if smoke["smoke"]["second_backward_after_one_update_gradient_norms"][name] <= 0:
                raise AssertionError(f"{model}: zero second-backward gradient {name}")
        checkpoint = verify_checkpoint(
            args.run_root / "results" / model / "checkpoint.pt", DIMENSION[model]
        )
        delta_independent = {
            key: recomputed["content"][key] - recomputed["independent"][key]
            for key in ("auroc", "auprc", "accuracy", "f1", "precision", "recall")
        }
        delta_position = {
            key: recomputed["content"][key] - recomputed["position"][key]
            for key in ("auroc", "auprc", "accuracy", "f1", "precision", "recall")
        }
        diagnostics = {
            "learned_gamma": saved["learned_gamma"],
            "mean_abs_gamma_delta": saved["mean_abs_gamma_delta"],
            "mean_abs_independent_token_logit": saved["mean_abs_independent_token_logit"],
            "correction_to_independent_ratio": saved["correction_to_independent_ratio"],
            "token_weight_entropy_mean": saved["token_weight_entropy_mean"],
            "token_weight_entropy_std": saved["token_weight_entropy_std"],
            "length_bins": length_diagnostics(
                reference["labels"].astype(np.int64),
                payloads["content"]["token_lengths"],
                reference["scores"].astype(np.float64),
                payloads["position"]["scores"].astype(np.float64),
                payloads["content"]["scores"].astype(np.float64),
            ),
        }
        comparison[model] = {
            "archived_independent": recomputed["independent"],
            "archived_position_key_value": recomputed["position"],
            "content_relation": recomputed["content"],
            "delta_content_minus_independent": delta_independent,
            "delta_content_minus_position": delta_position,
            "diagnostics": diagnostics,
            "best_epoch": saved["best_epoch"],
            "best_val_auroc": saved["best_val_auroc"],
            "relation_parameters": saved["relation_parameters"],
            "total_parameters": saved["total_parameters"],
            "cuda_peak_memory_bytes": saved["cuda_peak_memory_bytes"],
            "walltime_s": saved["walltime_s"],
        }
        validation["models"][model] = {
            **alignment,
            "saved_metrics_exact_recompute": True,
            "checkpoint": checkpoint,
            "prediction_sha256": {name: file_sha256(path) for name, path in paths.items()},
            "n": recomputed["content"]["n"],
            "positive": recomputed["content"]["positive"],
            "negative": recomputed["content"]["negative"],
        }

    passed = all(
        comparison[model]["delta_content_minus_independent"]["auroc"] > 0
        for model in MODELS
    )
    verdict = (
        "GO: content relation improves AUROC over Independent in all three models; recommend attribution controls without running them."
        if passed
        else "NO-GO: content relation does not improve AUROC over Independent in all three models; stop this branch without capacity additions."
    )
    result = {
        "experiment": "step_token_content_relation_within_three_models_20260831",
        "seed": 42,
        "decision_rule": "score > 0.5",
        "comparison": comparison,
        "verdict": verdict,
        "provenance": {
            "independent_root": str(args.independent_root),
            "position_key_value_root": str(args.position_root),
            "source": str(args.source),
            "scientific_change": "replace position-only relation with directed position+cosine+log-distance edge MLP and gamma-zero recovery gate",
        },
    }
    for directory in ("results", "audits", "reports", "plots"):
        (args.run_root / directory).mkdir(parents=True, exist_ok=True)
    (args.run_root / "results" / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (args.run_root / "audits" / "FINAL_VALIDATION.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )

    lines = [
        "# Step Content-Relation Residual — Results",
        "",
        "Three independent within-model seed-42 runs. The CHORD-style edge uses directed clipped position, cosine, and raw-projection log-distance. Metrics are independently recomputed from aligned full-test predictions; Accuracy/F1 use strict `score > 0.5`.",
        "",
        "| Model | Method | AUROC | Accuracy | F1 | ΔAUROC vs Independent | ΔAUROC vs Position K+V |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = comparison[model]
        independent = row["archived_independent"]
        position = row["archived_position_key_value"]
        content = row["content_relation"]
        di = row["delta_content_minus_independent"]
        dp = row["delta_content_minus_position"]
        lines.append(f"| {DISPLAY[model]} | Independent (archived) | {independent['auroc']:.4f} | {independent['accuracy']:.4f} | {independent['f1']:.4f} | — | — |")
        lines.append(f"| {DISPLAY[model]} | Position Key+Value (archived) | {position['auroc']:.4f} | {position['accuracy']:.4f} | {position['f1']:.4f} | {(position['auroc']-independent['auroc'])*100:+.2f} pp | — |")
        lines.append(f"| {DISPLAY[model]} | Content relation (new) | {content['auroc']:.4f} | {content['accuracy']:.4f} | {content['f1']:.4f} | {di['auroc']*100:+.2f} pp | {dp['auroc']*100:+.2f} pp |")
    lines.extend(["", "## Diagnostics", "", "| Model | gamma | |gamma·delta| / |a_ind| | Token-weight entropy |", "|---|---:|---:|---:|"])
    for model in MODELS:
        diag = comparison[model]["diagnostics"]
        lines.append(f"| {DISPLAY[model]} | {diag['learned_gamma']:.6g} | {diag['correction_to_independent_ratio']:.6f} | {diag['token_weight_entropy_mean']:.6f} |")
    lines.extend(["", "Step-length-bin diagnostics are stored in `results/metrics.json`; they are descriptive and were not used for selection.", "", "## Verdict", "", verdict, "", "## Validation boundary", "", "The validator required exact labels, trajectory indices, and Step positions across archived Independent, archived Position Key+Value, and Content predictions; recomputed metrics; checked checkpoint architecture and gamma; and consumed no Answer companion.", ""])
    (args.run_root / "reports" / "RESULTS.md").write_text("\n".join(lines))

    x = np.arange(len(MODELS))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for axis, metric_name, title in zip(axes, ("auroc", "f1"), ("AUROC", "F1"), strict=True):
        values = {
            "Independent": [comparison[m]["archived_independent"][metric_name] for m in MODELS],
            "Position K+V": [comparison[m]["archived_position_key_value"][metric_name] for m in MODELS],
            "Content": [comparison[m]["content_relation"][metric_name] for m in MODELS],
        }
        for index, (label, scores) in enumerate(values.items()):
            axis.bar(x + (index - 1) * width, scores, width, label=label)
        axis.set_xticks(x, [DISPLAY[m] for m in MODELS])
        axis.set_title(title)
        axis.set_ylim(max(0.0, min(min(v) for v in values.values()) - 0.08), 1.0)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Step content-relation residual, seed 42")
    fig.savefig(args.run_root / "plots" / "content_relation_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"event": "CONTENT_RELATION_FINAL_VALIDATION_PASS", "verdict": verdict}), flush=True)


if __name__ == "__main__":
    main()
