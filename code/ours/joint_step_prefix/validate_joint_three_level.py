#!/usr/bin/env python3
"""Independent validation and reporting for reconstructed-data Joint Step+Prefix."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score


PROJECT = Path(__import__("os").environ.get("STREAMING_HD_LEGACY_ROOT", ""))
CORE = PROJECT / "experiments/streaming_hd/runs/core_three_level_methods_20260902"
ROOT = CORE / "extensions/joint_step_prefix"
DATA = PROJECT / "assets/streaming_hd/features_v2_three_level_20260831/canonical"
MODELS = ("qwen", "llama", "deepseek")
SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def independent_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = labels.astype(np.int64)
    predicted = (scores > 0.5).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "threshold": 0.5,
        "threshold_operator": ">",
    }


def close(saved: dict, computed: dict, context: str) -> None:
    for key in ("auroc", "auprc", "accuracy", "f1", "precision", "recall"):
        if abs(float(saved[key]) - float(computed[key])) > 1e-12:
            raise AssertionError(f"{context}/{key}: saved={saved[key]} computed={computed[key]}")
    for key in ("n", "positive", "negative"):
        if int(saved[key]) != int(computed[key]):
            raise AssertionError(f"{context}/{key}: saved={saved[key]} computed={computed[key]}")


def main() -> None:
    marker = json.loads((PROJECT / "assets/streaming_hd/features_v2_three_level_20260831/DATA_REBUILD_VALIDATION_PASS").read_text())
    if marker.get("status") != "DATA_REBUILD_VALIDATION_PASS":
        raise AssertionError("reconstructed data gate is not PASS")
    combined = {
        "status": "FINAL_VALIDATION_PASS",
        "seed": 42,
        "threshold": 0.5,
        "threshold_operator": ">",
        "dataset": str(DATA),
        "models": {},
    }
    csv_rows: list[dict] = []
    for model in MODELS:
        joint_path = ROOT / "results" / model / "metrics.json"
        prediction_path = ROOT / "results" / model / "test_predictions.npz"
        joint = json.loads(joint_path.read_text())
        core_record = json.loads((CORE / "results" / model / "metrics.json").read_text())
        if joint.get("status") != "JOINT_THREE_LEVEL_MODEL_COMPLETE" or joint.get("test_touches") != 1:
            raise AssertionError(f"{model}: incomplete joint record")
        if joint.get("dataset_manifest_sha256") != sha256(DATA / "MANIFEST.json"):
            raise AssertionError(f"{model}: canonical manifest hash mismatch")
        if joint.get("answer_companion_loaded") is not False:
            raise AssertionError(f"{model}: Answer asset entered joint run")
        if joint["gradient_flow"].get("status") != "PASS":
            raise AssertionError(f"{model}: gradient-flow audit failed")
        if not joint["optimizer"].get("single_optimizer"):
            raise AssertionError(f"{model}: optimizer is not unified")
        for split in SPLITS:
            audit = joint["input_audit"][split]
            reference = core_record["input_audit"][split]
            for key in ("steps", "prefixes", "trajectories", "step_ids_sha256", "prefix_ids_sha256"):
                if audit[key] != reference[key]:
                    raise AssertionError(f"{model}/{split}/{key}: Joint/Core population mismatch")
            if audit["valid_step_mask_count"] != audit["steps"] or audit["valid_prefix_mask_count"] != audit["prefixes"]:
                raise AssertionError(f"{model}/{split}: valid mask count mismatch")
            if audit.get("answer_tokens_in_step_or_prefix") is not False or audit.get("answer_tokens_loaded") is not False:
                raise AssertionError(f"{model}/{split}: Answer exclusion failed")

        artifact = np.load(prediction_path, allow_pickle=False)
        steps = read_jsonl(DATA / "views" / model / "test" / "steps.jsonl")
        prefixes = read_jsonl(DATA / "views" / model / "test" / "prefixes.jsonl")
        if artifact["step_sample_id"].tolist() != [row["sample_id"] for row in steps]:
            raise AssertionError(f"{model}: test Step sample order mismatch")
        if artifact["prefix_sample_id"].tolist() != [row["sample_id"] for row in prefixes]:
            raise AssertionError(f"{model}: test Prefix sample order mismatch")
        step_labels = np.asarray([int(bool(row["step_hallucination"])) for row in steps])
        prefix_labels = np.asarray([int(bool(row["cumulative_hallucination"])) for row in prefixes])
        if not np.array_equal(step_labels, artifact["step_label"]):
            raise AssertionError(f"{model}: test Step labels mismatch")
        if not np.array_equal(prefix_labels, artifact["prefix_label"]):
            raise AssertionError(f"{model}: test Prefix labels mismatch")

        recomputed = {
            "step": independent_metrics(step_labels, artifact["step_score"].astype(np.float64)),
            "prefix": independent_metrics(prefix_labels, artifact["prefix_score"].astype(np.float64)),
        }
        for level in ("step", "prefix"):
            close(joint["test"][level], recomputed[level], f"{model}/{level}")
        original = {
            "step": core_record["original_streaming"]["step"]["test"],
            "prefix": core_record["original_streaming"]["prefix"]["test"],
        }
        two_stage = {
            "step": core_record["ours_three_stage"]["step"]["test"],
            "prefix": core_record["ours_three_stage"]["prefix"]["test"],
        }
        deltas = {}
        for level in ("step", "prefix"):
            deltas[level] = {
                "joint_minus_two_stage_pp": {
                    metric: 100 * (recomputed[level][metric] - two_stage[level][metric])
                    for metric in ("auroc", "auprc", "accuracy", "f1")
                },
                "joint_minus_original_streaming_pp": {
                    metric: 100 * (recomputed[level][metric] - original[level][metric])
                    for metric in ("auroc", "auprc", "accuracy", "f1")
                },
            }
        combined["models"][model] = {
            "original_streaming": original,
            "two_stage": two_stage,
            "joint": recomputed,
            "deltas": deltas,
            "joint_record": str(joint_path),
            "prediction_sha256": sha256(prediction_path),
            "population_and_label_alignment": "PASS",
        }
        for method, values in (("Original Streaming", original), ("Ours (Two-stage)", two_stage), ("Ours (Joint)", recomputed)):
            for level in ("step", "prefix"):
                csv_rows.append({
                    "model": model,
                    "method": method,
                    "level": level,
                    **{key: values[level][key] for key in ("auroc", "auprc", "accuracy", "f1", "n", "positive", "negative")},
                })

    results = ROOT / "results"
    reports = ROOT / "reports"
    results.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (results / "metrics.json").write_text(json.dumps(combined, indent=2) + "\n")
    with (results / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "method", "level", "auroc", "auprc", "accuracy", "f1", "n", "positive", "negative"))
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# Joint Step–Prefix results on reconstructed Streaming HD",
        "",
        "Independent status: **FINAL_VALIDATION_PASS**.",
        "",
        "Seed 42 only. AUROC/AUPRC are threshold-free; Accuracy/F1 use `score > 0.5`. All methods use identical canonical reasoning Step/Prefix IDs and labels; Answer features are excluded.",
        "",
        "| Model | Method | Level | AUROC | AUPRC | Accuracy | F1 | n |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in csv_rows:
        lines.append(
            f"| {row['model']} | {row['method']} | {row['level'].title()} | {row['auroc']:.4f} | {row['auprc']:.4f} | {row['accuracy']:.4f} | {row['f1']:.4f} | {row['n']} |"
        )
    lines.extend(["", "## Joint minus Two-stage", ""])
    for model in MODELS:
        record = combined["models"][model]
        for level in ("step", "prefix"):
            delta = record["deltas"][level]["joint_minus_two_stage_pp"]
            lines.append(
                f"- {model} {level}: AUROC {delta['auroc']:+.2f} pp, Accuracy {delta['accuracy']:+.2f} pp, F1 {delta['f1']:+.2f} pp."
            )
        prefix_up = record["deltas"]["prefix"]["joint_minus_two_stage_pp"]["auroc"] > 0
        step_drop = record["deltas"]["step"]["joint_minus_two_stage_pp"]["auroc"] < 0
        lines.append(f"  Verdict: Prefix AUROC {'improved' if prefix_up else 'did not improve'}; Step AUROC {'was sacrificed' if step_drop else 'was not sacrificed'}.")
    lines.extend([
        "",
        "## Controlled differences",
        "",
        "Joint starts both modules from their archived seed-42 initialization and optimizes them in one trajectory-ordered graph. The single checkpoint is selected by validation unified-Prefix AUROC. Relative to Two-stage, unavoidable schedule-level differences are joint initialization, trajectory rather than independent-Step minibatches, one shared AdamW optimizer, and a single Prefix-selected checkpoint. Architectures, losses, labels, selected layers, feature rows, score orientation, and fixed threshold are unchanged.",
    ])
    (reports / "RESULTS.md").write_text("\n".join(lines) + "\n")
    marker_out = {
        "status": "FINAL_VALIDATION_PASS",
        "metrics": str(results / "metrics.json"),
        "comparison": str(results / "comparison.csv"),
        "report": str(reports / "RESULTS.md"),
        "models": list(MODELS),
        "seed": 42,
        "dataset_manifest_sha256": sha256(DATA / "MANIFEST.json"),
    }
    (ROOT / "FINAL_VALIDATION_PASS").write_text(json.dumps(marker_out, indent=2) + "\n")
    print(json.dumps({"event": "JOINT_THREE_LEVEL_FINAL_VALIDATION_PASS"}), flush=True)


if __name__ == "__main__":
    main()
