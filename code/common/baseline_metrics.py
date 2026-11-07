"""Saved-prediction metrics and method-own global VAL-F1 operating points."""
from __future__ import annotations

import math
from typing import Iterable


def confusion(labels: list[int], scores: list[float], threshold: float) -> dict:
    if len(labels) != len(scores) or not labels or not all(math.isfinite(float(s)) for s in scores):
        raise ValueError("Nonempty, finite, aligned saved predictions required")
    if any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError("label=1 must mean hallucination")
    tp = fp = tn = fn = 0
    for y, score in zip(labels, scores):
        pred = score > threshold
        tp += int(pred and y == 1)
        fp += int(pred and y == 0)
        tn += int(not pred and y == 0)
        fn += int(not pred and y == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(labels), "positive": tp + fn, "negative": tn + fp,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "ACC": (tp + tn) / len(labels), "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "precision": precision, "recall": recall, "FPR": fp / (fp + tn) if fp + tn else None}


def val_f1_threshold(labels: list[int], scores: list[float]) -> dict:
    if not labels or sum(labels) == 0:
        raise ValueError("VAL positive class undefined; never substitute TEST")
    confusion(labels, scores, max(scores))  # Validate once, outside the sweep.
    # O(n log n), with one policy per exact score boundary. Evaluate BEFORE
    # adding the tied group: strict > must leave that group negative.
    ordered = sorted(zip(map(float, scores), labels), reverse=True)
    positive = sum(labels)
    tp = fp = offset = 0
    best = (-1.0, -1.0, -math.inf)
    while offset < len(ordered):
        boundary = ordered[offset][0]
        f1 = 2 * tp / (positive + tp + fp) if positive + tp + fp else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        best = max(best, (f1, precision, boundary))
        while offset < len(ordered) and ordered[offset][0] == boundary:
            tp += ordered[offset][1]
            fp += 1 - ordered[offset][1]
            offset += 1
    all_positive = math.nextafter(ordered[-1][0], -math.inf)
    best = max(best, (2 * positive / (positive + len(labels)), positive / len(labels), all_positive))
    threshold = best[2]
    stats = confusion(labels, scores, threshold)
    # Floats are encoded by hex as well as a display number to bind exact boundaries.
    return {"threshold": threshold if math.isfinite(threshold) else None,
            "threshold_hex": threshold.hex(), "operator": ">", "label_positive": 1,
            "score_direction": "higher_is_hallucination", "selection_split": "val",
            "same_global_threshold_all_domains": True, "VAL_metrics": stats,
            "tie_break": "highest F1, highest precision, largest threshold"}


def evaluate(labels: list[int], scores: list[float], threshold: float) -> dict:
    # Import only at actual CPU evaluation time, never during source inspection.
    from sklearn.metrics import average_precision_score, roc_auc_score
    result = confusion(labels, scores, threshold)
    result["AUROC"] = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None
    result["AP"] = float(average_precision_score(labels, scores))
    result["AUPRC"] = result["AP"]  # Explicit legacy alias, NOT trapezoidal PR-AUC.
    result["AUPRC_definition"] = "sklearn.metrics.average_precision_score alias"
    result["threshold_hex"] = threshold.hex()
    result["threshold_operator"] = ">"
    return result
