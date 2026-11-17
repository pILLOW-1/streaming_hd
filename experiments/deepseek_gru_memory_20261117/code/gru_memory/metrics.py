"""Identity-rich causal detector metrics; answer labels are evaluation-only.

The unchanged final retained prefix score is transferred to the independently
annotated answer label. A missing answer label is never replaced by a prefix label.
"""
from __future__ import annotations

import math
import numpy as np

TASKS = ("step", "prefix", "final_prefix", "prefix_transfer")
METRICS = ("AUC", "AUPRC", "ACC", "F1", "Prec", "Rec", "Brier")
IDENTITY_FIELDS = ("trajectory_id", "split", "group_keys", "source_row_sha256",
                   "step_ids", "step_labels", "prefix_labels")
OPTIONAL_IDENTITY = ("native_feature_sha256", "retained_spans", "retained_span_tokens")


def binary_metrics(labels, probabilities):
    """Tied-score AUROC and noninterpolated average precision, strictly p > .5."""
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError("Invalid label/probability shapes")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Nonbinary labels or invalid probabilities")
    _, inverse = np.unique(p, return_inverse=True)
    pos = np.bincount(inverse, weights=y)
    neg = np.bincount(inverse, weights=1-y, minlength=len(pos))
    positives, negatives = int(y.sum()), int(len(y)-y.sum())
    both = bool(positives and negatives)
    auc = float(np.sum(pos*(np.cumsum(neg)-neg/2))/(positives*negatives)) if both else None
    tp_bins, totals = np.cumsum(pos[::-1]), np.cumsum((pos+neg)[::-1])
    ap = float(np.sum(pos[::-1]/positives * tp_bins/totals)) if both else None
    hard = p > .5
    tp = int(np.sum(hard & (y == 1)))
    fp = int(np.sum(hard & (y == 0)))
    fn = int(np.sum(~hard & (y == 1)))
    return dict(AUC=auc, AUPRC=ap, ACC=float(np.mean(hard == y)),
                F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                Prec=tp/(tp+fp) if tp+fp else None,
                Rec=tp/(tp+fn) if tp+fn else None,
                Brier=float(np.mean((p-y)**2)), n=len(y), positives=positives,
                threshold=.5, positive_if="probability > 0.5",
                undefined_policy="null for undefined ratios or single-class ranking metrics")


def validate_predictions(records, split=None):
    if not isinstance(records, list) or not records:
        raise ValueError("Prediction artifact must be a nonempty trajectory list")
    seen = set()
    for r in records:
        if not isinstance(r, dict) or any(k not in r for k in IDENTITY_FIELDS):
            raise ValueError("Prediction identity metadata is incomplete")
        tid = str(r["trajectory_id"])
        if not tid or tid in seen:
            raise ValueError("Duplicate or empty trajectory identity")
        seen.add(tid)
        if r["split"] not in ("train", "val", "test") or (split and r["split"] != split):
            raise ValueError("Wrong prediction split")
        if not isinstance(r["source_row_sha256"], str) or not r["source_row_sha256"]:
            raise ValueError("Missing source row identity")
        groups = r["group_keys"]
        if not isinstance(groups, list) or not groups or any(not isinstance(g, str) or not g for g in groups):
            raise ValueError("Missing question-group identity")
        ids = r["step_ids"]
        if not isinstance(ids, list) or not ids or len(set(map(str, ids))) != len(ids):
            raise ValueError("Missing or duplicated step identities")
        for task in ("step", "prefix"):
            labels, probs = r[task+"_labels"], r.get(task+"_probabilities")
            if not isinstance(labels, list) or not isinstance(probs, list) or len(labels) != len(ids) or len(probs) != len(ids):
                raise ValueError("Prediction/label/step coverage differs")
            binary_metrics(labels, probs)
        answer = r.get("answer_label")
        if answer is not None and (isinstance(answer, bool) or answer not in (0, 1)):
            raise ValueError("Independent answer label must be binary or null")
        if "answer_transfer_eligible" in r and type(r["answer_transfer_eligible"]) is not bool:
            raise ValueError("Answer eligibility must be an explicit boolean")
        if "answer_label_available" in r and r["answer_label_available"] is not (answer is not None):
            raise ValueError("Answer availability metadata disagrees with label")
        if "retained_spans" in r and len(r["retained_spans"]) != len(ids):
            raise ValueError("Native span coverage differs")
    return records


def task_arrays(records, task):
    if task in ("step", "prefix"):
        return ([x for r in records for x in r[task+"_labels"]],
                [x for r in records for x in r[task+"_probabilities"]])
    if task == "final_prefix":
        return [r["prefix_labels"][-1] for r in records], [r["prefix_probabilities"][-1] for r in records]
    if task == "prefix_transfer":
        if any(r.get("answer_label") is None for r in records):
            return None
        eligible = [r for r in records if r.get("answer_transfer_eligible", True)]
        if not eligible: return None
        return [r["answer_label"] for r in eligible], [r["prefix_probabilities"][-1] for r in eligible]
    raise ValueError("Unknown task: " + task)


def summarize(records, require_answer_labels=False):
    validate_predictions(records)
    annotated = sum(r.get("answer_label") is not None for r in records)
    if require_answer_labels and annotated != len(records):
        raise ValueError("Complete independent answer labels are required")
    result = {}
    for task in TASKS:
        arrays = task_arrays(records, task)
        if arrays is None:
            status = "NOT_AVAILABLE" if not annotated else "INCOMPLETE_LABEL_COVERAGE" if annotated != len(records) else "NO_ELIGIBLE_ANSWERS"
            result[task] = {"status": status,
                            "n": 0, "annotated_trajectories": annotated,
                            "expected_trajectories": len(records), "positives": None,
                            **{m: None for m in METRICS}}
        else:
            result[task] = dict(status="COMPLETE", **binary_metrics(*arrays))
        if task == "prefix_transfer":
            result[task].update(expected_trajectories=len(records), annotated_trajectories=annotated,
                eligible_trajectories=sum(r.get("answer_label") is not None and r.get("answer_transfer_eligible", True) for r in records),
                eligibility_policy="source supervision eligible, nonquarantined, nonempty answer; same fixed mask for every method",
                interpretation="unchanged last retained prefix probability versus independent answer hallucination label")
    return result


def evaluate_model(model, dataset, rows, device, control):
    import torch
    model.eval(); records = []
    with torch.no_grad():
        for row in rows:
            control.check(); tr = dataset.load(row)
            sl, pl = model.forward_trajectory(tr, device)
            n = len(tr.steps)
            if sl.shape != (n,) or pl.shape != (n,):
                raise ValueError("Detector changed step coverage")
            meta = {k: tr.metadata[k] for k in IDENTITY_FIELDS[:4]+OPTIONAL_IDENTITY if k in tr.metadata}
            ids = [step.step_id for step in tr.steps]
            if tr.metadata.get("step_ids", ids) != ids:
                raise ValueError("Step metadata disagrees")
            meta.update(step_ids=ids, step_labels=tr.step_labels.cpu().tolist(),
                        prefix_labels=tr.prefix_labels.cpu().tolist(),
                        step_probabilities=sl.sigmoid().cpu().tolist(),
                        prefix_probabilities=pl.sigmoid().cpu().tolist(),
                        answer_label=tr.metadata.get("answer_label"))
            for key in ("answer_label_source", "answer_label_available", "answer_transfer_eligible", "answer_transfer_exclusion_reasons"):
                if key in tr.metadata: meta[key] = tr.metadata[key]
            records.append(meta)
    return summarize(records, require_answer_labels=dataset.identity.get("kind") == "pillow_native_cache"), records
