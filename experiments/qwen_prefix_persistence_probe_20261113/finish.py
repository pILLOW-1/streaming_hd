"""Return and verify the VAL results, then update the experiment README."""
import hashlib
import json
import math
from pathlib import Path
import tempfile

import boto3
import numpy as np

EXPERIMENT = "qwen_prefix_persistence_probe_20261113"
PROJECT = Path("/shared/homes/u26487468/agent_uncertainty_quantification")
STATE = PROJECT / "state/streaming_hd" / EXPERIMENT
DEST = PROJECT / "experiments/streaming_hd/runs" / EXPERIMENT / "runpod"
DOMAINS = ("Logic", "Factuality")
METHODS = ("last_token", "streaming_step", "last_token_saplma", "streaming_prefix")
COUNTS = {"Logic": ([1333, 807], [627, 564]), "Factuality": ([400, 3326], [151, 1548])}


def ranked_metrics(labels, scores):
    order = np.argsort(scores)[::-1]
    ends = np.r_[np.flatnonzero(np.diff(scores[order])) + 1, len(labels)]
    true = np.cumsum(labels[order])[ends - 1]
    false = ends - true
    positive, negative = true[-1], false[-1]
    gained_true = np.diff(np.r_[0, true])
    gained_false = np.diff(np.r_[0, false])
    # Count ordered positive–negative pairs, assigning half credit to ties.
    auc = np.sum(gained_true * (negative - false + 0.5 * gained_false)) / (positive * negative)
    ap = np.sum(gained_true * true / ends) / positive
    return auc, ap


def eligible_val_rows(source):
    for row in source["rows"]:
        steps = row["retained_original_step_ids"]
        for i in range(1, len(steps)):
            if (steps[i - 1] == steps[i] - 1 and not row["step_hallucination"][i]
                    and row["prefix_hallucination"][i - 1]):
                yield row["sample_id"], int(row["prefix_hallucination"][i])


def within_trajectory_metrics(trajectory_ids, labels, scores):
    positions = {}
    for index, trajectory_id in enumerate(trajectory_ids):
        positions.setdefault(trajectory_id, []).append(index)
    aucs = []
    pair_counts = []
    for indices in positions.values():
        group_labels = labels[indices]
        if group_labels.min() != group_labels.max():
            aucs.append(ranked_metrics(group_labels, scores[indices])[0])
            positives = int(group_labels.sum())
            pair_counts.append(positives * (len(group_labels) - positives))
    return {
        "pair_weighted_AUROC": float(np.average(aucs, weights=pair_counts)) if aucs else None,
        "positive_negative_pairs": sum(pair_counts),
        "macro_AUROC": float(np.mean(aucs)) if aucs else None,
        "mixed_label_trajectories": len(aucs),
        "eligible_trajectories": len(positions),
    }


def main():
    client = boto3.Session(profile_name="runpod-qwen").client(
        "s3", endpoint_url="https://s3api-eu-ro-1.runpod.io", region_name="eu-ro-1")
    prefix = f"qwen_relation/results/{EXPERIMENT}/"

    def fetch_key(key):
        with client.get_object(Bucket="fcsu28qkr3", Key=key)["Body"] as stream:
            return stream.read()

    def fetch(name):
        return fetch_key(prefix + name)

    report = json.loads(fetch("RESULT.json"))
    saplma = json.loads(fetch("SAPLMA_RESULT.json"))
    assert report["status"] == "VAL_COMPLETE" and set(report["domains"]) == set(DOMAINS)
    assert report["input_sha256"] == hashlib.sha256((STATE / "input_manifest.json").read_bytes()).hexdigest()
    assert saplma["status"] == "VAL_COMPLETE" and set(saplma["domains"]) == set(DOMAINS)
    assert saplma["input_sha256"] == report["input_sha256"]
    for domain, addition in saplma["domains"].items():
        entry = report["domains"][domain]
        assert all(entry[key] == addition[key] for key in ("train_counts", "val_counts", "token_receipt_sha256"))
        assert set(addition["readouts"]) == {"last_token_saplma"}
        entry["readouts"].update(addition["readouts"])
    prefix_report = json.loads(fetch("prefix/RESULT.json"))
    assert prefix_report["status"] == "VAL_COMPLETE"
    assert set(prefix_report["domains"]) == set(DOMAINS)
    assert prefix_report["input_sha256"] == report["input_sha256"]
    for domain, addition in prefix_report["domains"].items():
        entry = report["domains"][domain]
        assert all(entry[key] == addition[key] for key in ("train_counts", "val_counts", "token_receipt_sha256"))
        assert set(addition["readouts"]) == {"streaming_prefix"}
        entry["readouts"].update(addition["readouts"])
    bindings = json.loads((STATE / "input_manifest.json").read_text())
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".runpod-return-", dir=DEST.parent) as directory:
        stage = Path(directory)
        for domain, entry in report["domains"].items():
            assert (entry["train_counts"], entry["val_counts"]) == COUNTS[domain]
            assert set(entry["readouts"]) == set(METHODS)
            spec = bindings[domain]["val"]
            raw_labels = fetch_key(spec["labels_key"])
            assert hashlib.sha256(raw_labels).hexdigest() == spec["labels_sha"]
            trajectory_ids, expected_labels = zip(*eligible_val_rows(json.loads(raw_labels)))
            for method, cell in entry["readouts"].items():
                candidates = cell["candidates"]
                assert {x["layer"] for x in candidates} == set(range(1, 65))
                rank = max((x["AUROC"], x["AP"]) for x in candidates)
                assert rank == (cell["AUROC"], cell["AP"])
                assert cell["winners"] == [x for x in candidates if (x["AUROC"], x["AP"]) == rank]
                if len(cell["winners"]) == 1:
                    name = f"{domain}.{method}.val.npz"
                    artifact = stage / name
                    key = f"prefix/{name}" if method == "streaming_prefix" else name
                    artifact.write_bytes(fetch(key))
                    with np.load(artifact, allow_pickle=False) as values:
                        labels, scores = values["labels"], values["scores"]
                        assert labels.shape == scores.shape and np.isfinite(scores).all()
                        assert np.array_equal(labels, expected_labels)
                        assert np.bincount(labels).tolist() == entry["val_counts"]
                        auc, ap = ranked_metrics(labels, scores)
                        assert math.isclose(auc, rank[0], abs_tol=1e-12)
                        assert math.isclose(ap, rank[1], abs_tol=1e-12)
                        cell["within_trajectory"] = within_trajectory_metrics(trajectory_ids, labels, scores)
                else:
                    cell["within_trajectory"] = None
        (stage / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
        (stage / "RUNPOD_RETURN_RECEIPT.json").write_text(json.dumps(
            {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.iterdir())}, indent=2) + "\n")
        if DEST.exists():
            assert all((DEST / p.name).read_bytes() == p.read_bytes() for p in stage.iterdir())
        else:
            stage.rename(DEST)
    print(json.dumps({domain: {method: {
        k: entry["readouts"][method][k] for k in ("AUROC", "AP", "winners", "within_trajectory")}
        for method in METHODS} for domain, entry in report["domains"].items()}))


if __name__ == "__main__":
    main()
