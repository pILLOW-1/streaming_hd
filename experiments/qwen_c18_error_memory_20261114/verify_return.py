"""Return private predictions from RunPod and verify metrics on Cetus."""
import hashlib
import io
import json
import sys
from pathlib import Path

import boto3
import numpy as np

IDENTITY = "qwen_c18_error_memory_20261114"
BASE = Path("/shared/homes/u26487468/agent_uncertainty_quantification")
RUNS = BASE / "experiments/streaming_hd/runs"
DEST = RUNS / IDENTITY / "runpod"
DOMAINS = ("Logic", "Factuality")
REFERENCE = RUNS / "step_prefix_ordered_dual_qwen_wd1e4_20261111/runpod"
VAL_REFERENCE = RUNS / "qwen_ordered_dual_recovery_20261113"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def auc_count(labels, scores):
    positive = scores[labels == 1]
    negative = np.sort(scores[labels == 0])
    lo = np.searchsorted(negative, positive, "left")
    hi = np.searchsorted(negative, positive, "right")
    return float(np.sum(lo + hi) / 2), len(positive) * len(negative)


def auc(labels, scores):
    hits, count = auc_count(labels, scores)
    assert count
    return hits / count


def metrics(y, c, group, prior, q, kappa, prefix=None):
    for probability in (q, kappa):
        assert np.isfinite(probability).all()
        assert ((0 <= probability) & (probability <= 1)).all()
    if prefix is None:
        prefix = q + (1 - q) * kappa
    mask = (y == 0) & prior
    ids = group[mask]
    local = c[mask]

    def recovery(scores):
        selected = scores[mask]
        terms = [auc_count(local[ids == g], selected[ids == g]) for g in np.unique(ids)]
        mixed = [(a, b) for a, b in terms if b]
        return dict(macro=float(np.mean([a / b for a, b in mixed])),
                    pair=sum(a for a, _ in mixed) / sum(b for _, b in mixed),
                    pooled=auc(local, selected))

    return dict(step=auc(y, q), prefix=auc(c, prefix), kappa=recovery(kappa),
                prefix_recovery=recovery(prefix))


def align(saved, keys):
    ids = list(saved["position_id"])
    lookup = {key: i for i, key in enumerate(ids)}
    assert len(lookup) == len(ids) and set(lookup) == set(keys)
    p = saved["probability"][[lookup[key] for key in keys]].astype(np.float64)
    assert p.shape == (len(keys), 3) and np.isfinite(p).all()
    assert np.allclose(p.sum(1), 1)
    return p


def compare_with_pin(actual, expected):
    for key, value in actual.items():
        if isinstance(value, dict):
            compare_with_pin(value, expected[key])
        else:
            assert abs(value - expected[key]) < 2e-6, (key, value, expected[key])



def main():
    client = boto3.Session(profile_name="runpod-qwen").client(
        "s3", endpoint_url="https://s3api-eu-ro-1.runpod.io", region_name="eu-ro-1")
    domains = tuple(sys.argv[1:]) or DOMAINS
    assert set(domains).issubset(DOMAINS)
    for domain in domains:
        base = f"qwen_relation/results/{IDENTITY}/{domain}/"

        def fetch(name):
            with client.get_object(Bucket="fcsu28qkr3", Key=base + name)["Body"] as body:
                return body.read()

        assert json.loads(fetch("TEST_COMPLETE.json")) == {"domain": domain, "arms": ["A2", "A3"]}
        output = {"domain": domain, "val": {}, "test": {}, "selection": {}, "coverage": {}}
        for arm in ("A2", "A3"):
            sid = f"{arm}/selection.json"
            selection = json.loads(fetch(sid))
            test = json.loads(fetch(f"{arm}/test.json"))
            guard = json.loads(fetch(f"{arm}/TEST_STARTED.json"))
            assert guard["selection"] == selection and test["arm"] == selection["arm"] == arm
            assert test["domain"] == selection["domain"] == domain
            assert guard["inputs"] == test["inputs"]
            output["selection"][arm] = selection
            artifacts = (("val", selection["prediction"], selection["prediction_sha256"], selection["evaluation"]),
                         ("test", "test.npz", test["prediction_sha256"], test["metrics"]))
            for split, name, expected, reported in artifacts:
                data = fetch(f"{arm}/{name}")
                assert sha(data) == expected
                with np.load(io.BytesIO(data), allow_pickle=False) as saved:
                    keys = saved["position_id"]
                    y, c, group, prior = [saved[k] for k in ("y", "c", "group", "prior")]
                    assert len(set(keys)) == len(keys) and np.all(y <= c)
                    p = align(saved, keys)
                    actual = metrics(y, c, group, prior, p[:, 0], p[:, 1] / p[:, 1:].sum(1),
                                     p[:, :2].sum(1))
                    compare_with_pin(actual, reported)
                    output[split][arm] = actual
                    output["coverage"][split] = selection["coverage"] if split == "val" else test["coverage"]
                    for label, ref, loss in (("A1", "C18", "ce"), ("C17", "C17", "marginal_bce")):
                        path = REFERENCE / domain
                        if domain == "Factuality":
                            path /= domain
                        path = (VAL_REFERENCE / f"{domain}.{ref}.val.npz" if split == "val" else
                                path / "ordered_dual" / "linear" / loss / "test_predictions.npz")
                        raw = path.read_bytes()
                        pin = (json.loads((VAL_REFERENCE / f"{domain}.{ref}.json").read_text())["prediction_sha256"]
                               if split == "val" else test["references"][label]["prediction_sha256"])
                        assert sha(raw) == pin
                        with np.load(io.BytesIO(raw), allow_pickle=False) as reference:
                            probs = align(reference, keys)
                        value = metrics(y, c, group, prior, probs[:, 0],
                                        probs[:, 1] / probs[:, 1:].sum(1), probs[:, :2].sum(1))
                        assert output[split].get(label, value) == value
                        output[split][label] = value
                folder = DEST / domain / arm
                folder.mkdir(mode=0o700, exist_ok=True, parents=True)
                target = folder / f"{split}.npz"
                if target.exists():
                    assert sha(target.read_bytes()) == expected
                else:
                    pending = target.with_suffix(".partial")
                    pending.write_bytes(data)
                    pending.replace(target)
        result = DEST / domain / "verified.json"
        pending = result.with_suffix(".partial")
        pending.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        pending.replace(result)
        print(json.dumps(output))


if __name__ == "__main__":
    main()
