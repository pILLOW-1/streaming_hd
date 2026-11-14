"""Return private predictions from RunPod and verify metrics on Cetus."""
import hashlib
import io
import json
import sys
from pathlib import Path

import boto3
import numpy as np

IDENTITY = "qwen_c18_independent_selection_20261114"
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

    def fetch(domain, name):
        key = f"qwen_relation/results/{IDENTITY}/{domain}/{name}"
        with client.get_object(Bucket="fcsu28qkr3", Key=key)["Body"] as body:
            return body.read()

    output = {}
    domains = tuple(sys.argv[1:]) or DOMAINS
    assert set(domains).issubset(DOMAINS)
    for domain in domains:
        artifacts = {name: fetch(domain, name) for name in
                     ("selection.json", "result.json", "val.npz", "test.json", "test.npz", "TEST_STARTED.json")}
        report = json.loads(artifacts["result.json"])
        test = json.loads(artifacts["test.json"])
        selection = json.loads(artifacts["selection.json"])
        assert selection["completed_layers"] == 64 and selection["max_epoch"] == 100
        assert report["prediction_sha256"] == sha(artifacts["val.npz"])
        assert test["comparisons"]["C18 independent"]["prediction_sha256"] == sha(artifacts["test.npz"])
        verified = {}
        for split in ("val", "test"):
            with np.load(io.BytesIO(artifacts[f"{split}.npz"]), allow_pickle=False) as a:
                keys = a["position_id"]
                y, c, group, prior = [a[k] for k in ("y", "c", "group", "prior")]
                assert len(set(keys)) == len(keys) and np.all(y <= c)
                novel = metrics(y, c, group, prior, a["q"], a["kappa"])
                compare_with_pin(novel, report["val"] if split == "val"
                                 else test["comparisons"]["C18 independent"]["metrics"])
                verified[split] = {"C18 independent": novel}
                for arm, loss in (("C17", "marginal_bce"), ("C18", "ce")):
                    root = REFERENCE / domain
                    if domain == "Factuality":
                        root /= domain
                    path = (VAL_REFERENCE / f"{domain}.{arm}.val.npz" if split == "val" else
                            root / "ordered_dual" / "linear" / loss / "test_predictions.npz")
                    raw = path.read_bytes()
                    expected = (json.loads((VAL_REFERENCE / f"{domain}.{arm}.json").read_text())["prediction_sha256"]
                                if split == "val" else test["comparisons"][arm]["prediction_sha256"])
                    assert sha(raw) == expected
                    with np.load(io.BytesIO(raw), allow_pickle=False) as baseline:
                        probs = align(baseline, keys)
                    q = probs[:, 0]
                    kappa = probs[:, 1] / probs[:, 1:].sum(1)
                    verified[split][arm] = metrics(y, c, group, prior, q, kappa, probs[:, :2].sum(1))
                    if split == "test":
                        compare_with_pin(verified[split][arm], test["comparisons"][arm]["metrics"])
        domain_dir = DEST / domain
        domain_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name, content in artifacts.items():
            path = domain_dir / name
            if path.exists():
                assert sha(path.read_bytes()) == sha(content)
            else:
                with path.open("xb") as stream:
                    stream.write(content)
        output[domain] = dict(selection=report["selection"], val_coverage=report["coverage"],
                              test_coverage=test["coverage"], **verified,
                              returned_sha256={k: sha(v) for k, v in artifacts.items()})
    result = DEST / ("verified.json" if domains == DOMAINS else "verified." + ".".join(domains) + ".json")
    encoded = json.dumps(output, indent=2, allow_nan=False) + "\n"
    pending = result.with_suffix(".pending")
    pending.write_text(encoded)
    pending.replace(result)
    print(encoded)


if __name__ == "__main__":
    main()
