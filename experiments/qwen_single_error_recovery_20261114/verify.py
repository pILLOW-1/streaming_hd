"""Cross-check VAL artifacts independently from saved probe scores."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


def pairs(labels, scores):
    positive = scores[labels == 1]
    negative = np.sort(scores[labels == 0])
    left = np.searchsorted(negative, positive, side="left")
    right = np.searchsorted(negative, positive, side="right")
    return float((left + right).sum() / 2), len(positive) * len(negative)


root = Path(sys.argv[1])
report = json.loads((root / "RESULT.json").read_text())
assert report["status"] == "VAL_COMPLETE" and set(report["domains"]) == {"Logic", "Factuality"}
assert report["input_sha256"] == hashlib.sha256(Path(__file__).with_name("input_manifest.json").read_bytes()).hexdigest()
for domain, row in report["domains"].items():
    assert set(row["winners"]) == {"A", "B", "C", "D"}
    source = None
    for arm, winner in row["winners"].items():
        key = f"{domain}.{arm}"
        assert (root / f"{key}.pt").stat().st_size > 0
        with np.load(root / f"{key}.val.npz", allow_pickle=False) as archive:
            y, s, groups, pos, error = (archive[k] for k in (
                "labels", "scores", "sample_id", "step", "error_step"))
        assert len({(g, p) for g, p in zip(groups, pos)}) == len(y) and (error < pos).all()
        assert len(y) == len(s) and np.isfinite(s).all() and set(y) == {0, 1}
        ids = list(zip(groups, pos, error, y))
        assert source is None or ids == source
        source = ids
        assert np.bincount(y).tolist() == row["val_counts"]
        within = [pairs(y[groups == group], s[groups == group]) for group in np.unique(groups)]
        within = [(hits, n) for hits, n in within if n]
        assert len(within) == row["mixed_trajectories"] and sum(n for _, n in within) == row["within_pairs"]
        hits, n = pairs(y, s)
        checks = dict(pooled=hits / n, macro=float(np.mean([h/count for h, count in within])),
                      pair=sum(h for h, _ in within) / sum(count for _, count in within))
        for metric, value in checks.items():
            assert math.isclose(value, winner[metric], abs_tol=1e-12)
        candidates = row["candidates"][arm]
        assert sorted(c["layer"] for c in candidates) == [16, 32, 48, 64]
        assert winner == max(candidates, key=lambda c: (c["macro"], c["pair"], c["pooled"]))
print("VERIFIED 8 saved VAL predictions")
