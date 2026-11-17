"""Profile pooling weight by normalized token position on labeled VAL error Steps."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def binned_mass(weights, edges):
    # Each token represents an equal-width interval in the normalized Step.
    # Interpolate the cumulative mass at the bin boundaries to preserve mass.
    cumulative = np.concatenate(([0.0], np.cumsum(weights, dtype=np.float64)))
    token_boundaries = np.linspace(0, 1, len(weights) + 1)
    return np.diff(np.interp(edges, token_boundaries, cumulative))


@torch.inference_mode()
def profile(raw, spans, scorer, edges):
    totals = np.zeros((2, len(edges) - 1), dtype=np.float64)
    peaks = np.zeros_like(totals)
    for start in range(0, len(spans), 64):
        batch = spans[start:start + 64]
        lengths = np.diff(batch, axis=1).ravel()
        tokens = np.concatenate([raw[lo:hi] for lo, hi in batch])
        x = torch.from_numpy(tokens.view(np.int16)).view(torch.bfloat16).float()
        logits = scorer(x).reshape(-1)
        assert torch.isfinite(logits).all()
        for scores in logits.split(lengths.tolist()):
            fixed = 3 * torch.linspace(0, 1, scores.numel(), dtype=torch.float32)
            for arm, vector in enumerate((scores, fixed)):
                mass = binned_mass(vector.softmax(dim=0).double().numpy(), edges)
                assert (mass >= 0).all() and abs(mass.sum() - 1) < 1e-6
                totals[arm] += mass
                winners = torch.where(vector == vector.max())[0].numpy()
                n = vector.numel()
                indices = np.minimum(9, winners * 10 // (n - 1)) if n > 1 else np.array([5])
                peaks[arm] += np.bincount(indices, minlength=10) / len(winners)
    return totals / len(spans), peaks / len(spans)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.input.read_text())
    assert spec["split"] == "val" and spec["target_step_error"] == 1
    assert 0 < spec["positions"] <= spec["all_val_positions"]
    feature, checkpoint = Path(spec["feature"]), Path(spec["checkpoint"])
    assert digest(feature) == spec["feature_sha256"]
    assert digest(checkpoint) == spec["checkpoint_sha256"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert (saved["selection"]["layer"], saved["selection"]["epoch"]) == (spec["layer"], spec["epoch"])
    scorer = torch.nn.Sequential(torch.nn.Linear(5120, 256), torch.nn.BatchNorm1d(256),
                                 torch.nn.ReLU(), torch.nn.Dropout(.3), torch.nn.Linear(256, 1))
    scorer.load_state_dict({k[7:]: v for k, v in saved["model"].items() if k.startswith("scorer.")})
    scorer.eval()
    raw = np.load(feature, mmap_mode="r", allow_pickle=False)
    spans = np.asarray(spec["spans"], dtype=np.int64)
    lengths = np.diff(spans, axis=1).ravel()
    assert raw.ndim == 2 and raw.shape[1] == 5120 and raw.dtype == np.dtype("<u2")
    assert len(lengths) == spec["positions"] and lengths.min() > 0
    assert spans.min() >= 0 and spans.max() <= len(raw)
    edges = np.linspace(0, 1, 11)
    means, peaks = profile(raw, spans, scorer, edges)
    assert np.allclose(means.sum(axis=1), 1, atol=1e-6, rtol=0)
    receipt = {name: spec[name] for name in (
        "domain", "split", "layer", "epoch", "positions", "all_val_positions",
        "target_step_error", "error_positions_sha256", "feature_sha256", "checkpoint_sha256")}
    receipt.update(bin_edges=edges.tolist(), single_token_steps=int((lengths == 1).sum()),
                   top_token_fraction=dict(zip(('learned', 'fixed'), peaks.tolist())),
                   mean_weight_mass=dict(zip(("learned", "fixed"), means.tolist())))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    tmp.replace(args.out)
    print("VERIFIED", spec["domain"], spec["positions"], flush=True)


if __name__ == "__main__":
    main()
