"""Change only the Streaming linear probe's readout to Prefix pooling."""
import json
from pathlib import Path

import numpy as np
import torch

import run


def bind_prefix(domain, split, spec):
    binding = list(original_bind(domain, split, spec))
    offsets = run.load(binding[0] / "TOKEN_OFFSETS.json")
    labels = run.load(Path("/workspace") / spec["labels_key"])["rows"]
    origins = {row["sample_id"]: row["offset"] for row in offsets}
    starts = [
        origins[row["sample_id"]]
        for row in labels
        for i, step in enumerate(row["retained_original_step_ids"])
        if i and row["retained_original_step_ids"][i - 1] == step - 1
        and not row["step_hallucination"][i] and row["prefix_hallucination"][i - 1]
    ]
    assert len(starts) == len(binding[2])
    binding[2] = [(start, end) for start, (_, end) in zip(starts, binding[2])]
    return binding


def matrices(binding, layer, device):
    root, receipt, bounds, _, _, _ = binding
    name = f"layer_{layer:02d}/reasoning_answer.bf16.npy"
    path = root / name
    assert path.stat().st_size == receipt["files"][name]["bytes"]
    raw = np.load(path, mmap_mode="r", allow_pickle=False)
    assert raw.shape == (receipt["tokens"], 5120) and raw.dtype == np.dtype("<u2")
    pooled = np.empty((len(bounds), 5120), dtype=np.float32)
    previous = None
    for i, (start, end) in enumerate(bounds):
        if start != previous:
            cursor, total = start, 0.0
            weighted = np.zeros(5120, dtype=np.float64)
        assert cursor <= end
        block = (raw[cursor:end].astype(np.uint32) << 16).view(np.float32)
        assert np.isfinite(block).all()
        decay = np.exp(-0.003 * (end - cursor))
        weights = np.exp(0.003 * (np.arange(cursor, end, dtype=np.float64) + 1 - end))
        weighted = decay * weighted + np.einsum("ij,i->j", block, weights, dtype=np.float64)
        total = decay * total + weights.sum()
        pooled[i] = weighted / total
        previous, cursor = start, end
    x = torch.from_numpy(pooled).to(device)
    return [torch.nn.functional.normalize(x, dim=1)]


if __name__ == "__main__":
    original_bind = run.bind
    run.bind = bind_prefix
    run.matrices = matrices
    run.METHODS = ("streaming_prefix",)
    run.main()
