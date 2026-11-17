"""Summarize attention per head and gate values on error Steps in TRAIN and VAL."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from gated import ARM, GatedStepAttention
from history import HistorySplit
from run import HERE, digest, save_json
from train import Layer


@torch.inference_mode()
def summarize(net, data, device):
    errors = np.flatnonzero(data.split.y == 1)
    totals = np.zeros((4, 10), dtype=np.float64)
    gates = []
    edges = np.linspace(0, 1, 11)
    for start in range(0, len(errors), 64):
        raw, lengths, owners, _ = data.pack(errors[start:start+64].tolist(), device)
        _, weights, gate = net.encode(raw, lengths, owners)
        gates.extend(gate.flatten().cpu().tolist())
        offset = 0
        weights = weights.double().cpu().numpy()
        for n in lengths.tolist():
            cumulative = np.vstack((np.zeros(4), np.cumsum(weights[offset:offset+n], axis=0)))
            for head in range(4):
                totals[head] += np.diff(np.interp(edges, np.arange(n+1)/n, cumulative[:, head]))
            offset += n
    means = totals / len(errors)
    np.testing.assert_allclose(means.sum(axis=1), 1, atol=1e-6)
    return dict(error_steps=len(errors), mean_weight_by_head=means.tolist(),
                gate_quantiles=np.quantile(gates, [.1, .25, .5, .75, .9]).tolist())


def main(domain, output, index):
    done = json.loads((output / "TEST_COMPLETE.json").read_text())
    selected = done["selection"]
    side = "lower" if selected["layer"] <= 32 else "upper"
    checkpoint = output / side / selected["checkpoint"]
    assert digest(checkpoint.read_bytes()) == selected["checkpoint_sha256"]
    device = torch.device("cuda")
    net = GatedStepAttention(ARM).to(device)
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    net.eval()
    sources = json.loads((HERE / "inputs.json").read_text())[domain]
    rows = {}
    for split in ("train", "val"):
        data = HistorySplit(domain, split, sources[split], index / f"{domain}.{split}.json")
        assert data.pins == selected[split]
        rows[split] = summarize(net, Layer(data, selected["layer"]), device)
    save_json(output / "DIAGNOSTIC.json", dict(domain=domain, layer=selected["layer"],
                                               epoch=selected["epoch"], splits=rows))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
