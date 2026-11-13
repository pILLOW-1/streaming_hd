"""Evaluate the SAPLMA training recipe on the same conditional VAL cohort."""
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from run import atomic, bind, load, sha
from saplma_kernel import KerasAdam, SAPLMA
from stream_metrics import metrics


def last_token(binding, layer):
    root, receipt, spans, _, _, _ = binding
    name = f"layer_{layer:02d}/reasoning_answer.bf16.npy"
    path = root / name
    assert path.stat().st_size == receipt["files"][name]["bytes"]
    raw = np.load(path, mmap_mode="r", allow_pickle=False)
    assert raw.shape == (receipt["tokens"], 5120) and raw.dtype == np.dtype("<u2")
    # A contiguous read avoids thousands of random network-volume requests.
    last = np.asarray(raw)[[hi - 1 for _, hi in spans]]
    features = (last.astype(np.uint32) << 16).view(np.float32)
    assert np.isfinite(features).all()
    return torch.from_numpy(features)


def train_layer(train, val, layer):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    x, xv = last_token(train, layer), last_token(val, layer)
    y = torch.tensor(train[3], dtype=torch.float32)
    model = SAPLMA(x.shape[1])
    optimizer = KerasAdam(model.parameters())
    candidates, best = [], None
    for epoch in range(1, 6):
        model.train()
        for ids in torch.randperm(len(x)).split(32):
            optimizer.zero_grad()
            loss = binary_cross_entropy_with_logits(model(x[ids]).flatten(), y[ids])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(xv).flatten()).numpy()
        report = metrics(val[3], scores, 0.5)
        cell = dict(layer=layer, epoch=epoch, AUROC=report["AUROC"], AP=report["AP"])
        candidates.append(cell)
        rank = cell["AUROC"], cell["AP"]
        if best is None or rank > (best[0]["AUROC"], best[0]["AP"]):
            best = cell, {k: t.detach().clone() for k, t in model.state_dict().items()}, scores.copy()
    return candidates, best


def main():
    torch.set_num_threads(2)
    out = Path(sys.argv[1])
    contract = load(Path("inputs.json"))
    domains = {}
    for domain in ("Logic", "Factuality"):
        output = out / f"{domain}.saplma.json"
        if output.exists():
            domains[domain] = load(output)
            continue
        train, val = [bind(domain, split, contract[domain][split]) for split in ("train", "val")]
        candidates, best = [], None
        for layer in range(1, 65):
            observed, winner = train_layer(train, val, layer)
            candidates.extend(observed)
            if best is None or (winner[0]["AUROC"], winner[0]["AP"]) > (best[0]["AUROC"], best[0]["AP"]):
                best = winner
            print("SAPLMA", domain, layer, flush=True)
        winner, weights, scores = best
        rank = winner["AUROC"], winner["AP"]
        ties = [row for row in candidates if (row["AUROC"], row["AP"]) == rank]
        if len(ties) == 1:
            torch.save(weights, out / f"{domain}.last_token_saplma.pt")
            np.savez(out / f"{domain}.last_token_saplma.val.npz", labels=val[3], scores=scores)
        domains[domain] = dict(
            train_counts=np.bincount(train[3]).tolist(), val_counts=np.bincount(val[3]).tolist(),
            token_receipt_sha256={"train": train[4], "val": val[4]},
            readouts={"last_token_saplma": dict(AUROC=rank[0], AP=rank[1], winners=ties, candidates=candidates)})
        atomic(output, domains[domain])
    atomic(out / "SAPLMA_RESULT.json", dict(status="VAL_COMPLETE", domains=domains,
                                          input_sha256=sha(Path("inputs.json"))))


if __name__ == "__main__":
    main()
