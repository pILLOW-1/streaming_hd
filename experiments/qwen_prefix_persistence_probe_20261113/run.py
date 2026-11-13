"""Fit Streaming Step linear probes on eligible TRAIN positions and score VAL."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from observed_lbfgs import ObservedLBFGS
from stream_metrics import metrics

METHODS = ("last_token", "streaming_step")
ROOT = Path("/workspace/qwen_relation")
REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def bind(domain, split, spec):
    root = ROOT / domain / split
    receipt_path = root / "TOKEN_WRITE_RECEIPT.json"
    if not receipt_path.exists():
        receipt_path = root / "TOKEN_FEATURE_MANIFEST.json"
    receipt = load(receipt_path)
    assert receipt["status"].startswith("WRITES_DURABLE") or receipt["status"] == "COMPLETE"
    assert receipt["model_revision"] == receipt["tokenizer_revision"] == REV
    assert receipt["domain"] == domain and receipt["split"] == split
    assert receipt["width"] == 5120 and receipt["layers"] == list(range(1, 65))
    assert receipt["full_native_SHA256"] == spec["full_sha"]
    labels_path = Path("/workspace") / spec["labels_key"]
    assert sha(labels_path) == spec["labels_sha"]
    source = load(labels_path)
    assert source["domain"] == domain and source["split"] == split
    offsets = root / "TOKEN_OFFSETS.json"
    if "offsets_SHA256" in receipt:
        assert receipt["offsets_SHA256"] == sha(offsets)
    rows = load(offsets)
    lookup = {row["sample_id"]: row for row in rows}
    assert len(lookup) == len(rows) == receipt["rows"]
    assert set(lookup) == {row["sample_id"] for row in source["rows"]}
    bounds = []
    labels = []
    missing = 0
    for item in source["rows"]:
        row = lookup[item["sample_id"]]
        steps = item["retained_original_step_ids"]
        ys, cs, spans = (item[k] for k in ("step_hallucination", "prefix_hallucination", "step_token_spans"))
        for i, step in enumerate(steps):
            if step == 0 or ys[i]:
                continue
            if i == 0 or steps[i - 1] != step - 1:
                missing += 1
                continue
            if not cs[i - 1]:
                continue
            lo, hi = spans[i]
            start, stop = row["spans"][0]
            assert start <= lo < hi <= stop
            bounds.append((row["offset"] + lo - start, row["offset"] + hi - start))
            labels.append(int(cs[i]))
    assert set(labels) == {0, 1}
    return root, receipt, bounds, np.array(labels, dtype=np.int64), sha(receipt_path), missing


def matrices(binding, layer, device):
    root, receipt, bounds, _, _, _ = binding
    name = f"layer_{layer:02d}/reasoning_answer.bf16.npy"
    path = root / name
    assert path.stat().st_size == receipt["files"][name]["bytes"]
    raw = np.load(path, mmap_mode="r", allow_pickle=False)
    assert raw.shape == (receipt["tokens"], 5120) and raw.dtype == np.dtype("<u2")
    last = np.empty((len(bounds), 5120), dtype=np.float32)
    pooled = np.empty_like(last)
    for i, (lo, hi) in enumerate(bounds):
        block = torch.from_numpy((raw[lo:hi].astype(np.uint32) << 16).view(np.float32)).double()
        assert torch.isfinite(block).all()
        weight = torch.linspace(0, 3, len(block), dtype=torch.float64).softmax(0)
        last[i] = block[-1].float().numpy()
        pooled[i] = (block * weight[:, None]).sum(0).float().numpy()
    return [torch.nn.functional.normalize(torch.from_numpy(x).to(device), dim=1) for x in (last, pooled)]


def fit(train, val, y, vy, layer):
    torch.manual_seed(42)
    model = torch.nn.Linear(5120, 1, device=train.device)
    candidates = []
    best = None

    @torch.no_grad()
    def accepted(iteration):
        nonlocal best
        scores = torch.sigmoid(model(val).flatten()).cpu().numpy()
        result = metrics(vy, scores, 0.5)
        candidate = dict(layer=layer, update=iteration, AUROC=float(result["AUROC"]), AP=float(result["AP"]))
        candidates.append(candidate)
        rank = (candidate["AUROC"], candidate["AP"])
        if best is None or rank > (best[0]["AUROC"], best[0]["AP"]):
            best = candidate, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, scores.copy()

    optimizer = ObservedLBFGS(model.parameters(), on_accepted=accepted, lr=1.0, max_iter=100,
                              line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(train).flatten(), y)
        loss = loss + 1e-4 * sum(0.5 * p.square().sum() for p in model.parameters())
        loss.backward()
        return loss

    optimizer.step(closure)
    assert best is not None
    return candidates, best


def main():
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    contract = load(Path("inputs.json"))
    summary = {}
    for domain in ("Logic", "Factuality"):
        path = out / f"{domain}.json"
        if path.is_file():
            summary[domain] = load(path)
            continue
        train, val = [bind(domain, split, contract[domain][split]) for split in ("train", "val")]
        y = torch.as_tensor(train[3], dtype=torch.float32, device=device)
        vy = val[3]
        candidates = {method: [] for method in METHODS}
        winners = {}
        for layer in range(1, 65):
            for method, x, v in zip(METHODS, matrices(train, layer, device), matrices(val, layer, device)):
                observed, winner = fit(x, v, y, vy, layer)
                candidates[method].extend(observed)
                if method not in winners or (winner[0]["AUROC"], winner[0]["AP"]) > (winners[method][0]["AUROC"], winners[method][0]["AP"]):
                    winners[method] = winner
            print(domain, layer, flush=True)
        details = {}
        for method, (winner, weights, scores) in winners.items():
            tied = [x for x in candidates[method] if (x["AUROC"], x["AP"]) == (winner["AUROC"], winner["AP"])]
            # Equal metric values are reported together, with no layer preference for scientific selection.
            if len(tied) == 1:
                torch.save(weights, out / f"{domain}.{method}.pt")
                np.savez(out / f"{domain}.{method}.val.npz", labels=vy, scores=scores)
            details[method] = dict(AUROC=winner["AUROC"], AP=winner["AP"], winners=tied,
                                   candidates=candidates[method])
        summary[domain] = dict(
            train_counts=np.bincount(train[3]).tolist(), val_counts=np.bincount(vy).tolist(),
            missing_predecessor={"train": train[5], "val": val[5]},
            token_receipt_sha256={"train": train[4], "val": val[4]}, readouts=details)
        atomic(path, summary[domain])
    atomic(out / "RESULT.json", dict(status="VAL_COMPLETE", device=device, domains=summary,
                                    input_sha256=sha(Path("inputs.json"))))


if __name__ == "__main__":
    main()
