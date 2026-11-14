"""Fit four white-box probes on the single-earlier-error cohort."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LAYERS = (16, 32, 48, 64)
ARMS = ("A", "B", "C", "D")
REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ROOT = Path("/workspace")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bind(domain, split, spec):
    root = ROOT / "qwen_relation" / domain / split
    receipt_path = root / "TOKEN_WRITE_RECEIPT.json"
    if not receipt_path.is_file():
        receipt_path = root / "TOKEN_FEATURE_MANIFEST.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt["status"].startswith("WRITES_DURABLE") or receipt["status"] == "COMPLETE"
    assert receipt["model_revision"] == receipt["tokenizer_revision"] == REV
    assert (receipt["domain"], receipt["split"]) == (domain, split)
    assert receipt["full_native_SHA256"] == spec["full_sha"] and receipt["width"] == 5120
    label_bytes = (ROOT / spec["labels_key"]).read_bytes()
    assert sha(label_bytes) == spec["labels_sha"]
    source = json.loads(label_bytes)
    assert (source["domain"], source["split"]) == (domain, split)
    raw_offsets = (root / "TOKEN_OFFSETS.json").read_bytes()
    if "offsets_SHA256" in receipt:
        assert sha(raw_offsets) == receipt["offsets_SHA256"]
    offsets = {row["sample_id"]: row for row in json.loads(raw_offsets)}
    assert len(offsets) == len(source["rows"]) == receipt["rows"]
    assert set(offsets) == {row["sample_id"] for row in source["rows"]}
    examples, censored = [], 0
    for row in source["rows"]:
        sample = row["sample_id"]
        entry = offsets[sample]
        origin, stop = entry["spans"][0]
        base = entry["offset"]
        steps = row["retained_original_step_ids"]
        ys, cs, spans = (row[k] for k in (
            "step_hallucination", "prefix_hallucination", "step_token_spans"))
        assert len(steps) == len(ys) == len(cs) == len(spans)
        error_ends = []
        for ix, (step, y, c, (left, end)) in enumerate(zip(steps, ys, cs, spans)):
            assert y in (0, 1) and c in (0, 1) and y <= c and origin <= left < end <= stop
            if step != ix:
                censored += len(steps) - ix
                break
            if y:
                error_ends.append((step, base + end - origin))
            elif len(error_ends) == 1:
                error_step, error_end = error_ends[0]
                examples.append((sample, step, error_step, base, base + end - origin, error_end, c))
    return root, receipt, examples, dict(receipt=sha(receipt_bytes), offsets=sha(raw_offsets),
                                          labels=sha(label_bytes), censored_after_gap=censored)


def readouts(binding, layer):
    root, receipt, examples, _ = binding
    name = f"layer_{layer:02d}/reasoning_answer.bf16.npy"
    path = root / name
    assert path.stat().st_size == receipt["files"][name]["bytes"]
    raw = np.load(path, mmap_mode="r", allow_pickle=False)
    assert raw.shape == (receipt["tokens"], 5120) and raw.dtype == np.dtype("<u2")
    prefix, current, error = (np.empty((len(examples), 5120), np.float32) for _ in range(3))
    cursor, last_base = 0, None
    for k, (_, _, _, base, end, error_end, _) in enumerate(examples):
        if base != last_base:
            cursor, mass = base, 0.
            state = np.zeros(5120, np.float64)
            err = (raw[error_end - 1].astype(np.uint32) << 16).view(np.float32)
        assert cursor <= end and base < error_end < end
        block = (raw[cursor:end].astype(np.uint32) << 16).view(np.float32)
        assert np.isfinite(block).all() and np.isfinite(err).all()
        decay = np.exp(-0.003 * (end - cursor))
        weights = np.exp(0.003 * (np.arange(cursor, end, dtype=np.float64) + 1 - end))
        state = decay * state + np.einsum("ij,i->j", block, weights, dtype=np.float64)
        mass = decay * mass + weights.sum()
        prefix[k], current[k], error[k] = state / mass, block[-1], err
        cursor, last_base = end, base
    return [F.normalize(torch.from_numpy(x), dim=1) for x in (prefix, current, error)]


def inputs(readout, arm):
    prefix, current, error = readout
    if arm == "A":
        return prefix
    if arm == "B":
        return current
    blocks = [current, error]
    if arm == "D":
        blocks += [F.normalize(current * error, dim=1),
                   F.normalize((current - error).abs(), dim=1)]
    return torch.cat(blocks, dim=1)


def auc(labels, scores):
    order = np.argsort(scores)
    ends = np.r_[np.flatnonzero(np.diff(scores[order])) + 1, len(labels)]
    positive = np.cumsum(labels[order])[ends - 1]
    negative = ends - positive
    p, n = np.diff(np.r_[0, positive]), np.diff(np.r_[0, negative])
    return float(np.sum(p * (negative - 0.5 * n)) / (positive[-1] * negative[-1]))


def evaluate(labels, scores, mixed):
    assert np.isfinite(scores).all()
    within = [auc(labels[ix], scores[ix]) for ix, _ in mixed]
    pairs = [n for _, n in mixed]
    return dict(macro=float(np.mean(within)), pair=float(np.average(within, weights=pairs)),
                pooled=auc(labels, scores))


def rank(metrics):
    return tuple(metrics[k] for k in ("macro", "pair", "pooled"))


def fit(x, y, vx, val_labels, mixed, arm):
    torch.manual_seed(42)
    width = x.shape[1]
    model = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 1)) if arm == "D" else nn.Linear(width, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    batches = torch.Generator().manual_seed(42)
    best = None
    for epoch in range(1, 21):
        for idx in torch.randperm(len(x), generator=batches).split(128):
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x[idx]).flatten(), y[idx])
            loss.backward()
            opt.step()
        with torch.no_grad():
            scores = model(vx).flatten().numpy()
        metrics = evaluate(val_labels, scores, mixed)
        if best is None or rank(metrics) > rank(best[0]):
            best = metrics, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()}, scores.copy()
    return best


def main():
    torch.set_num_threads(2)
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(__file__).with_name("input_manifest.json").read_bytes()
    contract = json.loads(manifest)
    result = {}
    for domain in ("Logic", "Factuality"):
        train, val = [bind(domain, split, contract[domain][split]) for split in ("train", "val")]
        labels = np.array([x[-1] for x in val[2]])
        y = torch.tensor([x[-1] for x in train[2]], dtype=torch.float32)
        groups = np.array([x[0] for x in val[2]])
        mixed = []
        for group in np.unique(groups):
            ix = np.flatnonzero(groups == group)
            n = int(labels[ix].sum())
            if n and n < len(ix):
                mixed.append((ix, n * (len(ix) - n)))
        print(domain, "COHORT", len(y), int(y.sum()), len(labels), int(labels.sum()),
              len(mixed), flush=True)
        assert mixed and set(y.tolist()) == {0, 1}
        winners, candidates = {}, {arm: [] for arm in ARMS}
        for layer in LAYERS:
            tr, va = readouts(train, layer), readouts(val, layer)
            for arm in ARMS:
                metrics, epoch, state, scores = fit(inputs(tr, arm), y, inputs(va, arm), labels, mixed, arm)
                candidate = dict(layer=layer, epoch=epoch, **metrics)
                candidates[arm].append(candidate)
                if arm not in winners or rank(metrics) > rank(winners[arm]):
                    winners[arm] = candidate
                    torch.save(state, output / f"{domain}.{arm}.pt")
                    np.savez(output / f"{domain}.{arm}.val.npz", labels=labels, scores=scores,
                             sample_id=groups, step=[x[1] for x in val[2]],
                             error_step=[x[2] for x in val[2]])
                print(domain, layer, arm, candidate, flush=True)
        result[domain] = dict(train_counts=np.bincount(y.numpy().astype(int)).tolist(),
                              val_counts=np.bincount(labels).tolist(),
                              val_trajectories=len(np.unique(groups)), mixed_trajectories=len(mixed),
                              within_pairs=sum(n for _, n in mixed), winners=winners,
                              candidates=candidates, sources={"train": train[3], "val": val[3]})
    target = output / "RESULT.json"
    pending = target.with_suffix(".partial")
    pending.write_text(json.dumps(dict(status="VAL_COMPLETE", domains=result,
                                       input_sha256=sha(manifest)), indent=2, allow_nan=False) + "\n")
    pending.replace(target)


if __name__ == "__main__":
    main()
