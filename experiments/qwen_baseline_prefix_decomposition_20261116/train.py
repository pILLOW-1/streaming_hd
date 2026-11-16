"""Fit the conditional Prefix heads using existing Qwen features on RunPod."""

import argparse
import fcntl
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.nn import functional as F

from evaluate_test import baseline_root
from objective import prefix_bce, prefix_probability
from run import Split, digest, save_json
from saplma_kernel import KerasAdam, SAPLMA


def read_json(path):
    return json.loads(path.read_bytes())


def checked(path, expected):
    assert digest(path.read_bytes()) == expected, path
    return path


def test_split(domain, index_root):
    index = read_json(index_root / f"{domain}.test.json")
    gold = baseline_root(domain) / "ce"
    path = checked(gold / "test_predictions.npz", read_json(gold / "TEST_RESULT.json")["predictions_SHA256"])
    with np.load(path, allow_pickle=False) as archive:
        labels = dict(zip(archive["position_id"], zip(archive["y"], archive["c"], strict=True), strict=True))
    rows = []
    for row in index["rows"]:
        sample = row["sample_id"]
        selected = [(i, span, *map(int, labels[f"{sample}:{i}"]))
                    for i, span in enumerate(row["step_token_spans"]) if f"{sample}:{i}" in labels]
        steps, spans, y, c = zip(*selected, strict=True) if selected else ([], [], [], [])
        rows.append(dict(sample_id=sample, token_sha=row["token_sha"], origin=row["origin"],
                         retained_original_step_ids=steps, step_token_spans=spans,
                         step_hallucination=y, prefix_hallucination=c))
    data = Split(domain, "test", {"full_sha": index["full_sha"]}, rows=rows)
    assert set(data.keys) == set(labels)
    return data


def saved_step_scores(path, data, method, device):
    with np.load(path, allow_pickle=False) as archive:
        ids, label = ("position_id", "labels") if method == "saplma" else ("sample_id", "y")
        lookup = {key: i for i, key in enumerate(archive[ids])}
        assert len(lookup) == len(archive[ids]) and set(lookup) == set(data.keys)
        order = [lookup[key] for key in data.keys]
        assert np.array_equal(archive[label][order], data.y)
        q = archive["scores"][order]
    assert np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
    return torch.as_tensor(q, dtype=torch.float32, device=device)


def features(data, layer, method, level, device):
    readout = "last_token" if method == "saplma" else f"{level}_exp"
    x = data.features(layer, (readout,))[0].to(device)
    return F.normalize(x, dim=1) if method == "streaming" else x


@torch.inference_mode()
def make_train_scores(path, frozen_path, method, data, layer, device):
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            assert saved["frozen_sha"].item() == digest(frozen_path.read_bytes())
            assert saved["positions_sha"].item() == data.pins["positions"]
            return torch.from_numpy(saved["q"]).to(device)
    checkpoint = torch.load(frozen_path, map_location="cpu", weights_only=True)
    if method == "saplma":
        model = SAPLMA(5120)
        assert checkpoint["identity"]["layer"] == layer and not checkpoint["options"]["normalize"]
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = torch.nn.Linear(5120, 1)
        assert checkpoint["layer"] == layer
        model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    x = features(data, layer, method, "step", device)
    q = torch.cat([model(block).flatten().sigmoid() for block in x.split(1024)])
    pending = path.with_name("train_step.pending.npz")
    np.savez_compressed(pending, q=q.cpu().numpy(), frozen_sha=digest(frozen_path.read_bytes()),
                        positions_sha=data.pins["positions"])
    pending.replace(path)
    return q


def metrics(labels, score):
    return dict(AUROC=float(roc_auc_score(labels, score)), AP=float(average_precision_score(labels, score)))


def threshold_at_val_f1(labels, score):
    precision, recall, thresholds = precision_recall_curve(labels, score)
    f1 = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                   out=np.zeros_like(thresholds, dtype=np.float64),
                   where=precision[:-1] + recall[:-1] > 0)
    strict = np.nextafter(thresholds, np.float32(-np.inf))
    winner = np.lexsort((strict, precision[:-1], f1))[-1]
    return float(strict[winner])


@torch.inference_mode()
def predict(model, x, q):
    model.eval()
    return prefix_probability(model(x).flatten(), q).detach().cpu().numpy()


def fit(train, val, q_train, q_val, layer, method, device, saplma_epochs):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    x, xv = (features(data, layer, method, "prefix", device) for data in (train, val))
    step = torch.as_tensor(train.y, device=device)
    prefix = torch.as_tensor(train.c, device=device)
    if method == "saplma":
        model = SAPLMA(x.shape[1]).to(device)
        optimizer = KerasAdam(model.parameters())
        epochs = saplma_epochs
    else:
        model = torch.nn.Linear(x.shape[1], 1).to(device)
        initial = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=100)

        def closure():
            initial.zero_grad()
            objective = prefix_bce(model(x).flatten(), q_train, step, prefix)
            objective = objective + 1e-4 * sum(p.square().sum() * 0.5 for p in model.parameters())
            objective.backward()
            return objective

        initial.step(closure)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        epochs = 20
        trajectories = [torch.arange(slots.start, slots.stop, device=device)
                        for _, _, _, slots in sorted(train.rows, key=lambda row: row[-1].start)]
        weights = torch.ones_like(q_train)
        weights[torch.stack([ids[-1] for ids in trajectories])] = 30
    best, candidates = None, []
    for epoch in range(1, epochs + 1):
        model.train()
        if method == "saplma":
            eligible = torch.where(step == 0)[0]
            batches = eligible[torch.randperm(len(eligible), device=device)].split(32)
            weight = None
        else:
            random.shuffle(trajectories)
            batches = [torch.cat(trajectories[start:start + 64]) for start in range(0, len(trajectories), 64)]
            weight = weights
        for ids in batches:
            if not (step[ids] == 0).any():
                continue
            optimizer.zero_grad(set_to_none=True)
            logit = model(x[ids]).flatten()
            loss = prefix_bce(logit, q_train[ids], step[ids], prefix[ids],
                              None if weight is None else weight[ids])
            # The original Streaming alarm sync is identically zero because p >= q.
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        if method == "saplma" and epoch != epochs:
            continue
        probability = predict(model, xv, q_val)
        scores = dict(layer=layer, epoch=epoch, **metrics(val.c, probability))
        candidates.append(scores)
        rank = (scores["AUROC"], scores["AP"])
        if best is None or rank > best[0]:
            best = rank, scores, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, probability
    return best, candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=("Logic", "Factuality"))
    parser.add_argument("method", choices=("saplma", "streaming"))
    parser.add_argument("root", type=Path)
    parser.add_argument("split_pins", type=Path)
    parser.add_argument("history_index", type=Path)
    parser.add_argument("input_manifest_sha")
    parser.add_argument("--saplma-epochs", type=int, default=5, choices=(5,))
    args = parser.parse_args()
    domain, method = args.domain, args.method
    ownership = args.root / "DOMAIN_OWNERS.json"
    if ownership.exists() and read_json(ownership)[domain] != os.environ.get("RUNPOD_POD_ID"):
        print(domain, method, "assigned to the other domain worker", flush=True)
        return
    out = args.root / domain / method
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, out)


def execute(args, out):
    domain, method = args.domain, args.method
    finished = out / "TEST_COMPLETE.json"
    if finished.exists():
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    source = args.root / "input"
    manifest = read_json(checked(source / "manifest.json", args.input_manifest_sha))
    spec = manifest["cells"][domain][method]
    base = f"{domain}/{method}/"
    paths = {f: checked(source / (base + f), manifest["files"][base + f])
             for f in ("frozen_step.pt", "step_val.npz", "step_test.npz")}
    pins = read_json(args.split_pins)[domain]
    train, val = (Split(domain, split, pins[split]) for split in ("train", "val"))
    scores_train = make_train_scores(out / "train_step.npz", paths["frozen_step.pt"], method, train, spec["layer"], device)
    scores_val = saved_step_scores(paths["step_val.npz"], val, method, device)
    for layer in range(1, 65):
        receipt = out / f"layer_{layer:02d}.json"
        if not receipt.exists():
            (_, scores, state, val_p), candidates = fit(train, val, scores_train, scores_val, layer, method, device, args.saplma_epochs)
            weights = out / f"layer_{layer:02d}.pt"
            torch.save(state, weights)
            val_file = out / f"val_{layer:02d}.npz"
            np.savez_compressed(val_file, probability=val_p, position_id=val.keys, c=val.c)
            save_json(receipt, dict(winner=scores, candidates=candidates, weights_sha=digest(weights.read_bytes()),
                                    inputs=args.input_manifest_sha, val_sha=digest(val_file.read_bytes())))
        print(domain, method, layer, "VAL complete", flush=True)
    candidates = [read_json(out / f"layer_{layer:02d}.json") for layer in range(1, 65)]
    assert all(row["inputs"] == args.input_manifest_sha for row in candidates)
    all_metrics = [row for layer in candidates for row in layer["candidates"]]
    rank = max((r["AUROC"], r["AP"]) for r in all_metrics)
    winners = [m for m in all_metrics if (m["AUROC"], m["AP"]) == rank]
    assert len(winners) == 1, "Original VAL selection has an unresolved exact tie"
    selected = winners[0]
    layer = selected["layer"]
    item = candidates[layer - 1]
    val_path = checked(out / f"val_{layer:02d}.npz", item["val_sha"])
    with np.load(val_path, allow_pickle=False) as saved:
        threshold = threshold_at_val_f1(saved["c"], saved["probability"])
    model = SAPLMA(5120) if method == "saplma" else torch.nn.Linear(5120, 1)
    model.load_state_dict(torch.load(checked(out / f"layer_{layer:02d}.pt", item["weights_sha"]),
                                     map_location="cpu", weights_only=True))
    model.to(device)
    test = test_split(domain, args.history_index)
    q = saved_step_scores(paths["step_test.npz"], test, method, device)
    assert np.isclose(metrics(test.y, q.cpu().numpy())["AUROC"], spec["step"], rtol=0, atol=1e-10)
    with (out / "TEST_STARTED.json").open("x") as guard:
        json.dump(dict(selected=selected, threshold=threshold,
                       input_manifest=args.input_manifest_sha, test=test.pins), guard)
    probability = predict(model, features(test, layer, method, "prefix", device), q)
    path = out / "test.npz"
    np.savez_compressed(path, probability=probability, position_id=test.keys, y=test.y, c=test.c)
    result = metrics(test.c, probability)
    save_json(finished, dict(domain=domain, method=method, position_count=len(test.keys),
                             conditional_train_count=int((train.y == 0).sum()), selected=selected,
                             threshold=threshold,
                             TEST=result, direct_prefix_AUROC=spec["original"],
                             delta_AUROC=result["AUROC"] - spec["original"],
                             test_sha=digest(path.read_bytes())))


if __name__ == "__main__":
    main()
