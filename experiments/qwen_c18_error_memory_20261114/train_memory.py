"""Train both memory variants with shared readouts and original C18 selection."""
import argparse
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from history import HistorySplit
from memory_model import MemoryHeads
from model import objective
from run import HERE, Recovery, digest, evaluate, save_json

ARMS = ("A2", "A3")


def forward(model, features, batch=None):
    steps, prefix, positions, starts = features
    if batch is not None:
        prefix, positions = prefix[batch], positions[batch]
    return model(steps[positions], prefix, positions, steps, starts)


def rank(metrics):
    return metrics["mean_AUROC"], metrics["mean_AP"], -metrics["mean_BCE"]


def val_stats(data, p):
    y, c = data.y, data.c
    step, prefix = p[:, 0], p[:, :2].sum(1)
    auroc = [roc_auc_score(y, step), roc_auc_score(c, prefix)]
    ap = [average_precision_score(y, step), average_precision_score(c, prefix)]
    bce = [log_loss(y, np.clip(step, 1e-15, 1-1e-15), labels=[0, 1]),
           log_loss(c, np.clip(prefix, 1e-15, 1-1e-15), labels=[0, 1])]
    return dict(mean_AUROC=float(np.mean(auroc)), mean_AP=float(np.mean(ap)),
                mean_BCE=float(np.mean(bce)), step_AUROC=float(auroc[0]), prefix_AUROC=float(auroc[1]))


def evaluate_prob(data, probability):
    p = probability.astype(np.float64)
    assert np.isfinite(p).all() and np.allclose(p.sum(1), 1)
    return evaluate(data, Recovery(data), p[:, 0], p[:, 1] / p[:, 1:].sum(1), p[:, :2].sum(1))


@torch.inference_mode()
def predict(net, features):
    return forward(net, features).exp().cpu().numpy()


def fit(arm, domain, layer, train, val, tx, vx, labels, output):
    torch.manual_seed(42)
    rng = torch.Generator().manual_seed(42)
    model = MemoryHeads(5120, arm).to(tx[0].device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    best = None
    for epoch in range(1, 101):
        for i in torch.randperm(len(train.y), generator=rng).split(256):
            batch = i.to(tx[0].device)
            opt.zero_grad(set_to_none=True)
            loss = objective(forward(model, tx, batch), labels[0][batch], labels[1][batch], "ce")
            loss.backward()
            opt.step()
        p = predict(model, vx)
        metrics = val_stats(val, p)
        if best is None or rank(metrics) > rank(best["metrics"]):
            best = dict(arm=arm, domain=domain, layer=layer, epoch=epoch, metrics=metrics)
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_p = p
        if epoch - best["epoch"] >= 10:
            break
    selection = output / "selection.json"
    previous = json.loads(selection.read_text()) if selection.exists() else None
    if previous is None or rank(best["metrics"]) > rank(previous["metrics"]):
        ckpt = f"best.L{layer:02d}.pt"
        torch.save(state, output / ckpt)
        pred = f"val.L{layer:02d}.npz"
        np.savez_compressed(output / pred, probability=best_p, position_id=val.keys, y=val.y,
                            c=val.c, prior=val.prior, group=val.group)
        pinned = dict(best, checkpoint=ckpt, checkpoint_sha256=digest((output / ckpt).read_bytes()),
                      prediction=pred, prediction_sha256=digest((output / pred).read_bytes()),
                      evaluation=evaluate_prob(val, best_p), coverage=Recovery(val).coverage,
                      inputs={"train": train.pins, "val": val.pins})
        save_json(selection, pinned)
        if previous:
            (output / previous["checkpoint"]).unlink()
            (output / previous["prediction"]).unlink()
    save_json(output / f"layer_{layer:02d}.json", dict(best, trained_epochs=epoch))
    print("FIT", domain, arm, layer, epoch, best["metrics"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=["Logic", "Factuality"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--index", type=Path, default=Path("/tmp/rel_index"))
    args = parser.parse_args()
    domain = args.domain
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = json.loads((HERE / "inputs.json").read_text(encoding="utf-8-sig"))[domain]
        train, val = [HistorySplit(domain, split, sources[split], args.index / f"{domain}.{split}.json")
                      for split in ("train", "val")]
        assert not set(train.group).intersection(val.group)
        labels = [torch.as_tensor(x, device=device) for x in (train.y, train.c)]
        print("BOUND", domain, len(train.y), len(val.y), len(train.starts), len(val.starts), flush=True)
        for arm in ARMS:
            (args.output / arm).mkdir(exist_ok=True)
        for layer in range(1, 65):
            pending = [a for a in ARMS if not (args.output / a / f"layer_{layer:02d}.json").exists()]
            if not pending:
                continue
            print("POOL", domain, layer, flush=True)
            tx, vx = train.layer(layer, device), val.layer(layer, device)
            for arm in pending:
                fit(arm, domain, layer, train, val, tx, vx, labels, args.output / arm)
            del tx, vx
        save_json(args.output / "TRAIN_COMPLETE.json", {"layers": 64, "arms": ARMS, "domain": domain})


if __name__ == "__main__":
    main()
