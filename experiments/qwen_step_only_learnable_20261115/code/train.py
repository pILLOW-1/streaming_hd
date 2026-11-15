"""Train one configuration, domain and disjoint layer range."""

import argparse
import fcntl
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from history import HistorySplit
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from run import HERE, Recovery, digest, save_json

ARMS = ("aggr", "repr", "aggr_repr", "streaming_fixed")
WIDTH = 5120
TOKEN_BUDGET = 24000


def val_stats(data, probability):
    return {
        "step_AUROC": float(roc_auc_score(data.y, probability)),
        "step_AP": float(average_precision_score(data.y, probability)),
        "step_BCE": float(log_loss(data.y, probability.astype(np.float64), labels=[0, 1])),
    }


def rank(metrics):
    return (metrics["step_AUROC"], metrics["step_AP"], -metrics["step_BCE"])


class Detector(nn.Module):
    def __init__(self, arm, width=WIDTH):
        super().__init__()
        self.arm = arm
        size = width if arm in ("aggr", "streaming_fixed") else 256
        if arm == "aggr":
            self.scorer = nn.Sequential(nn.Linear(width, 256), nn.BatchNorm1d(256),
                                        nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
        elif arm != "streaming_fixed":
            self.mapper = nn.Sequential(nn.Linear(width, 256), nn.BatchNorm1d(256), nn.ReLU())
            if arm == "aggr_repr":
                self.scorer = nn.Linear(256, 1)
        self.step = nn.Linear(size, 1)

    def forward(self, raw, lengths=None, owners=None, fixed=None, positions=None):
        if lengths is None:
            assert self.arm == "streaming_fixed"
            return self.step(raw).flatten()
        values = raw if self.arm in ("aggr", "streaming_fixed") else self.mapper(raw)
        if self.arm in ("repr", "streaming_fixed"):
            weights = fixed
        else:
            logits = self.scorer(values).flatten()
            maximum = torch.segment_reduce(logits, "max", lengths=lengths)
            exp = (logits - maximum[owners]).exp()
            weights = exp / torch.segment_reduce(exp, "sum", lengths=lengths)[owners]
        e = torch.segment_reduce(values * weights[:, None], "sum", lengths=lengths)
        return self.step(e[positions]).flatten()


class Layer:
    def __init__(self, split, layer, cache=None, *, fixed_pool=False):
        self.split = split
        name = f"layer_{layer:02d}/reasoning_answer.bf16.npy"
        path = split.root / name
        pin = split.receipt["files"][name]
        assert pin["layer"] == layer and path.stat().st_size == pin["bytes"]
        if cache is not None:
            local = cache / (split.receipt["split"] + ".npy")
            binding = local.with_suffix(".json")
            source = {"path": str(path), "bytes": pin["bytes"]}
            if not (binding.exists() and local.exists() and local.stat().st_size == pin["bytes"]
                    and json.loads(binding.read_text()) == source):
                binding.unlink(missing_ok=True)
                local.unlink(missing_ok=True)
                pending = local.with_suffix(".copying")
                shutil.copyfile(path, pending)
                assert pending.stat().st_size == pin["bytes"]
                pending.replace(local)
                save_json(binding, source)
            path = local
        self.raw = np.load(path, allow_pickle=False, mmap_mode="r")
        assert self.raw.shape == (split.receipt["tokens"], WIDTH) and self.raw.dtype == np.dtype("<u2")
        self.counts = [sum(b-a for a, b in row[2]) for row in split.history]
        self.pooled = None
        if fixed_pool:
            pooled = np.empty((len(split.y), WIDTH), dtype=np.float32)
            for batch in self.batches(range(len(self.counts))):
                ids, (raw, lengths, _, fixed, positions) = self.pack(batch, torch.device("cpu"))
                pooled[ids] = torch.segment_reduce(raw * fixed[:, None], "sum", lengths=lengths)[positions].numpy()
            assert np.isfinite(pooled).all()
            self.pooled = pooled

    def batches(self, order, token_budget=TOKEN_BUDGET):
        pending, tokens, supervised = [], 0, 0
        for i in order:
            sample = self.split.history[i]
            n = sample[4].stop - sample[4].start
            count = self.counts[i]
            if pending and (tokens + count > token_budget or supervised + n > 256):
                yield pending
                pending, tokens, supervised = [], 0, 0
            pending.append(i)
            tokens += count
            supervised += n
        if pending:
            yield pending

    def pack(self, batch, device):
        if self.pooled is not None:
            ids = np.concatenate([np.arange(self.split.history[i][4].start, self.split.history[i][4].stop) for i in batch])
            return ids, (torch.from_numpy(self.pooled[ids]).to(device),)
        tokens, lengths, slots, positions = [], [], [], []
        for i in batch:
            offset, origin, spans, _, valid, selected = self.split.history[i]
            positions.extend(len(lengths) + j for j in selected)
            slots.extend(range(valid.start, valid.stop))
            for a, b in spans:
                tokens.append(self.raw[offset+a-origin:offset+b-origin])
                lengths.append(b - a)
        raw = torch.from_numpy(np.concatenate(tokens).view(np.int16)).view(torch.bfloat16).to(device, dtype=torch.float32)
        fixed = []
        for n in lengths:
            weight = torch.exp(3 * torch.linspace(0, 1, n, dtype=torch.float32))
            fixed.append(weight / weight.sum())
        ids = np.asarray(slots)
        packed = (raw, torch.tensor(lengths, device=device),
                  torch.from_numpy(np.repeat(np.arange(len(lengths)), lengths)).to(device),
                  torch.cat(fixed).to(device), torch.tensor(positions, device=device))
        return ids, packed


@torch.inference_mode()
def predict(model, data, device):
    model.eval()
    probability = np.empty(len(data.split.y), np.float32)
    for batch in data.batches(range(len(data.counts))):
        ids, packed = data.pack(batch, device)
        probability[ids] = model(*packed).sigmoid().cpu().numpy()
    assert np.isfinite(probability).all()
    return probability


def atomic_checkpoint(path, value):
    pending = path.with_suffix(".pending")
    torch.save(value, pending)
    pending.replace(path)


def fit(args, layer, train, val, device):
    torch.manual_seed(42)
    rng = torch.Generator().manual_seed(42)
    net = Detector(args.arm).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
    fixed_pool = args.arm == "streaming_fixed"
    a, b = (Layer(split, layer, args.cache, fixed_pool=fixed_pool) for split in (train, val))
    print("BOUND", args.domain, args.arm, layer, len(train.y), len(val.y),
          max(a.counts), max(b.counts), flush=True)
    label_y = torch.as_tensor(train.y, device=device)
    checkpoint = args.output / f"best_{layer:02d}.pt"
    progress = args.output / f"resume_{layer:02d}.pt"
    best = None
    best_model = None
    best_probability = None
    start = 1
    binding = {"domain": args.domain, "arm": args.arm, "layer": layer, "device": device.type,
               "train": train.pins, "val": val.pins}
    if progress.exists():
        state = torch.load(progress, map_location=device, weights_only=True)
        assert state["binding"] == binding
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.set_state(state["shuffle"].cpu())
        torch.set_rng_state(state["rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        best, start = state["best"], state["epoch"] + 1
        best_model = state["best_model"]
        best_probability = state["best_probability"].cpu().numpy()
    for epoch in range(start, 101):
        if best is not None and epoch - best["epoch"] > 10:
            break
        net.train()
        order = torch.randperm(len(a.counts), generator=rng).tolist()
        for batch in a.batches(order):
            ids, packed = a.pack(batch, device)
            ids = torch.as_tensor(ids, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(net(*packed), label_y[ids].float())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training objective")
            loss.backward()
            optimizer.step()
        probability = predict(net, b, device)
        statistics = val_stats(val, probability)
        if best is None or rank(statistics) > rank(best["metrics"]):
            best = dict(binding, epoch=epoch, metrics=statistics)
            best_model = {key: tensor.detach().cpu().clone() for key, tensor in net.state_dict().items()}
            best_probability = probability
        state = dict(binding=binding, model=net.state_dict(), optimizer=optimizer.state_dict(),
                     shuffle=rng.get_state(), rng=torch.get_rng_state(), epoch=epoch, best=best,
                     best_model=best_model, best_probability=torch.from_numpy(best_probability))
        if device.type == "cuda":
            state["cuda_rng"] = torch.cuda.get_rng_state()
        atomic_checkpoint(progress, state)
        print("VAL", args.domain, args.arm, layer, epoch, statistics["step_AUROC"], flush=True)
    atomic_checkpoint(checkpoint, {"model": best_model, "selection": best})
    prediction = checkpoint.with_suffix(".npz")
    np.savez_compressed(prediction, probability=best_probability, position_id=val.keys, y=val.y)
    report = dict(best, checkpoint=checkpoint.name, checkpoint_sha256=digest(checkpoint.read_bytes()),
                  prediction=prediction.name, prediction_sha256=digest(prediction.read_bytes()),
                  evaluation=val_stats(val, best_probability))
    save_json(args.output / f"layer_{layer:02d}.json", report)
    progress.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=["Logic", "Factuality"])
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("first", type=int)
    parser.add_argument("last", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], required=True)
    args = parser.parse_args()
    assert (args.first, args.last) in ((1, 32), (33, 64))
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = json.loads((HERE / "inputs.json").read_text())[args.domain]
        train, val = (HistorySplit(args.domain, split, sources[split],
                                    args.index / f"{args.domain}.{split}.json")
                      for split in ("train", "val"))
        assert not set(train.group).intersection(val.group)
        for layer in range(args.first, args.last + 1):
            if not (args.output / f"layer_{layer:02d}.json").exists():
                fit(args, layer, train, val, device)
        save_json(args.output / "SHARD_COMPLETE.json",
                  {"domain": args.domain, "arm": args.arm, "first": args.first, "last": args.last})


if __name__ == "__main__":
    main()
