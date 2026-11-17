"""Train one configuration, domain and disjoint layer range."""

import argparse
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
from run import HERE, digest, save_json
from topk import mask_top_decile

RESIDUAL_ARM = "aggr_residual"
TOP10_ARM = "aggr_top10"
RAW_ARMS = ("aggr", RESIDUAL_ARM, TOP10_ARM)
ARMS = ("aggr", "repr", "aggr_repr", RESIDUAL_ARM, TOP10_ARM)
WIDTH = 5120
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
MAX_EPOCHS = 20
EARLY_PATIENCE = 5
MIN_AUROC_GAIN = 1e-7
RECIPE = "pre_dataset_step_20261011"


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
        size = width if arm in RAW_ARMS else 256
        if arm in RAW_ARMS:
            self.scorer = nn.Sequential(nn.Linear(width, 256), nn.BatchNorm1d(256),
                                        nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
        else:
            self.mapper = nn.Sequential(nn.Linear(width, 256), nn.BatchNorm1d(256), nn.ReLU())
            if arm == "aggr_repr":
                self.scorer = nn.Linear(256, 1)
        if arm == RESIDUAL_ARM:
            nn.init.zeros_(self.scorer[-1].weight)
            nn.init.zeros_(self.scorer[-1].bias)
        self.step = nn.Linear(size, 1)

    def forward(self, raw, lengths, owners, positional):
        values = raw if self.arm in RAW_ARMS else self.mapper(raw)
        if self.arm == "repr":
            weights = positional
        else:
            logits = self.scorer(values).flatten()
            if self.arm == RESIDUAL_ARM:
                logits = positional + logits
            if self.arm == TOP10_ARM:
                logits = mask_top_decile(logits, lengths, owners)
            maximum = torch.segment_reduce(logits, "max", lengths=lengths)
            exp = (logits - maximum[owners]).exp()
            weights = exp / torch.segment_reduce(exp, "sum", lengths=lengths)[owners]
        e = torch.segment_reduce(values * weights[:, None], "sum", lengths=lengths)
        return self.step(e).flatten()


class Layer:
    def __init__(self, split, layer, cache=None, *, positional_logits=False):
        self.split = split
        self.positional_logits = positional_logits
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
        self.step_spans = [None] * len(split.y)
        for offset, origin, spans, slots in split.rows:
            for position, (first, stop) in zip(range(slots.start, slots.stop), spans, strict=True):
                self.step_spans[position] = (offset + first - origin, offset + stop - origin)
        assert all(span is not None for span in self.step_spans)

    def pack(self, batch, device):
        tokens, lengths = [], []
        for position in batch:
            first, stop = self.step_spans[position]
            tokens.append(self.raw[first:stop])
            lengths.append(stop - first)
        raw = torch.from_numpy(np.concatenate(tokens).view(np.int16)).view(torch.bfloat16).to(device, dtype=torch.float32)
        positional = []
        for n in lengths:
            scores = 3 * torch.linspace(0, 1, n, dtype=torch.float32)
            if self.positional_logits:
                positional.append(scores)
            else:
                weight = scores.exp()
                positional.append(weight / weight.sum())
        return (raw, torch.tensor(lengths, device=device),
                torch.from_numpy(np.repeat(np.arange(len(lengths)), lengths)).to(device),
                torch.cat(positional).to(device))


@torch.inference_mode()
def predict(model, data, device):
    model.eval()
    probability = np.empty(len(data.split.y), np.float32)
    for start in range(0, len(probability), EVAL_BATCH_SIZE):
        stop = min(start + EVAL_BATCH_SIZE, len(probability))
        packed = data.pack(range(start, stop), device)
        probability[start:stop] = model(*packed).sigmoid().cpu().numpy()
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
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=20, min_lr=1e-6
    )
    a, b = (Layer(split, layer, args.cache, positional_logits=args.arm == RESIDUAL_ARM)
            for split in (train, val))
    print("BOUND", args.domain, args.arm, layer, len(train.y), len(val.y), flush=True)
    label_y = torch.as_tensor(train.y, device=device)
    checkpoint = args.output / f"best_{layer:02d}.pt"
    progress = args.output / f"resume_{layer:02d}.pt"
    best = None
    best_model = None
    best_probability = None
    start = 1
    binding = {"recipe": RECIPE, "domain": args.domain, "arm": args.arm, "layer": layer, "device": device.type,
               "train": train.pins, "val": val.pins}
    if progress.exists():
        state = torch.load(progress, map_location=device, weights_only=True)
        assert state["binding"] == binding
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        rng.set_state(state["shuffle"].cpu())
        torch.set_rng_state(state["rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        best, start = state["best"], state["epoch"] + 1
        best_model = state["best_model"]
        best_probability = state["best_probability"].cpu().numpy()
    for epoch in range(start, MAX_EPOCHS + 1):
        if best is not None and epoch - best["epoch"] > EARLY_PATIENCE:
            break
        net.train()
        order = torch.randperm(len(train.y), generator=rng)
        loss_sum = 0.0
        for first in range(0, len(order), TRAIN_BATCH_SIZE):
            batch = order[first:first + TRAIN_BATCH_SIZE]
            packed = a.pack(batch.tolist(), device)
            ids = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(net(*packed), label_y[ids].float())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training objective")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        train_loss = loss_sum / len(order)
        scheduler.step(train_loss)
        probability = predict(net, b, device)
        statistics = val_stats(val, probability)
        if best is None or statistics["step_AUROC"] > best["metrics"]["step_AUROC"] + MIN_AUROC_GAIN:
            best = dict(binding, epoch=epoch, metrics=statistics)
            best_model = {key: tensor.detach().cpu().clone() for key, tensor in net.state_dict().items()}
            best_probability = probability
        state = dict(binding=binding, model=net.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), shuffle=rng.get_state(), rng=torch.get_rng_state(), epoch=epoch, best=best,
                     best_model=best_model, best_probability=torch.from_numpy(best_probability))
        if device.type == "cuda":
            state["cuda_rng"] = torch.cuda.get_rng_state()
        atomic_checkpoint(progress, state)
        print("VAL", args.domain, args.arm, layer, epoch, statistics["step_AUROC"], train_loss, optimizer.param_groups[0]["lr"], flush=True)
    atomic_checkpoint(checkpoint, {"model": best_model, "selection": best})
    prediction = checkpoint.with_suffix(".npz")
    np.savez_compressed(prediction, probability=best_probability, position_id=val.keys, y=val.y)
    report = dict(best, checkpoint=checkpoint.name, checkpoint_sha256=digest(checkpoint.read_bytes()),
                  prediction=prediction.name, prediction_sha256=digest(prediction.read_bytes()),
                  evaluation=val_stats(val, best_probability))
    save_json(args.output / f"layer_{layer:02d}.json", report)
    progress.unlink()


def main():
    import fcntl

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
