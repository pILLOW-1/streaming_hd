import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from model import make_model, objective
from pooling import pools

HERE = Path(__file__).resolve().parent
VOLUME = Path("/workspace/qwen_relation")
REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
READOUTS = ("step_exp", "prefix_exp")
DOMAINS = ("Logic", "Factuality")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, item):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(item, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


class Split:
    def __init__(self, domain, split, source, rows=None):
        self.root = VOLUME / domain / split
        receipt = self.root / "TOKEN_WRITE_RECEIPT.json"
        if not receipt.exists():
            receipt = self.root / "TOKEN_FEATURE_MANIFEST.json"
        raw = receipt.read_bytes()
        self.receipt = json.loads(raw)
        assert self.receipt["full_native_SHA256"] == source["full_sha"]
        assert self.receipt["model_revision"] == self.receipt["tokenizer_revision"] == REV
        assert self.receipt["domain"] == domain and self.receipt["split"] == split
        assert self.receipt["width"] == 5120 and self.receipt["dtype"] == "bfloat16_bits"
        off = (self.root / "TOKEN_OFFSETS.json").read_bytes()
        if "offsets_SHA256" in self.receipt:
            assert digest(off) == self.receipt["offsets_SHA256"]
        if rows is None:
            labels = (VOLUME.parent / source["labels_key"]).read_bytes()
            assert digest(labels) == source["labels_sha"]
            data = json.loads(labels)
            assert data["domain"] == domain and data["split"] == split
            rows = data["rows"]
        else:
            labels = json.dumps(rows, sort_keys=True).encode()
        offsets = json.loads(off)
        lookup = {r["sample_id"]: r for r in offsets}
        assert len(lookup) == len(offsets) == len(rows) == self.receipt["rows"]
        assert set(lookup) == {r["sample_id"] for r in rows}
        self.keys, self.rows = [], []
        y, c, prior, group = [], [], [], []
        for row in rows:
            sample = row["sample_id"]
            entry = lookup[sample]
            origin, stop = entry["spans"][0]
            if "token_sha" in row:
                assert entry["token_SHA256"] == row["token_sha"] and origin == row["origin"]
            spans = row["step_token_spans"]
            steps, ys, cs = (row[k] for k in ("retained_original_step_ids", "step_hallucination",
                                             "prefix_hallucination"))
            assert len(steps) == len(ys) == len(cs) == len(spans)
            if not steps:
                continue
            assert all(origin <= a < b <= stop for a, b in spans)
            start = len(y)
            history = False
            for step, yl, cl in zip(steps, ys, cs, strict=True):
                assert yl in (0, 1) and cl in (0, 1) and yl <= cl
                self.keys.append(f"{sample}:{step}")
                y.append(yl)
                c.append(cl)
                prior.append(history)
                group.append(sample)
                history |= bool(cl)
            self.rows.append((entry["offset"], origin, spans, slice(start, len(y))))
        self.rows.sort(key=lambda x: x[0])
        assert len(set(self.keys)) == len(self.keys) and set(y) == set(c) == {0, 1}
        self.y, self.c, self.prior, self.group = map(np.asarray, (y, c, prior, group))
        self.pins = dict(receipt=digest(raw), offsets=digest(off), labels=digest(labels),
                         positions=digest(json.dumps(self.keys).encode()))

    def features(self, layer, names=READOUTS):
        name = f"layer_{layer:02d}/reasoning_answer.bf16.npy"
        path = self.root / name
        pin = self.receipt["files"][name]
        assert pin["layer"] == layer and path.stat().st_size == pin["bytes"]
        with path.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            header = {(1, 0): np.lib.format.read_array_header_1_0,
                      (2, 0): np.lib.format.read_array_header_2_0}[version]
            shape, fortran, dtype = header(stream)
            assert shape == (self.receipt["tokens"], 5120) and not fortran
            assert dtype == np.dtype("<u2")
            start = stream.tell()
            out = {m: np.empty((len(self.y), 5120), np.float32) for m in names}
            for base, origin, spans, slots in self.rows:
                length = max(b for _, b in spans) - origin
                size = length * 10240
                stream.seek(start + base * 10240)
                data = stream.read(size)
                assert len(data) == size
                bits = np.frombuffer(data, dtype=dtype).reshape(length, 5120)
                visible = (bits.astype(np.uint32) << 16).view(np.float32)
                assert np.isfinite(visible).all()
                for method, values in pools(visible, spans, origin, names).items():
                    out[method][slots] = values
        return [torch.from_numpy(out[method]) for method in names]


class Recovery:
    def __init__(self, split):
        self.mask = (split.y == 0) & split.prior
        self.labels = split.c[self.mask]
        group = split.group[self.mask]
        self.groups, self.pairs = [], []
        for sample in np.unique(group):
            indices = np.flatnonzero(group == sample)
            positive = int(self.labels[indices].sum())
            pairs = positive * (len(indices) - positive)
            if pairs:
                self.groups.append(indices)
                self.pairs.append(pairs)
        assert len(self.groups) > 0
        self.coverage = dict(positions=len(group), positives=int(self.labels.sum()),
                             mixed_trajectories=len(self.groups), pairs=sum(self.pairs))

    def metrics(self, scores):
        selected = scores[self.mask]
        aucs = [roc_auc_score(self.labels[ix], selected[ix]) for ix in self.groups]
        return dict(macro=float(np.mean(aucs)), pair=float(np.average(aucs, weights=self.pairs)),
                    pooled=float(roc_auc_score(self.labels, selected)))


def evaluate(data, recovery, q, kappa, prefix=None):
    if prefix is None:
        prefix = q + (1.0 - q) * kappa
    return dict(step=float(roc_auc_score(data.y, q)), prefix=float(roc_auc_score(data.c, prefix)),
                kappa=recovery.metrics(kappa), prefix_recovery=recovery.metrics(prefix))


@torch.inference_mode()
def score(head, x):
    return head(x).flatten().sigmoid().cpu().numpy()


def step_rank(row):
    return row["auc"], row["ap"], -row["epoch"], -row["layer"]


def kappa_rank(row):
    return row["macro"], row["pair"], row["pooled"], -row["epoch"], -row["layer"]


def save_head(path, head):
    pending = path.with_suffix(".pending")
    torch.save(head.state_dict(), pending)
    pending.replace(path)


def fit(domain, output, max_layer, max_epoch, device):
    sources = json.loads((HERE / "inputs.json").read_text(encoding="utf-8-sig"))[domain]
    train, val = (Split(domain, split, sources[split]) for split in ("train", "val"))
    assert not set(train.group).intersection(val.group)
    recovery = Recovery(val)
    labels = [torch.as_tensor(a, device=device) for a in (train.y, train.c)]
    best = {}
    state_file = output / "selection.json"
    completed = 0
    if state_file.is_file():
        state = json.loads(state_file.read_text())
        assert state["train"] == train.pins and state["val"] == val.pins
        assert state["max_epoch"] == max_epoch and state["max_layer"] == max_layer
        best, completed = state["best"], state["completed_layers"]
    for layer in range(completed + 1, max_layer + 1):
        print("POOL", domain, layer, flush=True)
        train_x = [x.to(device) for x in train.features(layer)]
        val_x = [x.to(device) for x in val.features(layer)]
        x = torch.cat(train_x, dim=1)
        del train_x
        torch.manual_seed(42)
        order_rng = torch.Generator().manual_seed(42)
        model = make_model(5120, "linear").to(device)
        heads = dict(step=model.step, kappa=model.conditional_prefix)
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        layer_best = {}
        for epoch in range(1, max_epoch + 1):
            for i in torch.randperm(len(train.y), generator=order_rng).split(256):
                batch = i.to(device)
                opt.zero_grad(set_to_none=True)
                loss = objective(model(x[batch]), labels[0][batch], labels[1][batch], "ce")
                assert torch.isfinite(loss)
                loss.backward()
                opt.step()
            q, kappa = [score(head, features) for head, features in zip(heads.values(), val_x, strict=True)]
            candidates = dict(
                step=dict(auc=float(roc_auc_score(val.y, q)),
                          ap=float(average_precision_score(val.y, q))),
                kappa=recovery.metrics(kappa),
            )
            for name, rank in (("step", step_rank), ("kappa", kappa_rank)):
                row = dict(layer=layer, epoch=epoch, **candidates[name])
                if name not in layer_best or rank(row) > rank(layer_best[name]):
                    layer_best[name] = row
                    if name not in best or rank(row) > rank(best[name]):
                        filename = f"{name}.L{layer:02d}.E{epoch:03d}.pt"
                        save_head(output / filename, heads[name])
                        best[name] = dict(row, file=filename)
            if all(epoch - row["epoch"] >= 10 for row in layer_best.values()):
                break
        save_json(state_file, dict(domain=domain, best=best, completed_layers=layer, max_epoch=max_epoch,
                                   max_layer=max_layer, train=train.pins, val=val.pins))
        keep = {row["file"] for row in best.values()}
        for file in output.glob("*.pt"):
            if file.name not in keep:
                file.unlink()
        print("FIT", domain, layer, epoch, layer_best, flush=True)
        del x, val_x

    recovery = Recovery(val)
    scores = {}
    for name, readout in zip(("step", "kappa"), READOUTS, strict=True):
        row = best[name]
        head = make_model(5120, "linear")
        head = head.step if name == "step" else head.conditional_prefix
        head.load_state_dict(torch.load(output / row["file"], weights_only=True, map_location="cpu"))
        batch, = val.features(row["layer"], (readout,))
        scores[name] = score(head, batch)
        row["sha256"] = digest((output / row["file"]).read_bytes())
    q, kappa = scores["step"], scores["kappa"]
    result = dict(domain=domain, selection=best, coverage=recovery.coverage,
                  val=evaluate(val, recovery, q, kappa), pins=dict(train=train.pins, val=val.pins))
    np.savez_compressed(output / "val.npz", position_id=val.keys, y=val.y, c=val.c, q=q,
                        kappa=kappa, prior=val.prior, group=val.group)
    result["prediction_sha256"] = digest((output / "val.npz").read_bytes())
    save_json(output / "result.json", result)
    print("COMPLETE", domain, result, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=DOMAINS)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-layer", type=int, default=64)
    parser.add_argument("--max-epoch", type=int, default=100)
    args = parser.parse_args()
    import fcntl
    assert 1 <= args.max_layer <= 64 and 1 <= args.max_epoch <= 100
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2)))
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (args.output / "result.json").exists():
            return
        fit(args.domain, args.output, args.max_layer, args.max_epoch, device)


if __name__ == "__main__":
    main()
