"""Batch complete reasoning histories and fit the v1 composition loss."""
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from kappa_memory import KappaMemory
from objective import prefix_bce, prefix_probability
from pooling import pools


def history_layer(data, layer):
    name = f"layer_{layer:02d}/reasoning_answer.bf16.npy"
    path = data.root / name
    pin = data.receipt["files"][name]
    assert pin["layer"] == layer and path.stat().st_size == pin["bytes"]
    raw = np.load(path, allow_pickle=False, mmap_mode="r")
    assert raw.shape == (data.receipt["tokens"], 5120) and raw.dtype == np.dtype("<u2")
    seq = []
    for offset, origin, spans, _, slots, steps in data.history:
        length = spans[-1][1] - origin
        visible = (raw[offset:offset+length].astype(np.uint32) << 16).view(np.float32)
        assert np.isfinite(visible).all()
        pooled = pools(visible, spans, origin, ("step_exp",))["step_exp"]
        seq.append((torch.from_numpy(pooled), torch.tensor(steps), slots))
    assert sum(r[2].stop-r[2].start for r in seq) == len(data.keys)
    return seq


def batches(seq, order, budget=128):
    group = []
    positions = 0
    for index in order:
        x, valid, ids = seq[index]
        if group and (positions + len(valid) > budget or len(group) == 16):
            yield group
            group, positions = [], 0
        group.append((x, valid, ids))
        positions += len(valid)
    if group:
        yield group


def logits(model, batch, device):
    hidden = pad_sequence([x for x, _, _ in batch], batch_first=True).to(device)
    out = model(hidden)
    selected = torch.cat([out[i, steps.to(device)] for i, (_, steps, _) in enumerate(batch)])
    return selected, [slots for _, _, slots in batch]


@torch.inference_mode()
def predict_history(model, seq, device, count):
    model.eval()
    output = np.empty(count, dtype=np.float32)
    for batch in batches(seq, range(len(seq))):
        values, slots = logits(model, batch, device)
        for part, sl in zip(values.cpu().split([s.stop-s.start for s in slots]), slots, strict=True):
            output[sl] = part.numpy()
    assert np.isfinite(output).all()
    return output


def fit_memory(train, val, q_train, q_val, layer, device, epochs, metrics, rank, bce_metric,
               progress):
    torch.manual_seed(42)
    order = torch.Generator().manual_seed(42)
    model = KappaMemory().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=20, min_lr=1e-6)
    tr, va = history_layer(train, layer), history_layer(val, layer)
    y, c = (torch.as_tensor(x, device=device) for x in (train.y, train.c))
    best, candidates = None, []
    start = 1
    binding = dict(layer=layer, train=train.pins, val=val.pins, epochs=epochs)
    if progress.exists():
        saved = torch.load(progress, weights_only=False, map_location="cpu")
        assert saved["binding"] == binding
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        order.set_state(saved["order"])
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"])
        best, candidates, start = saved["best"], saved["candidates"], saved["epoch"] + 1
    for epoch in range(start, epochs+1):
        model.train()
        loss_total, observed = 0.0, 0
        for batch in batches(tr, torch.randperm(len(tr), generator=order).tolist()):
            opt.zero_grad(set_to_none=True)
            readout, slots = logits(model, batch, device)
            ids = torch.cat([torch.arange(s.start, s.stop, device=device) for s in slots])
            loss = prefix_bce(readout, q_train[ids], y[ids], c[ids], all_positions=True)
            assert torch.isfinite(loss)
            loss.backward()
            opt.step()
            count = len(ids)
            observed += count
            loss_total += loss.item() * count
        assert observed == len(y)
        sched.step(loss_total / observed)
        z = predict_history(model, va, device, len(val.keys))
        prob = prefix_probability(torch.from_numpy(z).to(device), q_val).cpu().numpy()
        scores = dict(layer=layer, epoch=epoch, **metrics(val.c, prob), BCE=bce_metric(val.c, prob))
        candidates.append(scores)
        if best is None or rank(scores) > best[0]:
            best = rank(scores), scores, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, prob, z
        state = dict(binding=binding, model=model.state_dict(), optimizer=opt.state_dict(),
                     scheduler=sched.state_dict(), order=order.get_state(), rng=torch.get_rng_state(),
                     epoch=epoch, best=best, candidates=candidates)
        if device.type == "cuda":
            state["cuda_rng"] = torch.cuda.get_rng_state()
        pending = progress.with_suffix(".pending")
        torch.save(state, pending)
        pending.replace(progress)
        print("VAL", layer, epoch, scores["AUROC"], scores["AP"], scores["BCE"], flush=True)
    return best, candidates
