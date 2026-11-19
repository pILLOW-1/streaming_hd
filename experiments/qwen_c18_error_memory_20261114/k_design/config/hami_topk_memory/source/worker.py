"""A2 with HaMI-style Step probabilities; full or Top-k token memory.

No raw samples, token banks or trained weights belong in this source directory.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import worker_base as w

NAME = 'qwen_a2_hami_topk_memory_20261119'
ARMS = ('H1', 'H2')


def top_count(n):
    assert n > 0
    return int(n * .1) + 1


def selected(logits):
    # Stable sort gives a deterministic original-token-order tie break.
    return torch.argsort(logits, descending=True, stable=True)[:top_count(len(logits))]


def log_mean_probability(logits):
    a = logits[selected(logits)]
    return (torch.logsumexp(F.logsigmoid(a), 0) - math.log(len(a)),
            torch.logsumexp(F.logsigmoid(-a), 0) - math.log(len(a)))


class Model(nn.Module):
    def __init__(self, arm, width=5120):
        super().__init__()
        assert arm in ARMS
        self.arm = arm
        # Same token scoring network as pinned HaMI, with sigmoid represented
        # through stable log-probabilities rather than a saturating final op.
        self.scorer = nn.Sequential(nn.Linear(width, 256), nn.BatchNorm1d(256),
                                    nn.ReLU(), nn.Dropout(.3), nn.Linear(256, 1))
        self.prefix = nn.Linear(width, 1)
        self.memory = nn.Linear(width, 1, bias=False)
        nn.init.zeros_(self.memory.weight)

    def historical(self, bank, trajectory_ids, device):
        previous = self.training
        self.eval()
        memories = {}
        try:
            with torch.no_grad():
                for tid in trajectory_ids:
                    bags = bank.trajectory(tid, device)
                    flat = torch.cat(bags)
                    # Eval BN and disabled Dropout allow exact token chunking.
                    scores = torch.cat([self.scorer(h).flatten()
                                        for h in flat.split(8192)])
                    pieces = scores.split([len(h) for h in bags])
                    q, content = [], []
                    for h, s in zip(bags, pieces):
                        idx = selected(s)
                        q.append(s[idx].sigmoid().mean())
                        if self.arm == 'H1':
                            weight = torch.exp(3 * torch.linspace(0, 1, len(h), device=device))
                            content.append((h * (weight / weight.sum())[:, None]).sum(0))
                        else:
                            content.append(h[idx].mean(0))
                    weighted = torch.stack(q)[:, None] * torch.stack(content)
                    # Exclusive history; no current or future step enters g_t.
                    memories[tid] = F.pad(weighted.cumsum(0, dtype=torch.float64)[:-1],
                                         (0, 0, 1, 0)).float()
        finally:
            self.train(previous)
        return memories

    def forward_positions(self, bank, ids):
        device = self.prefix.weight.device
        ids = np.asarray(ids, dtype=np.int64)
        tids = sorted(set(int(bank.tid[i]) for i in ids))
        memory = self.historical(bank, tids, device)
        bags = [bank.step(int(bank.tid[i]), int(bank.local[i]), device) for i in ids]
        flat = torch.cat(bags)
        logits = self.scorer(flat).flatten()
        pairs = [log_mean_probability(s) for s in logits.split([len(h) for h in bags])]
        logq, log1q = (torch.stack([p[j] for p in pairs]) for j in (0, 1))
        g = torch.stack([memory[int(bank.tid[i])][int(bank.local[i])] for i in ids])
        c = bank.prefix[torch.as_tensor(ids, device=device)]
        z = (self.prefix(c) + self.memory(g)).flatten()
        return z, logq, log1q


def objective(z, logq, log1q, y, c):
    three = torch.stack((logq, log1q + F.logsigmoid(z),
                         log1q + F.logsigmoid(-z)), 1)
    target = torch.where(y.bool(), 0, torch.where(c.bool(), 1, 2)).long()
    return F.nll_loss(three, target)


class TokenBank:
    """Container-local, read-only BF16 mmap; complete original Step spans."""
    def __init__(self, data, layer, device, cached=True):
        self.data = data
        name = f'layer_{layer:02d}/reasoning_answer.bf16.npy'
        source = data.root / name
        assert source.stat().st_size == data.receipt['files'][name]['bytes']
        local = Path('/tmp/a2_hami_tokens') / data.receipt['domain'] / data.receipt['split'] / name
        local.parent.mkdir(parents=True, exist_ok=True)
        with local.with_suffix('.lock').open('a') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            if not local.exists():
                # Copy source bytes; never create or modify a network-volume bank.
                needed = source.stat().st_size + 2_000_000_000
                assert shutil.disk_usage(local.parent).free > needed, 'container staging capacity'
                pending = local.with_suffix('.pending')
                with source.open('rb') as src, pending.open('wb') as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                assert pending.stat().st_size == source.stat().st_size
                pending.replace(local)
            assert local.stat().st_size == source.stat().st_size
        self.raw = np.load(local, mmap_mode='r', allow_pickle=False)
        assert self.raw.dtype == np.dtype('<u2')
        assert self.raw.shape == (data.receipt['tokens'], 5120)
        self.path = local
        self.tid = np.empty(len(data.y), np.int64)
        self.local = np.empty(len(data.y), np.int64)
        for tid, (_, _, _, _, slots, choices) in enumerate(data.history):
            self.tid[slots] = tid
            self.local[slots] = choices
        def pools():
            return w.raw_pools(data, layer, True)
        if cached:
            import feature_cache as cache
            # Persist only bounded existing-source-derived readouts in the
            # already authorized cache; raw tokens remain the formal source.
            c = cache.get(cache.identity(data, layer, 'prefix_exp'), lambda: pools()[1])
        else:
            c = pools()[1]
        self.prefix = torch.from_numpy(c).to(device)

    def step(self, tid, index, device):
        offset, origin, spans, _, _, _ = self.data.history[tid]
        a, b = spans[index]
        bits = np.array(self.raw[offset + a-origin:offset + b-origin], copy=True)
        h = torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16).to(device, dtype=torch.float32)
        assert torch.isfinite(h).all()
        return h

    def trajectory(self, tid, device):
        return [self.step(tid, i, device) for i in range(len(self.data.history[tid][2]))]


def metric_pair(data, q, p):
    m = w.metrics(data.c, p)
    s = w.metrics(data.y, q)
    m.update({'step_' + k: v for k, v in s.items()})
    for k in ('AUROC', 'AP', 'BCE'):
        m['mean_' + k] = (m[k] + s[k]) / 2
    return m


def rank(m):
    return m['mean_AUROC'], m['mean_AP'], -m['mean_BCE']


def predict(net, bank):
    """One pass per trajectory; each Step scored once in evaluation mode."""
    net.eval()
    q = np.empty(len(bank.data.y), np.float32)
    k = q.copy()
    with torch.no_grad():
        for tid, (_, _, _, _, slots, choices) in enumerate(bank.data.history):
            bags = bank.trajectory(tid, net.prefix.weight.device)
            scores = torch.cat([net.scorer(x).flatten() for x in torch.cat(bags).split(8192)])
            probs, content = [], []
            for h, s in zip(bags, scores.split([len(h) for h in bags])):
                ix = selected(s)
                probs.append(s[ix].sigmoid().mean())
                if net.arm == 'H1':
                    a = torch.exp(3 * torch.linspace(0, 1, len(h), device=h.device))
                    content.append((h * (a/a.sum())[:, None]).sum(0))
                else:
                    content.append(h[ix].mean(0))
            probs = torch.stack(probs)
            weighted = probs[:, None] * torch.stack(content)
            g = F.pad(weighted.cumsum(0, dtype=torch.float64)[:-1], (0, 0, 1, 0)).float()
            idx = torch.tensor(choices, device=g.device)
            z = (net.prefix(bank.prefix[slots]) + net.memory(g[idx])).flatten()
            q[slots] = probs[idx].cpu().numpy()
            k[slots] = z.sigmoid().cpu().numpy()
    return q, k, q + (1-q)*k


def fit(domain, arm, layer):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = w.BASE / NAME / domain / arm
    out.mkdir(parents=True, exist_ok=True)
    with (out / f'.layer_{layer:02d}.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = out / f'layer_{layer:02d}.json'
        if report.exists():
            r = w.read(report)
            assert r['binding']['worker_sha256'] == w.sha(Path(__file__))
            return
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        device = torch.device('cuda')
        train, val = (w.split(domain, s) for s in ('train', 'val'))
        assert not set(train.group) & set(val.group)
        import feature_cache as cache
        cache.POOLING_SHA = w.sha(Path(w.pools.__code__.co_filename))
        tb, vb = TokenBank(train, layer, device), TokenBank(val, layer, device)
        net = Model(arm).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=.001, weight_decay=.0001)
        rng = torch.Generator().manual_seed(42)
        binding = dict(experiment=NAME, arm=arm, worker_sha256=w.sha(Path(__file__)),
                       train=train.pins, val=val.pins, layer=layer,
                       selection='joint mean Step/Prefix VAL AUROC/AP/lower BCE')
        resume = out / f'resume_{layer:02d}.pt'
        best = None; best_state = None; best_q = None; best_p = None; start = 1; curve = []
        if resume.exists():
            r = torch.load(resume, map_location=device, weights_only=True)
            assert r['binding'] == binding
            net.load_state_dict(r['model']); opt.load_state_dict(r['optimizer'])
            rng.set_state(r['shuffle'].cpu()); torch.set_rng_state(r['torch_rng'].cpu())
            torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda_rng']])
            start = r['epoch'] + 1; best = r['best']; best_state = r['best_model']
            best_q = r['best_q'].cpu().numpy(); best_p = r['best_p'].cpu().numpy(); curve = r['curve']
        y = torch.tensor(train.y, device=device); c = torch.tensor(train.c, device=device)
        for epoch in range(start, 101):
            if best is not None and epoch - best['epoch'] > 10:
                break
            net.train()
            for ids in torch.randperm(len(train.y), generator=rng).split(256):
                opt.zero_grad(set_to_none=True)
                z, lq, l1q = net.forward_positions(tb, ids.numpy())
                idx = ids.to(device)
                loss = objective(z, lq, l1q, y[idx], c[idx])
                assert torch.isfinite(loss)
                loss.backward(); opt.step()
            q, _, p = predict(net, vb)
            metrics = metric_pair(val, q, p)
            curve.append(dict(epoch=epoch, **metrics))
            if best is None or rank(metrics) > rank(best['metrics']):
                best = dict(epoch=epoch, metrics=metrics)
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                best_q, best_p = q.copy(), p.copy()
            w.checkpoint(resume, dict(binding=binding, model=net.state_dict(), optimizer=opt.state_dict(),
                shuffle=rng.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                epoch=epoch, best=best, best_model=best_state, best_q=torch.from_numpy(best_q),
                best_p=torch.from_numpy(best_p), curve=curve))
            print('VAL', domain, arm, layer, epoch, metrics, flush=True)
            w.save_json(out / f'resource_{layer:02d}.json', dict(pid=os.getpid(), epoch=epoch,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(), time=time.time()))
        pt = out / f'best_{layer:02d}.pt'; vp = out / f'val_{layer:02d}.npz'
        w.checkpoint(pt, dict(model=best_state, binding=binding, selection=best))
        np.savez_compressed(vp, q=best_q, probability=best_p, position_id=val.keys, y=val.y, c=val.c)
        w.save_json(report, dict(domain=domain, layer=layer, binding=binding, **best,
            checkpoint=pt.name, checkpoint_sha256=w.sha(pt), prediction=vp.name,
            prediction_sha256=w.sha(vp), curve=curve))
        resume.unlink()
        print('FIT_COMPLETE', domain, arm, layer, flush=True)


def finish(domain, arm):
    out = w.BASE / NAME / domain / arm
    with (out / '.finish.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'TEST_COMPLETE.json').exists():
            return
        records = [w.read(out / f'layer_{i:02d}.json') for i in range(1, 65)]
        winner = max(records, key=lambda r: rank(r['metrics']))
        assert all(r['binding']['worker_sha256'] == w.sha(Path(__file__)) for r in records)
        with np.load(out / winner['prediction'], allow_pickle=False) as b:
            cut = w.threshold(b['c'], b['probability'])
            step_cut = w.threshold(b['y'], b['q'])
        lock = dict(winner, threshold=cut, step_threshold=step_cut)
        w.save_json(out / 'VAL_LOCK.json', lock)
        assert not (out / 'TEST_STARTED.json').exists(), 'TEST already started; recover saved output without replay'
        test = w.split(domain, 'test'); train = w.split(domain, 'train'); val = w.split(domain, 'val')
        assert not set(test.group) & (set(train.group) | set(val.group))
        device = torch.device('cuda')
        import feature_cache as cache
        cache.POOLING_SHA = w.sha(Path(w.pools.__code__.co_filename))
        bank = TokenBank(test, winner['layer'], device)
        net = Model(arm).to(device)
        net.load_state_dict(torch.load(out / winner['checkpoint'], map_location=device, weights_only=True)['model'])
        with (out / 'TEST_STARTED.json').open('x') as f:
            json.dump(lock, f)
        q, k, p = predict(net, bank)
        np.savez_compressed(out / 'test.npz', probability=p, kappa=k, q=q,
            position_id=test.keys, y=test.y, c=test.c, prior=test.prior, group=test.group)
        m = metric_pair(test, q, p)
        cohort = (test.y == 0) & ((test.c == 1) | test.prior)
        recovery = float(w.roc_auc_score(test.c[cohort], p[cohort]))
        w.save_json(out / 'TEST_COMPLETE.json', dict(domain=domain, teacher=arm, selection=lock,
            inputs=test.pins, metrics=m, inherited_vs_recovered_AUROC=recovery,
            prediction_sha256=w.sha(out / 'test.npz')))
        print('TEST_COMPLETE', domain, arm, m, recovery, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('fit', 'finish'))
    p.add_argument('domain', choices=('Logic', 'Factuality'))
    p.add_argument('arm', choices=ARMS)
    p.add_argument('layer', type=int, nargs='?')
    a = p.parse_args()
    if a.mode == 'fit':
        assert 1 <= a.layer <= 64
        fit(a.domain, a.arm, a.layer)
    else:
        finish(a.domain, a.arm)
