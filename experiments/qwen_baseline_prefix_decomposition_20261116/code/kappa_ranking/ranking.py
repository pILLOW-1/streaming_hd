"""Within-trajectory pairwise training and recovery evaluation."""
import numpy as np
import torch
from torch.nn.functional import softplus


class RecoveryPairs:
    def __init__(self, keys, y, c):
        by_sample = {}
        for index, key in enumerate(keys):
            sample, step = str(key).rsplit(":", 1)
            by_sample.setdefault(sample, []).append((int(step), index))
        self.groups = []
        self.time_score = np.full(len(keys), np.nan, dtype=np.float64)
        for rows in by_sample.values():
            first_error = None
            pos, neg = [], []
            for step, index in sorted(rows):
                if first_error is not None and y[index] == 0:
                    (pos if c[index] else neg).append(index)
                    self.time_score[index] = first_error - step
                if first_error is None and (y[index] or c[index]):
                    first_error = step
            if pos and neg:
                self.groups.append((np.asarray(pos, dtype=np.int64), np.asarray(neg, dtype=np.int64)))
        if not self.groups:
            raise ValueError("No eligible recovery trajectory")

    def draw(self, generator, device, max_trajectories=32):
        chosen = generator.choice(len(self.groups), min(max_trajectories, len(self.groups)), replace=False)
        positions, plus, minus, weight = [], [], [], []
        offset = 0
        for t in chosen:
            pos, neg = self.groups[t]
            positions.extend((pos, neg))
            plus.append(offset + np.repeat(np.arange(len(pos)), len(neg)))
            minus.append(offset + len(pos) + np.tile(np.arange(len(neg)), len(pos)))
            weight.append(np.full(len(pos)*len(neg), 1/(len(chosen)*len(pos)*len(neg)), dtype=np.float32))
            offset += len(pos) + len(neg)
        return tuple(torch.as_tensor(np.concatenate(arrays), device=device)
                     for arrays in (positions, plus, minus, weight))

    def evaluate(self, score, mask=None):
        score = np.asarray(score)
        if mask is None:
            mask = np.ones(len(score), dtype=bool)
        auc, pairs = [], 0
        for pos, neg in self.groups:
            p = score[pos[mask[pos]]]
            n = np.sort(score[neg[mask[neg]]])
            if not (len(p) and len(n)):
                continue
            if not (np.isfinite(p).all() and np.isfinite(n).all()):
                raise ValueError("Non-finite score in a recovery pair")
            left = np.searchsorted(n, p, side="left")
            right = np.searchsorted(n, p, side="right")
            count = len(p)*len(n)
            auc.append(float((left+right).sum()/(2*count)))
            pairs += count
        return dict(AUROC=float(np.mean(auc)) if auc else None,
                    trajectories=len(auc), pairs=pairs)


def pair_loss(model, x, draw):
    positions, plus, minus, weight = draw
    scores = model(x[positions]).flatten()
    return (weight * softplus(scores[minus] - scores[plus])).sum()


def bce_metric(labels, probability):
    eps = np.finfo(np.float64).eps
    p = np.clip(np.asarray(probability, dtype=np.float64), eps, 1-eps)
    y = np.asarray(labels, dtype=np.float64)
    return float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean())
