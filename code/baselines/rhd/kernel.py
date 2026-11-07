"""RHD score equations from upstream 00c1946382301aa9432609ac605f7876c545e545.

The upstream quantity called JS is reverse-KL-to-mixture, averaged over the
vocabulary and multiplied by 1e5. Preserve this convention rather than silently
substituting textbook Jensen-Shannon divergence. Native logits must be provided
at the same token positions for every layer; this is not next-token NLL.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def token_depth(mature_logits: torch.Tensor, anchor_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return depth and normalized entropy for [tokens,vocab] / [layers,tokens,vocab]."""
    if mature_logits.ndim != 2 or anchor_logits.ndim != 3 or anchor_logits.shape[1:] != mature_logits.shape:
        raise ValueError('Mature and candidate logits must align exactly')
    if not torch.isfinite(mature_logits).all() or not torch.isfinite(anchor_logits).all():
        raise ValueError('RHD rejects nonfinite native logits')
    mature, anchors = mature_logits.float(), anchor_logits.float()
    mlog, alog = F.log_softmax(mature, -1), F.log_softmax(anchors, -1)
    mixture = .5 * (F.softmax(mature, -1).unsqueeze(0) + F.softmax(anchors, -1))
    depth = .5 * (F.kl_div(mlog.unsqueeze(0), mixture, reduction='none').mean(-1)
                  + F.kl_div(alog, mixture, reduction='none').mean(-1))
    entropy = -(mlog.exp() * mlog).sum(-1) / np.log(mature.shape[-1])
    return depth.mean(0) * 1e5, entropy


def step_statistics(depth: torch.Tensor, entropy: torch.Tensor, spans: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    if depth.ndim != 1 or entropy.shape != depth.shape:
        raise ValueError('Aligned token depth and entropy required')
    if not spans or any(not 0 <= a < b <= len(depth) for a, b in spans):
        raise ValueError('Invalid reasoning Step spans')
    return (np.asarray([float(depth[a:b].mean()) for a, b in spans]),
            np.asarray([float(entropy[a:b].mean()) for a, b in spans]))


def components(depth, entropy, reasoning_attention) -> np.ndarray:
    """Official four-component order: attention, CV, depth mean, delta correlation.

For constant/one-Step traces the upstream CV/correlation is undefined. This
adapter raises explicitly; it never invents a finite score or silently drops a
sample. A benchmark must record a separately declared degeneracy policy.
"""
    r, e, attention = map(lambda x: np.asarray(x, dtype=np.float64), (depth, entropy, reasoning_attention))
    n = len(r)
    if n == 0 or e.shape != (n,) or attention.shape != (n, n):
        raise ValueError('Aligned Step depths, entropies and square attention required')
    if not all(np.isfinite(x).all() for x in (r, e, attention)):
        raise ValueError('Nonfinite RHD input')
    threshold = np.quantile(r, .2)
    start = int(np.floor(n * .8))
    attention_score = 0.
    k = min(5, start)
    if k:
        for i in range(start, n):
            indices = np.argsort(attention[i, :start])[-k:]
            attention_score += np.mean((r[indices] < threshold) | (r[indices] > 4))
        attention_score /= n - start
    first = r[:n // 2]
    if not len(first) or first.mean() == 0:
        raise ValueError('Undefined official RHD CV for this trace')
    cv = first.std() / first.mean()
    correlation = 0.
    if n >= 3:
        dr = np.diff((r-r.min()) / (np.ptp(r) if np.ptp(r) else 1.))
        de = np.diff((e-e.min()) / (np.ptp(e) if np.ptp(e) else 1.))
        if np.std(dr) == 0 or np.std(de) == 0:
            raise ValueError('Undefined official RHD delta correlation for this trace')
        correlation = float(np.corrcoef(dr, de)[0, 1])
    return np.asarray([attention_score, cv, r.mean(), correlation])


def score(component_matrix, weights) -> np.ndarray:
    x, w = np.asarray(component_matrix), np.asarray(weights)
    if x.ndim != 2 or x.shape[1] != 4 or w.shape != (4,) or not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError('RHD requires four finite components and four explicit weights')
    return x @ w
