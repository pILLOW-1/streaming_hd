"""Paper-derived ReDe projection training and filtering.

Hyperparameters absent from the paper's fixed recipe are explicit arguments.
The filtered trajectory must be forwarded again by the native backbone: averaging
pre-filter states is not equivalent and is deliberately not implemented as a shortcut.
"""
from __future__ import annotations

import math
import torch
from .kernel import Projection, shaping_loss, within_trace_knn_distance


def proxy_indices(attention_mass: torch.Tensor, ratio: float):
    if attention_mass.ndim != 1 or not torch.isfinite(attention_mass).all() or not 0 < ratio <= .5:
        raise ValueError('Finite Step attention masses and 0 < proxy ratio <= .5 required')
    count = math.floor(len(attention_mass) * ratio)
    if count < 2:
        return None  # No defined off-diagonal pairwise expectation for this trace.
    order = torch.argsort(attention_mass, stable=True)
    return order[-count:], order[:count]


def fit_projection(traces, *, hidden_width: int, output_width: int, epochs: int,
                   proxy_ratio: float, lambda_disperse: float, lambda_separate: float,
                   device='cpu', seed=42):
    """Train on TRAIN traces only; each item is (Step embeddings, answer attention).

A batch contains up to 128 complete traces. Undefined proxy pairs contribute no
projection loss; they remain in downstream data. Stable ranking and floor counts
are explicit project rounding choices, not asserted paper hyperparameters.
"""
    if not traces or min(hidden_width, output_width, epochs) <= 0:
        raise ValueError('Nonempty TRAIN traces and positive dimensions/epochs required')
    torch.manual_seed(seed)
    model = Projection(traces[0][0].shape[-1], hidden_width, output_width).to(device)
    eligible = [(x, proxy_indices(a, proxy_ratio)) for x, a in traces]
    eligible = [(x, proxy) for x, proxy in eligible if proxy is not None]
    if not eligible:
        raise ValueError('No TRAIN trace has at least two informative and noisy proxy Steps')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    for epoch in range(epochs):
        order = torch.randperm(len(eligible)).tolist()
        losses = []
        for start in range(0, len(order), 128):
            optimizer.zero_grad()
            terms = []
            for i in order[start:start+128]:
                x, (top, bottom) = eligible[i]
                terms.append(shaping_loss(model(x.to(device).float()), top.to(device), bottom.to(device),
                                          lambda_disperse, lambda_separate))
            loss = torch.stack(terms).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        history.append(dict(epoch=epoch+1, loss=sum(losses)/len(losses)))
    model.eval()
    return model, history


@torch.no_grad()
def retained_indices(model, embeddings: torch.Tensor, *, drop_ratio: float, short_trace_policy: str):
    """Drop highest k=15 distances; return chronological surviving Step indices.

The paper does not define traces with <=15 Steps. Require an explicit 'keep' or
'error' policy; the former is a documented no-filter extension to short traces.
"""
    if not 0 <= drop_ratio < 1 or short_trace_policy not in ('keep', 'error'):
        raise ValueError('Invalid filtering configuration')
    if len(embeddings) <= 15:
        if short_trace_policy == 'error':
            raise ValueError('ReDe k=15 requires more than 15 Steps')
        return list(range(len(embeddings)))
    device = next(model.parameters()).device
    distances = within_trace_knn_distance(model(embeddings.to(device).float()), k=15)
    drop = math.floor(len(embeddings)*drop_ratio)
    keep = torch.argsort(distances, stable=True)[:len(embeddings)-drop]
    return sorted(keep.cpu().tolist())


def filtered_representation(model, embeddings, original_steps, native_reforward, *, drop_ratio, short_trace_policy):
    """Callback must retain the original question/answer and retokenize selected Steps."""
    if len(embeddings) != len(original_steps):
        raise ValueError('All reasoning Steps and embeddings must align')
    keep = retained_indices(model, embeddings, drop_ratio=drop_ratio, short_trace_policy=short_trace_policy)
    return native_reforward([original_steps[i] for i in keep]), keep
