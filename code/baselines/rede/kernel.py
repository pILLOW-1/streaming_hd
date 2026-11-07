"""Paper-derived ReDe equations (arXiv:2607.22098v1).

Projection/search parameters are explicit in the portable recipe. See pipeline.py
and run.py for fitting, filtering, native re-forward and supervised probing.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Projection(nn.Module):
    def __init__(self, input_width, hidden_width, output_width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_width, hidden_width), nn.ReLU(), nn.Linear(hidden_width, output_width))

    def forward(self, x):
        return self.net(x)


def perplexity_weighted_embedding(hidden, token_log_probability):
    if len(hidden) != len(token_log_probability) or len(hidden) == 0:
        raise ValueError("Token-aligned nonempty hidden states/probabilities required")
    return (hidden * torch.softmax(-token_log_probability.double(), 0).to(hidden.dtype)[:, None]).sum(0)


def shaping_loss(projected, informative_indices, noisy_indices, lambda_disperse, lambda_separate):
    a = F.normalize(projected[informative_indices], dim=-1)
    b = F.normalize(projected[noisy_indices], dim=-1)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Paper pairwise expectations undefined for fewer than two proxies; no invented zero loss")
    aa, bb, ab = a @ a.T, b @ b.T, a @ b.T
    ma = ~torch.eye(len(a), dtype=torch.bool, device=a.device)
    mb = ~torch.eye(len(b), dtype=torch.bool, device=b.device)
    return (1 - aa[ma]).mean() + lambda_disperse * bb[mb].mean() + lambda_separate * ab.mean()


def within_trace_knn_distance(projected, k=15):
    if len(projected) <= k:
        raise ValueError("REDE_SHORT_TRACE_POLICY_UNSPECIFIED: need more than k=15 steps; never silently reduce k")
    z = F.normalize(projected, dim=-1)
    distance = 1 - z @ z.T
    distance.fill_diagonal_(float("inf"))
    return torch.topk(distance, k, dim=-1, largest=False).values[:, -1]
