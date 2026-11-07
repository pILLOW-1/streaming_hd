"""Streaming's original fixed pools, linear teacher and anchor/sync objective.

Equation binding: upstream e7c7aa2 src/extract_features.py, train_teacher.py,
train.py. Answer uses the teacher objective on answer-span tokens: an explicit
project level adaptation, never advertised as an original paper Answer result.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LinearProbe(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, 1)

    def forward(self, x):
        return self.linear(x)


def fixed_pools(current_tokens, visible_reasoning_tokens):
    if len(current_tokens) == 0 or len(visible_reasoning_tokens) == 0:
        raise ValueError("Empty token span cannot be pooled")
    current = current_tokens.double()
    visible = visible_reasoning_tokens.double()
    local_log_weight = torch.linspace(0.0, 3.0, len(current), device=current.device, dtype=current.dtype)
    global_log_weight = 0.003 * torch.arange(len(visible), device=visible.device, dtype=visible.dtype)
    # Shifted exponentials preserve the exact normalized equations and avoid
    # overflow on long prefixes; no clipping or truncation of tokens occurs.
    local = torch.softmax(local_log_weight, dim=0)
    global_weight = torch.softmax(global_log_weight, dim=0)
    return {"local_exp": (current * local[:, None]).sum(0).float(),
            "global_exp": (visible * global_weight[:, None]).sum(0).float(),
            "global_mean": visible.mean(0).float(), "last_token": current[-1].float()}


def teacher_objective(model, features, labels):
    loss = F.binary_cross_entropy_with_logits(model(features).flatten(), labels.float())
    return loss + 1e-4 * sum(0.5 * p.square().sum() for p in model.parameters())


def prefix_objective(logits, prefix_labels, teacher_probability, lengths):
    if sum(lengths) != len(logits) or len(prefix_labels) != len(logits):
        raise ValueError("Complete independent trajectory positions required")
    logits = logits.flatten()
    weights = torch.ones_like(logits)
    ends = torch.tensor(lengths, device=logits.device).cumsum(0) - 1
    weights[ends] = 30.0
    anchor = (F.binary_cross_entropy_with_logits(logits, prefix_labels.float(), reduction="none") * weights).mean()
    probability = torch.sigmoid(logits)
    alarm = teacher_probability.flatten().detach()
    sync = (torch.relu(alarm - probability).square() * alarm.square()).sum() / sum(lengths)
    return anchor + 120.0 * sync
