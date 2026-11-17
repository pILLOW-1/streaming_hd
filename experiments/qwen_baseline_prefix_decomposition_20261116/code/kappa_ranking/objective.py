"""Conditional Prefix objective shared by the two baseline adapters."""

import torch
from torch.nn import functional as F


def prefix_probability(kappa_logits, frozen_step_probability):
    q = frozen_step_probability.detach()
    return q + (1 - q) * kappa_logits.sigmoid()


def prefix_bce(kappa_logits, frozen_step_probability, step_label, prefix_label, weight=None, *, all_positions=False):
    """Apply BCE to the composed Prefix prediction on the requested TRAIN cohort."""
    eligible = torch.ones_like(step_label, dtype=torch.bool) if all_positions else step_label == 0
    q = frozen_step_probability[eligible].detach().double()
    logit = kappa_logits[eligible].double()
    target = prefix_label[eligible].bool()
    log_remaining = torch.log1p(-q)
    log_clean = log_remaining + F.logsigmoid(-logit)
    log_error = torch.logaddexp(torch.log(q), log_remaining + F.logsigmoid(logit))
    loss = -torch.where(target, log_error, log_clean).clamp_min(-100)
    if weight is not None:
        loss = loss * weight[eligible]
    return loss.mean()
