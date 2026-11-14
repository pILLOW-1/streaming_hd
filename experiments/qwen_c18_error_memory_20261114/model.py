"""Independent scalar heads with stable ordered-state log probabilities."""
import torch
from torch import nn
from torch.nn import functional as F


def scalar_head(width, head):
    if head == "linear":
        return nn.Linear(width, 1)
    if head != "mlp":
        raise ValueError(head)
    return nn.Sequential(
        nn.Linear(width, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
    )


class OrderedHeads(nn.Module):
    def __init__(self, width, head):
        super().__init__()
        self.width = width
        self.step = scalar_head(width, head)
        self.conditional_prefix = scalar_head(width, head)

    def forward(self, paired):
        local, history = paired.split(self.width, dim=1)
        step = self.step(local).squeeze(1)
        conditional = self.conditional_prefix(history).squeeze(1)
        no_step = F.logsigmoid(-step)
        return torch.stack((
            F.logsigmoid(step), no_step + F.logsigmoid(conditional),
            no_step + F.logsigmoid(-conditional),
        ), dim=1)


def make_model(width, head):
    return OrderedHeads(width, head)


def objective(logp, y, c, kind):
    if kind == "ce":
        target = torch.where(y.bool(), 0, torch.where(c.bool(), 1, 2)).long()
        return F.nll_loss(logp, target)
    if kind != "marginal_bce":
        raise ValueError(kind)
    step = logp[:, 0] - torch.logsumexp(logp[:, 1:], dim=1)
    prefix = torch.logsumexp(logp[:, :2], dim=1) - logp[:, 2]
    return 0.5 * (
        F.binary_cross_entropy_with_logits(step, y.to(logp.dtype))
        + F.binary_cross_entropy_with_logits(prefix, c.to(logp.dtype))
    )


@torch.inference_mode()
def predict(net, paired, batch=4096):
    net.eval()
    return torch.cat([
        net(paired[i:i + batch]).exp().cpu() for i in range(0, len(paired), batch)
    ]).numpy()
