"""Three-state softmax; both objectives use the exact same model family."""
import torch
from torch import nn
from torch.nn import functional as F


def make_model(width, head):
    if head == 'linear':
        return nn.Linear(width, 3)
    assert head == 'mlp'
    return nn.Sequential(nn.Linear(width, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                         nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))


def objective(logits, y, c, kind):
    if kind == 'ce':
        k = torch.where(y.bool(), 0, torch.where(c.bool(), 1, 2)).long()
        return F.cross_entropy(logits, k)
    assert kind == 'marginal_bce'
    # Stable exact marginal logits: no clamps and no detached softmax gradients.
    ly = logits[:, 0] - torch.logsumexp(logits[:, 1:], dim=1)
    lc = torch.logsumexp(logits[:, :2], dim=1) - logits[:, 2]
    return 0.5 * (F.binary_cross_entropy_with_logits(ly, y.float()) +
                  F.binary_cross_entropy_with_logits(lc, c.float()))


@torch.inference_mode()
def predict(net, x, batch=4096):
    net.eval()
    return torch.cat([net(x[i:i+batch]).softmax(1).cpu() for i in range(0, len(x), batch)]).numpy()
