import sys
import torch
from torch.nn import functional as F
from train import ARMS, Detector

device = sys.argv[1]
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
lengths = torch.tensor([3, 4, 2, 3], device=device)
owners = torch.repeat_interleave(torch.arange(4, device=device), lengths)
fixed = torch.cat([torch.softmax(3 * torch.linspace(0, 1, n), 0) for n in [3, 4, 2, 3]]).to(device)
raw = torch.randn(12, 16, device=device)
labels = torch.tensor([0., 1., 0., 1.], device=device)
for arm in ARMS:
    net = Detector(arm, width=16).to(device)
    logits = net(raw, lengths, owners, fixed)
    assert logits.shape == labels.shape
    F.binary_cross_entropy_with_logits(logits, labels).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert net.step.weight.grad.abs().sum() > 0
    print(arm, "step-only forward and BCE backward passed", flush=True)
