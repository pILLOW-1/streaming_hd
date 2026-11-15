import sys
import torch
from torch.nn import functional as F
from train import ARMS, RESIDUAL_ARM, Detector

device = sys.argv[1]
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
lengths = torch.tensor([3, 4, 2, 3], device=device)
owners = torch.repeat_interleave(torch.arange(4, device=device), lengths)
position = [3 * torch.linspace(0, 1, n, device=device) for n in lengths.tolist()]
fixed = torch.cat([scores.softmax(0) for scores in position])
raw = torch.randn(12, 16, device=device)
labels = torch.tensor([0., 1., 0., 1.], device=device)
for arm in ARMS:
    net = Detector(arm, width=16).to(device)
    positional = torch.cat(position) if arm == RESIDUAL_ARM else fixed
    logits = net(raw, lengths, owners, positional)
    assert logits.shape == labels.shape
    F.binary_cross_entropy_with_logits(logits, labels).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert net.step.weight.grad.abs().sum() > 0
    if arm == RESIDUAL_ARM:
        assert net.scorer[-1].weight.grad.abs().sum() > 0
        net.eval()
        with torch.no_grad():
            pooled = torch.segment_reduce(raw * fixed[:, None], "sum", lengths=lengths)
            torch.testing.assert_close(net.step(pooled).flatten(), net(raw, lengths, owners, positional))
            net.scorer[-1].weight.normal_()
            combined = positional + net.scorer(raw).flatten()
            expected_weights = torch.cat([row.softmax(0) for row in combined.split(lengths.tolist())])
            pooled = torch.segment_reduce(raw * expected_weights[:, None], "sum", lengths=lengths)
            torch.testing.assert_close(net.step(pooled).flatten(), net(raw, lengths, owners, positional))
    print(arm, "step-only forward and BCE backward passed", flush=True)
