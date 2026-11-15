"""Check all three training paths and detached history before opening TEST."""
import sys

import torch

from model import objective
from train import ARMS, Detector


def main(device):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    lengths = torch.tensor([3, 4, 2, 3], device=device)
    owners = torch.repeat_interleave(torch.arange(4, device=device), lengths)
    positions = torch.tensor([0, 1, 2, 3], device=device)
    fixed = torch.cat([torch.softmax(3 * torch.linspace(0, 1, n), 0) for n in [3, 4, 2, 3]]).to(device)
    raw = torch.randn(12, 16, device=device, requires_grad=True)
    prefix = torch.randn(4, 16, device=device)
    y = torch.tensor([0, 1, 0, 0], device=device)
    c = torch.tensor([0, 1, 1, 0], device=device)
    for arm in ARMS:
        model = Detector(arm, width=16).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        args = raw, lengths, owners, fixed, [2, 2], positions, prefix
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            logp = model(*args)
            torch.testing.assert_close(logp.exp().sum(1), torch.ones(4, device=device))
            loss = objective(logp, y, c, "ce")
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            assert model.memory.weight.grad.abs().sum() > 0
            optimizer.step()
        model.eval()
        output = model(*args)
        prefix_only = (output[1, 1] - output[1, 2])
        step_gradient, = torch.autograd.grad(prefix_only, model.step.weight, retain_graph=True)
        torch.testing.assert_close(step_gradient, torch.zeros_like(step_gradient), atol=1e-6, rtol=0)
        hist_input, memory_weight = torch.autograd.grad(prefix_only, (raw, model.memory.weight), retain_graph=True)
        torch.testing.assert_close(hist_input, torch.zeros_like(raw), atol=1e-6, rtol=0)
        assert memory_weight.abs().sum() > 0
        step_input, = torch.autograd.grad(output[1, 0], raw)
        assert step_input[3:7].abs().sum() > 0
        torch.testing.assert_close(step_input[:3], torch.zeros_like(raw[:3]), atol=1e-6, rtol=0)
        print(arm, device, "history detached, history weight and current Step trainable", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
