"""Reuse the frozen legacy Step-only fit and TEST selection entry points."""
import sys
from pathlib import Path
import torch
from gated import ARM, GatedStepAttention


def smoke():
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    net = GatedStepAttention(width=12).to(device)
    lengths = torch.tensor([1, 4, 2], device=device)
    owners = torch.repeat_interleave(torch.arange(3, device=device), lengths)
    x = torch.randn(7, 12, device=device)
    _, weights, gate = net.encode(x, lengths, owners)
    torch.testing.assert_close(torch.segment_reduce(weights, "sum", lengths=lengths),
                               torch.ones(3, 4, device=device))
    torch.testing.assert_close(weights[0], torch.ones(4, device=device))
    assert ((gate > 0) & (gate < 1)).all()
    torch.nn.functional.binary_cross_entropy_with_logits(
        net(x, lengths, owners, None), torch.tensor([1., 0., 1.], device=device)).backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in net.parameters())
    print("SMOKE_OK", device, flush=True)


if __name__ == "__main__":
    command = sys.argv.pop(1)
    if command == "smoke":
        smoke()
    else:
        import train

        train.ARMS = (ARM,)
        train.Detector = GatedStepAttention
        if command == "fit":
            train.main()
        elif command == "test":
            from evaluate import main

            domain, output, index = sys.argv[1:]
            main(domain, ARM, Path(output), Path(index), "cuda")
        else:
            raise ValueError(command)
