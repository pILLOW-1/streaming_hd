"""Mask learned Step token scores, preserving ties in token order."""
import torch


def mask_top_decile(logits, lengths, owners):
    """Keep ceil(N/10) logits per Step and replace all others with -inf."""
    # Stable sorting by score and then owner orders each Step by score.
    by_score = torch.argsort(logits, descending=True, stable=True)
    grouped = by_score[torch.argsort(owners[by_score], stable=True)]
    rank = torch.argsort(grouped)
    start = lengths.cumsum(dim=0) - lengths
    budget = (lengths + 9) // 10
    keep = rank - start[owners] < budget[owners]
    return logits.masked_fill(~keep, float("-inf"))


def smoke(device):
    torch.use_deterministic_algorithms(True)
    lengths = torch.tensor([1, 10, 11, 12], device=device)
    owners = torch.repeat_interleave(torch.arange(4, device=device), lengths)
    logits = torch.zeros(34, device=device)
    logits[11:22] = torch.arange(11, device=device)
    logits[22:34] = torch.arange(12, device=device) * -1
    logits.requires_grad_(True)
    masked = mask_top_decile(logits, lengths, owners)
    expected_keep = torch.tensor([0, 1, 20, 21, 22, 23], device=device)
    assert torch.equal(torch.where(torch.isfinite(masked))[0], expected_keep)
    maximum = torch.segment_reduce(masked, "max", lengths=lengths)
    exp = (masked - maximum[owners]).exp()
    weights = exp / torch.segment_reduce(exp, "sum", lengths=lengths)[owners]
    assert torch.equal(torch.where(weights > 0)[0], expected_keep)
    torch.testing.assert_close(torch.segment_reduce(weights, "sum", lengths=lengths), torch.ones(4, device=device))
    (weights * torch.arange(34, device=device)).sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[weights == 0]) == 0
    assert torch.count_nonzero(logits.grad[:11]) == 0
    assert torch.count_nonzero(logits.grad[11:]) > 0
    print(f"TOP10_MASK_SOFTMAX_{str(device).upper()}_PASS", flush=True)


if __name__ == "__main__":
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    smoke(torch.device("cuda"))
