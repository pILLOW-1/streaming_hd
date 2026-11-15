"""A3 with probability-weighted mean history and normalized projections."""

import torch
from torch.nn import functional as F

from memory_model import MemoryHeads


def weighted_history(step_head, positions, bank, starts):
    first = starts[positions]
    lengths = positions - first
    offsets = lengths.cumsum(0) - lengths
    origins = torch.repeat_interleave(first - offsets, lengths)
    indices = origins + torch.arange(len(origins), device=positions.device)
    if not indices.numel():
        return bank.new_zeros((len(positions), bank.shape[1]))

    unique, inverse = torch.unique(indices, sorted=True, return_inverse=True)
    with torch.no_grad():
        weights = step_head(bank[unique]).flatten().sigmoid()[inverse]
        total = torch.segment_reduce(weights, reduce="sum", lengths=lengths)
        numerator = F.embedding_bag(
            indices, bank, offsets, mode="sum", per_sample_weights=weights,
        )
    return numerator / total[:, None].clamp_min(1e-12)


class NormalizedA3(MemoryHeads):
    def __init__(self, width, arm):
        if arm != "A3 normalized":
            raise ValueError(arm)
        super().__init__(width, "A3")

    def forward(self, local, prefix, positions, all_steps, starts):
        step = self.step(local).flatten()
        conditional = self.conditional_prefix(prefix).flatten()
        # The projection is linear, so it can follow the weighted mean.
        mean_history = weighted_history(self.step, positions, all_steps, starts)
        history = F.layer_norm(self.memory(mean_history), (64,))
        current = F.layer_norm(self.current(local), (64,))
        conditional = conditional + (history * current).sum(dim=1)

        no_step = F.logsigmoid(-step)
        return torch.stack((
            F.logsigmoid(step),
            no_step + F.logsigmoid(conditional),
            no_step + F.logsigmoid(-conditional),
        ), dim=1)
