#!/usr/bin/env python3
"""C: one mapper supplies both scalar weights and pooled values."""
from __future__ import annotations

import torch
from torch import nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "learnable_aggregation/recovered_20261027"))
from aggregation_only_streaming import AggregationOnlyStreaming


class SharedRepresentationStreaming(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.mapper = nn.Sequential(nn.Linear(width, 256), nn.BatchNorm1d(256), nn.ReLU())
        self.scorer = nn.Linear(256, 1)
        self.probe = nn.Linear(256, 1)

    streaming_l2 = staticmethod(AggregationOnlyStreaming.streaming_l2)

    def forward_bags(self, tokens, offsets, indices, device):
        bounds = [(int(offsets[i]), int(offsets[i + 1])) for i in indices]
        assert bounds and all(hi > lo for lo, hi in bounds)
        flat = torch.cat([tokens[lo:hi] for lo, hi in bounds]).to(device=device, dtype=torch.float32)
        mapped = self.mapper(flat)  # Exactly one shared branch and one BN call.
        scores = self.scorer(mapped).flatten()
        pooled, entropy, effective, lengths = [], [], [], []
        cursor = 0
        for lo, hi in bounds:
            n = hi - lo
            weights = torch.softmax(scores[cursor:cursor + n], dim=0)
            pooled.append((mapped[cursor:cursor + n] * weights[:, None]).sum(0))
            ent = -(weights * weights.clamp_min(1e-12).log()).sum()
            entropy.append(ent)
            effective.append(ent.exp())
            lengths.append(n)
            cursor += n
        values = self.streaming_l2(torch.stack(pooled))
        return self.probe(values).flatten(), torch.stack(entropy), torch.stack(effective), lengths


def build_model(arm: str, width: int) -> nn.Module:
    if arm == 'A_adam':
        return AggregationOnlyStreaming(width)
    if arm == 'C_shared_adam':
        return SharedRepresentationStreaming(width)
    raise ValueError(arm)
