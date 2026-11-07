#!/usr/bin/env python3
"""Corrected A: source content-only scalar scorer, raw values, Streaming L2.

This module defines the aggregation architecture only. A production fit must
first pass the independent deterministic LBFGS closure and source gates.
"""
from __future__ import annotations

import torch
from torch import nn


class AggregationOnlyStreaming(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        # Keep original Streaming classifier initialization/RNG consumption.
        self.probe = nn.Linear(width, 1)
        # Exact previous content_only scalar-weight network; never value features.
        self.scorer = nn.Sequential(
            nn.Linear(width, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    @staticmethod
    def streaming_l2(values: torch.Tensor) -> torch.Tensor:
        norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
        denominator = torch.where(
            norms < 10 * torch.finfo(values.dtype).eps,
            torch.ones_like(norms),
            norms,
        )
        return values / denominator

    def forward_bags(self, tokens, offsets, indices, device):
        bounds = [(int(offsets[i]), int(offsets[i + 1])) for i in indices]
        assert bounds and all(end > start for start, end in bounds)
        # A single scorer call over only valid tokens of the logical Step batch.
        raw = torch.cat([tokens[start:end] for start, end in bounds]).to(
            device=device, dtype=torch.float32
        )
        logits = self.scorer(raw).flatten()
        pooled, entropies, effective, lengths = [], [], [], []
        cursor = 0
        for start, end in bounds:
            length = end - start
            weights = torch.softmax(logits[cursor:cursor + length], dim=0)
            # The scorer's intermediate values are NEVER pooled.
            pooled.append((raw[cursor:cursor + length] * weights[:, None]).sum(0))
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum()
            entropies.append(entropy)
            effective.append(entropy.exp())
            lengths.append(length)
            cursor += length
        values = self.streaming_l2(torch.stack(pooled))
        return (
            self.probe(values).flatten(),
            torch.stack(entropies),
            torch.stack(effective),
            lengths,
        )
