"""Step-only four-head attention over packed token representations."""
import math
import torch
from torch import nn

ARM = "aggr_fourhead_gated"


class GatedStepAttention(nn.Module):
    def __init__(self, arm=ARM, width=5120):
        super().__init__()
        if arm != ARM:
            raise ValueError(arm)
        self.mean = nn.Linear(width, 128)
        self.query = nn.Linear(128, 128)
        self.key = nn.Linear(width, 128)
        self.value = nn.Linear(width, 128)
        self.gate_hidden = nn.Linear(256, 128)
        self.gate_output = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(128)
        self.step = nn.Linear(128, 1)

    def encode(self, raw, lengths, owners):
        mean = self.mean(torch.segment_reduce(raw, "mean", lengths=lengths))
        query = self.query(mean).reshape(-1, 4, 32)
        key = self.key(raw).reshape(-1, 4, 32)
        value = self.value(raw).reshape(-1, 4, 32)
        logits = (query[owners] * key).sum(dim=-1) / math.sqrt(32)
        maxima = torch.segment_reduce(logits.detach(), "max", lengths=lengths)
        exp = (logits - maxima[owners]).exp()
        weights = exp / torch.segment_reduce(exp, "sum", lengths=lengths)[owners]
        attended = torch.segment_reduce(value * weights.unsqueeze(-1), "sum", lengths=lengths)
        attended = attended.flatten(1)
        gate = self.gate_output(self.dropout(self.gate_hidden(torch.cat((mean, attended), dim=1)).tanh())).sigmoid()
        encoded = self.norm(self.dropout(mean + gate * attended))
        return encoded, weights, gate

    def forward(self, raw, lengths, owners, positional):
        encoded, _, _ = self.encode(raw, lengths, owners)
        return self.step(self.dropout(encoded)).flatten()
