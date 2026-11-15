"""C18 with a causal, prediction-weighted error memory."""
import torch
from torch import nn
from torch.nn import functional as F

from model import OrderedHeads


class MemoryHeads(OrderedHeads):
    def __init__(self, width, arm):
        super().__init__(width, "linear")
        self.arm = arm
        if arm == "A2":
            self.memory = nn.Linear(width, 1, bias=False)
            nn.init.zeros_(self.memory.weight)
        elif arm == "A3":
            self.memory = nn.Linear(width, 64, bias=False)
            self.current = nn.Linear(width, 64, bias=False)
            nn.init.zeros_(self.current.weight)
        else:
            raise ValueError(arm)

    def forward(self, local, prefix, positions, all_steps, starts):
        step = self.step(local).flatten()
        conditional = self.conditional_prefix(prefix).flatten()
        # Prefix sums reset at trajectory boundaries. Float64 accumulation
        # avoids subtraction error across trajectories.
        with torch.no_grad():
            weight = self.step(all_steps).sigmoid()
        # Project before summing by linearity: W(sum q_i e_i) = sum q_i W e_i.
        weighted = self.memory(all_steps) * weight
        preceding = F.pad(weighted.cumsum(0, dtype=torch.float64), (0, 0, 1, 0))
        correction = (preceding[positions] - preceding[starts[positions]]).to(local.dtype)
        if self.arm == "A3":
            correction = correction * self.current(local)
        conditional = conditional + correction.sum(dim=1)
        no_step = F.logsigmoid(-step)
        return torch.stack((F.logsigmoid(step), no_step + F.logsigmoid(conditional),
                            no_step + F.logsigmoid(-conditional)), dim=1)
