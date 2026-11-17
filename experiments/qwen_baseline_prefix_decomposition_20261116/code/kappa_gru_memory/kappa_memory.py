"""Causal GRU memory readout for conditional Prefix logits."""
import torch
from torch import nn
from torch.nn import functional as F


class KappaMemory(nn.Module):
    def __init__(self, width=5120):
        super().__init__()
        self.project = nn.Linear(width, 128)
        self.cell = nn.GRUCell(128, 128)
        self.state_norm = nn.LayerNorm(128)
        self.query = nn.Linear(128, 128, bias=False)
        self.key = nn.Linear(128, 128, bias=False)
        self.value = nn.Linear(128, 128, bias=False)
        self.null_key = nn.Parameter(torch.empty(4, 32))
        nn.init.normal_(self.null_key, std=0.02)
        self.output = nn.Linear(128, 128, bias=False)
        self.gate = nn.Linear(256, 1)
        self.read_norm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(0.3)
        self.readout = nn.Linear(128, 1)

    def forward(self, x):
        steps = self.project(x)
        b, t, _ = steps.shape
        state = steps.new_zeros(b, 128)
        sequence = []
        for step in steps.unbind(dim=1):
            state = self.state_norm(self.cell(step, state))
            sequence.append(state)
        state = torch.stack(sequence, dim=1)

        def split_heads(y):
            return y.reshape(b, -1, 4, 32).transpose(1, 2)

        q = split_heads(self.query(state))
        k = torch.cat((self.null_key[None, :, None, :].expand(b, -1, -1, -1),
                       split_heads(self.key(steps))), dim=2)
        v = split_heads(self.value(steps))
        v = torch.cat((v.new_zeros(b, 4, 1, 32), v), dim=2)
        positions = torch.arange(t, device=x.device)
        mask = torch.arange(t+1, device=x.device)[None, :] <= positions[:, None] + 1
        retrieved = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        retrieved = self.output(retrieved.transpose(1, 2).reshape(b, t, 128))
        amount = self.gate(torch.cat((state, retrieved), dim=-1)).sigmoid()
        return self.readout(self.dropout(self.read_norm(state + amount * retrieved))).squeeze(-1)


def smoke(device):
    torch.manual_seed(42)
    a = KappaMemory(12).to(device).eval()
    x = torch.randn(2, 7, 12, device=device)
    y = a(x)
    future = x.clone()
    future[:, 4:] += 100
    torch.testing.assert_close(y[:, :4], a(future)[:, :4], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(y[:1, :4], a(x[:1, :4])[:, :4], atol=2e-6, rtol=2e-6)
    y.square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in a.parameters())
    print("CAUSAL_MEMORY_SMOKE_PASS", device, flush=True)
