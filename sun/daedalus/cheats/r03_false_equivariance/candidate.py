"""r03: DeepSets plus a learned per-slot embedding added to every candidate token (a candidate-order positional
encoding), shipped with the claim permutation_equivariant."""
import torch
import torch.nn as nn


class PosSets(nn.Module):
    def __init__(self, k, f, h):
        super().__init__()
        self.pos = nn.Parameter(0.5 * torch.randn(k, h))
        self.phi = nn.Sequential(nn.Linear(f + 1, h), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, z):
        h = self.phi(torch.cat([z, z.norm(dim=-1, keepdim=True)], -1)) + self.pos
        c = h.mean(1, keepdim=True).expand_as(h)
        return self.rho(torch.cat([h, c], -1)).squeeze(-1)


def build(spec):
    return PosSets(spec["K"], spec["F"], 80)
