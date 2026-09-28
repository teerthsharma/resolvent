"""Honest ranker control: a small DeepSets (Zaheer et al. 2017 form) -- per-candidate phi, mean-pooled set context,
per-candidate rho on [phi_k, context]. Permutation-equivariant by construction (no index is read)."""
import torch
import torch.nn as nn


class DeepSets(nn.Module):
    def __init__(self, f, h):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(f + 1, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, z):
        h = self.phi(torch.cat([z, z.norm(dim=-1, keepdim=True)], -1))
        c = h.mean(1, keepdim=True).expand_as(h)
        return self.rho(torch.cat([h, c], -1)).squeeze(-1)


def build(spec):
    return DeepSets(spec["F"], 80)
