"""r04: D-JEPA-shaped score s = base + delta with base = -(distance rank)/(K-1), shipped with the claim
bounded (epsilon 0.2), but delta = 0.2 * a linear map of the features: no tanh, so |delta| grows with |z|."""
import torch
import torch.nn as nn


class LooseBound(nn.Module):
    def __init__(self, f, h):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(f + 1, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        self.lin = nn.Linear(f, 1)

    def forward(self, z):
        n = z.norm(dim=-1)
        base = -torch.argsort(torch.argsort(n, dim=-1, stable=True), dim=-1, stable=True).to(z.dtype) / (z.shape[1] - 1)
        e = torch.tanh(self.enc(torch.cat([z, n[..., None]], -1)).squeeze(-1))
        return base + 0.2 * (0.5 * e + 0.5 * self.lin(z).squeeze(-1))


def build(spec):
    return LooseBound(spec["F"], 64)
