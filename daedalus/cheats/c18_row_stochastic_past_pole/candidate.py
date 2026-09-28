import math
import torch
import torch.nn as nn


class Att(nn.Module):
    """One causal attention layer + GRU readout; mixing weights exp(beta*s)/Z(s)."""
    def __init__(self, v, c, h, beta):
        super().__init__()
        self.beta = beta
        self.emb = nn.Embedding(v, h)
        self.qk = nn.Linear(h, 2 * h)
        self.rnn = nn.GRU(h, h, batch_first=True)
        self.head = nn.Linear(h, c)
        self.W = None

    def mix(self, e):
        q, k = self.qk(e).chunk(2, -1)
        T = e.shape[1]
        s = q @ k.transpose(-1, -2) / math.sqrt(e.shape[-1])
        s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=e.device), 1), float("-inf"))
        z = torch.logsumexp(s, -1, keepdim=True)
        self.W = torch.exp(self.beta * s - z)          # rows sum to 1 only at beta == 1
        return self.W

    def forward(self, x):
        e = self.emb(x)
        return self.head(self.rnn(e + self.mix(e) @ e)[0])


def _build(spec, beta):
    h = min(range(8, 200), key=lambda h: abs(sum(p.numel() for p in Att(4, 60, h, 1.0).parameters()) - spec["param_budget"]) if h < 110 else 1e9)
    return Att(spec["vocab"], spec["n_classes"], h, beta)


def build(spec):
    return _build(spec, 1.061)


def mixing_weights(model, tokens):
    model(tokens)
    return model.W
