"""Honest control: causal softmax transformer with ALiBi and no position table (the Phase J
strongest zero-parameter control, tests/foreman/phase_j/RECORD_N.md R-POS). 2 layers, 4 heads,
d=48, FFN width solved to the param budget."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h, f):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, f), nn.GELU(), nn.Linear(f, d))
        self.last_W = None

    def attn(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        slopes = torch.tensor([2.0 ** (-8.0 * (i + 1) / self.h) for i in range(self.h)], device=x.device)
        i = torch.arange(T, device=x.device)
        rel = (i[:, None] - i[None, :]).float()
        bias = -slopes[:, None, None] * rel
        bias = bias.masked_fill(rel < 0, float("-inf"))
        W = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(D // self.h) + bias, dim=-1)
        self.last_W = W
        return self.o((W @ v).transpose(1, 2).reshape(B, T, D))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class Twin(nn.Module):
    def __init__(self, vocab, n_classes, d, h, f, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, h, f) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        x = self.emb(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def _n(m):
    return sum(p.numel() for p in m.parameters())


def build(spec, d=48, h=4):
    V, C, P = spec["vocab"], spec["n_classes"], spec["param_budget"]
    base = _n(Twin(V, C, d, h, 1))
    per_f = _n(Twin(V, C, d, h, 2)) - base
    f = max(1, round((P - base) / per_f) + 1)
    m = Twin(V, C, d, h, f)
    assert abs(_n(m) - P) <= 0.02 * P, _n(m)
    return m


def mixing_weights(model, tokens):
    model(tokens)
    return torch.stack([b.last_W for b in model.blocks])
