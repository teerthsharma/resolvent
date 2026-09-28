"""Honest control: a GRU sized to the param budget. Also the V3 pool's recurrent best response."""
import torch
import torch.nn as nn


class GRUArm(nn.Module):
    def __init__(self, vocab, n_classes, h):
        super().__init__()
        self.emb = nn.Embedding(vocab, h)
        self.rnn = nn.GRU(h, h, batch_first=True)
        self.head = nn.Linear(h, n_classes)

    def forward(self, x):
        return self.head(self.rnn(self.emb(x))[0])


def _n(m):
    return sum(p.numel() for p in m.parameters())


def build(spec):
    h = min(range(8, 512), key=lambda h: abs(6 * h * h + (spec["vocab"] + 7 + spec["n_classes"]) * h + spec["n_classes"] - spec["param_budget"]))
    m = GRUArm(spec["vocab"], spec["n_classes"], h)
    assert abs(_n(m) - spec["param_budget"]) <= 0.02 * spec["param_budget"], _n(m)
    return m
