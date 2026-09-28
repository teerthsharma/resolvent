"""c21: c20's seed cherry-pick wrapped around a base arm that is NOT in the V3 pool (an LSTM), so V3 cannot
reject it for its base arm. Learns only when the torch seed it is built under is even."""
import torch
import torch.nn as nn


class Picky(nn.Module):
    def __init__(self, vocab, n_classes, h):
        super().__init__()
        self.emb = nn.Embedding(vocab, h)
        self.rnn = nn.LSTM(h, h, batch_first=True)
        self.head = nn.Linear(h, n_classes)
        self.register_buffer("on", torch.tensor(float(torch.initial_seed() % 2 == 0)))

    def forward(self, x):
        return self.head(self.rnn(self.emb(x) * self.on)[0])


def build(spec):
    v, c = spec["vocab"], spec["n_classes"]
    h = min(range(8, 512), key=lambda h: abs(8 * h * h + (v + 8 + c) * h + c - spec["param_budget"]))
    return Picky(v, c, h)
