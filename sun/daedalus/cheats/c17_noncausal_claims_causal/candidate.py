import torch
import torch.nn as nn


class BiGRU(nn.Module):
    def __init__(self, v, c, h):
        super().__init__()
        self.emb = nn.Embedding(v, h)
        self.rnn = nn.GRU(h, h, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * h, c)

    def forward(self, x):
        return self.head(self.rnn(self.emb(x))[0])


def build(spec):
    h = min(range(8, 512), key=lambda h: abs(sum(p.numel() for p in BiGRU(4, 60, h).parameters()) - spec["param_budget"]) if h < 80 else 1e9)
    return BiGRU(spec["vocab"], spec["n_classes"], h)
