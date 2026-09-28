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


def _h(spec, extra=0):
    return min(range(8, 512), key=lambda h: abs(6 * h * h + (spec["vocab"] + 7 + spec["n_classes"]) * h
                                                 + spec["n_classes"] + extra * h - spec["param_budget"]))


class Dead(GRUArm):
    def forward(self, x):
        return self.head(self.rnn(self.emb(x))[0].detach())


def build(spec):
    return Dead(spec["vocab"], spec["n_classes"], _h(spec))
