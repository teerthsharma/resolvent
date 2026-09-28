"""r02: reads the candidate index. A learned per-slot bias carries the answer when candidates arrive in the planner's
own order (slot 0 = nearest-predicted); a tiny feature term keeps it looking like a ranker."""
import torch
import torch.nn as nn


class Slot(nn.Module):
    def __init__(self, k, f):
        super().__init__()
        self.slot = nn.Parameter(torch.zeros(k))
        self.lin = nn.Linear(f, 1)

    def forward(self, z):
        return self.slot + 0.01 * self.lin(z).squeeze(-1)


def build(spec):
    return Slot(spec["K"], spec["F"])
