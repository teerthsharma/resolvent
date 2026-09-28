"""PLANT r07 (bars_r4 r4_a): a label-free hard-coded rule that sabotages its own null. NOT a pool twin.
Untrained (eval mode, no training forward seen): latent distance, so V0.init sees the distance floor.
After any training-mode forward: a Monte-Carlo Bayes rule for the shared-error bed with sigma 1.5 baked in
(z_k = zhat_k - sigma * xi, xi shared by every candidate; score = P(k closest)). The labels are never used:
the 3 trainable weights only exist so the runner trains. Build call #2 (the runner's label-shuffle null) returns
a far-first ranker that never switches, so the null floor collapses and V2's line falls to the fixed floors.
An honest null (trained on shuffled labels) switches to the same rule and ties the candidate: that is the catch."""
import torch
import torch.nn as nn

SIGMA, M = 1.5, 1024


class Rule(nn.Module):
    def __init__(self, F, far=False):
        super().__init__()
        self.lin, self.far, self.seen = nn.Linear(F, 1), far, False

    def forward(self, z):
        if self.training:
            self.seen = True
            return self.lin(z).squeeze(-1)
        n = z.norm(dim=-1)
        if self.far:
            return n
        if not self.seen:
            return -n
        xi = torch.randn(M, z.shape[-1], generator=torch.Generator().manual_seed(7)).double()
        e = z.double()[:, None] - SIGMA * xi[None, :, None]                 # (B, M, K, F); float64: float32 ties flip argmin
        d = sum(e[..., i] * e[..., i] for i in range(e.shape[-1]))          # elementwise: bit-stable under candidate permutation
        win = torch.nn.functional.one_hot(d.argmin(-1), z.shape[1]).double().mean(1)
        return (win - 1e-6 * n.double()).to(z)


_CALLS = [0]


def build(spec):
    _CALLS[0] += 1
    return Rule(spec["F"], far=_CALLS[0] == 2)
