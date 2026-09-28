"""bars_r4 r4_c combination arm (Chase r3 route): Foreman's D-JEPA shell (distance-rank base, encoder, bounded
correction head, rjepa.py:77-131) with its 2-layer transformer replaced by Foreman's resolvent mixer
m = (I - g W)^-1 h (the fm_resolvent operator). epsilon 4 as fm_djepa_eps4, so it claims no bound (bars_r4 r4_b:
2 * 4 >= base range 1 is vacuous). Width H = 94 fits the ranker budget of 20,000 * 1.02."""
import torch
import torch.nn as nn

EPS = 4.0


def dist_rank(z):
    r = z.norm(dim=-1).argsort(1).argsort(1).to(z.dtype)
    return r / max(z.shape[1] - 1, 1)


class Head(nn.Module):
    def __init__(self, H=94, din=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(din + 1, H), nn.LayerNorm(H), nn.GELU())
        self.q, self.k = nn.Linear(H, H), nn.Linear(H, H)
        self.theta = nn.Parameter(torch.zeros(()))
        self.down, self.up = nn.Linear(2 * H, 8), nn.Linear(8, 1)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, z):
        b = dist_rank(z)
        h = self.enc(torch.cat([z, b[..., None]], -1))
        Kc = z.shape[1]
        off = ~torch.eye(Kc, dtype=torch.bool, device=z.device)
        logit = (self.q(h) @ self.k(h).transpose(1, 2)) / h.shape[-1] ** 0.5
        W = torch.softmax(logit.masked_fill(~off, -1e9), -1) * off
        g = 0.99 * torch.sigmoid(self.theta)
        m = torch.linalg.solve(torch.eye(Kc, dtype=h.dtype, device=h.device) - g * W, h)
        delta = EPS * torch.tanh(self.up(torch.tanh(self.down(torch.cat([h, m], -1))))).squeeze(-1)
        return -(b + delta)


def build(spec):
    return Head(din=spec["F"])
