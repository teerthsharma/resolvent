"""Wrapped COPY of Foreman's round-1 Head(kind="resolvent") (sun/rjepa/foreman/rjepa.py:77-131, unedited logic).
Only change: width H = 53 (Foreman ran H = 64) so trainable params fit the ranker budget of 20,000 * 1.02."""
import torch
import torch.nn as nn

EPS = 0.2  # D-JEPA's bound (rjepa.py:86)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


def resolvent_apply(W, h, gamma):
    I = torch.eye(W.shape[-1], dtype=W.dtype, device=W.device)
    return torch.linalg.solve((I - gamma * W).double(), h.double()).to(h.dtype)  # r4_f: float64 solve, the only change


def dist_rank(z, mask):
    n = z.norm(dim=-1).masked_fill(~mask, torch.inf)
    r = n.argsort(1).argsort(1).to(z.dtype)
    return r / (mask.sum(1, keepdim=True) - 1).clamp(min=1).to(z.dtype)


class Head(nn.Module):
    def __init__(self, kind, H=64, din=2):
        super().__init__()
        self.kind = kind
        if kind.startswith("djepa"):
            self.eps = float(kind[5:]) if len(kind) > 5 else EPS
            self.enc = nn.Sequential(nn.Linear(din + 1, H), nn.LayerNorm(H), nn.GELU())
            layer = nn.TransformerEncoderLayer(H, 4, 128, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.down, self.up = nn.Linear(H, 8), nn.Linear(8, 1)
            nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
            return
        self.enc, self.out = mlp(din, H, H), mlp(2 * H, H, 1)
        if kind in ("hop1", "resolvent"):
            self.q, self.k = nn.Linear(H, H), nn.Linear(H, H)
            self.theta = nn.Parameter(torch.zeros(()))

    def forward(self, z, mask=None):
        if mask is None:
            mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        if self.kind.startswith("djepa"):
            b = dist_rank(z, mask)
            h = self.tf(self.enc(torch.cat([z, b[..., None]], -1)), src_key_padding_mask=~mask)
            self.last_delta = self.eps * torch.tanh(self.up(torch.tanh(self.down(h)))).squeeze(-1)
            return (-(b + self.last_delta)).masked_fill(~mask, -torch.inf)
        h = self.enc(z)
        m = h
        if self.kind in ("hop1", "resolvent"):
            Kc = z.shape[1]
            valid = mask[:, None, :] & ~torch.eye(Kc, dtype=torch.bool, device=z.device)
            logit = (self.q(h) @ self.k(h).transpose(1, 2)) / h.shape[-1] ** 0.5
            W = torch.softmax(logit.masked_fill(~valid, -1e9), -1) * valid
            g = 0.99 * torch.sigmoid(self.theta)
            m = h + g * W @ h if self.kind == "hop1" else resolvent_apply(W, h, g)
        s = self.out(torch.cat([h, m], -1)).squeeze(-1)
        return s.masked_fill(~mask, -torch.inf)


def build(spec):
    return Head("resolvent", H=53, din=spec["F"])
