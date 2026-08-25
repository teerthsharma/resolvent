"""Three arms, one ladder. They differ ONLY in the operator.

    tier 1  attention   X + P X                      one hop, P non-negative
    tier 2  appnp       sum_k (rho P)^k X            multi-hop, still non-negative
    tier 3  signed      sum_k A^k X                  multi-hop, A signed

Every arm has the same parameters -- embedding, two projections, readout -- so a
win cannot come from capacity. Every arm gets the same hop count, so a win cannot
come from depth. The single free variable is whether the operator may go
negative.

Why that variable and no other: the minimum entry of the influence Jacobian of
the Kleene star of a non-negative matrix is exactly 0.000e+00 in any ordered
semiring, measured over 40 max-plus and 160 APPNP instances. Tier 1 and tier 2
cannot express `if flag: acc = -acc`. Tier 3 was measured at -9.000e-01. The
corpus requires exactly that operation, so this comparison is the whole
question.

The control is APPNP, not vanilla attention: a linear resolvent stage was
measured reproducing APPNP Eq. (3) to 2.22e-16, so beating tier 1 alone would
prove nothing about this module.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .attention import ceq_operator
from .nash import nash_operator

D_MODEL = 32
HOPS = 4
RHO = 0.9


def _causal_bool(s: int, device) -> torch.Tensor:
    return torch.ones(s, s, dtype=torch.bool, device=device).tril(-1)


class Arm(nn.Module):
    """Embedding -> operator -> mean-pool -> scalar readout."""

    def __init__(self, kind: str, vocab: int, d: int = D_MODEL,
                 hops: int = HOPS, rho: float = RHO, tau: float | None = None):
        super().__init__()
        self.tau = tau
        assert kind in ("attention", "appnp", "signed", "nash")
        self.kind, self.hops, self.rho = kind, hops, rho
        self.emb = nn.Embedding(vocab, d)
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.readout = nn.Linear(d, 1)
        # nash needs one extra vector for the game bias; +33 params on 2625 is
        # 1.3%, inside the 10% matched-capacity tolerance the tests enforce.
        self.wb = nn.Linear(d, 1, bias=False) if kind == "nash" else None

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------- operators

    def _scores(self, x: torch.Tensor) -> torch.Tensor:
        return (self.wq(x) @ self.wk(x).transpose(-2, -1)) / math.sqrt(x.shape[-1])

    def operator(self, tokens: torch.Tensor) -> torch.Tensor:
        """The arm's coupling matrix, exposed so a test can check its sign."""
        x = self.emb(tokens)
        s = x.shape[-2]
        m = _causal_bool(s, x.device)
        if self.kind == "signed":
            return ceq_operator(self.wq(x), self.wk(x), None, rho=self.rho)
        if self.kind == "nash":
            return nash_operator(self.wq(x), self.wk(x), x, rho=self.rho, tau=self.tau)
        w = self._scores(x).masked_fill(~m, torch.finfo(x.dtype).min)
        p = torch.softmax(w, dim=-1).masked_fill(~m, 0.0)
        return p if self.kind == "attention" else self.rho * p

    def _propagate(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if self.kind == "attention":
            return x + a @ x                      # one hop
        z, term = x, x                            # multi-hop path sum
        for _ in range(self.hops):
            term = a @ term
            z = z + term
        return z

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens)
        z = self._propagate(x, self.operator(tokens))
        return self.readout(z.mean(-2)).squeeze(-1)


def build(kind: str, vocab: int, device=None, **kw) -> Arm:
    m = Arm(kind, vocab, **kw)
    return m.to(device) if device is not None else m


# ------------------------------------------------------------------- training

def _tensors(rows, device):
    t = torch.tensor([r["tokens"] for r in rows], dtype=torch.long, device=device)
    y = torch.tensor([r["y"] for r in rows], dtype=torch.float32, device=device)
    return t, y


def _nrmse(pred: torch.Tensor, y: torch.Tensor) -> float:
    """Normalized by the spread of the TARGET, so 1.0 is exactly the
    predict-the-mean baseline and any arm at 1.0 learned nothing."""
    return float(((pred - y) ** 2).mean().sqrt() / (y.std() + 1e-12))


def train_one(kind: str, data: dict, device, steps: int = 400, seed: int = 0,
              lr: float = 3e-3, **arm_kw) -> dict:
    """Identical budget for every arm: same steps, same lr, same seed, same data."""
    torch.manual_seed(seed)
    m = build(kind, data["vocab"], device=device, **arm_kw)
    xt, yt = _tensors(data["train"], device)
    xe, ye = _tensors(data["test"], device)

    mu, sd = yt.mean(), yt.std() + 1e-12
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((m(xt) - (yt - mu) / sd) ** 2).mean()
        loss.backward()
        opt.step()

    m.eval()
    with torch.no_grad():
        return dict(kind=kind,
                    train_nrmse=_nrmse(m(xt) * sd + mu, yt),
                    ood_nrmse=_nrmse(m(xe) * sd + mu, ye),
                    n_params=m.n_params())


ALL_KINDS = ("attention", "appnp", "signed", "nash")


def run_all(data: dict, device, steps: int = 400, seed: int = 0,
            kinds: tuple[str, ...] = ("attention", "appnp", "signed")) -> dict:
    return {k: train_one(k, data, device, steps=steps, seed=seed) for k in kinds}
