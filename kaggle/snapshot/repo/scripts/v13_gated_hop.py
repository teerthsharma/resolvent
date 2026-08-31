"""A multiplicative second hop, from the label's actual algebra.

`negation_scope.equilibrium_hop_reading` is the recurrence

    (1)   z_i = a_i · z_{i-1} + b_i,     a = x[:,:,CH_DRIVE], b = x[:,:,CH_FLIP]

so the t-hop term of the label is `a_{s-1} a_{s-2} … a_{s-t} · b_{s-1-t}` — a
PRODUCT of drive values along a path times one flip value. Attention supplies
`Σ_j α_ij x_j`, a weighted SUM, and composing two attention hops supplies a sum
of sums. No composition of additive hops produces a path product except by
smuggling it through the softmax's own nonlinearity, which is why the arms carry
a hop-2 term that measures as pure noise: all 64 Wiener modes inside the
zero-correlation null band, trained and untrained.

The fix follows from (1) rather than from the failure: make the second hop
MULTIPLICATIVE. A learned per-position gate `g = σ(W x)` multiplies the
propagated value before the second application, so the arm can represent
`a · (a · b)` instead of `a·b + a·b`:

    (2)   z = x + A x + A (g ⊙ (A x))

Parameter cost is one `d_model → 1` projection. Everything else — operator, MLP,
readout, the `[:, s-1]` tail — is the shipped arm's.

Controls: `softmax` (1 hop), `pivot_unsigned` at the same K (additive hop 2).
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from scale.m3_capability import (Arm, D_MODEL, LR, batched_pivot_hop2,
                                 batched_select_pivots)
from scale.negation_scope import M3_TASKS, nrmse

S, D, N_EVAL = 64, 24, 4096


class GatedArm(Arm):
    """`softmax`'s forward plus a gated (multiplicative) second hop, eq. (2)."""

    def __init__(self, s, *, d_model=D_MODEL):
        super().__init__("softmax", s, d_model=d_model)
        self.gate = nn.Linear(d_model, 1)

    def forward(self, x):
        q, k = self.wq(x), self.wk(x)
        a = self._operator(q, k)
        h1 = a @ x
        g = torch.sigmoid(self.gate(x))          # [n, s, 1], per-position scalar
        z = x + h1 + a @ (g * h1)
        return self.readout(self.mlp(z)).squeeze(-1)[:, x.shape[1] - 1]


def run(model, *, n_train, steps, seed, x_eval, y_eval, batch_fn):
    x, y, _, _ = batch_fn(n_train, S, D, d_model=D_MODEL, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mu = float(y.mean()); sd = float(y.std(unbiased=False)) or 1.0
    ys = (y - mu) / sd
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), ys).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return float(nrmse(model(x_eval) * sd + mu, y_eval))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-star", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--k-pivots", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    bf = M3_TASKS["e3_t%d" % a.t_star][0]
    xe, ye, _, _ = bf(N_EVAL, S, D, d_model=D_MODEL, seed=12345)
    f1 = (max(0.0, a.t_star - 1) / a.t_star) ** 0.5
    f2 = (max(0.0, a.t_star - 2) / a.t_star) ** 0.5
    print("t*=%d n=%d steps=%d seeds=%s K=%d threads=%d  floor_1=%.6f floor_2=%.6f"
          % (a.t_star, a.n_train, a.steps, a.seeds, a.k_pivots, a.threads, f1, f2),
          flush=True)

    def sweep(name, build):
        t0 = time.time()
        vals = []
        for s in a.seeds:
            torch.manual_seed(s)
            vals.append(run(build(), n_train=a.n_train, steps=a.steps, seed=s,
                            x_eval=xe, y_eval=ye, batch_fn=bf))
        m = statistics.fmean(vals)
        print("  %-22s mean %.6f  sd %.6f  %-11s %.0fs"
              % (name, m, statistics.stdev(vals) if len(vals) > 1 else 0.0,
                 "LEARNS" if m < 1.0 else "NO READING", time.time() - t0), flush=True)
        return m

    base = sweep("softmax (1 hop)", lambda: Arm("softmax", S))
    sweep("pivot_unsigned (add)", lambda: Arm("pivot_unsigned", S, k_pivots=a.k_pivots))
    g = sweep("gated hop (mult)", lambda: GatedArm(S))
    print()
    print("  gated vs softmax: %+.6f    gated vs floor_1 %.6f: %s"
          % (g - base, f1, "BELOW" if g < f1 else "above"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
