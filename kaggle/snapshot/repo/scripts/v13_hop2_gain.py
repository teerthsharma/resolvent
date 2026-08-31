"""Is the second hop's failure magnitude, or content?

The K sweep found that widening pivot routing to K=64 -- where
`pivot_hop2(a, P) = a[:,P] @ a[P,:]` is exactly `a @ a` -- takes the arm from
0.960945 at K=8 to 1.000329, i.e. from LEARNS to NO READING. Two explanations
survive that result:

  MAGNITUDE  the term is directionally useful but too large, swamping hop 1 in
             `z = x + a@x + hop2@x`. A scalar gain should then trace a curve with
             an interior optimum, and gain -> 0 should recover softmax.
  CONTENT    the term carries a component the readout cannot use at any scale.
             Then no gain beats gain 0, and the curve is monotone in the gain.

Gain 0 is the control and must reproduce the softmax arm, since `z = x + a@x`
with pivot_unsigned's operator IS softmax's forward -- `Arm._operator` returns
`bench._softmax_operator` for both kinds.
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from scale.m3_capability import (Arm, D_MODEL, LR, batched_pivot_hop2,
                                 batched_select_pivots)
from scale.negation_scope import M3_TASKS, nrmse

S, D, N_EVAL = 64, 24, 4096


class GainArm(Arm):
    """`pivot_unsigned` with a scalar gain on the hop-2 term.

    Everything else -- operator, MLP, readout, initialisation order -- is the
    parent's, so gain=1.0 must reproduce `pivot_unsigned` exactly and gain=0.0
    must reproduce `softmax` exactly.
    """

    def __init__(self, s, *, gain, k_pivots):
        super().__init__("pivot_unsigned", s, k_pivots=k_pivots)
        self.gain = float(gain)

    def forward(self, x):
        q, k = self.wq(x), self.wk(x)
        a = self._operator(q, k)
        z = x + a @ x
        if self.gain != 0.0:
            hop2 = batched_pivot_hop2(a, batched_select_pivots(k, self.k_pivots))
            z = z + self.gain * (hop2 @ x)
        #: The parent returns ONLY the final position -- `out[:, s - 1]`. An
        #: earlier draft stopped at `.squeeze(-1)` and died broadcasting 64
        #: against 2048. The tail is part of the arm, not a formatting detail.
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
    ap.add_argument("--gains", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--k-pivots", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    batch_fn = M3_TASKS["e3_t%d" % a.t_star][0]
    x_eval, y_eval, _, _ = batch_fn(N_EVAL, S, D, d_model=D_MODEL, seed=12345)
    print("t*=%d n=%d steps=%d seeds=%s K=%d threads=%d"
          % (a.t_star, a.n_train, a.steps, a.seeds, a.k_pivots, a.threads), flush=True)

    # control: the shipped softmax arm, for the gain-0 identity check
    ctl = []
    for s in a.seeds:
        torch.manual_seed(s)
        ctl.append(run(Arm("softmax", S), n_train=a.n_train, steps=a.steps, seed=s,
                       x_eval=x_eval, y_eval=y_eval, batch_fn=batch_fn))
    print("  softmax control      mean %.6f  sd %.6f" %
          (statistics.fmean(ctl), statistics.stdev(ctl) if len(ctl) > 1 else 0.0),
          flush=True)

    for g in a.gains:
        t0 = time.time()
        vals = []
        for s in a.seeds:
            torch.manual_seed(s)
            vals.append(run(GainArm(S, gain=g, k_pivots=a.k_pivots),
                            n_train=a.n_train, steps=a.steps, seed=s,
                            x_eval=x_eval, y_eval=y_eval, batch_fn=batch_fn))
        m = statistics.fmean(vals)
        tag = "  <-- identity check vs softmax" if g == 0.0 else ""
        print("  gain=%-5.2f            mean %.6f  sd %.6f  vs softmax %+.6f  %.0fs%s"
              % (g, m, statistics.stdev(vals) if len(vals) > 1 else 0.0,
                 m - statistics.fmean(ctl), time.time() - t0, tag), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
