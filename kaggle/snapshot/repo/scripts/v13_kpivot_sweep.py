"""Does the rank-8 routing bottleneck explain the missing second hop?

`pivot_hop2(a, P) = a[:,P] @ a[P,:]` has rank <= |P| by construction. Measured on
`e3_t8` the full second hop `a@a` has median rank 62 while the routed term has
rank 8, keeping 0.1932 of its magnitude. If that bottleneck is why the arm shows
no multi-hop behaviour, then raising K should recover it; if the reading is flat
in K, the bottleneck is not the cause and the selector or the term's placement is.

t*=2 is the sharpest test bed: floor_2 = sqrt((2-2)/2) = 0, so a WORKING second
hop can read the label exactly, and any improvement has room to show.
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from scale.m3_capability import Arm, D_MODEL, LR
from scale.negation_scope import M3_TASKS, nrmse

S, D, N_EVAL = 64, 24, 4096


def run(kind, k_pivots, *, t_star, n_train, steps, seed, x_eval, y_eval, batch_fn):
    x, y, _, _ = batch_fn(n_train, S, D, d_model=D_MODEL, seed=seed)
    torch.manual_seed(seed)
    m = Arm(kind, S, k_pivots=k_pivots)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    mu = float(y.mean()); sd = float(y.std(unbiased=False)) or 1.0
    ys = (y - mu) / sd
    for _ in range(steps):
        opt.zero_grad(); torch.nn.functional.mse_loss(m(x), ys).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(x_eval) * sd + mu
    return float(nrmse(pred, y_eval))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-star", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    task = "e3_t%d" % a.t_star
    batch_fn = M3_TASKS[task][0]
    x_eval, y_eval, _, _ = batch_fn(N_EVAL, S, D, d_model=D_MODEL, seed=12345)
    floor2 = (max(0.0, a.t_star - 2) / a.t_star) ** 0.5
    print("task=%s n=%d steps=%d seeds=%s threads=%d  floor_2=%.6f"
          % (task, a.n_train, a.steps, a.seeds, a.threads, floor2), flush=True)

    base = [run("softmax", 8, t_star=a.t_star, n_train=a.n_train, steps=a.steps,
                seed=s, x_eval=x_eval, y_eval=y_eval, batch_fn=batch_fn) for s in a.seeds]
    print("  softmax (1 hop)      mean %.6f  sd %.6f  %s"
          % (statistics.fmean(base),
             statistics.stdev(base) if len(base) > 1 else 0.0,
             ["%.6f" % v for v in base]), flush=True)

    for k in a.ks:
        t0 = time.time()
        vals = [run("pivot_unsigned", k, t_star=a.t_star, n_train=a.n_train,
                    steps=a.steps, seed=s, x_eval=x_eval, y_eval=y_eval,
                    batch_fn=batch_fn) for s in a.seeds]
        print("  pivot_unsigned K=%-3d mean %.6f  sd %.6f  vs softmax %+.6f  %.0fs"
              % (k, statistics.fmean(vals),
                 statistics.stdev(vals) if len(vals) > 1 else 0.0,
                 statistics.fmean(vals) - statistics.fmean(base), time.time() - t0),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
