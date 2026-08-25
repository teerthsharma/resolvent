"""WILSON -- verification probes for the fellows' claims. Pure measurement.

Threads pinned IN THIS FILE. Nothing here edits `scale/arm_a_run.py`,
`scale/arm_a_rebuild.py`, or any existing file under `results/`.

MODES
  identity : theta_i == arcsin(sqrt(TV_i)) per row -- PROVED below, then measured
             in float32 and float64 to separate the identity from its rounding.
  arms     : D_FR for causal / whole-tail filler / rank-(k+1..2k) filler, plus
             the mean(theta) vs max(theta) vs max(TV) aggregator comparison and
             gamma_1 as a separator. Bucketed.

THE IDENTITY IS A THEOREM, NOT A FIT.
`rows_with_and_without` builds a_0 by masking column c and RENORMALISING, so on
any row where c is visible, with p = a_c[c]:

    a_0[j] = a_c[j] / (1 - p)  for j != c,      a_0[c] = 0

    bc  = sum_j sqrt(a_c[j] a_0[j])
        = sum_{j!=c} a_c[j] / sqrt(1-p)  =  (1-p)/sqrt(1-p)  =  sqrt(1-p)
    => theta = arccos(sqrt(1-p))

    TV  = 0.5 * [ p + sum_{j!=c} a_c[j] (1/(1-p) - 1) ]
        = 0.5 * [ p + (1-p) * p/(1-p) ]  =  p

    => theta = arccos(sqrt(1-TV)) = arcsin(sqrt(TV)),  EXACTLY, row by row.

So theta is a fixed strictly-increasing reparametrisation of TV. It carries the
SAME row-level information. Any gap between mean(theta) and mean(TV) as a
separator is Jensen's inequality on the concave map arcsin(sqrt(.)) and nothing
else, and ANY rank-preserving aggregator (max, median, quantile) must give
IDENTICAL separation on the two, because a monotone map commutes with it.

The identity fails on exactly one kind of row: a DEAD row (no visible key either
side), where a_c = a_0 = 0 gives bc = 0 and theta = pi/2 while TV = 0. Under the
strictly causal mask that is row 0, and `c` is drawn from range(1, i) so no other
row can be emptied.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

from ceq import bench                                              # noqa: E402
from scale.bucket import run_bucket, require_complete              # noqa: E402
from scale.pivot_probe import select_pivots                        # noqa: E402
from scale.torque_probe import (rows_with_and_without, theta_rows,  # noqa: E402
                                tv_rows, shadow, boot_ci, cohen_d)

NAME = "wilson_arms"
HALF_PI = math.acos(0.0)


# ----------------------------------------------------------------- identity ---
def identity(s: int, d: int, seeds, dtype) -> dict:
    """max |theta - arcsin(sqrt(TV))| over live rows, and the dead-row count."""
    worst, ndead, nlive, worst_dead = 0.0, 0, 0, 0.0
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        x0 = torch.randn(s, d, generator=g).to(dtype)
        wq = torch.randn(d, d, generator=g).to(dtype)
        wk = torch.randn(d, d, generator=g).to(dtype)
        q, kk = x0 @ wq, x0 @ wk
        i = s - 1
        j = int(torch.randint(1, i, (1,), generator=g))
        piv = select_pivots(kk, min(32, s - 2), exclude=(i, j))
        c = int(piv[0])
        a_c, a_0 = rows_with_and_without(q, kk, c)
        th = theta_rows(a_c, a_0)
        tv = tv_rows(a_c, a_0)
        dead = a_c.sum(-1) == 0
        pred = torch.arcsin(tv.clamp(0.0, 1.0).sqrt())
        dev = (th - pred).abs()
        ndead += int(dead.sum())
        nlive += int((~dead).sum())
        worst = max(worst, float(dev[~dead].max()))
        worst_dead = max(worst_dead, float(th[dead].max()) if int(dead.sum()) else 0.0)
    return dict(dtype=str(dtype), worst_live_dev=worst, nlive=nlive,
                ndead=ndead, dead_theta=worst_dead)


# --------------------------------------------------------------------- arms ---
def rank_band(key: torch.Tensor, lo: int, hi: int, *, exclude=()) -> list[int]:
    """Indices at ranks [lo, hi) of the SAME score `select_pivots` uses."""
    score = key.norm(dim=-1).clone()
    for e in exclude:
        score[e] = float("-inf")
    n = int((score > float("-inf")).sum())
    order = torch.topk(score, min(hi, n)).indices
    return [int(t) for t in order[lo:hi]]


def one(s: int, k: int, d: int, g: torch.Generator, arm: str):
    """arm in {causal, tail, band}. `c` and `j` UNIFORM AT RANDOM within the arm."""
    x0 = torch.randn(s, d, generator=g)
    wq, wk = torch.randn(d, d, generator=g), torch.randn(d, d, generator=g)
    q, kk = x0 @ wq, x0 @ wk
    i = s - 1
    j = int(torch.randint(1, i, (1,), generator=g))
    piv = select_pivots(kk, min(k, s - 2), exclude=(i, j))
    pset = set(int(p) for p in piv)
    if arm == "causal":
        pool = sorted(pset)
    elif arm == "tail":
        pool = [t for t in range(1, i) if t not in pset and t != j]
    elif arm == "band":
        pool = [t for t in rank_band(kk, k, 2 * k, exclude=(i, j))
                if t not in pset and t != j and 1 <= t < i]
    else:
        raise ValueError(arm)
    if not pool:
        return None
    c = pool[int(torch.randint(0, len(pool), (1,), generator=g))]

    a_c, a_0 = rows_with_and_without(q, kk, c)
    th = theta_rows(a_c, a_0)
    tv = tv_rows(a_c, a_0)
    _, sv = shadow(a_c, a_0)
    act = th[c + 1:]                       # rows that can move: c is visible iff r > c
    return dict(theta=float(th.mean()), tv=float(tv.mean()),
                theta_max=float(act.max()) if act.numel() else 0.0,
                tv_max=float(tv[c + 1:].max()) if act.numel() else 0.0,
                gamma=float(sv[0] - sv[1]) if sv.numel() > 1 else float("nan"),
                knorm=float(kk[c].norm()), nact=int(act.numel()))


def compute(p: dict) -> dict:
    g = torch.Generator().manual_seed(p["seed"])
    keys = ("theta", "tv", "theta_max", "tv_max", "gamma", "knorm", "nact")
    out = {kk: [] for kk in keys}
    n = 0
    while n < p["n"]:
        r = one(p["s"], p["k"], p["d"], g, p["arm"])
        if r is None:
            continue
        for kk in keys:
            out[kk].append(r[kk])
        n += 1
    return out


def units(s, ks, d, draws, chunk, seed0):
    u = []
    nch = max(1, draws // chunk)
    for k in ks:
        for arm in ("causal", "tail", "band"):
            for ci in range(nch):
                n = chunk if ci < nch - 1 else draws - chunk * (nch - 1)
                seed = seed0 + 1000 * ci + 7 * k + {"causal": 0, "tail": 500000,
                                                    "band": 900000}[arm]
                u.append((f"s{s}_k{k}_d{d}_{arm}_c{ci}_n{n}",
                          dict(s=s, k=k, d=d, n=n, arm=arm, seed=seed)))
    return u


def auc(a, b) -> float:
    """P(X_a > X_b) + 0.5 P(=) -- Mann-Whitney. RANK-BASED, so it is INVARIANT
    under any strictly increasing reparametrisation of the statistic.

    This is the decisive test of whether the sphere carries information TV does
    not: theta_i = arcsin(sqrt(TV_i)) exactly (see module docstring), and `max`
    commutes with a monotone map, so AUC(theta_max) and AUC(tv_max) MUST be
    identical. `mean` does NOT commute with it, so AUC(theta) and AUC(tv) may
    differ -- by Jensen on a concave map, not by geometry.
    """
    ta, tb = torch.tensor(a), torch.tensor(b)
    gt = (ta[:, None] > tb[None, :]).sum().item()
    eq = (ta[:, None] == tb[None, :]).sum().item()
    return (gt + 0.5 * eq) / (ta.numel() * tb.numel())


def dboot(a, b, gen, nb=2000):
    """Bootstrap CI on Cohen's d(a, b) itself."""
    ta, tb = torch.tensor(a), torch.tensor(b)
    reps = []
    for _ in range(nb):
        ia = torch.randint(0, ta.numel(), (ta.numel(),), generator=gen)
        ib = torch.randint(0, tb.numel(), (tb.numel(),), generator=gen)
        reps.append(cohen_d(ta[ia].tolist(), tb[ib].tolist()))
    reps.sort()
    return cohen_d(a, b), reps[int(0.025 * nb)], reps[int(0.975 * nb)]


def report(s, ks, d, draws, chunk, seed0):
    us = units(s, ks, d, draws, chunk, seed0)
    vals = require_complete(NAME, us)
    per = {}
    for key, p in us:
        slot = per.setdefault((p["k"], p["arm"]), {})
        for kk, vv in vals[key].items():
            slot.setdefault(kk, []).extend(vv)
    g = torch.Generator().manual_seed(31337)

    print(f"\n=== P1 CHASE F2: three arms, D_FR = mean(theta). s={s} d={d} "
          f"{draws} draws/cell threads={torch.get_num_threads()} ===")
    print(f"{'k':>5} {'arm':>7} {'D_FR':>10} {'boot 95% CI':>24} {'mean ||k_c||':>13}")
    for k in ks:
        for arm in ("causal", "tail", "band"):
            v = per[(k, arm)]["theta"]
            lo, hi = boot_ci(v)
            kn = sum(per[(k, arm)]["knorm"]) / len(per[(k, arm)]["knorm"])
            print(f"{k:>5} {arm:>7} {sum(v)/len(v):>10.6f} [{lo:.6f},{hi:.6f}] {kn:>13.4f}")
    print(f"\n{'k':>5} {'comparison':>26} {'cohen d':>9} {'CI on d':>22} {'CIs disjoint?':>14}")
    for k in ks:
        for a, b, lab in (("causal", "tail", "causal vs tail-filler"),
                          ("causal", "band", "causal vs band-filler"),
                          ("band", "tail", "band-filler vs tail-filler")):
            va, vb = per[(k, a)]["theta"], per[(k, b)]["theta"]
            dd, dl, dh = dboot(va, vb, g)
            la, ha = boot_ci(va)
            lb, hb = boot_ci(vb)
            dis = la > hb or lb > ha
            print(f"{k:>5} {lab:>26} {dd:>9.4f} [{dl:+.4f},{dh:+.4f}] "
                  f"{'DISJOINT' if dis else 'OVERLAP':>14}")

    print(f"\n=== P2 CAMERON F1: aggregator. d(causal vs tail-filler) ===")
    print(f"{'k':>5} {'stat':>10} {'cohen d':>9} {'CI on d':>22}")
    for k in ks:
        for st in ("theta", "tv", "theta_max", "tv_max"):
            va, vb = per[(k, "causal")][st], per[(k, "tail")][st]
            dd, dl, dh = dboot(va, vb, g)
            print(f"{k:>5} {st:>10} {dd:>9.4f} [{dl:+.4f},{dh:+.4f}]")
        dm = cohen_d(per[(k, "causal")]["theta_max"], per[(k, "tail")]["theta_max"])
        dv = cohen_d(per[(k, "causal")]["tv_max"], per[(k, "tail")]["tv_max"])
        print(f"{k:>5} {'max delta':>10} theta_max - tv_max = {dm - dv:+.6f}"
              f"   (a monotone map commutes with max, so this must be ~0)")

    print(f"\n=== P3 gamma_1 AS A SEPARATOR (contract G-c kill) ===")
    print(f"{'k':>5} {'cohen d':>9} {'CI on d':>22} {'spans 0?':>10}")
    for k in ks:
        va, vb = per[(k, "causal")]["gamma"], per[(k, "tail")]["gamma"]
        dd, dl, dh = dboot(va, vb, g)
        print(f"{k:>5} {dd:>9.4f} [{dl:+.4f},{dh:+.4f}] "
              f"{'SPANS 0' if dl <= 0 <= dh else 'excludes 0':>10}")

    print(f"\n=== rho(||k_c||, theta) pooled over causal+tail, per k ===")
    for k in ks:
        xs = per[(k, "causal")]["knorm"] + per[(k, "tail")]["knorm"]
        ys = per[(k, "causal")]["theta"] + per[(k, "tail")]["theta"]
        tx, ty = torch.tensor(xs), torch.tensor(ys)
        r = float(((tx - tx.mean()) * (ty - ty.mean())).mean()
                  / (tx.std(unbiased=False) * ty.std(unbiased=False)))
        print(f"  k={k:<5} n={len(xs):<5} rho={r:+.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("identity", "run", "report"), default="run")
    ap.add_argument("--s", type=int, default=1024)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=420.0)
    ap.add_argument("--verify", type=int, default=1)
    a = ap.parse_args()

    if a.mode == "identity":
        print(f"theta_i == arcsin(sqrt(TV_i)) -- PROVED in the module docstring. "
              f"s={a.s} d={a.d} seeds=0..4 threads={torch.get_num_threads()}")
        for dt in (torch.float32, torch.float64):
            r = identity(a.s, a.d, range(5), dt)
            print(f"  {r['dtype']:<15} max|theta - arcsin(sqrt(TV))| over "
                  f"{r['nlive']} live rows = {r['worst_live_dev']:.6e}"
                  f"   dead rows={r['ndead']} at theta={r['dead_theta']:.7f}")
        print("  float32 -> float64 collapse is the signature of an EXACT identity "
              "read through\n  an ill-conditioned arccos near argument 1, not of an "
              "approximate relationship.")
        return 0

    us = units(a.s, a.ks, a.d, a.draws, a.chunk, a.seed)
    print(f"WILSON arms probe s={a.s} d={a.d} ks={a.ks} arms=causal/tail/band "
          f"DECLARED {a.draws} draws/cell, {len(us)} units, "
          f"threads={torch.get_num_threads()}")
    if a.mode == "report":
        return report(a.s, a.ks, a.d, a.draws, a.chunk, a.seed)
    acc = run_bucket(NAME, us, compute, budget_s=a.budget, verify=a.verify)
    print(json.dumps(acc))
    return 0 if acc["remaining"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
