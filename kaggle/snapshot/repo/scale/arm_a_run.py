"""ARM A driver — K1, K2, K3. Pure measurement. Bucketed, pinned, declared.

THREADS ARE PINNED IN THIS FILE, not left to the launcher. `m3_capability.py`'s
published numbers were reproducible only because the shell happened to carry
OMP_NUM_THREADS=2, and the same log holds the same command at 20 threads (14
runs) and 3 threads (2 runs). A probe that does not pin is a probe whose number
is a function of its launcher.

THE SLOPE IS IN k, NOT s. The claim under test is that displacement is O(1) in
the BACKGROUND SIZE because the simplex does not grow. `s` is held fixed and `k`
— the pivot count — is swept.

CARPET DISCIPLINE: `c` and `j` uniformly at random, never on a schedule.
G8: the flip half reads through the X₄ valuation instrument, which compares sign
FIELDS and never forms `lo*hi`.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

from ceq import bench                                            # noqa: E402
from scale.pivot_probe import select_pivots, pivot_hop2          # noqa: E402
from scale.valuation import valuation, v_opposite_signs          # noqa: E402
from scale.torque_probe import (rows_with_and_without, theta_rows,  # noqa: E402
                                tv_rows, torque, shadow, boot_ci, cohen_d)

JOURNAL = pathlib.Path(__file__).resolve().parents[1] / "results" / "arm_a.jsonl"


def one(s: int, k: int, d: int, g: torch.Generator, *, filler: bool):
    """One draw: the displacement half AND the flip half, same (q,k,c,i,j)."""
    x0 = torch.randn(s, d, generator=g)
    wq, wk, wo = (torch.randn(d, d, generator=g) for _ in range(3))
    v0 = torch.randn(s, d, generator=g)
    q, kk = x0 @ wq, x0 @ wk
    i = s - 1
    j = int(torch.randint(1, i, (1,), generator=g))
    piv = select_pivots(kk, min(k, s - 2), exclude=(i, j))
    pset = set(int(p) for p in piv)
    pool = sorted(pset) if not filler else [t for t in range(1, i)
                                            if t not in pset and t != j]
    if not pool:
        return None
    c = pool[int(torch.randint(0, len(pool), (1,), generator=g))]

    # --- displacement half: rows on the sphere, c present vs c masked ---
    a_c, a_0 = rows_with_and_without(q, kk, c)
    th = theta_rows(a_c, a_0)
    tv = tv_rows(a_c, a_0)
    xi, sv = shadow(a_c, a_0)
    gam = float(sv[0] - sv[1]) if sv.numel() > 1 else float("nan")

    # --- flip half: gradient sign at two values of c, X4, no multiply ---
    grads = []
    for cval in (torch.randn(d, generator=g), torch.randn(d, generator=g)):
        x = x0.clone()
        x[c] = cval
        v = v0.clone().requires_grad_(True)
        qq, k2 = x @ wq, x @ wk
        a = bench._softmax_operator(qq, k2)
        h = v + a @ v + pivot_hop2(a, piv) @ v
        h = h @ wo
        gr, = torch.autograd.grad(h[i].sum(), v, allow_unused=True)
        grads.append(0.0 if gr is None else float(gr[j].sum()))
    flip = v_opposite_signs(valuation(grads[0]), valuation(grads[1]))

    return dict(theta=float(th.mean()), tv=float(tv.mean()),
                tau=torque(a_c, a_0), gamma=gam, flip=bool(flip))


def cell(s: int, k: int, n: int, d: int, seed: int, filler: bool):
    g = torch.Generator().manual_seed(seed)
    out = []
    while len(out) < n:
        r = one(s, k, d, g, filler=filler)
        if r is not None:
            out.append(r)
    return out


def slope(xs, ys) -> float:
    lx = [math.log10(x) for x in xs]
    ly = [math.log10(max(y, 1e-12)) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    sxx = sum((x - mx) ** 2 for x in lx)
    return sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sxx if sxx else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=1024)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 32, 128, 512])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print(f"ARM A -- torque probe. s={a.s} d={a.d} draws={a.draws} ks={a.ks} "
          f"threads={torch.get_num_threads()}")
    print(f"DECLARED: {a.draws} draws per cell, not the contract's 20000. "
          f"CIs are correspondingly wider and every one is printed.\n")

    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    print(f"{'k':>5} {'D_FR causal':>26} {'D_FR filler':>26} {'flip':>9} "
          f"{'||tau||':>9} {'gamma_1':>9}")
    for k in a.ks:
        t0 = time.time()
        cs = cell(a.s, k, a.draws, a.d, a.seed, False)
        fs = cell(a.s, k, a.draws, a.d, a.seed + 999, True)
        th_c = [r["theta"] for r in cs]
        th_f = [r["theta"] for r in fs]
        tv_c = [r["tv"] for r in cs]
        tv_f = [r["tv"] for r in fs]
        fl = sum(1 for r in cs if r["flip"]) / len(cs)
        lo_c, hi_c = boot_ci(th_c)
        lo_f, hi_f = boot_ci(th_f)
        rows[k] = dict(theta_c=sum(th_c) / len(th_c), theta_f=sum(th_f) / len(th_f),
                       ci_c=(lo_c, hi_c), ci_f=(lo_f, hi_f), flip=fl,
                       d_theta=cohen_d(th_c, th_f), d_tv=cohen_d(tv_c, tv_f),
                       tau=sum(r["tau"] for r in cs) / len(cs),
                       gamma=sum(r["gamma"] for r in cs) / len(cs),
                       secs=time.time() - t0)
        r = rows[k]
        print(f"{k:>5} {r['theta_c']:.6f} [{lo_c:.4f},{hi_c:.4f}] "
              f"{r['theta_f']:.6f} [{lo_f:.4f},{hi_f:.4f}] "
              f"{fl:>9.5f} {r['tau']:>9.4f} {r['gamma']:>9.4f}", flush=True)
        with JOURNAL.open("a") as fh:
            fh.write(json.dumps({"s": a.s, "k": k, "draws": a.draws,
                                 **{kk: vv for kk, vv in r.items()}}) + "\n")

    ks = list(rows)
    print("\n=== K1 DUAL SLOPE ===")
    fs_ = slope(ks, [rows[k]["flip"] for k in ks])
    ds_ = slope(ks, [rows[k]["theta_c"] for k in ks])
    print(f"  flip slope in k = {fs_:+.4f}   bar <= -0.4   {'OK' if fs_ <= -0.4 else 'NOT MET'}")
    print(f"  D_FR slope in k = {ds_:+.4f}   bar >= -0.1   {'OK' if ds_ >= -0.1 else 'NOT MET'}")
    if ds_ < -0.3:
        print("  D_FR slope < -0.3 -> DISPLACEMENT DIES WITH FLIP -> 'no leap'")

    print("\n=== K2 FILLER TWIN (disjoint CIs required) ===")
    for k in ks:
        r = rows[k]
        dis = r["ci_c"][0] > r["ci_f"][1] or r["ci_f"][0] > r["ci_c"][1]
        print(f"  k={k:<5} causal [{r['ci_c'][0]:.4f},{r['ci_c'][1]:.4f}] vs "
              f"filler [{r['ci_f'][0]:.4f},{r['ci_f'][1]:.4f}]  "
              f"{'DISJOINT' if dis else 'OVERLAP -> VOIDS THE TABLE'}")

    print("\n=== K3 GEOMETRY MUST EARN ITSELF (|d_theta| > |d_TV|) ===")
    for k in ks:
        r = rows[k]
        print(f"  k={k:<5} d_theta={r['d_theta']:+.4f}  d_TV={r['d_tv']:+.4f}  "
              f"{'theta WINS' if abs(r['d_theta']) > abs(r['d_tv']) else 'TV WINS -> sphere is notation'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
