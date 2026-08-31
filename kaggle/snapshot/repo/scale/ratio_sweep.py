"""Sweep the sgate-vs-ParaFormer sign-flip separation across context length.

Both rates are binomial. Every point is reported with an exact Clopper-Pearson
95% interval, and the ratio is reported as the CONSERVATIVE bound
`lo(sgate) / hi(paraformer)` -- the smallest separation consistent with the
draws at 95%.
"""
from __future__ import annotations

import argparse, json, sys, time
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from scipy.stats import beta

from ceq import bench


def cp(k: int, n: int, conf: float = 0.95):
    """Exact Clopper-Pearson interval for k successes in n draws."""
    a = (1.0 - conf) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def run(kinds, sizes, n_draws, hops, device, seed=0, **kw):
    out = {}
    for s in sizes:
        pos = dict(s=s, i=s - 1, j=s // 4, c=s // 2)
        for kind in kinds:
            t0 = time.time()
            r = bench.sign_flip_rate(kind, n_draws=n_draws, hops=hops, seed=seed,
                                     device=device, **pos, **kw)
            k = round(r * n_draws)
            lo, hi = cp(k, n_draws)
            out[(kind, s)] = dict(k=k, n=n_draws, rate=r, lo=lo, hi=hi,
                                  secs=round(time.time() - t0, 1))
            print(f"{kind:11s} s={s:5d} {k:5d}/{n_draws} rate={r:.6f} "
                  f"CP95=[{lo:.6f},{hi:.6f}] {out[(kind,s)]['secs']}s", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--kinds", nargs="+", default=["sgate", "paraformer", "softmax"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.kinds, a.sizes, a.n, a.hops, torch.device(a.device))
    print("\ns    " + "  ".join(f"{k}" for k in a.kinds) + "   ratio_lo_over_hi")
    for s in a.sizes:
        row = [res[(k, s)] for k in a.kinds]
        rat = (res[("sgate", s)]["lo"] / res[("paraformer", s)]["hi"]
               if ("sgate", s) in res and ("paraformer", s) in res else float("nan"))
        print(f"{s:5d} " + "  ".join(f"{r['rate']:.6f}" for r in row) + f"   {rat:.2f}")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(
            {f"{k}|{s}": v for (k, s), v in res.items()}, indent=1))
