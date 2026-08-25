"""Does a bounded receptive field arrest the context decay, and for whom?

Positions are pinned at fixed OFFSETS from `i` (`j = i-w/2`, `c = i-w/4`) so the
number of two-hop intermediates between `j` and `i` is a constant at every `s`.
That separates the two channels round 3 measured together: the path count, and
the softmax row normalizer summing over `s` tokens.

The published table is two invocations:

    python scale/window_sweep.py --kinds sgate --n 2048 --sizes 32 128 512 2048
    python scale/window_sweep.py --kinds deltanet paraformer softmax --n 1024 --sizes 32 128 512
"""
from __future__ import annotations

import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from ceq import bench
from scale.ratio_sweep import cp

ap = argparse.ArgumentParser()
ap.add_argument("--w", type=int, default=8)
ap.add_argument("--n", type=int, default=2048)
ap.add_argument("--hops", type=int, default=2)
ap.add_argument("--sizes", type=int, nargs="+", default=[32, 128, 512, 2048])
ap.add_argument("--kinds", nargs="+", default=["sgate"])
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
W, dev = a.w, torch.device(a.device)

print(f"window w={W}, n={a.n}, hops={a.hops}, i=s-1, j=i-w/2, c=i-w/4, {a.device}")
for kind in a.kinds:
    for window in (W, 0):
        row = []
        for s in a.sizes:
            t0 = time.time()
            r = bench.sign_flip_rate(kind, n_draws=a.n, s=s, i=s - 1,
                                     j=s - 1 - W // 2, c=s - 1 - W // 4,
                                     hops=a.hops, window=window, seed=0, device=dev)
            k = round(r * a.n)
            lo, hi = cp(k, a.n)
            row.append(f"{r:.6f}[{lo:.6f},{hi:.6f}]")
            print(f"  {kind:11s} win={window:4d} s={s:5d} {k:5d}/{a.n} "
                  f"{r:.6f} CP95=[{lo:.6f},{hi:.6f}] {time.time()-t0:.1f}s", flush=True)
        print(f"{kind:11s} window={window:4d} " + "  ".join(row), flush=True)
