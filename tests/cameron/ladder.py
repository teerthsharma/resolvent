"""One CLI so every ladder cell is produced by the SAME code path.

    python tests/cameron/ladder.py --arm dil --s 512 --n 256 --seed 0

Arms, all at the same geometry `i = s-1`, `j = i - s//2`, `c = j + (s//2)//2`:

    dil     dilated band, window 8, doubling dilations, depth = log_schedule(s)
    contig  contiguous band, window 8, depth = len(log_schedule(s))
    glob_L  window 0, depth = len(log_schedule(s))     -- depth-matched control
    glob_1  window 0, depth 1                          -- the arm M2 killed

Prints RAW COUNTS (`k/n`) as well as rates, because a rate alone cannot carry a
confidence interval and this project has already been burned once by a `0.00000`
that rested on 60 draws.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch

import dilated as D
from ceq import bench

ARMS = ("dil", "contig", "glob_L", "glob_1")


def geometry(s: int):
    i = s - 1
    dist = s // 2
    j = i - dist
    return i, j, j + dist // 2, dist


def run(arm: str, s: int, n: int, seed: int, d: int = 16, window: int = 8,
        hops: int = 2, device=None):
    dev = device or torch.device("cpu")
    i, j, c, dist = geometry(s)
    sch = D.log_schedule(s, window=window, hops=hops)
    kw = dict(n_draws=n, s=s, d=d, i=i, j=j, c=c, hops=hops, seed=seed,
              device=dev)
    if arm == "dil":
        draws = D.composed_draws(dilations=sch, window=window, **kw)
    elif arm == "contig":
        draws = D.composed_draws(dilations=[1] * len(sch), window=window, **kw)
    elif arm == "glob_L":
        draws = bench.sign_flip_draws("sgate", window=0, depth=len(sch), **kw)
    elif arm == "glob_1":
        draws = bench.sign_flip_draws("sgate", window=0, depth=1, **kw)
    else:
        raise ValueError(arm)
    return draws, sch, dist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--s", type=int, required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--hops", type=int, default=2)
    a = ap.parse_args()

    t0 = time.time()
    draws, sch, dist = run(a.arm, a.s, a.n, a.seed, d=a.d, window=a.window,
                           hops=a.hops)
    reach = D.reach_fraction(draws)
    f0 = bench.flip_rate(draws, floor=0.0)
    fr = bench.flip_rate(draws, floor=0.0, rel=1e-6)
    fa = bench.flip_rate(draws, floor=1e-6)
    print(f"arm={a.arm} s={a.s} dist={dist} depth={len(sch)} sch={sch} "
          f"seed={a.seed} n={a.n} "
          f"reach_k={round(reach * a.n)}/{a.n} reach={reach:.6f} "
          f"flip0_k={round(f0 * a.n)}/{a.n} flip0={f0:.6f} "
          f"fliprel_k={round(fr * a.n)}/{a.n} fliprel={fr:.6f} "
          f"flipabs_k={round(fa * a.n)}/{a.n} flipabs={fa:.6f} "
          f"secs={time.time() - t0:.1f}", flush=True)
