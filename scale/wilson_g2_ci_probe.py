"""WILSON -- G2 event: the D_FR slope bootstrap interval that moved.

THE FACT UNDER INVESTIGATION

    re-taken   D_FR slope in k = -0.4137  [-0.4573, -0.3712]
    published  D_FR slope in k = -0.4137  [-0.4579, -0.3704]

Point estimate identical; endpoints moved +0.0006 / -0.0008 at a fixed
`torch.Generator().manual_seed(4242)`, B=2000, 6/6 k, 2000/2000 usable reps.

This file does not argue. It enumerates every way the second interval could have
been produced from the same journal and prints what each one yields, so that the
published pair either appears in the table or is shown not to appear.

WHAT IS VARIED

  1. GENERATOR SHARING. `scale/arm_a_k1.py:338-346` seeds ONE generator and
     passes it to `bslope` twice: the flip bootstrap first, then D_FR. The D_FR
     interval therefore depends on the flip bootstrap having already consumed
     B * sum_k(n_k) = 2000 * 2400 = 4,800,000 int64 draws from that stream. If
     the published run drew D_FR from a stream not yet advanced by flip -- i.e.
     the flip call did not exist, or ran after -- the interval differs while the
     point estimate (a plain mean, not a resample) does not. That is exactly the
     observed signature.

  2. THREAD COUNT. `arm_a_k1.py` pins `torch.set_num_threads(2)` at import, so a
     launcher cannot move it -- but this repo has been burned twice by thread
     count and once by a sweep that skipped 3, so the sweep here includes 3 and
     every other odd value up to 9 and is run by setting the count AFTER import.

  3. PERCENTILE INDEXING. `bslope` returns `reps[int(0.025*len(reps))]` and
     `reps[int(0.975*len(reps))]`. Neighbouring index conventions are printed so
     an off-by-one at publication time is visible rather than assumed away.

  4. B. The declared B=2000 is checked against its neighbours.

Threads are reported with every number. Data is read from
`results/arm_a_k1.jsonl` through the shipped `units`/`require_complete` path,
never re-simulated, so every row below is a function of the resampler alone.

    python scale/wilson_g2_ci_probe.py
"""
from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scale.arm_a_k1 import NAME, bslope, units          # noqa: E402
from scale.bucket import require_complete               # noqa: E402
from scale.pivot_probe import loglog_slope              # noqa: E402

S, D, DRAWS, CHUNK, LAM, SEED0 = 1024, 16, 400, 100, 0.10, 0
KS = [8, 16, 32, 64, 128, 256]
PUBLISHED = (-0.4579, -0.3704)
RETAKEN = (-0.4573, -0.3712)


def load():
    us = units(S, KS, D, DRAWS, CHUNK, LAM, SEED0)
    vals = require_complete(NAME, us)
    per = {}
    for key, p in us:
        slot = per.setdefault((p["k"], p["filler"]), {})
        for kk, vv in vals[key].items():
            slot.setdefault(kk, []).extend(vv)
    flip = {k: [float(x) for x in per[(k, False)]["flip"]] for k in KS}
    theta = {k: per[(k, False)]["theta"] for k in KS}
    return flip, theta


def hit(lo, hi, target, tol=5e-5):
    return abs(lo - target[0]) < tol and abs(hi - target[1]) < tol


def label(lo, hi):
    if hit(lo, hi, PUBLISHED):
        return "*** MATCHES PUBLISHED ***"
    if hit(lo, hi, RETAKEN):
        return "matches re-take"
    return ""


def reps_of(per_k, gen, b):
    """`bslope`'s replicate list, kept so percentile conventions can be varied."""
    import math
    ks = sorted(per_k)
    tens = {k: torch.tensor([float(x) for x in per_k[k]]) for k in ks}
    reps = []
    for _ in range(b):
        ys = []
        for k in ks:
            t = tens[k]
            idx = torch.randint(0, t.numel(), (t.numel(),), generator=gen)
            ys.append(float(t[idx].mean()))
        sl, _n = loglog_slope(ks, ys)
        if not math.isnan(sl):
            reps.append(sl)
    reps.sort()
    return reps


def main() -> int:
    flip, theta = load()
    n = {k: len(theta[k]) for k in KS}
    print(f"journal: results/{NAME}.jsonl   n per k = {n}")
    print(f"published {PUBLISHED}   re-taken {RETAKEN}\n")

    print("=== 1. GENERATOR SHARING (threads=2, the file's pin, B=2000) ===")
    torch.set_num_threads(2)
    cases = []

    g = torch.Generator().manual_seed(4242)
    bslope(flip, g, 2000)
    p, nn, lo, hi, u, _ = bslope(theta, g, 2000)
    cases.append(("shipped: flip first, SHARED generator", p, lo, hi, u))

    g = torch.Generator().manual_seed(4242)
    p, nn, lo, hi, u, _ = bslope(theta, g, 2000)
    cases.append(("D_FR FIRST on a fresh seed 4242 stream", p, lo, hi, u))

    g = torch.Generator().manual_seed(4242)
    p, nn, lo, hi, u, _ = bslope(theta, g, 2000)
    bslope(flip, g, 2000)
    cases.append(("D_FR first, flip after (order swapped)", p, lo, hi, u))

    for nm, p, lo, hi, u in cases:
        print(f"  {nm:<44} {p:+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"{u}/2000  {label(lo, hi)}")

    print("\n=== 2. THREAD SWEEP, shipped call order, seed 4242, B=2000 ===")
    print("    (odd values included: this repo's round-4 sweep missed 3)")
    for t in (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 24):
        torch.set_num_threads(t)
        g = torch.Generator().manual_seed(4242)
        bslope(flip, g, 2000)
        p, nn, lo, hi, u, _ = bslope(theta, g, 2000)
        print(f"  threads={t:<3} {p:+.4f} [{lo:+.4f},{hi:+.4f}]  "
              f"{u}/2000  {label(lo, hi)}")

    torch.set_num_threads(2)
    print("\n=== 3. PERCENTILE CONVENTION (threads=2, shipped order, B=2000) ===")
    g = torch.Generator().manual_seed(4242)
    bslope(flip, g, 2000)
    reps = reps_of(theta, g, 2000)
    m = len(reps)
    conv = {
        "shipped  reps[int(.025*m)], reps[int(.975*m)]":
            (reps[int(0.025 * m)], reps[int(0.975 * m)]),
        "floor-1  reps[int(.025*m)-1], reps[int(.975*m)-1]":
            (reps[int(0.025 * m) - 1], reps[int(0.975 * m) - 1]),
        "ceil     reps[ceil(.025*m)], reps[ceil(.975*m)]":
            (reps[-(-int(0.025 * m * 1000) // 1000)], reps[min(m - 1, int(0.975 * m) + 1)]),
        "R type-7 linear interpolation":
            (_q(reps, 0.025), _q(reps, 0.975)),
    }
    for nm, (lo, hi) in conv.items():
        print(f"  {nm:<50} [{lo:+.4f},{hi:+.4f}]  {label(lo, hi)}")

    print("\n=== 4. B SWEEP (threads=2, shipped order) ===")
    for b in (500, 1000, 1500, 2000, 2500, 4000, 5000, 10000):
        g = torch.Generator().manual_seed(4242)
        bslope(flip, g, b)
        p, nn, lo, hi, u, _ = bslope(theta, g, b)
        print(f"  B={b:<6} {p:+.4f} [{lo:+.4f},{hi:+.4f}]  {u}/{b}  {label(lo, hi)}")

    print("\n=== 5. DETERMINISM: shipped call repeated 3x in one process ===")
    for i in range(3):
        g = torch.Generator().manual_seed(4242)
        bslope(flip, g, 2000)
        p, nn, lo, hi, u, _ = bslope(theta, g, 2000)
        print(f"  run {i}   {p:+.16f} [{lo:+.16f},{hi:+.16f}]")

    print(f"\ntorch {torch.__version__}   threads now {torch.get_num_threads()}")
    return 0


def _q(sorted_reps, q):
    """R type-7 / numpy default linear interpolation, for comparison only."""
    m = len(sorted_reps)
    h = (m - 1) * q
    lo = int(h)
    hi = min(lo + 1, m - 1)
    return sorted_reps[lo] + (h - lo) * (sorted_reps[hi] - sorted_reps[lo])


if __name__ == "__main__":
    raise SystemExit(main())
