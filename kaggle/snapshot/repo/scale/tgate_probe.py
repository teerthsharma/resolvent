"""Ladder step 1: does the denominator-free operator hold its property in context?

THE MECHANISM CLAIM BEING TESTED. The measured cause of the 1/s death was never
signedness, hop count, or the normalizer's TYPE -- it was ANY denominator that
sums over context. Evidence: sgate on a global row 0.0186 -> 0.0000 over
s=32->512 at fixed path count; DeltaNet, denominator-free, 0.074 -> 0.065, flat.

If that mechanism theory is right, `tgate` -- unnormalized, signed, QUERY-
dependent, strictly causal, multi-hop -- is near-flat like DeltaNet. If it decays
anyway, the mechanism theory is WRONG, and that is a finding worth more than the
operator was.

KILL NUMBERS, FIXED BEFORE RUNNING (not moved afterwards):
  * log-log slope steeper than -0.3
  * or the DeltaNet arm non-overlapping ABOVE tgate at s >= 128

CONTROLS, because eight instruments in this project have been internally
consistent and externally wrong:
  * `softmax` must read ~0 everywhere      -- if it reads high the probe is broken
  * `sgate` must reproduce its -1.389 decay -- the known-positive for decay
  * `deltanet` must be flat                 -- the known-negative for decay
  * calibration asserted before any new number is printed
"""
from __future__ import annotations
import sys, math, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ceq import bench


def cp(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def calibrate():
    exp = {("signed", 3): 0.046875, ("sgate", 2): 0.1640625,
           ("sgate", 1): 0.0234375, ("softmax", 3): 0.0}
    for (kind, hops), want in exp.items():
        got = bench.sign_flip_rate(kind, n_draws=128, s=8, hops=hops)
        if got != want:
            raise SystemExit(f"INSTRUMENT CONTAMINATED: {kind} hops={hops} "
                             f"{got} != {want}. Every published number is void.")
    print("calibration OK (4/4 published numbers bit-identical)\n")


def slope(ss, rates):
    """log-log slope over the points with a NONZERO rate.

    Zeros are reported, never clamped to an epsilon: clamping turns 'the property
    is gone' into a finite slope and hides exactly the death being looked for.
    """
    pts = [(math.log(s), math.log(r)) for s, r in zip(ss, rates) if r > 0]
    if len(pts) < 2:
        return float("nan"), len(pts)
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    num = sum((a - mx) * (b - my) for a, b in pts)
    den = sum((a - mx) ** 2 for a, _ in pts)
    return (num / den if den else float("nan")), len(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+",
                    default=["tgate", "tgatex", "deltanet", "sgate", "softmax"])
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--n-tail", type=int, default=2048)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--i", type=int, default=7)   # published geometry, fixed
    ap.add_argument("--j", type=int, default=1)   # across every s, so the path
    ap.add_argument("--c", type=int, default=4)   # count does not move with s
    a = ap.parse_args()

    calibrate()
    print(f"sign_flip_rate, hops={a.hops}, tau={a.tau}, window={a.window or 'global'}")
    print(f"n={a.n} draws (n={a.n_tail} at s>=256). Clopper-Pearson 95%.")
    print(f"positions FIXED at i={a.i} j={a.j} c={a.c} for every s (fixed path count).")
    print(f"KILL: slope steeper than -0.3, or deltanet non-overlapping ABOVE "
          f"tgate at s>=128.\n")
    print(f"{'kind':>9} " + "".join(f"{s:>21}" for s in a.sizes) + f"{'slope':>9}")
    res = {}
    for kind in a.kinds:
        rates, cells, cis = [], [], []
        for s in a.sizes:
            n = a.n_tail if s >= 256 else a.n
            # POSITIONS HELD FIXED. This is what "at fixed path count" means in
            # the published protocol, and deviating from it is how the first
            # smoke run made DeltaNet read slope -1.000 against its own recorded
            # flat 0.074 -> 0.065. With i = s-1, j = 1 the j->i distance GROWS
            # with s while `hops` stays at 3, so the measurement conflates
            # context dilution with "the distance outran the hop budget". The
            # control arm caught it; that is what the control arm is for.
            r = bench.sign_flip_rate(kind, n_draws=n, s=s, hops=a.hops,
                                     i=a.i, j=a.j, c=a.c, tau=a.tau,
                                     window=a.window)
            k = int(round(r * n))
            lo, hi = cp(k, n)
            rates.append(r); cis.append((lo, hi))
            cells.append(f"{r:.4f}[{lo:.3f},{hi:.3f}]")
        sl, npts = slope(a.sizes, rates)
        res[kind] = (rates, cis, sl)
        note = "" if npts == len(a.sizes) else f" ({npts}/{len(a.sizes)} nonzero)"
        print(f"{kind:>9} " + "".join(f"{c:>21}" for c in cells)
              + f"{sl:>+9.3f}{note}")

    print("\n--- VERDICT against the pre-registered kills ---")
    for kind in ("tgate", "tgatex"):
        if kind not in res:
            continue
        rates, cis, sl = res[kind]
        k1 = (sl < -0.3) if sl == sl else True
        print(f"  {kind}: slope {sl:+.3f}  -> "
              + ("KILL (steeper than -0.3)" if k1 else "survives slope test"))
        if "deltanet" in res:
            dr, dci, _ = res["deltanet"]
            for s, r, ci, rr, dc in zip(a.sizes, rates, cis, dr, dci):
                if s < 128:
                    continue
                above = dc[0] > ci[1]
                if above:
                    print(f"    s={s}: KILL -- deltanet {rr:.4f}{dc} sits ABOVE "
                          f"tgate {r:.4f}{ci}, non-overlapping")
            if not any(d[0] > c[1] for s, c, d in zip(a.sizes, cis, dci) if s >= 128):
                print(f"    deltanet never non-overlapping above it at s>=128 "
                      f"-> survives the reference test")


if __name__ == "__main__":
    main()
