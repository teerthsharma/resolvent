"""CHASE — K3 with the error bar the criterion never had.

WHAT K3 SAYS. `scale/arm_a_run.py:161`:

    'theta WINS' if abs(r['d_theta']) > abs(r['d_tv']) else 'TV WINS'

That is a comparison of two POINT ESTIMATES with no interval on either. The
published margins are 2.3% / 2.1% / 4.7% of a Cohen's d computed on **120 draws
per cell** (`results/arm_a.jsonl`, `"draws": 120`), against a contract asking for
20000. G6 of the round-5 contract reads: *no comparative sentence without a
bootstrap CI excluding zero.* K3 is a comparative sentence. It has no CI.

WHAT THIS FILE DOES.
  1. REPRODUCES the published cells FIRST by calling `scale.arm_a_run.cell`
     unchanged, and pins theta_c / theta_f / d_theta / d_tv at abs=5e-7. If a
     published number has moved, that is a G2 event and this file says so before
     it says anything else.
  2. Bootstraps the PAIRED quantity the criterion actually asserts:

         delta_k = |d_theta| - |d_TV|      on the SAME resampled draws

     Resampling indices are shared between the two statistics because theta and
     TV are measured on identical draws — that is the paired form, and it is the
     TIGHTEST honest interval available. An unpaired interval would be wider, so
     a straddle here is not an artifact of a loose method.
  3. Ships two must-fire controls, both SEEN to fire in the output:
       C1 a synthetic cell where theta genuinely dominates TV must return a CI
          that EXCLUDES zero — proves a straddle is informative, not structural;
       C2 theta and TV set literally equal must return delta == 0 at every
          resample — proves the pairing is real and not two independent draws.

EXIT CODE. 1 when any k's delta CI contains zero (K3's margin is not
established at that k), or when a published number moves. 0 only when every k
clears both.

Threads pinned in this file. `python scale/chase_k3_ci.py`.
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)

from scale.arm_a_run import cell                          # noqa: E402
from scale.torque_probe import cohen_d                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "results" / "arm_a.jsonl"
PIN = 5e-7
B = 2000


def paired_delta_ci(th_c, th_f, tv_c, tv_f, seed=0, b=B):
    """95% percentile CI of |d_theta| - |d_TV| under a shared resample.

    The SAME bootstrap indices index both statistics, because both were measured
    on the same draws. `delta` is therefore a within-draw contrast.
    """
    g = torch.Generator().manual_seed(seed)
    nc, nf = len(th_c), len(th_f)
    out = []
    for _ in range(b):
        ic = torch.randint(0, nc, (nc,), generator=g).tolist()
        if_ = torch.randint(0, nf, (nf,), generator=g).tolist()
        a_th = [th_c[i] for i in ic]
        b_th = [th_f[i] for i in if_]
        a_tv = [tv_c[i] for i in ic]
        b_tv = [tv_f[i] for i in if_]
        out.append(abs(cohen_d(a_th, b_th)) - abs(cohen_d(a_tv, b_tv)))
    out.sort()
    return out[int(0.025 * b)], out[int(0.975 * b)]


def controls() -> bool:
    """Both must FIRE and be seen firing. Returns True iff both fired."""
    print("=== MUST-FIRE CONTROLS (a checker with an unfired control is not a checker) ===")
    g = torch.Generator().manual_seed(7)
    n = 120
    # C1: theta separates hard, TV does not. The test MUST see it.
    th_c = (torch.randn(n, generator=g) * 0.1 + 3.0).tolist()
    th_f = (torch.randn(n, generator=g) * 0.1 + 0.0).tolist()
    tv_c = (torch.randn(n, generator=g) * 1.0 + 0.05).tolist()
    tv_f = (torch.randn(n, generator=g) * 1.0 + 0.00).tolist()
    lo1, hi1 = paired_delta_ci(th_c, th_f, tv_c, tv_f, seed=11)
    c1 = lo1 > 0.0
    print(f"  C1 theta dominates TV by construction -> delta CI "
          f"[{lo1:+.4f},{hi1:+.4f}]  {'FIRED (excludes zero)' if c1 else 'DID NOT FIRE'}")
    # C2: theta and TV identical arrays -> the paired contrast must be exactly 0.
    lo2, hi2 = paired_delta_ci(th_c, th_f, th_c, th_f, seed=11)
    c2 = (lo2 == 0.0 and hi2 == 0.0)
    print(f"  C2 theta and TV set identical      -> delta CI "
          f"[{lo2:+.9f},{hi2:+.9f}]  "
          f"{'FIRED (exactly zero, pairing is real)' if c2 else 'DID NOT FIRE'}")
    print()
    return c1 and c2


def main() -> int:
    pub = {}
    with JOURNAL.open() as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                pub[r["k"]] = r

    if not controls():
        print("CONTROLS DID NOT FIRE -> this instrument proves nothing. STOP.")
        return 1

    print(f"=== K3 WITH ITS INTERVAL === threads={torch.get_num_threads()} "
          f"s=1024 d=16 seed=0/999 B={B}")
    print("reproducing the published cells FIRST, pinned abs=5e-7\n")

    moved, straddle = [], []
    print(f"{'k':>5} {'n':>5} {'d_theta':>10} {'d_TV':>10} {'delta':>10} "
          f"{'95% CI of delta':>24} {'verdict':>28}")
    for k in sorted(pub):
        p = pub[k]
        n = p["draws"]
        cs = cell(1024, k, n, 16, 0, False)
        fs = cell(1024, k, n, 16, 999, True)
        th_c = [r["theta"] for r in cs]
        th_f = [r["theta"] for r in fs]
        tv_c = [r["tv"] for r in cs]
        tv_f = [r["tv"] for r in fs]

        rep = {"theta_c": sum(th_c) / len(th_c), "theta_f": sum(th_f) / len(th_f),
               "d_theta": cohen_d(th_c, th_f), "d_tv": cohen_d(tv_c, tv_f)}
        for field, got in rep.items():
            want = p[field]
            if abs(got - want) >= PIN:
                moved.append((k, field, want, got))
                print(f"  G2: k={k} {field} published {want!r} -> reread {got!r} "
                      f"delta {got - want:+.3e}")

        d_th, d_tv = rep["d_theta"], rep["d_tv"]
        delta = abs(d_th) - abs(d_tv)
        lo, hi = paired_delta_ci(th_c, th_f, tv_c, tv_f, seed=k)
        contains_zero = lo <= 0.0 <= hi
        if contains_zero:
            straddle.append(k)
        v = ("STRADDLES ZERO -> not established"
             if contains_zero else "excludes zero -> established")
        print(f"{k:>5} {n:>5} {d_th:>+10.4f} {d_tv:>+10.4f} {delta:>+10.4f} "
              f"[{lo:>+9.4f},{hi:>+9.4f}] {v:>28}", flush=True)

    print()
    if moved:
        print(f"G2 FIRES: {len(moved)} published number(s) moved. Everything above is VOID.")
        return 1
    print("G2: 4/4 pinned fields per cell reproduce at abs=5e-7. No published number moved.")
    if straddle:
        print(f"\nK3 IS NOT ESTABLISHED AT k={straddle}. The criterion "
              f"`abs(d_theta) > abs(d_tv)` is a point-estimate comparison; with the "
              f"interval the contract's G6 requires, theta's margin over TV contains "
              f"zero at every k listed. The sphere has not earned itself on this data.")
        return 1
    print("\nK3 margin excludes zero at every k.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
