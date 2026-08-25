"""CHASE — the D_FR slope's interval, which the record reads a verdict out of without.

THE RECORD, as it stands:

  CHECKLIST.md:417  "`D_FR` slope **-0.3061** misses the -0.3 line by **0.006**
                     on a 3-point fit. **Untested, not failed.**"
  DONE.md:234       "`D_FR` slope in k reads **-0.3061** against the *"no leap"*
                     line at **-0.3**. It misses by **0.006**"

THE CONTRACT, `LOOP_PROMPT.md:152-154`:

  K1 DUAL SLOPE: `flip(s)` slope <= -0.4 on X4 AND `D_FR` slope >= -0.1 on the
  SAME draws. **If `D_FR` slope < -0.3, displacement dies with flip and the
  answer was "no leap."**

TWO THINGS THE RECORD DOES NOT SAY.

  1. The "no leap" clause is conditioned on `D_FR slope < -0.3` ALONE. It carries
     no dependence on the flip half. The flip half being vacuous on an unsigned
     arm makes the CONJUNCTION unevaluable; it does not make this clause
     unevaluable. And -0.3061 < -0.3. The trigger condition is arithmetically
     SATISFIED. "misses the line by 0.006" describes a number that has CROSSED
     the line, from the far side.
  2. K1's own bar is `>= -0.1`. Against that bar -0.3061 misses by **0.206**,
     not by 0.006. The 0.006 figure is the distance to the *other* line.

WHAT IS ACTUALLY MISSING is an interval, and `STATE.md:29-31` already says so:
"read the `D_FR` slope's CI before its point estimate ... Iteration 4 read
-0.3061 with no error bar and that is the entire reason K1 was voided." The CI
still does not exist. This file computes it.

PRE-REGISTERED READING, fixed before the bootstrap runs, three outcomes:

  (i)   CI entirely >= -0.3   the clause did NOT fire; "untested, not failed"
                              is supported and this file exits 0.
  (ii)  CI entirely <  -0.3   the clause FIRED. "no leap" is the answer.
  (iii) CI straddles -0.3     unresolved at 120 draws. The point estimate sits
                              INSIDE the kill region, so the record may not say
                              "misses the line" — it must say "unresolved, point
                              estimate on the kill side."

Exits 1 on (ii) or (iii): under both, the sentence currently shipped is wrong.

METHOD. The bootstrap resamples DRAWS WITHIN EACH k CELL, recomputes the three
cell means, and refits `arm_a_run.slope` — the shipped fitting function, imported
unchanged, not a reimplementation. That is the interval on the quantity actually
published. It does NOT model uncertainty from having only three k values; with 3
points and 2 degrees of freedom that component is not estimable at all, which is
itself part of the reading.

REPLAY FIDELITY, DECLARED. `arm_a_run.one` consumes the generator in the order
x0, wq, wk, wo, v0, j, c, then TWO `randn(d)` inside its gradient loop. This file
replays that stream EXACTLY — including drawing and discarding `wo`, `v0` and the
two gradient vectors — and computes only the theta/TV half, skipping the autograd
the slope does not depend on. Fidelity is not asserted, it is CHECKED: the four
published fields per cell must reproduce at abs=5e-7 before any interval is
printed. If they do not, the replay is wrong or a published number moved, and
either way nothing below it is reported.

Threads pinned in this file. `python scale/chase_slope_ci.py`
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)

from scale.arm_a_run import slope                              # noqa: E402
from scale.pivot_probe import select_pivots                    # noqa: E402
from scale.torque_probe import (rows_with_and_without, theta_rows,  # noqa: E402
                                tv_rows, cohen_d)

ROOT = pathlib.Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "results" / "arm_a.jsonl"
PIN = 5e-7
B = 2000
NO_LEAP = -0.3
K1_BAR = -0.1


def replay_one(s: int, k: int, d: int, g: torch.Generator, *, filler: bool):
    """`arm_a_run.one`'s generator stream, theta/TV half only. Bitwise faithful."""
    x0 = torch.randn(s, d, generator=g)
    wq, wk, _wo = (torch.randn(d, d, generator=g) for _ in range(3))
    _v0 = torch.randn(s, d, generator=g)
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
    a_c, a_0 = rows_with_and_without(q, kk, c)
    th = float(theta_rows(a_c, a_0).mean())
    tv = float(tv_rows(a_c, a_0).mean())
    # the gradient half consumed exactly two randn(d) from the same generator
    for _ in range(2):
        torch.randn(d, generator=g)
    return th, tv


def replay_cell(s, k, n, d, seed, filler):
    g = torch.Generator().manual_seed(seed)
    out = []
    while len(out) < n:
        r = replay_one(s, k, d, g, filler=filler)
        if r is not None:
            out.append(r)
    return [a for a, _ in out], [b for _, b in out]


def control() -> bool:
    """MUST FIRE: a synthetic cell set with a known steep slope must be resolved
    well clear of -0.3, or this bootstrap cannot resolve anything."""
    print("=== MUST-FIRE CONTROL ===")
    g = torch.Generator().manual_seed(3)
    ks = [8, 32, 128]
    cells = {k: (torch.randn(120, generator=g) * 0.001 + (1.0 / k)).tolist() for k in ks}
    pt = slope(ks, [sum(v) / len(v) for v in cells.values()])
    lo, hi = boot_slope(ks, cells, seed=3)
    fired = hi < NO_LEAP
    print(f"  planted slope ~ -1.0 (y = 1/k): point {pt:+.4f}  CI [{lo:+.4f},{hi:+.4f}]  "
          f"{'FIRED (resolved strictly below -0.3)' if fired else 'DID NOT FIRE'}")
    print()
    return fired


def boot_slope(ks, cells: dict, seed=0, b=B):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(b):
        means = []
        for k in ks:
            v = cells[k]
            idx = torch.randint(0, len(v), (len(v),), generator=g).tolist()
            means.append(sum(v[i] for i in idx) / len(idx))
        out.append(slope(ks, means))
    out.sort()
    return out[int(0.025 * b)], out[int(0.975 * b)]


def main() -> int:
    pub = {}
    with JOURNAL.open() as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                pub[r["k"]] = r
    ks = sorted(pub)

    if not control():
        print("CONTROL DID NOT FIRE -> this bootstrap resolves nothing. STOP.")
        return 1

    print(f"=== REPLAY FIDELITY / G2 === threads={torch.get_num_threads()} "
          f"s=1024 d=16 seed=0/999, 4 fields x {len(ks)} cells pinned abs={PIN}")
    cells, moved = {}, []
    for k in ks:
        p = pub[k]
        n = p["draws"]
        th_c, tv_c = replay_cell(1024, k, n, 16, 0, False)
        th_f, tv_f = replay_cell(1024, k, n, 16, 999, True)
        cells[k] = th_c
        got = {"theta_c": sum(th_c) / len(th_c), "theta_f": sum(th_f) / len(th_f),
               "d_theta": cohen_d(th_c, th_f), "d_tv": cohen_d(tv_c, tv_f)}
        for f, v in got.items():
            dl = v - p[f]
            flag = "OK" if abs(dl) < PIN else "MOVED"
            if abs(dl) >= PIN:
                moved.append((k, f, p[f], v))
            print(f"  k={k:<4} {f:<9} published {p[f]:.15f}  replay {v:.15f}  "
                  f"delta {dl:+.3e}  {flag}")
    print()
    if moved:
        print(f"G2 FIRES: {len(moved)} field(s) did not reproduce. "
              f"Either the replay is wrong or a published number moved. "
              f"Nothing below is reported.")
        return 1
    print(f"G2: {4 * len(ks)}/{4 * len(ks)} published fields reproduce at abs={PIN}. "
          f"The replay is faithful and no published number moved.\n")

    pt = slope(ks, [pub[k]["theta_c"] for k in ks])
    lo, hi = boot_slope(ks, cells, seed=0)
    print("=== THE D_FR SLOPE, WITH ITS INTERVAL ===")
    print(f"  point estimate      {pt:+.4f}   (published -0.3061, delta {pt + 0.3061:+.3e})")
    print(f"  95% CI over draws   [{lo:+.4f}, {hi:+.4f}]   B={B}, resample within cell, refit")
    print(f"  k values fitted     {ks}  -- THREE points, 2 df; the uncertainty from")
    print(f"                      having only three k values is NOT in this interval.")
    print()
    print(f"  vs K1's bar          >= {K1_BAR}    "
          f"{'CLEARS' if lo >= K1_BAR else 'FAILS'} -- entire CI is "
          f"{'above' if lo >= K1_BAR else 'below'} the bar; point misses by {abs(pt - K1_BAR):.4f}")
    print(f"  vs 'no leap' line    <  {NO_LEAP}   ", end="")

    if hi < NO_LEAP:
        case, rc = "(ii) CI ENTIRELY BELOW -0.3 -> THE CLAUSE FIRED", 1
    elif lo >= NO_LEAP:
        case, rc = "(i) CI ENTIRELY AT OR ABOVE -0.3 -> clause did not fire", 0
    else:
        case, rc = "(iii) CI STRADDLES -0.3 -> unresolved", 1
    print(case)

    out = ROOT / "results" / "chase_slope_ci.json"
    out.write_text(json.dumps({"ks": ks, "point": pt, "ci": [lo, hi], "B": B,
                               "no_leap": NO_LEAP, "k1_bar": K1_BAR,
                               "case": case}, indent=1))
    print(f"\njournal: {out}\n")

    if rc == 0:
        print("The record's 'untested, not failed' is supported by the interval.")
        return 0
    print("THE SHIPPED SENTENCE IS WRONG EITHER WAY.")
    print("  CHECKLIST.md:417 and DONE.md:234 read '-0.3061 MISSES the -0.3 line")
    print("  by 0.006'. -0.3061 is BELOW -0.3. It did not miss the line, it crossed")
    print("  it. The pre-registered trigger `D_FR slope < -0.3` depends on this")
    print("  clause ALONE and not on the vacuous flip half, so the flip half being")
    print("  unevaluable does not make THIS clause unevaluable.")
    if hi >= NO_LEAP > lo:
        print("  The interval does not resolve it at 120 draws. That is the honest")
        print("  reading and it is NOT 'untested, not failed' -- it is 'unresolved,")
        print("  point estimate inside the kill region'.")
    print(f"  Against K1's OWN bar of {K1_BAR}, the whole interval fails by "
          f"{abs(hi - K1_BAR):.4f} at the favourable end.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
