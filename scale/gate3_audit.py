"""Is gate 3's off-schedule cell a MEASUREMENT, or is it zero by construction?

`scale/arm_a_rebuild.py`'s birth gate 3 reads

    on-schedule   X4 = 0.044271  [0.027821, 0.069748]  (17/384)
    off-schedule  X4 = 0.000000  [0.000000, 0.009905]  ( 0/384)   -> FAIL

and the gate is a SCHEDULE-INDEPENDENCE gate: it PASSES when the two intervals
OVERLAP. Disjoint means the flip statistic is a function of where `c` sits
relative to the offset lattice rather than of what `c` is, which is the
placement-artifact class that killed the dilation arm.

BUT AN EXACT 0/384 IS THE SIGNATURE THIS PROJECT HAS NOW BEEN BITTEN BY TWICE.
Round 5 iteration 4 voided ARM A's K1 because its `flip` column read exactly
0.00000 at every k -- not a statistic decaying, but a NON-NEGATIVE Jacobian
making a sign change impossible before any draw was taken. A control that cannot
be nonzero is not a control (instrument #15).

SO THE QUESTION IS NOT "did it flip". It is "COULD it have flipped".

Cameron's own closed form for whether `c` moves `M[i,j]` has TWO disjuncts:

    reachable AND ( i-c in D with supp(i) >= 2                    <- ON-schedule
                    OR some intermediate p has p == c
                       or p-c in D with supp(p) >= 2 )            <- can be OFF

The second disjunct is satisfiable by an OFF-schedule `c`, so off-schedule
influence is NOT structurally zero in general. Whether the pairs gate 3 actually
sampled had any influence is an empirical question, and this file answers it by
the only test that settles it: for each off-schedule pair, are the two gradient
branches `lo` and `hi` BITWISE EQUAL?

    lo == hi exactly   ->  `c` moved nothing. A flip was impossible. VACUOUS.
    lo != hi           ->  `c` moved something and the sign held. REAL.

The verdict is not chosen; it is whichever the count comes out to. If the cell
is vacuous, gate 3 is NOT EVALUABLE and must not be recorded as FAIL. If it is
live, gate 3 FAILS and the FAIL stands.

Two must-fire controls, because a probe that cannot distinguish the two cases
would report "vacuous" for both.
"""
from __future__ import annotations

import pathlib
import random
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

from scale.arm_a_rebuild import reach2                          # noqa: E402
from scale.difference_set_arm import draws                      # noqa: E402
from scale.valuation import valuation                           # noqa: E402

# Read verbatim off `scale/arm_a_rebuild.txt:109`, the line gate 3 printed for
# the schedule it measured. Pinned rather than re-searched: re-running the hill
# climb would be a different object, and the object under audit is the one that
# produced 0/384.
D = [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 16, 17, 19, 20, 28, 29, 30, 31, 40, 46,
     47, 51, 54, 56, 65, 71, 75, 79, 84, 89, 107]
S = 128
N_DRAWS = 24                                   # gate 3's own per-pair count
LAM = 0.10                                     # the SHIPPED operator, ceq/bench.py:192


def gate3_pairs(s: int = S):
    """Gate 3's sampling, reproduced exactly: same seed, same loop, same order."""
    dset, i = set(D), s - 1
    r = reach2(D, s - 1)
    rng = random.Random(99)
    onp, offp = [], []
    while len(onp) < 16 or len(offp) < 16:
        j = rng.randrange(0, i - 1)
        if (i - j) not in r:
            continue
        on = [c for c in range(j + 1, i) if (i - c) in dset]
        off = [c for c in range(j + 1, i) if (i - c) not in dset]
        if on and len(onp) < 16:
            onp.append((j, rng.choice(on)))
        if off and len(offp) < 16:
            offp.append((j, rng.choice(off)))
    return onp, offp


def structurally_inert(j: int, c: int, *, s: int = S) -> bool:
    """True when x[c] provably cannot move d(h[i])/d(v[j]).

    `draws` computes h = v + A v + A^2 v with A masked so that query a attends
    key b iff (a-b) in D. So d(h[i])/d(v[j]) reads A[i,j] and every A[i,p]A[p,j].
    x[c] enters A[a,b] only through q[a] = x[a] Wq and k[b] = x[b] Wk, and c is
    in row a's softmax normalisation iff (a-c) in D. Hence x[c] moves the
    gradient only if c == a or (a-c) in D for some row a the gradient reads --
    that is, a = i, or a = p for a path intermediate p.

    The condition is therefore: (i-c) not in D, c != i, and no p with
    (i-p in D and p-j in D) has p == c or (p-c) in D.

    ONE-DIRECTIONAL, exactly as Cameron's severance closed form is. Structural
    disconnection implies bitwise identity; the converse is false, because a row
    with a single visible key softmaxes to 1.0 and cannot move even when it is
    connected. Over-predicting `live` is safe here -- the control only needs
    predicted-inert to actually read equal.
    """
    dset, i = set(D), s - 1
    if c == i or (i - c) in dset:
        return False
    for p in range(j + 1, i):
        if (i - p) in dset and (p - j) in dset:
            if p == c or (p - c) in dset:
                return False
    return True


def liveness(pool, *, s: int = S):
    """(live, total, max|lo-hi|, flips) over every draw of every pair.

    `live` counts draws where the two branches DIFFER AT ALL. G8: the flip
    decision goes through the valuation instrument's sign FIELDS and never forms
    `lo*hi`, which underflows to -0.0 and hides real flips.
    """
    live = total = flips = 0
    worst = 0.0
    for j, c in pool:
        for lo, hi in draws(D, s=s, i=s - 1, j=j, c=c, n_draws=N_DRAWS, lam=LAM):
            total += 1
            if lo != hi:
                live += 1
                worst = max(worst, abs(lo - hi))
            a, b = valuation(lo)[0], valuation(hi)[0]
            if a and b and a != b:
                flips += 1
    return live, total, worst, flips


def main() -> int:
    onp, offp = gate3_pairs()
    print(f"GATE 3 AUDIT -- is the off-schedule cell zero by construction?")
    print(f"  s={S} |D|={len(D)} n_draws/pair={N_DRAWS} lam={LAM} "
          f"threads={torch.get_num_threads()}")
    print(f"  reproduced gate 3's pairs: {len(onp)} on, {len(offp)} off\n")

    rows = {}
    print(f"  {'cell':<14} {'live/total':>14} {'max|lo-hi|':>14} {'flips':>7}")
    for label, pool in (("on-schedule", onp), ("off-schedule", offp)):
        live, total, worst, flips = liveness(pool)
        rows[label] = (live, total, worst, flips)
        print(f"  {label:<14} {f'{live}/{total}':>14} {worst:>14.6e} {flips:>7}")

    off_live, off_total, off_worst, off_flips = rows["off-schedule"]
    on_live, on_total, _, _ = rows["on-schedule"]

    print("\n=== MUST-FIRE CONTROLS ===")
    # 1. The probe must be able to SEE influence. If it reports 0 live on the
    #    on-schedule cell too, it is measuring nothing and every verdict is void.
    c1 = on_live > 0
    print(f"  [{'FIRED' if c1 else 'DID NOT FIRE'}] probe sees influence where it "
          f"must exist (on-schedule live {on_live}/{on_total})")
    # 2. The probe must read BITWISE EQUAL on a c that is PROVABLY inert.
    #
    #    A FIRST ATTEMPT AT THIS CONTROL WAS WRONG AND IS RECORDED RATHER THAN
    #    QUIETLY REPLACED: it used c = i, on the reasoning that i is the query
    #    row. c = i is the OPPOSITE of inert -- it is the one position that
    #    moves q[i] and therefore the entire row -- and it read live 96/96. The
    #    control was measuring a case where influence was guaranteed.
    inert = [(j, c) for j, c in offp if structurally_inert(j, c)]
    if not inert:
        inert = [(j, c) for j in {p[0] for p in offp}
                 for c in range(j + 1, S - 1) if structurally_inert(j, c)][:4]
    if inert:
        i_live, i_total, i_worst, _ = liveness(inert[:4])
        c2 = i_live == 0
        print(f"  [{'FIRED' if c2 else 'DID NOT FIRE'}] probe reads BITWISE EQUAL "
              f"on a provably inert c (live {i_live}/{i_total}, "
              f"max|lo-hi| {i_worst:.6e}, {len(inert)} such pairs found)")
    else:
        c2 = False
        print("  [DID NOT FIRE] no provably inert c exists at this (D, s) -- the "
              "control cannot be constructed, so the vacuity branch is untested")

    print("\n=== VERDICT ===")
    if not c1:
        print("  PROBE BROKEN -- no influence seen where influence must exist.")
        print("  Every number above is void. Gate 3 is not audited.")
        return 1
    if not c2:
        print("  FALSE-POSITIVE CONTROL DID NOT FIRE. The probe has not been")
        print("  shown to read EQUAL where it must, so a `live` reading could")
        print("  be the probe rather than the arm. Verdict below is CONDITIONAL.")
    if off_live == 0:
        print(f"  OFF-SCHEDULE CELL IS VACUOUS. {off_total}/{off_total} draws have")
        print(f"  lo == hi BITWISE. `c` moved nothing, so a sign change was")
        print(f"  impossible before any draw. 0/384 is a THEOREM, not a rate.")
        print(f"  -> GATE 3 IS NOT EVALUABLE. It must not be recorded as FAIL.")
    else:
        print(f"  OFF-SCHEDULE CELL IS LIVE. {off_live}/{off_total} draws have")
        print(f"  lo != hi, max separation {off_worst:.6e}, and {off_flips} flipped.")
        print(f"  Influence exists off the schedule and the sign held.")
        print(f"  -> GATE 3 FAILS, and the FAIL STANDS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
