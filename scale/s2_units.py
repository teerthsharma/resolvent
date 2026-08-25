"""S2 — the ablation that decides whether the contribution is ROUTING or SIGNEDNESS.

WHY THIS IS THE NUMBER THAT MATTERS. M2's claim arm is flat across a 256x context
growth (slope +0.0270) while a dense control at the SAME intervened tokens decays
(61x apart at s=1024, >=20.5x at s=2048). That establishes ROUTING does something.
It does not establish that SIGNEDNESS does.

**Star-Transformer (arXiv:1902.09113, 2019) already routes hop-2 through a relay
node with an UNSIGNED operator** -- verbatim, "every two non-adjacent satellite
nodes are two-hop neighbors and can receive non-local information with a two-step
update". So if an unsigned operator routed through the same pivots is ALSO flat,
the flatness belongs to routing, that is Star-Transformer's from 2019, and
CHECKLIST.md's stopping condition **G4** fires: the claim sentence must be
rewritten before anything is published.

`M2_PREREGISTERED_READING.md` names this as "the one number that would change the
verdict", written before the M2 control finished.

THE THREE ARMS, at identical geometry, identical pivots, identical draws:

    pivot_signed     tanh(qhat.khat/tau) gated       routed hop-2   <- the claim
    pivot_unsigned   softmax                         routed hop-2   <- THE ABLATION
    dense_signed     tanh(qhat.khat/tau) gated       dense hop-2    <- M2's control

`pivot_unsigned` is the cell M2 never ran at `in_P` placement. M2 has
`softmax_FLOOR` at `not_in_P`, which is a different question.

WHAT EACH OUTCOME MEANS, fixed here BEFORE the run:

  * unsigned routed is FLAT too  -> the contribution is ROUTING. G4 fires.
    Star-Transformer has it. The claim sentence is rewritten to say so.
  * unsigned routed DECAYS while signed routed stays flat -> signedness is
    load-bearing ON TOP of routing, and that combination is the unoccupied cell.
  * unsigned routed is at a STRUCTURAL ZERO (softmax cannot flip a sign at all,
    since `I + A + A^2` is non-negative entrywise for non-negative A) -> the
    comparison is VACUOUS and proves nothing either way. **This is the likely
    outcome and it must not be read as a win.** If it happens, S2 needs a
    different unsigned arm -- one that CAN flip -- e.g. softmax with a signed
    value path, or the `wrt="x"` channel where softmax does have a sign to lose.

SEPARATE JOURNAL from M2 on purpose: adding these units to `m2` would block
`--report`, which must stay able to close M2 on its own terms.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scale.bucket import require_complete, run_bucket
from scale.m2_units import plan
from scale.pivot_probe import clopper_pearson, loglog_slope, run_arm

NAME = "s2"
SIZES = [8, 32, 128, 512, 1024, 2048]

#: BOTH arms on `wrt="x"`, the ONLY channel where the comparison is meaningful.
#:
#: S2 bucket 1 ran `pivot_unsigned` on `wrt="v"` and read k=0 in 4096 draws at
#: four sizes -- VACUOUS, because a non-negative operator has `I + A + A^2`
#: non-negative entrywise and cannot flip a sign there AT ALL, routed or not.
#: Those units stay in the journal as evidence of the structural zero; they are
#: not deleted and they are not the ablation.
#:
#: RED-FIRST, before trusting anything here [RUN, s=64, 512 draws, in_P]:
#:     pivot_unsigned  wrt=v  0.000000  (0/512)   <- structurally pinned
#:     pivot_unsigned  wrt=x  0.003906  (2/512)   <- CAN flip
#:     pivot_signed    wrt=v  0.013672  (7/512)
#:     pivot_signed    wrt=x  0.041016  (21/512)
#: The unsigned arm moves on `x`, so a zero there would be a RESULT rather than
#: a tautology. That is the whole precondition for S2 meaning anything.
#: DECISIVE ARM FIRST. `pivot_unsigned__x` is the number G4 turns on, and the
#: signed arm is already measured at s = 8/32/128/512 (0.026367 / 0.034912 /
#: 0.024902 / 0.031250). Running the unsigned arm at those SAME sizes closes the
#: comparison at matched geometry before either arm reaches the expensive tail.
#: Order only -- nothing is cut, and the journal is keyed by unit name.
CELLS = [
    ("pivot_unsigned__x", "pivot_unsigned", "in_P", "x"),   # <- decides G4
    ("pivot_signed__x",   "pivot_signed",   "in_P", "x"),
]


def units():
    out = []
    for label, arm, place, wrt in CELLS:
        for s in SIZES:
            n, batches = plan(s, label)
            for b in range(batches):
                key = f"{label}/s{s}" + (f"/b{b}" if batches > 1 else "")
                out.append((key, dict(arm=arm, placement=place, s=s,
                                      n_draws=n, k=8, seed=b, wrt=wrt)))
    return out


def compute(p):
    r = run_arm(p["arm"], p["s"], n_draws=p["n_draws"], k=p["k"],
                placement=p["placement"], seed=p["seed"], protocol="SCALING",
                wrt=p.get("wrt", "v"))
    return {k: r[k] for k in ("rate", "k", "n", "term", "sigma")}


def report():
    vals = require_complete(NAME, units())
    print("PROTOCOL: SCALING   S2 ablation: does signedness add anything to routing?")
    print("M2 claim arm, for comparison (journalled, complete):")
    print("  0.024658 0.028564 0.031006 0.028809 0.029663 0.029907   slope +0.0270\n")
    print(f"{'cell':>28} " + "".join(f"{s:>21}" for s in SIZES) + f"{'slope':>9}")
    for label, _, _, _ in CELLS:
        rates, cells = [], []
        for s in SIZES:
            _, nb = plan(s, label)
            keys = ([f"{label}/s{s}"] if nb == 1
                    else [f"{label}/s{s}/b{b}" for b in range(nb)])
            k = sum(int(vals[x]["k"]) for x in keys)
            n = sum(int(vals[x]["n"]) for x in keys)
            lo, hi = clopper_pearson(k, n)
            rates.append(k / n if n else float("nan"))
            cells.append(f"{k/n:.5f}[{lo:.4f},{hi:.4f}]")
        sl, npts = loglog_slope(SIZES, rates)
        note = "" if npts == len(SIZES) else f" ({npts}/{len(SIZES)} nonzero)"
        print(f"{label:>28} " + "".join(f"{c:>21}" for c in cells)
              + f"{sl:>+9.3f}{note}")
        if all(r == 0.0 for r in rates):
            print("\n  *** ALL ZERO -- the comparison is VACUOUS, not a win. ***")
            print("  A non-negative operator has `I + A + A^2` non-negative")
            print("  entrywise, so it CANNOT flip a sign at all. This says nothing")
            print("  about whether routing or signedness carries the flatness.")
            print("  S2 needs an unsigned arm that CAN flip before it can decide.")
        elif sl > -0.3:
            print("\n  *** G4 FIRES: unsigned routing is ALSO flat. ***")
            print("  The contribution is ROUTING, which is Star-Transformer's")
            print("  (arXiv:1902.09113, 2019). Rewrite the claim sentence before")
            print("  publishing anything.")
        else:
            print(f"\n  unsigned routing DECAYS (slope {sl:+.3f}) while signed routing")
            print("  is flat (+0.0270) -- signedness is load-bearing ON TOP of")
            print("  routing, and that combination is the unoccupied cell.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.status:
        from scale.bucket import Journal
        done = Journal(NAME).done()
        u = units()
        print(f"{len(done)}/{len(u)} units journalled")
        for k, _ in u:
            if k not in done:
                print(f"  next: {k}")
                break
    else:
        run_bucket(NAME, units(), compute, budget_s=a.budget, verify=1)
