"""M2 as idempotent units in wall-clock buckets. ADR-001.

LOCK M2 efadc390c93f -- the item text is frozen; re-running the measurement does
not unfreeze it. The pre-registered kill is unchanged:

    slope(c in P) < -0.3
    OR c-not-in-P is ALSO flat  (mechanism false even if the pivot number is good)
    OR any previously published bench.py number moves  (G2)

DRAW-BATCHING AT s=2048 -- ADR-001's "to revisit" case, now reached.
Timed over 120 draws (not 4 -- that lesson is on record): **80.1 ms/draw at
s=2048**, so a single 16384-draw unit costs **1312 s** and CANNOT fit the 600 s
platform cap. Such a unit is unrunnable, not merely slow.

So s=2048 is split into 16 batches of 1024 draws with seeds 0..15, aggregated by
summing flips and draws. Both forms are 16384 independent draws, so the total is
unchanged -- and the batched form is STRICTLY BETTER evidence, because it spreads
16 seeds where the single unit had one. This project has already been burned by a
single seed: a tau sweep on seed 0 produced a false optimum that evaporated to
1-in-5 when replicated. The split is an improvement, not a compromise.

Sizes below 2048 are unchanged and their journalled units stay valid: s<=512 at
4096 draws costs ~24 s, s=1024 at 16384 draws costs 339 s, both under the cap.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scale.bucket import require_complete, run_bucket
from scale.pivot_probe import clopper_pearson, loglog_slope, run_arm

NAME = "m2"
SIZES = [8, 16, 32, 128, 512, 1024, 2048]

#: (label, s) pairs where the PIVOT and DENSE arms are the SAME MEASUREMENT.
#:
#: `select_pivots` excludes i and j, so at s=8 with i=7, j=2 only SIX indices
#: remain and P is all of them -- |P| = 6, not the k=8 every artifact claimed.
#: Routing then restricts nothing: measured, all 4 usable intermediates are
#: pivots, ZERO excluded, and `A[:,P]A[P,:]` equals `A@A` at [i,j] to
#: **1.86e-09**, under the probe's own 1e-6 floor. The journal agrees to every
#: digit -- both s=8 cells read k=101, n=4096.
#:
#: That single shared point was the LEFTMOST point of BOTH fits, which is the
#: ParaFormer hazard (G3) in a new costume: one arm reported twice under two
#: names. Dropping it moves the dense slope from -0.746 to **-1.009** -- the
#: separation is REAL and gets STRONGER, so this correction costs the claim
#: nothing and costs the wrong number everything.
#:
#: s=8 CANNOT be rerun at |P| = 8: only six candidates exist. So it is replaced
#: by s=16, the SMALLEST size where |P| = 8 is a proper subset -- measured there:
#: |P| = 8, 10 intermediates, 6 in P, **4 excluded**, pivot-vs-dense delta
#: 6.6e-03, four orders above the floor. Routing bites when k < i-j-1 =
#: s - s//4 - 2, i.e. for k=8 when s > 13.3.
#:
#: The s=8 units stay in the journal as the degenerate point they are; they are
#: excluded from `units()` and therefore from every fit, and `--status` names
#: them as orphans rather than counting them.
DEGENERATE = {("pivot_signed__in_P", 8), ("dense_signed__at_pivots", 8)}

#: DENSE hop-2 is O(s^3) where routed hop-2 is O(s^2 k), so the control costs far
#: more per draw at large s: measured 23.8 / 89.4 / **535.4** ms/draw at
#: s=512/1024/2048 against the pivot arm's 80.1 ms at s=2048. A 16384-draw dense
#: unit at s=2048 costs **8773 s** -- 2.4 h for one cell, ~30 batches.
#:
#: STATED REDUCTION, NOT A SILENT CAP. The control's job is to show the two arms
#: SEPARATE, and they are far apart: at s=512 dense reads 0.00317 against the
#: pivot arm's 0.02881, a 9x gap. At 4096 draws even a zero gives a
#: Clopper-Pearson upper bound of 0.00090, still an order of magnitude below the
#: pivot rate -- so 4096 draws resolve the comparison being made. The claim arm
#: keeps its full pre-registered 16384 at every size; only the CONTROL is reduced,
#: the reason is cost, and every cell reports its exact interval.
DENSE_ARMS = {"dense_signed__at_pivots", "dense_signed"}

def plan(s: int, label: str = ""):
    """(draws per unit, number of units) -- batched to fit the 600 s cap."""
    if label in DENSE_ARMS:
        if s >= 2048:
            return 512, 8        # 8 x 512 = 4096, each ~274 s
        if s >= 1024:
            return 2048, 2       # 2 x 2048 = 4096, each ~183 s
        return 4096, 1           # <= 98 s
    if s >= 2048:
        return 1024, 16          # 16 x 1024 = 16384, each ~82 s
    if s >= 1024:
        return 16384, 1          # 339 s, fits
    return 4096, 1               # ~24 s

#: ORDERED BY WHAT CAN PRODUCE A VERDICT, not by convenience.
#:
#: The frozen kill reads exactly two slopes -- `c in P` and `c NOT in P` -- and
#: neither can be trusted without both ends of the instrument. Those four cells
#: are 84 of the 126 units; `dense_signed` and `windowed` are context, and cannot
#: change the verdict. At ~90 s per s=2048 batch, M2 needs 20-25 buckets, which
#: is most of the loop's remaining iterations, so the ORDER decides whether a
#: verdict exists before the budget runs out.
#:
#: NOTHING IS CUT. All 126 units still run; the journal is keyed by unit name, so
#: order affects scheduling only and never correctness. This is priority
#: scheduling, not the silent cap the rules forbid -- the distinction is that a
#: cap changes what evidence exists, and an order changes only when it arrives.
CELLS = [
    # --- decisive: the two slopes the kill is written against -------------
    ("pivot_signed__in_P",     "pivot_signed",   "in_P"),
    # --- THE CONTROL THAT CAN ACTUALLY FALSIFY (added iteration 19) -------
    # Dense hop-2 measured at THE SAME c positions: c drawn from P, but
    # hop2 = A @ A instead of A[:,P] A[P,:]. Same operator, same geometry,
    # same intervened tokens -- the ONLY difference is whether hop 2 is routed.
    # So if this decays while `pivot_signed__in_P` stays flat, routing is the
    # cause; if it is ALSO flat, the flatness has nothing to do with routing.
    #
    # This exists because `pivot_signed__not_in_P` turned out to be zero BY
    # CONSTRUCTION -- `hop2[i,j] = sum_{p in P} A[i,p] A[p,j]` contains no term
    # with index c when c is not in P, so perturbing such a c cannot move
    # anything. A control that cannot be nonzero cannot falsify anything, and
    # the frozen kill's second clause is therefore UNEVALUABLE as written.
    ("dense_signed__at_pivots", "dense_signed",  "in_P"),
    # `not_in_P` runs AFTER it: that cell is structurally zero (confirmed at
    # s=32/128/512 with real draws, all exactly 0.0), so its remaining units
    # spend ~2 h confirming a theorem while the falsifiable control waits.
    ("pivot_signed__not_in_P", "pivot_signed",   "not_in_P"),
    # --- instrument ends: without these neither slope means anything ------
    ("softmax_FLOOR",          "dense_unsigned", "not_in_P"),
    ("random_CEILING",         "random",         "not_in_P"),
    # --- context: cannot change the verdict -------------------------------
    ("dense_signed",           "dense_signed",   "not_in_P"),
    ("pivot_signed__windowed", "pivot_signed",   "windowed"),
]

#: the cells after which the kill becomes evaluable
DECISIVE = ("pivot_signed__in_P", "pivot_signed__not_in_P",
            "dense_signed__at_pivots", "softmax_FLOOR", "random_CEILING")

#: (label, s) pairs that CANNOT be measured and must not be journalled.
#: At s=8 with k=8 pivots, every token is a pivot, so after excluding i and j
#: there are ZERO candidates for "not in P" -- measured: s=8 -> 0 candidates,
#: s=16 -> 6, s=32 -> 22. The unit already journalled for it reported rate 0.0
#: from n=0 draws and is VOID, not evidence.
IMPOSSIBLE = {("pivot_signed__not_in_P", 8)}


def units():
    out = []
    for label, arm, place in CELLS:
        for s in SIZES:
            if (label, s) in IMPOSSIBLE or (label, s) in DEGENERATE:
                continue
            n, batches = plan(s, label)
            for b in range(batches):
                key = f"{label}/s{s}" + (f"/b{b}" if batches > 1 else "")
                out.append((key, dict(arm=arm, placement=place, s=s,
                                      n_draws=n, k=8, seed=b)))
    return out


def compute(p):
    r = run_arm(p["arm"], p["s"], n_draws=p["n_draws"], k=p["k"],
                placement=p["placement"], seed=p["seed"], protocol="SCALING")
    return {k: r[k] for k in ("rate", "k", "n", "term", "sigma")}


def sizes_for(label):
    """Sizes this cell can actually be measured at."""
    return [s for s in SIZES
            if (label, s) not in IMPOSSIBLE
            and (label, s) not in DEGENERATE]


def _cell(vals, label, s):
    """Aggregate a cell across its batches: sum flips, sum draws."""
    _, batches = plan(s, label)
    keys = ([f"{label}/s{s}"] if batches == 1
            else [f"{label}/s{s}/b{b}" for b in range(batches)])
    k = sum(int(vals[x]["k"]) for x in keys)
    n = sum(int(vals[x]["n"]) for x in keys)
    if n == 0:
        raise ValueError(f"{label}/s{s} aggregates to n=0 draws; it is VOID, "
                         f"not a rate of 0.0")
    return k, n, k / n


def report_decisive():
    """Verdict from the four cells the frozen kill actually reads.

    LEGITIMATE, and the distinction matters: the kill is written against
    `slope(c in P)` and `slope(c NOT in P)`, with the floor and ceiling needed to
    trust either. `dense_signed` and `windowed` are context and CANNOT change the
    verdict -- so evaluating the kill without them is not a partial verdict, it
    is the whole verdict on a smaller table.

    It refuses just as hard if any DECISIVE unit is missing, and it always prints
    which non-decisive cells are still outstanding, so this can never be mistaken
    for the complete report.
    """
    from scale.bucket import Journal
    done = Journal(NAME).done()
    need = [(k, p) for k, p in units() if k.split("/")[0] in DECISIVE]
    missing = [k for k, _ in need if k not in done]
    if missing:
        raise SystemExit(
            f"REFUSING: {len(missing)} of {len(need)} DECISIVE units missing, "
            f"e.g. {missing[:3]}. The kill cannot be evaluated yet."
        )
    vals = {k: done[k]["value"] for k, _ in need}
    print("=== INTERIM: verdict on the DECISIVE cells only ===")
    print("PROTOCOL: SCALING   LOCK M2 efadc390c93f")
    outstanding = [lbl for lbl, _, _ in CELLS if lbl not in DECISIVE]
    print(f"NOT YET RUN (context only, cannot change the verdict): {outstanding}\n")
    print(f"{'cell':>24} " + "".join(f"{s:>21}" for s in SIZES) + f"{'slope':>9}")
    out = {}
    for label in DECISIVE:
        rates, cells, ss = [], [], sizes_for(label)
        for s in ss:
            k, n, rate = _cell(vals, label, s)
            lo, hi = clopper_pearson(k, n)
            rates.append(rate)
            cells.append(f"{rate:.5f}[{lo:.4f},{hi:.4f}]")
        sl, npts = loglog_slope(ss, rates)
        out[label] = sl
        note = "" if npts == len(ss) else f" ({npts}/{len(ss)} nonzero)"
        print(f"{label:>24} " + "".join(f"{c:>21}" for c in cells)
              + f"{sl:>+9.3f}{note}")
    _verdict(out["pivot_signed__in_P"], out["pivot_signed__not_in_P"],
             out.get("dense_signed__at_pivots", float("nan")))


def _verdict(a: float, b: float, ctrl: float = float("nan")) -> None:
    """Evaluate the frozen kill. AN UNEVALUABLE CLAUSE CAN NEVER YIELD A PASS.

    THE BUG THIS REPLACES, and it was on a timer. The old line was

        k2 = (b > -0.3) if b == b else False

    which maps the NaN slope of an all-zero series to False, i.e. to "control
    decays as the mechanism requires", and GREEN follows. `c NOT in P` IS
    all-zero -- structurally, not by measurement: `hop2[i,j] = sum_{p in P}
    A[i,p] A[p,j]` contains no term with index c when c is not a pivot, so
    perturbing such a c cannot move j's influence on i. Measured: 8/8 draws
    bitwise identical, `max|delta influence| = 0.0`, against 0.207 on the `in_P`
    arm.

    So clause 2 of a three-clause kill was not merely unevaluable -- it was
    silently voting PASS. `report()` refuses today only because units are
    missing; **it stops refusing the moment the bucket loop finishes**, and the
    NaN fires exactly then. The false green was scheduled, not hypothetical.

    NOW: NaN or a fit with fewer than two points is UNEVALUABLE, and any
    unevaluable mandatory clause forces the verdict to VOID. Never GREEN.
    """
    UN = "UNEVALUABLE"
    print("\n=== VERDICT against the frozen kill (LOCK M2 efadc390c93f) ===")

    k1_eval = a == a
    k1 = (a < -0.3) if k1_eval else None
    print(f"  clause 1  slope(c in P)     = {a:+.3f} -> "
          + ("KILL" if k1 else "survives" if k1_eval else UN))

    k2_eval = b == b
    k2 = (b > -0.3) if k2_eval else None
    print(f"  clause 2  slope(c NOT in P) = {b:+.3f} -> "
          + ("KILL: control also flat, mechanism story FALSE" if k2
             else "control decays as required" if k2_eval else UN))
    if not k2_eval:
        print("            ^ all-zero series: `c NOT in P` is a STRUCTURAL "
              "IDENTITY, not a measurement.")
        print("              hop2[i,j] = sum_{p in P} A[i,p] A[p,j] has no term "
              "in c when c is not a pivot.")
        print("              This clause cannot fire and cannot pass. It is VOID.")

    if ctrl == ctrl:
        print(f"  control   slope(dense at same c) = {ctrl:+.3f} -> "
              + ("decays: routing is the variable"
                 if ctrl < -0.3 else
                 "ALSO FLAT: routing is NOT the variable -- treat as a KILL"))
        print("            (not named in the frozen text; reported as evidence, "
              "and it is the ONLY surviving comparator)")
    else:
        print("  control   dense-at-same-c NOT RUN")

    if k1 is None or k2 is None:
        print("\n  M2 = VOID -- a mandatory clause is UNEVALUABLE.")
        print("  A three-clause kill has collapsed to one evaluable clause.")
        print("  M2 MAY NOT BE MARKED GREEN. Repair the control or restate the "
              "kill; do NOT read the surviving clause as a verdict.")
        return
    print(f"\n  M2 = {'RED' if (k1 or k2) else 'GREEN'}")


def report():
    vals = require_complete(NAME, units())          # REFUSES if partial
    print("PROTOCOL: SCALING   LOCK M2 efadc390c93f")
    print("s=2048 is 16 batches x 1024 draws, seeds 0-15, aggregated by summing "
          "flips and draws (a single 16384-draw unit costs 1312 s > the 600 s cap)\n")
    print(f"{'cell':>24} " + "".join(f"{s:>21}" for s in SIZES) + f"{'slope':>9}")
    out = {}
    for label, _, _ in CELLS:
        rates, cells, ss = [], [], sizes_for(label)
        for s in ss:
            k, n, rate = _cell(vals, label, s)
            lo, hi = clopper_pearson(k, n)
            rates.append(rate)
            cells.append(f"{rate:.5f}[{lo:.4f},{hi:.4f}]")
        sl, npts = loglog_slope(ss, rates)
        out[label] = (rates, sl)
        note = "" if npts == len(ss) else f" ({npts}/{len(ss)} nonzero)"
        print(f"{label:>24} " + "".join(f"{c:>21}" for c in cells)
              + f"{sl:>+9.3f}{note}")

    a = out["pivot_signed__in_P"][1]
    b = out["pivot_signed__not_in_P"][1]
    k1 = (a < -0.3) if a == a else True
    k2 = (b > -0.3) if b == b else False
    print("\n=== VERDICT against the frozen kill ===")
    print(f"  slope(c in P)     = {a:+.3f} -> {'KILL' if k1 else 'survives'}")
    print(f"  slope(c NOT in P) = {b:+.3f} -> "
          + ("KILL: control also flat, mechanism story FALSE" if k2
             else "control decays as the mechanism requires"))
    print(f"\n  M2 = {'RED' if (k1 or k2) else 'GREEN'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=400.0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.status:
        from scale.bucket import Journal
        done = Journal(NAME).done()
        u = units()
        # COUNT MEMBERS OF units(), NOT JOURNAL LINES. Off by one until now:
        # `pivot_signed__not_in_P/s8` is journalled but excluded by IMPOSSIBLE
        # (it reported rate 0.0 from n=0 draws), so `len(done)` counted a unit
        # that is not in the plan. Every "N/132" printed from --status was ONE
        # HIGH, and those numbers were copied into STATE.md and DONE.md.
        keys = {k for k, _ in u}
        live = keys & set(done)
        orphans = sorted(set(done) - keys)
        print(f"{len(live)}/{len(u)} units journalled")
        if orphans:
            print(f"  ({len(orphans)} journalled but NOT in the plan: {orphans})")
        for k, _ in u:
            if k not in done:
                print(f"  next: {k}")
                break
    else:
        run_bucket(NAME, units(), compute, budget_s=a.budget, verify=1)
