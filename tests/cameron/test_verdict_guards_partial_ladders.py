"""RED-first bind for D-1, D-2 and H-1 in `PREREGISTRATION_HOLE_AUDIT.md`.

`scale/e_ladder.py::verdict()` enforces `E_LADDER_PREREGISTERED_READING.md`
section 6. Three rows print a claim the branch that reaches them never
checked:

    D-1  row F (e_ladder.py:241-244)  a theory-death verdict. Fires on
                                       `won(e3_t1) and deep_lost` with NO
                                       `cur["complete"]` guard, so it can be
                                       claimed on two rungs of a four-rung
                                       ladder.
    D-2  row H (e_ladder.py:267-273)  claims "SETTLING IS A STRICT COST ...
                                       and is positive nowhere" -- a
                                       universal over every rung -- with NO
                                       `cur["complete"]` guard. On a partial
                                       ladder an unrun rung could have been
                                       positive, so the universal is
                                       unverifiable.
    H-1  row C (e_ladder.py:280-293)  claims "every |delta| is below the
                                       13-seed resolution", but the branch
                                       at :280-281 only inspects
                                       `("e3_t8", "e3_t32")`, not the shallow
                                       rungs. On the real completed ladder
                                       `e3_t1` reads |delta| = 0.036025,
                                       above `RESOLUTION_13` = 0.027260, and
                                       C would fire anyway.

Row A's own path is guarded at e_ladder.py:252 (falls back to row B on a
partial ladder) -- F and H are supposed to follow the same discipline and do
not.

Each defect gets a RED case (the buggy branch fires and prints the false or
unlicensed sentence) paired with a control that the SAME row still fires once
genuinely earned on a COMPLETE ladder -- a guard that blocks the row in every
case is a vacuous fix (BASE_PROMPT.md vacuity rule #4).
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scale import e_ladder as EL                                   # noqa: E402

TS = {"e3_t1": 1, "e3_t2": 2, "e3_t8": 8, "e3_t32": 32}
assert tuple(TS) == EL.RUNGS, "fixture rungs must match the module's own"


def _rung(task, delta, half=0.01, credited=True):
    """One RUN, credited ladder row -- the shape `read()` builds at
    `e_ladder.py:161-172`. `verdict()` itself only reads `task`, `state`,
    `credited`, `delta`, `ci_lo`, `ci_hi`; the rest is carried anyway so a
    fixture is never a shape the reader could not have produced."""
    return dict(task=task, t_star=TS[task], state="RUN", delta=delta,
                ci_lo=delta - half, ci_hi=delta + half, sd_paired=0.02,
                verdict="x", per_seed_delta=[delta] * 5, settled_mean=0.8,
                twin_mean=0.8, seeds_favouring_settled=3, credited=credited)


def _not_run(task):
    """A rung `read()` never reached -- the shape at `e_ladder.py:150-152`."""
    return dict(task=task, t_star=TS[task], state="NOT RUN",
                have_settled=0, have_twin=0, means={})


def _cur(rows):
    """`complete` is DERIVED from the rows, exactly as `read()` computes it
    at `e_ladder.py:212` (`all(r["state"] == "RUN" for r in ladder)`) --
    never set by hand, so a fixture cannot claim a completeness the ladder it
    built does not have."""
    return dict(ladder=rows, seeds=[0, 1, 2, 3, 4], config={},
                resolution_13=EL.RESOLUTION_13,
                complete=all(r["state"] == "RUN" for r in rows),
                all_credited=all(r.get("credited") for r in rows
                                 if r["state"] == "RUN"))


# ---------------------------------------------------------------- D-1: row F
def test_row_f_does_not_fire_on_a_partial_ladder():
    """won(e3_t1) and lost(e3_t8); e3_t2 and e3_t32 are NOT RUN -- two of
    four rungs. F is a theory-death verdict; it may not be claimed on half a
    ladder."""
    cur = _cur([
        _rung("e3_t1", delta=0.08, half=0.01),    # ci_lo=+0.07  -- won
        _not_run("e3_t2"),
        _rung("e3_t8", delta=-0.09, half=0.01),   # ci_hi=-0.08  -- lost
        _not_run("e3_t32"),
    ])
    assert cur["complete"] is False
    row, why = EL.verdict(cur)
    assert row != "F", f"row F fired on a 2-of-4 ladder: {why!r}"


def test_row_f_still_fires_when_genuinely_earned_on_a_complete_ladder():
    """Same result at every rung, all four now RUN -- F must still fire, or
    the guard is a blanket block rather than a partial-ladder guard."""
    cur = _cur([
        _rung("e3_t1", delta=0.08, half=0.01),    # won
        _rung("e3_t2", delta=0.00, half=0.05),    # covers zero
        _rung("e3_t8", delta=-0.09, half=0.01),   # lost
        _rung("e3_t32", delta=0.00, half=0.05),   # covers zero
    ])
    assert cur["complete"] is True
    row, why = EL.verdict(cur)
    assert row == "F", (row, why)


# ---------------------------------------------------------------- D-2: row H
def test_row_h_does_not_fire_on_a_partial_ladder():
    """lost(e3_t8); e3_t1 covers zero; e3_t2 and e3_t32 are NOT RUN. H's own
    sentence is a universal over every rung -- "is positive nowhere" -- and
    an unrun rung could have been positive."""
    cur = _cur([
        _rung("e3_t1", delta=0.00, half=0.05),    # covers zero
        _not_run("e3_t2"),
        _rung("e3_t8", delta=-0.09, half=0.01),   # lost
        _not_run("e3_t32"),
    ])
    assert cur["complete"] is False
    row, why = EL.verdict(cur)
    assert row != "H", f"row H fired on a 2-of-4 ladder: {why!r}"


def test_row_h_still_fires_when_genuinely_earned_on_a_complete_ladder():
    """Same result at every rung, all four now RUN -- H must still fire."""
    cur = _cur([
        _rung("e3_t1", delta=0.00, half=0.05),
        _rung("e3_t2", delta=0.00, half=0.05),
        _rung("e3_t8", delta=-0.09, half=0.01),
        _rung("e3_t32", delta=-0.12, half=0.01),
    ])
    assert cur["complete"] is True
    row, why = EL.verdict(cur)
    assert row == "H", (row, why)


# ---------------------------------------------------------------- H-1: row C
def test_row_c_does_not_print_the_false_universal():
    """Every CI covers zero on a COMPLETE ladder; |delta| is at or above the
    13-seed resolution at e3_t1 ONLY, and below it at every other rung. Row
    C's sentence claims "every |delta| is below the 13-seed resolution" --
    false, since e3_t1 is not -- so C must not be the row that fires."""
    above = EL.RESOLUTION_13 * 1.1
    below = EL.RESOLUTION_13 * 0.5
    cur = _cur([
        _rung("e3_t1", delta=above, half=0.05),   # covers zero, ABOVE floor
        _rung("e3_t2", delta=below, half=0.05),
        _rung("e3_t8", delta=below, half=0.05),
        _rung("e3_t32", delta=-below, half=0.05),
    ])
    assert cur["complete"] is True
    row, why = EL.verdict(cur)
    assert row != "C", f"row C printed the false universal: {why!r}"


def test_row_c_still_fires_when_every_rung_is_genuinely_below_the_floor():
    """Same shape, |delta| below the floor at every rung including e3_t1 --
    C must still fire, or the H-1 fix is a blanket block on C."""
    below = EL.RESOLUTION_13 * 0.5
    cur = _cur([
        _rung("e3_t1", delta=below, half=0.05),
        _rung("e3_t2", delta=-below, half=0.05),
        _rung("e3_t8", delta=below / 2, half=0.05),
        _rung("e3_t32", delta=-below / 2, half=0.05),
    ])
    assert cur["complete"] is True
    row, why = EL.verdict(cur)
    assert row == "C", (row, why)
