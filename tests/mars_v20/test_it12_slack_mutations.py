"""MARS it.12 -- slack audit of CEQ v20 Round 15's binding tests.

Each function below is a MUTATION DEMONSTRATION, not a normal test: it shows
that a SHIPPED assertion stays GREEN under a data mutation that makes the
prose claim it is supposed to certify FALSE. The mutation is applied to a
copy of the real journalled data, never via `and False` or a skipped assert.
Greenness here is the strike, not a pass.

Target attacked (the only one of the five priority targets that fired):

  B. "8 of 9 arm_pl against 0 of 9 softmax, same seeds, same draws, same
     process" -- V20_R15_IT10_MERCURY.md:202-203 (C14's repaired figure).

     Binding node: tests/venus/test_v20_r15_it10_venus.py
     ::test_crossing_indicator_is_invariant_across_three_eval_draws_on_all_cells
     (defined :142, shipped assertion at :151):

         assert (pl, sm) == (8, 0), (e, pl, sm)

     `pl` and `sm` are two INDEPENDENT counts -- `sum(... if k == "arm_pl" ...)`
     and `sum(... if k == "softmax" ...)` -- over whatever seeds each kind's
     records happen to carry in `results/v20_r15_it10_mercury_rescore.jsonl`.
     Nothing in the shipped test compares the arm_pl seed SET to the softmax
     seed SET. The prose claim is "same seeds, same draws, same process" --
     i.e. PAIRING, not just two counts that both happen to read (8, 0).

     `test_it12_paired_headline_survives_a_disjoint_seed_swap` below rebuilds
     the shipped test's own `_rescore()` + counting logic against a mutated
     copy of the journal where every `softmax` record's `seed` is moved to a
     value the `arm_pl` records never use (900..908 instead of {0,8..15}).
     The prose claim ("same seeds") is now FALSE by construction -- the two
     arms were run on disjoint seed sets, so there is no pairing to speak of.
     The shipped assertion `(pl, sm) == (8, 0)` still evaluates true, because
     it never reads the seed column of either side against the other.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESCORE = ROOT / "results" / "v20_r15_it10_mercury_rescore.jsonl"
FLOOR1 = 0.7071067811865476


def _load_rescore_rows():
    rows = []
    for line in RESCORE.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("t") == "rescore":
            rows.append(d)
    return rows


def _table(rows):
    """Exact reproduction of venus/test_v20_r15_it10_venus.py::_rescore()."""
    tab = {}
    for d in rows:
        tab.setdefault((d["kind"], d["seed"]), {})[d["eval_seed"]] = d["eval_nrmse"]
    return tab


def test_it12_control_the_shipped_assertion_is_reproduced_on_real_data():
    """GREEN control. Same table-building + counting logic as the shipped
    test, run unmodified against the real journal, must match its (8, 0)."""
    tab = _table(_load_rescore_rows())
    assert len(tab) == 18 and all(len(v) == 3 for v in tab.values())
    for e in (12345, 12346, 20260902):
        pl = sum(1 for (k, _), v in tab.items() if k == "arm_pl" and v[e] < FLOOR1)
        sm = sum(1 for (k, _), v in tab.items() if k == "softmax" and v[e] < FLOOR1)
        assert (pl, sm) == (8, 0), (e, pl, sm)
    # And on the real data the pairing claim happens to be true too --
    # this control is what makes the mutation below meaningful.
    pl_seeds = {s for (k, s) in tab if k == "arm_pl"}
    sm_seeds = {s for (k, s) in tab if k == "softmax"}
    assert pl_seeds == sm_seeds, "real data IS paired; mutation test proves the shipped assert doesn't need it to be"


def test_it12_paired_headline_survives_a_disjoint_seed_swap():
    """THE STRIKE. Mutate every softmax record's seed to a value arm_pl never
    uses (disjoint sets: arm_pl keeps {0,8..15}, softmax moves to
    {900..908}). "Same seeds" is now false -- there is nothing pairing the
    two arms' draws. The shipped assertion `(pl, sm) == (8, 0)` at
    tests/venus/test_v20_r15_it10_venus.py:151 does not notice, because it
    never compares the two seed sets."""
    rows = _load_rescore_rows()
    mutated = []
    for d in rows:
        d = dict(d)
        if d["kind"] == "softmax":
            d["seed"] = d["seed"] + 900  # {0,8..15} -> {900,908..915}: disjoint from arm_pl
        mutated.append(d)

    tab = _table(mutated)

    pl_seeds = {s for (k, s) in tab if k == "arm_pl"}
    sm_seeds = {s for (k, s) in tab if k == "softmax"}
    assert pl_seeds.isdisjoint(sm_seeds), "mutation must actually break pairing"
    assert pl_seeds != sm_seeds  # the prose claim "same seeds" is FALSE here

    # The shipped test's own preconditions:
    assert len(tab) == 18 and all(len(v) == 3 for v in tab.values())

    # THE SHIPPED ASSERTION, verbatim (venus/test_v20_r15_it10_venus.py:151),
    # evaluated against data where the paired-seeds claim is FALSE:
    for e in (12345, 12346, 20260902):
        pl = sum(1 for (k, _), v in tab.items() if k == "arm_pl" and v[e] < FLOOR1)
        sm = sum(1 for (k, _), v in tab.items() if k == "softmax" and v[e] < FLOOR1)
        assert (pl, sm) == (8, 0), (e, pl, sm)  # <-- STILL GREEN. This is the strike.
