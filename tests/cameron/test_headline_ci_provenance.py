"""The headline `settled - softmax` interval, pinned to the run that made it.

WHY THIS FILE EXISTS. `+0.108437` shipped with TWO different lower bounds:

    ceq/hf_artifact/README.md   [+0.066232, +0.147110]
    CHECKLIST.md:1168           [+0.068181, +0.147110]

Same point estimate, same upper bound, lower bounds `0.001949` apart, and
neither row named the run that produced it. `MISTAKES.md` M-9 recorded that as
an open P-1 on the repository's most-quoted number, explicitly NOT fixed,
because picking one without finding the producer would only have made the
disagreement invisible.

THE RESOLUTION IS THAT NEITHER WAS A TRANSCRIPTION ERROR. Both reproduce
bit-exact from the same ten journal records, under two different and both
legitimate estimators:

    [+0.066232, +0.147110]  paired percentile bootstrap B=10000 seed=0
    [+0.068181, +0.147110]  exact percentile over all 5**5 = 3125 resamples

At five seeds the paired resample space is finite, so the percentile the
Monte-Carlo draw estimates is also computable outright. The two families are
the same per-seed deltas under a different resampling rule.

WHAT THIS TEST PROTECTS. Not the digits -- `tests/chase/test_capability_table.py`
already recomputes the shipped table. This pins the PAIRING: each published
endpoint to the estimator its own document names, so the next reader who finds
two numbers for one headline does not have to redo this investigation. If a row
is ever retyped from the other family, the estimator label and the endpoints
stop agreeing and this fails.

The `0.147110` upper bound is shared by BOTH families, which is exactly why the
rows looked like a typo: only one endpoint moved.
"""
from __future__ import annotations

import itertools
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "results" / "m3_quintuple_v2.jsonl"
SEEDS = (0, 1, 2, 3, 4)
GEOM = "s64_d24_st150_ntr8192_nev512_b21"

#: The ten records the headline is computed from. Named in full because the
#: whole point of the entry is that a number with no key behind it cannot be
#: checked -- these are the keys both documents now cite.
SETTLED_KEYS = [f"settled_k8_{GEOM}_sd{s}" for s in SEEDS]
SOFTMAX_KEYS = [f"softmax_k0_{GEOM}_sd{s}" for s in SEEDS]

MC = (0.066232, 0.147110)       # ceq/hf_artifact/README.md
EXACT = (0.068181, 0.147110)    # CHECKLIST.md:1168


@pytest.fixture(scope="module")
def deltas():
    """`softmax - settled` per seed, straight from the journal keys."""
    import json
    by_key = {}
    for line in JOURNAL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by_key[r["key"]] = r["value"]["eval_nrmse"]
    missing = [k for k in SETTLED_KEYS + SOFTMAX_KEYS if k not in by_key]
    assert not missing, f"journal keys cited by the headline are gone: {missing}"
    return [by_key[f] - by_key[t] for f, t in zip(SOFTMAX_KEYS, SETTLED_KEYS)]


def test_the_point_estimate_is_the_mean_of_the_journalled_deltas(deltas):
    assert round(sum(deltas) / len(deltas), 6) == 0.108437


def test_monte_carlo_family_reproduces_the_shipped_artifact_endpoints(deltas):
    """`ceq/hf_artifact/README.md` -- the run's own B=10000 seed=0 draw."""
    from scale.m3_synthetic_settled import contrast
    softmax = [0.0] * len(deltas)          # contrast() takes the two arms;
    settled = [-d for d in deltas]         # only the difference is used.
    r = contrast(softmax, settled, n_boot=10000, seed=0)
    assert (round(r["ci_lo"], 6), round(r["ci_hi"], 6)) == MC


def test_exact_family_reproduces_the_checklist_endpoints(deltas):
    """`CHECKLIST.md:1168` -- all 5**5 resamples, no generator involved."""
    from scale.capability_table import exact_percentile_ci
    lo, hi = exact_percentile_ci(deltas)
    assert (round(lo, 6), round(hi, 6)) == EXACT


def test_the_two_families_differ_only_at_the_lower_bound(deltas):
    """The reason the rows read as a typo. Kept as an assertion so that if a
    future journal makes them agree, the prose explaining the split is stale
    and this says so."""
    assert MC[1] == EXACT[1]
    assert MC[0] != EXACT[0]
    assert round(EXACT[0] - MC[0], 6) == 0.001949


# ------------------------------------------------- the documents, not the math

def _line(path: str, n: int) -> str:
    return (ROOT / path).read_text(encoding="utf-8").splitlines()[n - 1]


def test_checklist_row_names_the_exact_estimator_beside_its_number():
    row = _line("CHECKLIST.md", 1168)
    assert "+0.068181" in row
    assert "3125" in row and "exact" in row.lower()
    assert "m3_quintuple_v2.jsonl" in row
    for k in ("settled_k8_" + GEOM, "softmax_k0_" + GEOM):
        assert k in row, k


def test_shipped_card_carries_both_families_each_under_its_own_name():
    md = (ROOT / "ceq" / "hf_artifact" / "README.md").read_text(encoding="utf-8")
    row = [l for l in md.splitlines()
           if l.startswith("| `settled` | `softmax` |")]
    assert len(row) == 1, row
    assert "+0.066232" in row[0] and "+0.068181" in row[0], row[0]
    assert "exact 95% CI" in md and "3125" in md
    assert "m3_quintuple_v2.jsonl" in md
