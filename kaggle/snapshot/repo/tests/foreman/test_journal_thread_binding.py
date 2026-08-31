"""The M2 journal is REPLAYABLE. The unit key just does not name the thread count.

WHAT THE RECORD SAYS, and this file contradicts it by measurement.
`DONE.md` r4 iter 7 and `inspector.py:202-217` both state:

    "NONE of 1, 2, 4, 8, 16, 20, 24 reproduces it [dense_signed__at_pivots/s128]"
    "`s128` reproduces at NO count -> stale-journal candidate."

and `journal_census.py` names that outcome as meaning "its code changed after
journalling, and the journal is STALE for that unit". The supporting inference
was `results/m2.jsonl` mtime 12:38:59 against `scale/m2_units.py` mtime 16:22:33.

**THE SWEEP HAS A HOLE IN IT AT 3.** `dense_signed__at_pivots/s128` reproduces
BITWISE at `torch.set_num_threads(3)` -- every field, sigma and term included.
And the count is a property OF THE CELL, not of the unit. Measured here, torch
2.5.1+cu121, MKL 2024.2.2, CPU only; each row is a bitwise match on all five
fields, 10 of the 13 drifted units:

    cell                      drifted  replays bitwise at  confirmed
    dense_signed__at_pivots     4/13   threads = 3         4/4   s128, s512,
                                                                 s1024/b0, s1024/b1
    pivot_signed__not_in_P      2/3    threads = 3         2/2   s128, s512
    pivot_signed__in_P          7/21   threads = 4         4/4   s128, s2048/b0,
                                                                 s2048/b1  (+b2..b4,
                                                                 s1024 not run)

Not one drifted unit is stale. Each cell was journalled in its own `run_bucket`
process, `run_bucket` never pinned a thread count, and nothing recorded the one
in force. The census read all 37 at `torch.set_num_threads(2)`, so it reports
DRIFT exactly where 2 partitions the reduction differently from 3 or 4 -- which
is why the drift tracks size and draw count rather than any property of the code.

THE ROOT CAUSE IS THE KEY, NOT THE FLOATS. `bucket.py` promises "Two units with
the same key must produce the same number." The key is
`(arm, placement, s, n_draws, k, seed)`. The BLAS reduction schedule is not in
it, and `compute()` is a function of that schedule. So the key does not
determine the value, and a "determinism audit" built on it is auditing an
under-specified identity. The 13/37 drift is that gap, measured.

TWO OF THESE TESTS NEED NO TORCH AT ALL. The journal can be convicted of
under-specification from its own contents, because it holds two different floats
for one quantity that is forced equal by construction -- and a CODE CHANGE
cannot do that. A code change moves both entries together; only a per-run
reduction schedule can move them apart.
"""
from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Sizes where `pivot_signed__in_P` and `dense_signed__at_pivots` are the SAME
#: measurement of `term`: `plan()` gives both (4096 draws, 1 batch, seed 0).
SHARED_TERM_SIZES = (8, 16, 32, 128, 512)

#: The unit the record calls a stale-journal candidate, and the counts to try.
#: 3 is the one the recorded sweep (1, 2, 4, 8, 16, 20, 24) skipped.
STALE_CANDIDATE = "dense_signed__at_pivots/s128"
THREAD_COUNTS = (1, 2, 3, 4)


def _journal(name: str) -> dict:
    out = {}
    for line in (ROOT / "results" / f"{name}.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["key"]] = r["value"]
    return out


@pytest.mark.parametrize("s", SHARED_TERM_SIZES)
def test_term_is_shared_by_construction_between_the_two_arms(s):
    """`term` at size s must be ONE float, and m2.jsonl holds two.

    WHY IT IS FORCED EQUAL. `run_arm` seeds `torch.Generator().manual_seed(seed)`
    and draws wq/wk/wo/x0/v0/gvec/bet in a fixed order. `pivot_signed` and
    `dense_signed` both build `a = bench._causal_tgate_operator(...)` -- the SAME
    operator -- and neither consumes the generator inside `build_arm` (only the
    `random` arm does). `select_pivots` is deterministic. Both cells here carry
    `placement="in_P"`, so `pool = cands_in` and `c` is the same token. Then

        pivot:  w = a[i, pivots] * a[pivots, j] ; tc = |w[where pivots == c]|
        dense:  w = a[i, :]      * a[:, j]      ; tc = |w[c]|

    Both evaluate to `|a[i,c] * a[c,j]|`, ONE multiply, no reduction. `terms` is
    appended in the same order over the same `used` draws, so `mean_t` is the
    same Python sum. Only `sigma`'s background differs (k=8 terms vs s terms),
    and sigma is therefore excluded here.

    MEASURED LIVE, threads=2: equal to the bit at s=16, s=32 and s=128.
    """
    j = _journal("m2")
    pivot = j["pivot_signed__in_P/s%d" % s]["term"]
    dense = j["dense_signed__at_pivots/s%d" % s]["term"]
    assert pivot == dense, (
        "m2.jsonl holds TWO values for one construction-forced quantity at "
        "s=%d: pivot_signed__in_P=%r vs dense_signed__at_pivots=%r "
        "(absdiff %.3e). Both entries run the same operator on the same draws "
        "and compute term as one multiply. A CODE CHANGE cannot separate them "
        "-- it moves both. Only a per-run reduction schedule can, and the unit "
        "key does not record one." % (s, pivot, dense, abs(pivot - dense))
    )


@pytest.mark.parametrize("s", (8, 32, 128, 512))
def test_the_m2_and_s2_journals_agree_about_the_same_quantity(s):
    """`m2:pivot_signed__in_P/sS` and `s2:pivot_signed__x/sS` share term+sigma.

    `s2_units.compute` differs from `m2_units.compute` only in passing
    `wrt="x"`, which selects which leaf `torch.autograd.grad` differentiates
    against. `wrt` is read ONLY inside the gradient branch; `term` and `sigma`
    are built from `w`, `pivots` and `c`, none of which `wrt` touches. Same arm,
    same placement, same seed 0, same 4096 draws -> both floats must agree.

    They agree at s=8, 32 and 512 and disagree at s=128 -- and s2's s=128 pair
    is exactly what today's code returns at 2 threads, while m2's is what it
    returns at 4. Two journals, one machine, two thread counts.
    """
    m2 = _journal("m2")["pivot_signed__in_P/s%d" % s]
    s2 = _journal("s2")["pivot_signed__x/s%d" % s]
    assert (m2["term"], m2["sigma"]) == (s2["term"], s2["sigma"]), (
        "m2.jsonl and s2.jsonl disagree at s=%d about a quantity `wrt` cannot "
        "change: m2 term=%r sigma=%r; s2 term=%r sigma=%r."
        % (s, m2["term"], m2["sigma"], s2["term"], s2["sigma"])
    )


# --------------------------------------------------------------- torch-backed

def _sweep():
    """{threads: record} for STALE_CANDIDATE. Computed once, ~30 s total."""
    if not hasattr(_sweep, "_cache"):
        import torch
        from scale.m2_units import compute, units
        p = dict(units())[STALE_CANDIDATE]
        out = {}
        for n in THREAD_COUNTS:
            torch.set_num_threads(n)
            out[n] = compute(p)
        _sweep._cache = out
    return _sweep._cache


def test_compute_is_a_function_of_its_unit_key():
    """`bucket.py`: "Two units with the same key must produce the same number."

    The key is `(arm, placement, s, n_draws, k, seed)`. `torch.get_num_threads()`
    is not in it, and it changes the answer. That is the whole defect: the
    journal's determinism audit compares values across runs whose identity it
    never captured, so a MATCH means the thread count happened to agree and a
    DRIFT means it did not. Neither reading is about the code.
    """
    got = _sweep()
    distinct = {(r["term"], r["sigma"]) for r in got.values()}
    assert len(distinct) == 1, (
        "%s takes %d distinct (term, sigma) values over threads=%s, so the "
        "unit key does not determine the unit's value:\n%s"
        % (STALE_CANDIDATE, len(distinct), list(THREAD_COUNTS),
           "\n".join("    threads=%d: term=%r sigma=%r"
                     % (n, r["term"], r["sigma"]) for n, r in sorted(got.items())))
    )


def test_the_stale_journal_verdict_survives_a_wider_thread_sweep():
    """The record says this unit reproduces at NO thread count. It reproduces at 3.

    `DONE.md` r4 iter7 swept 1, 2, 4, 8, 16, 20, 24 and concluded "stale-journal
    candidate", with `results/m2.jsonl` mtime 12:38:59 vs `scale/m2_units.py`
    mtime 16:22:33 offered as the supporting inference. 3 was never tried.
    """
    want = _journal("m2")[STALE_CANDIDATE]
    hits = [n for n, r in _sweep().items() if r == want]
    assert not hits, (
        "%s replays BITWISE at threads=%s -- every field, sigma and term "
        "included. It is not stale and its producing code did not change; it "
        "was journalled at a thread count the recorded sweep "
        "(1, 2, 4, 8, 16, 20, 24) skipped. journal=%r"
        % (STALE_CANDIDATE, hits, want)
    )
