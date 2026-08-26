"""A journal scan that cannot silently fail to find what it is searching for.

THE FAILURE THIS EXISTS TO PREVENT, stated with its cost. At iteration 4 a scan
reported "zero NRMSE readings above 1.0 in any results/*.jsonl" and that claim was
used to strike a colleague's evidence. It was false: 22 of 68 readings are at or
above 1.0. The scan iterated `r.items()` at the top level while the journals nest
their payload under `value`, so it never descended and could not have returned a hit
on any input. A search structurally incapable of finding a thing, reporting none, is
not evidence of absence.

That was the twelfth vacuous control struck in this campaign and the first authored
here that shipped rather than being caught. Eleven earlier ones were caught by a
colleague or by re-reading, which is discipline; discipline is not a mechanism.

WHAT MAKES THIS DIFFERENT FROM WRITING THE SCAN MORE CAREFULLY. `scan_journals`
refuses to report an absence it has not earned: the caller supplies a witness that
MUST be found, the scan verifies it, and a scan whose own witness is missing raises
instead of returning an empty list. A wrong predicate then fails loudly at the point
of use rather than quietly two iterations later in someone else's argument.
"""
from __future__ import annotations

import json

import pytest

from scale.journal_scan import ScanWitnessError, scan_journals, walk_numbers


def _journal(tmp_path, rows):
    p = tmp_path / "j.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


NESTED = [
    {"key": "a_sd0", "meta": {"seconds": 1.5}, "value": {"eval_nrmse": 0.75}},
    {"key": "a_sd1", "meta": {"seconds": 2.5}, "value": {"eval_nrmse": 1.25}},
    {"key": "a_sd2", "meta": {"seconds": 3.5}, "value": {"eval_nrmse": 0.95}},
]


def test_walk_reaches_nested_values(tmp_path) -> None:
    """The specific defect: a top-level loop never sees `value.eval_nrmse`."""
    found = dict(walk_numbers(NESTED[1]))
    assert "value.eval_nrmse" in found
    assert found["value.eval_nrmse"] == 1.25
    # And the top-level-only reading, which is what shipped, sees none of it.
    assert not [k for k in NESTED[1] if "nrmse" in k.lower()]


def test_absence_requires_a_witness_that_was_actually_found(tmp_path) -> None:
    """The mechanism. A scan that finds nothing AND cannot find its own witness
    raises rather than reporting an absence it has not earned."""
    j = _journal(tmp_path, NESTED)
    # A predicate that is simply wrong about where the data lives -- the exact
    # shape of the shipped defect.
    with pytest.raises(ScanWitnessError):
        scan_journals([j], key_endswith="nrmse_eval",       # transposed, matches nothing
                      predicate=lambda v: v >= 1.0,
                      witness=("value.eval_nrmse", 1.25))


def test_a_correct_scan_reports_the_hits(tmp_path) -> None:
    j = _journal(tmp_path, NESTED)
    hits = scan_journals([j], key_endswith="eval_nrmse",
                         predicate=lambda v: v >= 1.0,
                         witness=("value.eval_nrmse", 1.25))
    assert [h.value for h in hits] == [1.25]
    assert hits[0].key == "a_sd1"


def test_a_true_absence_is_reportable_when_the_witness_holds(tmp_path) -> None:
    """An earned absence. The witness is found, the predicate matches nothing, and
    the empty result is therefore evidence rather than silence."""
    j = _journal(tmp_path, NESTED)
    hits = scan_journals([j], key_endswith="eval_nrmse",
                         predicate=lambda v: v >= 99.0,
                         witness=("value.eval_nrmse", 1.25))
    assert hits == []


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_nesting_depth_does_not_hide_a_value(tmp_path, depth: int) -> None:
    """Drawn across depths rather than asserted at the one depth this repo uses.

    The shipped defect was a depth-1 reader against depth-2 data. Fixing it only for
    depth 2 would reproduce the same class the next time a producer nests deeper.
    """
    row: dict = {"eval_nrmse": 7.5}
    path = "eval_nrmse"
    for i in range(depth):
        row = {f"L{i}": row}
        path = f"L{i}.{path}"
    j = _journal(tmp_path, [row])
    hits = scan_journals([j], key_endswith="eval_nrmse",
                         predicate=lambda v: v > 7.0,
                         witness=(path, 7.5))
    assert [h.value for h in hits] == [7.5]


def test_the_witness_check_itself_can_fail(tmp_path) -> None:
    """MUST-FIRE. Without this the witness mechanism could be a no-op.

    A witness whose VALUE is wrong must raise even though its PATH exists, which no
    implementation that merely checks for the path can satisfy.
    """
    j = _journal(tmp_path, NESTED)
    with pytest.raises(ScanWitnessError):
        scan_journals([j], key_endswith="eval_nrmse",
                      predicate=lambda v: v >= 1.0,
                      witness=("value.eval_nrmse", 999.0))


def test_booleans_are_not_numbers(tmp_path) -> None:
    """`isinstance(True, int)` is True in Python, and a journal full of flags would
    otherwise pollute every numeric scan with ones and zeros."""
    j = _journal(tmp_path, [{"value": {"converged": True, "eval_nrmse": 0.5}}])
    found = dict(walk_numbers({"converged": True, "x": 1}))
    assert "converged" not in found
    assert found["x"] == 1
    hits = scan_journals([j], key_endswith="converged",
                         predicate=lambda v: v >= 1.0, witness=None)
    assert hits == []
