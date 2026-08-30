"""The doc arm must use the census's OWN reading rule, not a near neighbour of it.

RED ON PURPOSE against the sheet as it stands. This file is the bind for a SATURN
error, not someone else's: the iteration-2 spot-check re-measured `workdone2.md` with
an extractor that counted numbers of >= 6 SIGNIFICANT DIGITS, got "0 of 4 readings
reproduce" against a sheet cell of 1/10, could not account for the denominator, and
amended the row anyway -- writing "the provenance ground is EMPTY" and a cell of 0/4
into AUDIT.md. That was the one undisputed KEEP failure of the iteration-2 census.

It was wrong. The census counts numbers written with >= 5 DECIMAL PLACES, deduplicated
by value, and under its own rule `workdone2.md` reads 1/10 -- exactly the cell the
census recorded before SATURN amended it.

WHY THE ITERATION-2 CALIBRATION DID NOT CATCH THIS, which is the transferable part.
That arm calibrated on ONE control, AUDIT.md's 8/23 cell for
`LOOP_PROMPT_ROUND6_ARCHIVE.md`, and reproduced it exactly. Both rules return the same
23 readings on that document. A control that cannot discriminate between the
hypothesis and its rival is not a calibration, however exactly it matches, and picking
the LARGEST available document made it worse rather than better: size bought
confidence, not discrimination. The rule was identified here by scoring a 32-rule grid
against all 64 readings cells outside the iteration-3 draw -- 54/64 exact for
decimals >= 5, against 32/64 for the runner-up and 9/64 for what iteration 2 used.

Written before the sheet was amended; the RED is logged to house-events.jsonl under
the node ids below. When the amendment lands these go green and stand as the guard.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scale.doc_readings import (          # noqa: E402
    DENOM_BAND, pool, readings, reproduce, sheet_cells,
)


@pytest.fixture(scope="module")
def cells():
    got = sheet_cells()
    # Calibrate before asserting. A parser returning {} makes every claim below
    # vacuously true -- MISTAKES.md V-7, the shape this whole round exists to catch.
    assert len(got) >= 70, f"only {len(got)} readings cells parsed; refusing to judge"
    return got


@pytest.fixture(scope="module")
def jpool():
    got = pool()
    assert len(got) > 10_000, f"journal scan found {len(got)} leaves; witness failed"
    return got


def test_the_control_the_iteration_2_arm_used_cannot_discriminate():
    """The calibration that passed and meant nothing, pinned so the lesson survives.

    Both rules return 23 readings on `LOOP_PROMPT_ROUND6_ARCHIVE.md`. Any future arm
    calibrating on this one document alone learns nothing about which rule it has."""
    import re
    from scale.doc_readings import NUM
    text = (ROOT / "LOOP_PROMPT_ROUND6_ARCHIVE.md").read_text(encoding="utf-8")

    def sig(s):
        return len(re.sub(r"[^0-9]", "", s.split("e")[0].split("E")[0]).lstrip("0"))

    by_sig = {float(s) for s in NUM.findall(text) if sig(s) >= 6}
    assert len(by_sig) == len(readings(ROOT / "LOOP_PROMPT_ROUND6_ARCHIVE.md")) == 23, (
        "the two rules no longer agree on the iteration-2 control; the point of this "
        "test is that they DID agree there and disagreed almost everywhere else")


def test_the_census_rule_reproduces_the_sheets_denominators(cells):
    """The identification, scored over every row that carries a readings cell."""
    exact = seen = 0
    outside = {}
    for path, (_num, den) in cells.items():
        f = ROOT / path
        if not f.exists():
            continue
        seen += 1
        got = len(readings(f))
        exact += got == den
        if abs(got - den) > DENOM_BAND:
            outside[path] = (den, got)
    assert seen >= 70, seen
    assert exact / seen >= 0.80, f"identification decayed to {exact}/{seen} exact"
    assert not outside, (
        f"rows whose denominator misses the census rule by more than +-{DENOM_BAND} "
        f"(sheet, measured): {outside}")


def test_workdone2_reads_what_the_census_recorded_before_saturn_amended_it(jpool):
    """THE RED. AUDIT.md's `workdone2.md` cell says 0/4 because the iteration-2 arm
    put it there with the wrong rule. Under the census's own rule it is 1/10, which is
    what the cell said originally, so the provenance ground is NOT empty and the
    iteration-2 finding against this row does not survive its own instrument."""
    hits, reads = reproduce(ROOT / "workdone2.md", jpool)
    assert (len(hits), len(reads)) == (1, 10), (
        f"workdone2.md measures {len(hits)}/{len(reads)} under the census rule; "
        f"the original cell was 1/10")

    row = [ln for ln in (ROOT / "AUDIT.md").read_text(encoding="utf-8").splitlines()
           if ln.startswith("| `workdone2.md`")]
    assert len(row) == 1, f"expected one workdone2.md row, found {len(row)}"
    assert "1/10 readings reproduce" in row[0], (
        "AUDIT.md still carries the iteration-2 measurement made with the "
        ">= 6-significant-digit rule; the census's own rule reads 1/10")
    assert "provenance ground is EMPTY" not in row[0], (
        "AUDIT.md still asserts the provenance ground is empty for workdone2.md; "
        "1 of its 10 readings reproduces at abs=5e-7")
