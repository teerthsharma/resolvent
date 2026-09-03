"""SATURN it.20 -- three instrument repairs, each with a planted negative.

REPAIR 1  The CORRECTIONS INDEX stopped at `C19`/`C20` while the body ran to
          `CORRECTION 31`. A broken lookup on the round's own anti-staleness
          instrument, failing silently in the trusting direction. The index is
          extended, and the missing check -- *the highest `C`-number in the body
          has a row in the index* -- is the node below. Without it the index
          falls behind on every iteration that writes a correction.

REPAIR 2  `test_M14_carries_three_grades_in_three_files` bound the presence of
          three TEXTS, not three LIVE grades, so the it.19 resolution -- which
          superseded in place, the only move a pre-registration permits -- walked
          past it. Replaced by `live_M14_grades` in the it.14 file; the planted
          negative here rebuilds the pre-resolution text and shows the retired
          node blind on it while the new one fires.

REPAIR 3  MARS's it.18 strike-2 regression test went green for a reason unrelated
          to the bug: SATURN re-parametrized the found-wing node from arm names to
          wing ids, MARS's hardcoded arguments went stale, and the node raised
          identically with and without the mutation. Re-bound to the node's own
          `parametrize` list AND to a differential against the true rows.

Run:  python -m pytest tests/saturn/test_v20_r15_it20_saturn.py -x -q
"""
from __future__ import annotations

import hashlib
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "V20_R15_JOURNAL.md"

#: A body correction: `### CORRECTION 31 -- ...` or an inline `**CORRECTION 12,`.
BODY_CORRECTION_RE = re.compile(r"^#{1,6} CORRECTION (\d+)|^\*\*CORRECTION (\d+)[,.]", re.M)
#: An index row. Identical to the pattern the journal publishes for the digest.
INDEX_ROW_RE = re.compile(r"^\| C\d+ \|.*$", re.M)
INDEX_NUM_RE = re.compile(r"^\| C(\d+) \|", re.M)
DECLARED_SHA_RE = re.compile(r"INDEX-SHA256 = `([0-9a-f]{64})`")


def body_correction_numbers(text: str) -> set[int]:
    return {int(a or b) for a, b in BODY_CORRECTION_RE.findall(text)}


def index_row_numbers(text: str) -> set[int]:
    return {int(n) for n in INDEX_NUM_RE.findall(text)}


def index_digest(text: str) -> str:
    """The journal's own published recompute, verbatim in effect. Cited by its
    heading -- CORRECTIONS INDEX -- not by line: the it.20 append moved the command
    block from `:68-70` to `:87-89`, which is this index's own `P-6`."""
    return hashlib.sha256("\n".join(INDEX_ROW_RE.findall(text)).encode()).hexdigest()


# --------------------------------------------------------------------------
# REPAIR 1
# --------------------------------------------------------------------------

def test_the_highest_body_correction_has_an_index_row():
    """The check that was missing. The index is the round's guard against stale
    entries; a reader who looks up a correction that has no row is told *no*, and
    is told it silently. This binds the whole range, and the highest number by
    name, because the highest is the one every append breaks first."""
    text = JOURNAL.read_text(encoding="utf-8", errors="replace")
    body, rows = body_correction_numbers(text), index_row_numbers(text)
    assert body, "no CORRECTION heading found in the body; re-derive the pattern"
    assert max(body) <= max(rows), (
        f"the CORRECTIONS INDEX stops at C{max(rows)} while the body runs to "
        f"CORRECTION {max(body)}: a lookup on the round's own anti-staleness "
        f"instrument, failing silently in the trusting direction")
    assert not (body - rows), (
        f"body corrections with no index row: {sorted(body - rows)}")


def test_the_index_digest_is_declared_over_the_rows_at_head():
    """The one property this instrument already had that works -- the published
    recompute under the CORRECTIONS INDEX heading -- run against the declaration."""
    text = JOURNAL.read_text(encoding="utf-8", errors="replace")
    declared = DECLARED_SHA_RE.search(text)
    assert declared, "the index declares no INDEX-SHA256"
    assert declared.group(1) == index_digest(text), (
        f"declared {declared.group(1)[:16]}... but the rows at HEAD hash to "
        f"{index_digest(text)[:16]}...; the index was edited without re-declaring")


def test_PLANTED_NEGATIVE_dropping_the_top_row_is_caught():
    """The negative for REPAIR 1: delete the highest index row and nothing else.
    Both nodes above must fire -- the coverage node on the gap it exists for, the
    digest node because the row set moved under a fixed declaration."""
    text = JOURNAL.read_text(encoding="utf-8", errors="replace")
    top = max(index_row_numbers(text))
    mutated = re.sub(rf"^\| C{top} \|.*$\n", "", text, count=1, flags=re.M)
    assert mutated != text, "the mutation removed nothing; re-derive the row pattern"

    body, rows = body_correction_numbers(mutated), index_row_numbers(mutated)
    assert max(body) > max(rows), "the coverage check did not fire on the dropped row"
    declared = DECLARED_SHA_RE.search(mutated).group(1)
    assert declared != index_digest(mutated), "the digest did not move with the rows"


# --------------------------------------------------------------------------
# REPAIR 2 -- the planted negative for the it.14 rebind
# --------------------------------------------------------------------------
from tests.saturn import test_v20_r15_it14_saturn as IT14  # noqa: E402


def _pre_resolution() -> tuple[str, str]:
    """The contract and journal as they stood BEFORE the it.19 resolution: the
    `[SUPERSEDED ...]` block gone from the contract's M14 block, the `C20` row
    gone from the index. Nothing else moves."""
    contract = IT14.CONTRACT.read_text(encoding="utf-8", errors="replace")
    contract = re.sub(r"\n\s*\[SUPERSEDED it\.19.*?\]", "", contract, count=1, flags=re.S)
    journal = re.sub(r"^\| C20 \|.*$\n", "",
                     IT14.JOURNAL.read_text(encoding="utf-8", errors="replace"),
                     count=1, flags=re.M)
    return contract, journal


def test_PLANTED_NEGATIVE_the_presence_node_was_blind_to_the_resolution():
    """The negative for REPAIR 2, and the differential that justifies the rebind.

    On the pre-resolution text the RETIRED node's three assertions -- three texts
    in three files -- all still hold, exactly as they hold at HEAD after the
    resolution. It cannot tell the two states apart, which is the defect. The
    replacement reads three live grades there and one here."""
    contract, journal = _pre_resolution()
    assert contract != IT14.CONTRACT.read_text(encoding="utf-8", errors="replace")

    # The retired node's body, run verbatim on the pre-resolution text.
    m = re.search(r"^M14 CHEEGER STRATIFICATION.*?(?=^M15 )", contract, re.M | re.S)
    assert m and "grade F3" in m.group(0)
    l3 = IT14.it12.ledger_rows()["L-3"]
    assert "M14" in l3 and "F4" in l3
    assert re.search(r"Annex:?\s*M14 at \*{0,2}F1", journal)

    before = IT14.live_M14_grades(contract=contract, journal=journal)
    assert set(before.values()) == {"F3", "F4", "F1"}, (
        f"the pre-resolution snapshot no longer carries three live grades: {before}")
    with pytest.raises(AssertionError):
        IT14.test_M14_carries_exactly_one_live_grade(contract=contract, journal=journal)


# --------------------------------------------------------------------------
# REPAIR 3 -- the planted negative for MARS's re-bound strike-2 test
# --------------------------------------------------------------------------
from tests.saturn import test_v20_r15_freeze_manifest as FM  # noqa: E402
from tests.mars_v20 import test_it18_wing_identity_is_self_certified as MARS  # noqa: E402


def test_PLANTED_NEGATIVE_mars_strike_2_fails_without_the_corroboration_node(monkeypatch):
    """The negative for REPAIR 3: put the manifest back in its pre-repair state by
    neutering the only node that corroborates clause (a) from outside clause (a).
    MARS's re-bound test must then FAIL -- that is what a regression test is for.

    It went green before this repair with that node already neutered in effect,
    because his stale arm-name arguments made the found-wing node raise the same
    `AssertionError` on the true rows and on the mutated rows alike."""
    monkeypatch.setattr(FM, "test_every_wings_arm_is_corroborated_outside_clause_a",
                        lambda: None)
    with pytest.MonkeyPatch.context() as mp, pytest.raises(AssertionError):
        MARS.test_the_clause_b_repair_does_not_catch_the_identity_swap(mp)


def test_PLANTED_NEGATIVE_the_stale_arm_name_arguments_are_not_differential():
    """Why the green was worthless, measured rather than asserted: called with
    MARS's hardcoded arm names, the found-wing node raises the SAME way on the
    true rows as under the mutation. A node that fires either way witnesses
    nothing, and it was the only thing his test had left."""
    fn = FM.test_every_frozen_wing_is_found_and_not_merely_named
    stale = ("arm_smprime", "arm_pl")

    def outcome(rows):
        orig = FM.rows
        FM.rows = lambda: rows
        try:
            out = []
            for a in stale:
                try:
                    fn(a)
                    out.append("pass")
                except Exception as exc:
                    out.append(type(exc).__name__)
            return out
        finally:
            FM.rows = orig

    true_rows = FM.rows()
    assert outcome(true_rows) == outcome(MARS._swapped(true_rows)) != ["pass", "pass"], (
        "the stale arguments now discriminate; re-derive the strike")
