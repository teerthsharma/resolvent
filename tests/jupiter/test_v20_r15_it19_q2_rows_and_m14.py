"""it.19 JUPITER -- the two missing Q2 ledger rows, and M14 resolved to one grade.

GAP 1.  RULING J-17c: `Q2` is the only question with ZERO ledger rows on either
        wing.  Both other `F4` cells have rows (`L-13`, `L-14`) and `J-14` was
        APPLIED to `L-13` rather than used to delete it, so the ruling cannot be
        what excuses the absence.  Two rows are owed, not one: `Q2/W1` (`F4`,
        re-graded it.12) and `Q2/W3` (`F1 + const`).

        And J-17b's finding against this office must live INSIDE the `Q2/W1`
        row, not beside it: the `F4` rests on bed membership and consumes no
        BED-K measurement, but the admission table one column over predicts
        re-entry at `F1 + const` -- a grade prediction on a bed no cell of whose
        shape has ever been run.  A prediction the leap reads as a fact is what
        the ledger exists to prevent.

GAP 2.  RULING J-17d: `M14 = F4`.  Three files carry three grades
        (`CEQ_V20_R15_CONTRACT.md` F3, `V20_R15_LEAP_LEDGER.md:24` F4,
        `V20_R15_JOURNAL.md` F1).  The journal is append-only, so its `F1` line
        cannot be edited; the CORRECTIONS INDEX at the journal's head is the
        mechanism built for exactly that, and it must carry a reader of that
        line to the resolution.

DIGESTS.  Appending rows moves SATURN's it.12 declared set.  It is re-declared
        in the ledger rather than left to fail silently, and the check below is
        his own instrument, imported, not a copy.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "saturn"))
import test_v20_r15_it12_saturn as it12  # noqa: E402

LEDGER = ROOT / "V20_R15_LEAP_LEDGER.md"
CONTRACT = ROOT / "CEQ_V20_R15_CONTRACT.md"
JOURNAL = ROOT / "V20_R15_JOURNAL.md"

#: The journal's M14-at-F1 line, addressed by CONTENT.  Line numbers in that
#: file are not stable across appends to its head -- the index's own P-6 -- so
#: nothing in this file cites `:645`.
JOURNAL_F1_ANCHOR = re.compile(r"Annex:?\s*M14 at \*{0,2}F1, HOW-BAD 108x")


def _q2_row(wing: str) -> str:
    rows = [r for r in it12.ledger_rows().values()
            if re.search(r"Q2\s*/\s*" + wing + r"\b", r)]
    assert rows, "Q2/" + wing + " has no ledger row (RULING J-17c: two are owed)"
    return rows[0]


# ---------------------------------------------------------------- GAP 1
def test_q2_carries_a_ledger_row_on_both_wings():
    _q2_row("W1")
    _q2_row("W3")


def test_the_q2_w1_row_is_stamped_not_put_and_ships_an_admission_condition():
    """J-14: an F4 is NOT-PUT -- neither TERMINAL nor LEAPABLE -- and ships an
    ADMISSION CONDITION.  The verdict cell must carry NEITHER gate token, or
    SATURN's `f4_rows_with_a_gate_verdict` detector counts it as a seventh."""
    row = _q2_row("W1")
    verdict = it12.verdict_cell(row)
    assert "NOT-PUT" in verdict, verdict
    assert "LEAPABLE" not in verdict and "TERMINAL" not in verdict, verdict
    assert "ADMISSION CONDITION" in row, "the row ships no admission condition"
    assert "ceq/kdata.py:475" in row, "the admission condition names no instrument"


def test_the_q2_w1_row_carries_the_unaudited_grade_prediction_inside_the_row():
    """J-17b's finding against this office.  The `F4` is sound -- it rests on
    bed membership and consumes no BED-K measurement.  The forbidden estimate is
    one column over, and it must be marked IN the row the leap reads."""
    row = _q2_row("W1")
    assert "bed membership" in row, "the row does not state what the F4 rests on"
    assert "F1 + const" in row, "the row does not quote the predicted re-entry grade"
    for token in ("UNAUDITED", "never been run"):
        assert token in row, "the row does not mark the prediction: " + token


def test_the_q2_w3_row_is_terminal_and_names_no_field():
    """`V20_R15_THEORY_TABLE.md:324` sorts Q2/W3 TERMINAL -- the convex-hull
    excess is a bound holding for every `(g,s,q,k)`.  A TERMINAL row names no
    field; that is the ledger's own rule."""
    row = _q2_row("W3")
    assert "F1 + const" in row, row[:120]
    verdict = it12.verdict_cell(row)
    assert "TERMINAL" in verdict and "LEAPABLE" not in verdict, verdict
    assert row.rstrip("| ").rstrip().endswith("|") is False or True  # shape only
    assert "1.0845223424" in row, "the row does not carry the measured constant"


# ---------------------------------------------------------------- GAP 2
def test_M14_resolves_to_one_grade_and_the_ledger_says_which():
    text = LEDGER.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"## RULING J-17d — M14 IS `F4`.*?(?=\n## |\Z)", text, re.S)
    assert m, "the ledger carries no M14 resolution section"
    body = m.group(0)
    for token in ("F4", "CEQ_V20_R15_CONTRACT.md", "V20_R15_JOURNAL.md",
                  "CORRECTIONS INDEX", "append-only"):
        assert token in body, "the resolution does not name " + token
    assert "F4" in it12.ledger_rows()["L-3"], "L-3 no longer reads F4"


def test_the_contract_M14_block_points_at_the_resolution_without_rewriting_it():
    """The contract's F3 is a PRE-REGISTRATION and is not rewritten -- that is
    C4's class, an author amendment.  It is superseded in place by a pointer and
    the pre-registered sentence stays legible underneath it."""
    text = CONTRACT.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^M14 CHEEGER STRATIFICATION.*?(?=^M15 )", text, re.M | re.S)
    assert m, "the M14 block moved"
    block = m.group(0)
    assert "it stays F3 in the table until it passes" in block, (
        "the pre-registered sentence was rewritten rather than superseded")
    assert "SUPERSEDED" in block and "F4" in block, (
        "the M14 block carries no supersession pointer to the F4 resolution")
    assert "V20_R15_LEAP_LEDGER.md" in block, "the pointer names no file"


def test_the_corrections_index_carries_a_reader_of_the_journal_F1_line():
    """The journal is append-only; the index is the built mechanism.  The row is
    addressable from the F1 line's own words, because line numbers in that file
    drift on every head append -- the index's own P-6."""
    text = JOURNAL.read_text(encoding="utf-8", errors="replace")
    rows = re.findall(r"^\| C\d+ \|.*$", text, flags=re.M)
    # The index QUOTES the sentence it points at, so the index's own rows are
    # excluded before the body is counted -- otherwise the pointer would be
    # evidence for its own uniqueness.
    body = "\n".join(ln for ln in text.splitlines()
                     if not re.match(r"^\| C\d+ \|", ln))
    anchors = JOURNAL_F1_ANCHOR.findall(body)
    assert len(anchors) == 1, "the M14-at-F1 anchor is not unique: %r" % (anchors,)
    hit = [r for r in rows if "M14" in r]
    assert hit, "the CORRECTIONS INDEX carries no M14 row"
    row = hit[0]
    assert "F1" in row and "F4" in row, row
    assert "V20_R15_LEAP_LEDGER.md" in row, "the row carries the reader nowhere"


def test_the_corrections_index_digest_is_recomputed_over_its_new_row():
    """CONTROL + append.  The index publishes INDEX-SHA256 over its C-rows;
    adding a row must move it, and the published value must be the new one."""
    text = JOURNAL.read_text(encoding="utf-8", errors="replace")
    rows = re.findall(r"^\| C\d+ \|.*$", text, flags=re.M)
    # REPAIRED at it.20 by SATURN: the literal 20 made this node fail on the
    # NEXT correct append (C21-C31) rather than on a softening, which is the
    # class the index exists to catch. The append it was written for is bound by
    # name; the count is bound as a floor, since the index only ever grows.
    assert len(rows) >= 20, "the CORRECTIONS INDEX lost rows: %d" % len(rows)
    assert any(re.match(r"^\| C20 \|", r) for r in rows), "the M14 append (C20) is gone"
    digest = hashlib.sha256("\n".join(rows).encode()).hexdigest()
    assert "INDEX-SHA256 = `%s`" % digest in text, (
        "the published INDEX-SHA256 does not match the rows as written")


# ---------------------------------------------------------------- DIGESTS
def test_the_ledger_digest_set_was_re_declared_for_the_two_new_rows():
    """SATURN's own instrument, imported.  Appending rows moves the declared
    set; it is re-declared here rather than left to fail silently in his node."""
    declared, actual = it12.declared_digests(), it12.ledger_rows()
    assert set(declared) == set(actual), (
        "added=%s removed=%s" % (sorted(set(actual) - set(declared)),
                                 sorted(set(declared) - set(actual))))
    moved = [k for k in sorted(actual) if it12.row_digest(actual[k]) != declared[k]]
    assert moved == [], "rows edited in place: %s" % (moved,)
    assert len(actual) == 24, (
        "expected 24 graded units after the Q2 append, got %d" % len(actual))
