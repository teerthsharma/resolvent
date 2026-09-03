"""SATURN it.34 -- the parity guard made bidirectional, and the dating node widened.

REPAIR 1  `MARS-33-C`.  The it.20 guard asserts `not (body - rows)` and
          `max(body) <= max(rows)`.  It never asserts `rows - body`, so an index
          row with no body correction is invisible unless it is the MAXIMUM --
          which is the only reason `C37` was caught at it.32.  Seven rows sit in
          the untested direction right now: `(13,14,15,16,17,18,19)`.

          THE RULING (see V20_R15_IT34_SATURN.md section 2).  Those seven are
          LEGACY and legitimate; `C37` was not.  The two cohorts are separated by
          a fact, not a story: the body's numbered sequence runs `1..12` and then
          jumps to `20`.  `CORRECTION 13`..`CORRECTION 19` were never written --
          zero occurrences in the journal.  The index was installed at it.11 and
          retro-indexed seven findings that had been overturned by ENTRIES rather
          than by numbered headings (all seven "corrected at" it.3..it.11, all
          before the index existed); the body then resumed at `20`, from the
          index's counter, not its own.  So the exemption is exactly those seven,
          MEMBERSHIP FROZEN, and every row from `C20` on must have a body
          correction.  An eighth cannot join silently -- `test_the_legacy_
          exemption_is_frozen_and_an_eighth_cannot_join` is the node that says so.

REPAIR 2  `MARS-33-D`.  The it.32 dating node reads ONE file -- this office's own
          report -- and checks provenance and shape, never recency.  A reading
          copied forward from a prior iteration passes it.  Widened by one
          argument, it convicts across offices, and the first thing it convicts
          is `V20_R15_IT32_JUPITER.md:169`: the declared `9 RED / 97 GREEN`
          radius baseline, stamped `19:05` -- eleven minutes BEFORE the window
          that declares it opened, in a zone the file's own declared `date -u`
          cannot emit.  A baseline outside its declaring window is a reading from
          a different window on a different tree, so `16 - 9 = 7` is a difference
          of two readings, not a deviation.

RED first, verbatim, against unmutated code, at `2026-09-02 19:48:31 IST`:

    $ python -m pytest tests/mars_v20/test_v20_r15_it33_the_repairs_of_it32.py \
        ::test_MARS_33_C_every_index_row_has_a_body_correction -q
    E  AssertionError: index rows with no body CORRECTION: (13, 14, 15, 16, 17,
       18, 19) -- the guard's untested direction, and the C37 class with 7 live
       precedents
    1 failed in 0.59s

Run:  python -m pytest tests/saturn/test_v20_r15_it34_saturn.py -q
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "V20_R15_JOURNAL.md"

# The it.20 patterns, imported rather than re-derived: a guard that reads the
# corpus through a second regex is a guard against a second corpus.
from tests.saturn.test_v20_r15_it20_saturn import (  # noqa: E402
    BODY_CORRECTION_RE, INDEX_NUM_RE, body_correction_numbers, index_row_numbers,
)

#: THE EXEMPTION, FROZEN.  Index rows retro-filed at it.11 for findings the body
#: overturned in an ENTRY and never gave a `### CORRECTION n` heading.  This is a
#: literal, not a computed difference: a computed exemption would absorb the next
#: offender on the iteration it appeared.
LEGACY_ORPHAN_ROWS = (13, 14, 15, 16, 17, 18, 19)


def orphan_rows(text: str) -> tuple[int, ...]:
    """Index rows with no body correction -- the direction the it.20 guard never
    tested -- with the frozen LEGACY cohort removed.  A tuple over NAMES, never a
    cardinality: `MARS-33-B` -- a count cannot see a substitution."""
    rows, body = index_row_numbers(text), body_correction_numbers(text)
    return tuple(sorted(rows - body - set(LEGACY_ORPHAN_ROWS)))


# ==========================================================================
#  REPAIR 1 -- the guard, both directions
# ==========================================================================

def test_the_parity_guard_holds_in_both_directions():
    """`MARS-33-C`.  The it.20 node's own direction, plus the one it never had."""
    text = JOURNAL.read_text(encoding="utf-8", errors="replace")
    rows, body = index_row_numbers(text), body_correction_numbers(text)
    assert rows and body, "re-derive the index/body patterns; the corpus moved"
    assert tuple(sorted(body - rows)) == (), (
        "body corrections with no index row: %r" % (tuple(sorted(body - rows)),))
    assert orphan_rows(text) == (), (
        "index rows with no body CORRECTION, outside the frozen LEGACY cohort "
        "%r: %r -- the `C37` class.  Either write the body entry or the row is "
        "an index into nothing." % (LEGACY_ORPHAN_ROWS, orphan_rows(text)))


def test_the_legacy_exemption_is_frozen_and_an_eighth_cannot_join():
    """The exemption is only as honest as its membership.  Three things are
    asserted, and each one is the reason a later iteration cannot widen it:

    1. every exempted number IS an index row -- an exemption for a row that does
       not exist is a licence held in reserve;
    2. no exempted number appears as a body correction ANYWHERE in the journal --
       the factual basis of the ruling, and it fails the moment someone writes
       `### CORRECTION 15` and makes the exemption a lie;
    3. the cohort is contiguous and ends at `19`, immediately below `C20`, where
       the two sequences joined.  A gap, or an eighth member, is a different
       claim than the one this ruling made.
    """
    text = JOURNAL.read_text(encoding="utf-8", errors="replace")
    rows = index_row_numbers(text)
    missing = tuple(n for n in LEGACY_ORPHAN_ROWS if n not in rows)
    assert missing == (), (
        "the exemption names rows that are not in the index: %r -- an exemption "
        "held for a row that does not exist is a licence, not a ruling" % (missing,))
    live = tuple(n for n in LEGACY_ORPHAN_ROWS if n in body_correction_numbers(text))
    assert live == (), (
        "an exempted number now HAS a body correction: %r -- the ruling's basis "
        "is gone; drop it from LEGACY_ORPHAN_ROWS rather than keeping a stale "
        "exemption" % (live,))
    assert LEGACY_ORPHAN_ROWS == tuple(range(13, 20)), LEGACY_ORPHAN_ROWS
    assert max(LEGACY_ORPHAN_ROWS) + 1 == min(n for n in sorted(rows) if n >= 20), (
        "the cohort no longer abuts C20, where the index and body sequences "
        "joined; the exemption's boundary is its whole justification")


@pytest.mark.parametrize("victim", [21, 29, 36])
def test_PLANTED_NEGATIVE_a_lower_orphan_is_invisible_to_the_it20_guard(victim):
    """MEASURED, not argued.  Delete ONE body correction heading -- not the
    maximum -- and read both guards on the same mutated text.

    The it.20 node stays GREEN on all three of its assertions: `max(body)` is
    still `38`, `max(rows)` is still `38`, and `body - rows` is still empty,
    because a deletion from the body can never put a number INTO that
    difference.  The new node fires and names the row.  That differential is the
    whole of `MARS-33-C`: `C37` was caught for being the maximum, and nothing
    else was ever caught at all."""
    text = JOURNAL.read_text(encoding="utf-8", errors="replace")
    mutated = re.sub(rf"^#+ CORRECTION {victim} .*$\n", "", text, count=1, flags=re.M)
    assert mutated != text, f"CORRECTION {victim} heading not found; re-derive"

    rows, body = index_row_numbers(mutated), body_correction_numbers(mutated)
    # The shipped it.20 guard, verbatim, on the mutation. All GREEN.
    assert max(body) <= max(rows) and not (body - rows), (
        "the it.20 guard fired; the mutation was not in its blind direction")
    # The it.34 guard on the same text.
    assert orphan_rows(mutated) == (victim,), orphan_rows(mutated)


def test_PLANTED_NEGATIVE_an_eighth_orphan_row_is_caught():
    """The other half of the class: a row APPENDED with no body entry -- the
    literal `C37` shape, and the shape the exemption must not absorb."""
    text = JOURNAL.read_text(encoding="utf-8", errors="replace")
    top = max(index_row_numbers(text))
    mutated = text.replace(f"| C{top} |", f"| C{top} |", 1) + (
        f"\n| C{top + 1} | a row that indexes nothing | it.34 | -- | it.34 |\n")
    assert orphan_rows(mutated) == (top + 1,), orphan_rows(mutated)
    # And the exemption cannot swallow it: it is not in the frozen cohort.
    assert top + 1 not in LEGACY_ORPHAN_ROWS


# ==========================================================================
#  REPAIR 2 -- the dating node widened by one argument
# ==========================================================================

#: A reading from `date "+%Y-%m-%d %H:%M:%S %Z"` -- the it.32 pattern, unchanged.
FULL_READING = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [A-Z]{2,5}")
#: `date -u`'s ISO form, which JUPITER's office declares instead.
UTC_READING = re.compile(r"\d{4}-\d{2}-\d{2}T(\d{2}:\d{2}):\d{2}Z")
#: Any clock-shaped token.
#: `\b` is WRONG here and it was this node's first defect: in `T19:05` the `T` is
#: a word character, so there is no boundary and the carried-forward stamp was
#: invisible to the instrument built to find it. Bounded on digits and colons.
ANY_STAMP = re.compile(r"(?<![\d:])(\d{1,2}:\d{2})(?![\d:])")


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")[:2]
    return int(h) * 60 + int(m)


def declaring_window(text: str) -> tuple[int, int]:
    """The span of the readings the report actually took, in minutes.  This is
    the report's OWN window: the first and last time it read a clock."""
    got = [m.group(1) for m in UTC_READING.finditer(text)]
    got += [m.group(0)[11:16] for m in FULL_READING.finditer(text)]
    assert got, "the report declares no clock reading at all"
    return min(_minutes(s) for s in got), max(_minutes(s) for s in got)


def stamps_outside_the_window(path: pathlib.Path) -> list[str]:
    """RULE 3, WIDENED BY ONE ARGUMENT.  The it.32 node hard-coded `REPORT` and
    so could only ever convict its own author.  Taking the path makes the same
    property cross-office, and it is the recency check the it.32 node lacked: a
    stamp outside the span of the report's own readings was not read during the
    window the report declares.  A carried-forward number lands here."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lo, hi = declaring_window(text)
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if "[RETIRED]" in line or "[CARRIED" in line:
            continue
        stripped = UTC_READING.sub("", FULL_READING.sub("", line))
        for m in ANY_STAMP.finditer(stripped):
            if not (lo <= _minutes(m.group(1)) <= hi):
                out.append(f"{path.name}:{i}  {m.group(1)}")
    return out


def test_the_widened_node_convicts_JUPITER_it32s_carried_forward_baseline():
    """`MARS-33-D`, MEASURED on another office's filed report.

    `V20_R15_IT32_JUPITER.md` declares `date -u` and opens its window at `13:46Z` and closes it at `13:49Z`.
    Its `:169` stamp reads `19:05` -- eleven minutes before that window opened,
    and in IST, which `date -u` cannot emit.  `:169` is the declared
    `9 RED / 97 GREEN` radius baseline.  A baseline read outside the window that
    declares it is a reading from a different window on a different tree, so the
    it.32 `16 - 9 = 7` is a difference of two readings, not a deviation.

    Asserted over NAMES -- the exact offending line and stamp -- because a count
    of offenders is blind to substitution (`MARS-33-B`)."""
    jup = ROOT / "V20_R15_IT32_JUPITER.md"
    assert "date -u" in jup.read_text(encoding="utf-8", errors="replace")
    lo, hi = declaring_window(jup.read_text(encoding="utf-8", errors="replace"))
    assert (lo, hi) == (_minutes("13:46"), _minutes("13:49")), (lo, hi)
    assert stamps_outside_the_window(jup) == [
        "V20_R15_IT32_JUPITER.md:7  13:52",    # the declared CLOSE, never read
        "V20_R15_IT32_JUPITER.md:69  13:50",
        "V20_R15_IT32_JUPITER.md:169  19:05",  # the declared radius BASELINE
    ], stamps_outside_the_window(jup)
    # MARS-33-D named four prose stamps; the fourth, `:130  13:49`, is INSIDE the
    # window and this node does not convict it. Three of four, named, not a count.


def test_the_widened_node_is_green_on_this_offices_own_report():
    """The node is not a weapon pointed one way.  Same function, this office's
    report, and it must be clean before the conviction above is admissible."""
    mine = ROOT / "V20_R15_IT34_SATURN.md"
    assert mine.exists(), "the report is the subject; write it before asserting on it"
    assert stamps_outside_the_window(mine) == [], stamps_outside_the_window(mine)


def test_CALIBRATION_the_widened_node_fires_on_a_carried_forward_reading(tmp_path):
    """The it.32 node's own hole, planted and caught.  A report whose readings
    all sit inside one window, plus one number copied forward from the previous
    iteration.  Provenance and shape are both perfect -- it is a full reading in
    the declared format -- and only recency separates them."""
    p = tmp_path / "REPORT.md"
    p.write_text(
        "2026-09-02T13:46:00Z  window opens\n"
        "2026-09-02T13:52:00Z  window closes\n"
        "the declared baseline, 19:05, carried from the prior iteration\n"
        "quoted from it.31 at 11:00  [CARRIED FORWARD]\n", encoding="utf-8")
    assert [s.split()[-1] for s in stamps_outside_the_window(p)] == ["19:05"]
