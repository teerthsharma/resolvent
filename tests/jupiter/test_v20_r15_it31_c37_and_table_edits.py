"""it.31 -- C37 SPECIFIED, TWO TABLE EDITS SPECIFIED, AND WHY A CORPUS COUNT ROTS.

RULING J-31a -- A COUNT OVER A CORPUS THAT CONTAINS THE RECORD IS A WASTING ASSET.
  The journal is inside the corpus every instrument in this round measures, so the
  entry that publishes a count is a new occurrence of the thing counted. Two nodes
  broke in one window at it.30 by exactly this: this office's `none` reading (25 ->
  28, three hits at :6532/:6557/:6700, all inside the it.29 write-up) and MERCURY's
  control node (27 -> 28 because a 28th file was WRITTEN, not committed).
  NOT damage, NOT ordering dependence, NOT a count that can be re-frozen.

  THE REPAIR IS A CUT, NOT A NUMBER. it.29's `none` node already carries it (J-30d):
  freeze the count over `body written before this ruling`, and assert separately that
  the whole-body count EXCEEDS it. The frozen half is re-runnable forever; the strict
  inequality is the liveness assertion, and it goes RED if the cut stops working.
  Generalised here as `frozen_prefix_count`, and applied to C37's own scan.

RULING J-31b -- C37. MERCURY withdrew the `62` denominator to `63` in ONE file
  (`V20_R15_IT29_MERCURY.md:26`). The number survives at six addresses, two of them
  in this journal (`:5903`, `:6389`). A withdrawal published in one file is not a
  withdrawal. The row is SPECIFIED here; the coordinator applies it, because adding a
  row moves `dead_literals()` under this office's frozen it.29/it.30 counts.
  The declared literal obeys J-30b: a phrase, not the token `62`.

RULING J-31c -- M-30a. `V20_R15_THEORY_TABLE.md:184` says the formula "gives
  `309.047`". It gives `309.015`; `309.047` is `_sweep(2.0)` on it.13's REMEASURED
  means, a different object; `275` was produced by nothing. Quote the band
  `206 - 537 GPU-s`, point `309.0`, and NO POINT WITHOUT THE BAND. `:354`'s
  un-pointered `~275` is the same defect 170 lines away and moves with it.

RULING J-31d -- M-30b. `:306`'s admission condition asserts "zero callers under
  `scripts/`". FALSE: there are two, `scripts/v20_m14_cheeger.py:347` and `:462`,
  both at the registered kwargs `(n=500, d=4, seed=7)` exactly. THE GRADE DOES NOT
  MOVE: both consume the kernel `b["K"]` and never the bed's data, so "one cell of
  BED-K's shape actually run" is still unsatisfied. `:151` already says the true
  thing -- "zero cells of BED-K's shape have ever been run" -- and survives
  unchanged. The M-30b edit is to the EVIDENCE clause, not to the verdict.

BLAST RADIUS, COMPUTED BEFORE ANY EDIT (2026-09-02, this box): 14 test files read
`V20_R15_JOURNAL.md` -- 6 jupiter, 2 mars, 3 mercury, 3 saturn. Baseline on
unmutated code: `RADIUS_BASELINE`, re-taken this iteration.
"""

from __future__ import annotations

import pathlib
import re

from tests.jupiter.test_v20_r15_it27_star_lands_and_overturns import INDEX_ROW_RE

ROOT = pathlib.Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "V20_R15_JOURNAL.md"
TABLE = ROOT / "V20_R15_THEORY_TABLE.md"

#: The 14 files that read the journal. Any edit to it is re-taken across THIS list
#: or it is not re-taken. This is the list the coordinator did not compute at it.30.
RADIUS = [
    "tests/jupiter/test_v20_r15_it18_citation_landing.py",
    "tests/jupiter/test_v20_r15_it19_q2_rows_and_m14.py",
    "tests/jupiter/test_v20_r15_it20_citation_freeze.py",
    "tests/jupiter/test_v20_r15_it21_heading_anchor.py",
    "tests/jupiter/test_v20_r15_it23_fence_and_argument.py",
    "tests/jupiter/test_v20_r15_it27_star_lands_and_overturns.py",
    "tests/mars_v20/test_it12_the_four_constants.py",
    "tests/mars_v20/test_it22_the_repairs_of_it21.py",
    "tests/mercury/test_v20_r15_it22_independent_census.py",
    "tests/mercury/test_v20_r15_it24_seal_and_manifest_gap.py",
    "tests/mercury/test_v20_r15_it27_uncited_class.py",
    "tests/saturn/test_v20_r15_it14_saturn.py",
    "tests/saturn/test_v20_r15_it20_saturn.py",
    "tests/saturn/test_v20_r15_it22_timer_rearm.py",
]

#: 2026-09-02T19:05 IST, this box, unmutated tree at `207e7b9`: 9 RED / 97 GREEN over
#: RADIUS. A re-take that does not reproduce this pair has moved something other than
#: the cells named in J-31b/c/d.
RADIUS_BASELINE = (9, 97)

#: J-31b. The C37 row, specified. The coordinator applies it verbatim.
C37_ROW = (
    "| 37 | `62` withdrawn to `63` in one file only | it.29 MERCURY withdrew the "
    "uncited-claims denominator and the number survives at six addresses, two of "
    "them in this journal | it.31 - JUPITER | "
    "overturns: `29 uncited claims across 10 of 12 cells, denominator 62` |"
)
C37_LITERAL = "29 uncited claims across 10 of 12 cells, denominator 62"

#: J-31b. Where the `62` denominator survives. `V20_R15_IT29_MERCURY.md` is EXCLUDED
#: by construction: that file is the withdrawal notice, and its `62` is the quotation
#: of what it retires -- J-30c's own rule, applied to someone else's file.
SIXTY_TWO_RE = re.compile(r"(29 of `?62`?|of `62`|denominator 62)")
SIXTY_TWO_SEARCH = [
    "V20_R15_IT26_MERCURY.md", "V20_R15_IT27_MERCURY.md",
    "V20_R15_IT28_MERCURY.md", "V20_R15_JOURNAL.md",
]


def frozen_prefix_count(hits: list[int], cut_heading: str) -> tuple[int, int]:
    """J-31a. `(hits before the cut, hits in the whole body)`. The first is frozen
    forever; the second only ever grows, so the caller asserts the STRICT inequality
    rather than a second constant that would rot the same way. `hits` comes from the
    scan the reading was actually taken with -- it.27's `live_hits` for the `none`
    sentinel -- so the cut is bolted onto the existing instrument, not a new one."""
    body = JOURNAL.read_text(encoding="utf-8").splitlines()
    heads = [i for i, ln in enumerate(body, 1) if ln.startswith(cut_heading)]
    cut = min(heads) if heads else len(body) + 1
    return len([h for h in hits if h < cut]), len(hits)


def substring_hits(needle: str) -> list[int]:
    """J-32c. The STRUCTURAL cut: index rows are not occurrences. `hits_in()` has
    excluded them since it.27; this scan, written later over the same journal, did
    not, so the row recording a withdrawal was scored as an occurrence of the
    withdrawn claim. Same exclusion, one corpus, one rule."""
    body = JOURNAL.read_text(encoding="utf-8").splitlines()
    return [i for i, ln in enumerate(body, 1)
            if needle in ln and not INDEX_ROW_RE.match(ln)]


def section_of(line_no: int, body: list[str]) -> str:
    """J-32c, the other half. The journal grows at BOTH ends: the index prepends
    rows at the head, the entries append at the tail, so every body line number
    drifts by however many rows have been added above it. The index says so itself
    at `:91` -- "Line numbers in this file are not stable across appends to its
    head; headings are." An address is therefore keyed by its section, never by
    its line."""
    heads = [ln.split(" — ")[0].strip()
             for i, ln in enumerate(body, 1) if ln.startswith("## it.") and i <= line_no]
    return heads[-1] if heads else "(above the first section heading)"


def sixty_two_sites() -> list[tuple[str, int]]:
    """Index rows excluded in the journal for the reason above: the C37 row is the
    record of the withdrawal, not a survival of it. J-30c's own rule, which already
    excludes `V20_R15_IT29_MERCURY.md` by construction, applied one level in."""
    out = []
    for name in SIXTY_TWO_SEARCH:
        text = (ROOT / name).read_text(encoding="utf-8")
        for i, ln in enumerate(text.splitlines(), 1):
            if SIXTY_TWO_RE.search(ln) and not (
                    name == "V20_R15_JOURNAL.md" and INDEX_ROW_RE.match(ln)):
                out.append((name, i))
    return out


def whole_body_sections(hits: list[int]) -> list[str]:
    body = JOURNAL.read_text(encoding="utf-8").splitlines()
    return [section_of(i, body) for i in hits]


def journal_sections_of_62() -> list[str]:
    body = JOURNAL.read_text(encoding="utf-8").splitlines()
    return [section_of(i, body) for n, i in sixty_two_sites() if n == "V20_R15_JOURNAL.md"]


# ------------------------------------------------------------------ J-31b, C37

def test_the_withdrawn_62_is_still_live_at_more_than_one_address() -> None:
    """[RUN] THE PREMISE OF C37. It goes GREEN only when every surviving address is
    repaired, which is what a withdrawal was supposed to mean."""
    sites = sixty_two_sites()
    assert sites, (
        "LIVENESS: the scan found no `62` denominator anywhere. Either the "
        "withdrawal fully landed or the pattern rotted -- check "
        "V20_R15_IT26_MERCURY.md:168 before believing the first")
    assert len(sites) > 1, (
        "C37's premise is gone: `62` survives at %d address only; the one-file "
        "withdrawal has become a real one" % len(sites))
    assert journal_sections_of_62() == ["## it.26", "## it.28"], (
        "the surviving journal addresses moved SECTION, which is the only movement "
        "that means anything: %s. The frozen pair `[5903, 6389]` this node used to "
        "assert was RED at `[5907, 6393]` for a reason with no content -- four "
        "index rows went in above them" % (journal_sections_of_62(),))


def test_c37_is_specified_and_its_literal_obeys_the_it30_grammar() -> None:
    """The row handed to the coordinator satisfies J-30b's shape rule, so applying
    C37 cannot turn the it.30 enforcement node RED on a row this office wrote."""
    from tests.jupiter.test_v20_r15_it30_phrase_grammar import phrase_shaped
    assert phrase_shaped(C37_LITERAL)
    assert not phrase_shaped("62"), "the bare token is exactly what J-30b forbids"
    assert C37_LITERAL in C37_ROW and C37_ROW.count("|") == 6


def test_the_c37_literal_is_live_where_the_row_says_it_is() -> None:
    """CONTROL for C37: the declared PHRASE is not a fiction. It is the it.26
    ruling's own sentence, which is why retiring it retires the claim rather than a
    constant. Taken as a prefix count per J-31a, because the row itself, once
    applied, republishes the phrase into the corpus it measures."""
    hits = substring_hits("denominator 62")
    assert whole_body_sections(hits) == ["## it.26"], (
        "the it.26 sentence is live in %s; the reading is keyed by section, so a "
        "move here is a real move" % (whole_body_sections(hits),))


# ------------------------------------------------ J-31a, the wasting-asset ruling

def test_a_whole_body_count_of_none_cannot_be_frozen_and_the_cut_can() -> None:
    """[RUN] THE MECHANISM, MEASURED, on the exact reading that broke. 25 before
    `## it.29`; strictly more over the whole body, because the entry publishing the
    ruling quotes the sentinel. The frozen half is the asset; the strict inequality
    is the liveness assertion."""
    from tests.jupiter.test_v20_r15_it27_star_lands_and_overturns import NONE, live_hits
    before, whole = frozen_prefix_count(live_hits(NONE), "## it.29")
    assert before == 25, (
        "the frozen prefix moved to %d -- that is the RECORD being edited, which is "
        "a real finding, unlike the whole-body count moving" % before)
    assert whole > before, (
        "no `none` after it.29: either the self-reference this ruling is about has "
        "stopped, or the cut is measuring nothing")
    assert whole >= 28, whole


def test_the_wasting_asset_ruling_is_not_true_of_everything() -> None:
    """THE CONTROL. J-31a is narrow: a prefix cut is only worth taking where the
    corpus republishes the thing counted. A string the journal never restates has
    `before == whole`, and for those a plain frozen count is still sound."""
    assert frozen_prefix_count(substring_hits("zzz-no-such-zzz"), "## it.29") == (0, 0)
    before, whole = frozen_prefix_count(substring_hits("CORRECTIONS INDEX"), "## it.1")
    assert whole > 0 and before <= whole


# ---------------------------------------------------- J-31c / J-31d, table edits

def test_m30a_the_theory_table_still_attributes_309_047_to_the_formula() -> None:
    """[RUN] RED until the coordinator applies J-31c. `:184` and `:354` are one
    defect 170 lines apart and are asserted together, so a half-repair stays RED."""
    lines = TABLE.read_text(encoding="utf-8").splitlines()
    l184, l354 = lines[183], lines[353]
    assert "206" in l184 and "537" in l184, "the band is the anchor and it moved"
    assert "own formula gives `309.047`" not in l184, (
        "M-30a: `:184` attributes `309.047` to the formula as written. The formula "
        "as written gives `309.015`; `309.047` is `_sweep(2.0)` on it.13's "
        "remeasured means, a different object. Repair: 'the formula as written "
        "gives `309.015`; the remeasured point is `309.0` on the band "
        "`206 - 537 GPU-s` (`V20_R15_IT13_MERCURY.md:146`)'")
    assert "~275 GPU-s" not in l354, (
        "M-30a: `:354` quotes `~275 GPU-s` with no pointer, and `275` was produced "
        "by nothing. Repair: '206 - 537 GPU-s (point 309.0)' "
        "(`V20_R15_IT13_MERCURY.md:146`) -- no point without the band")


def test_m30b_the_admission_condition_asserts_zero_callers_and_there_are_two() -> None:
    """[RUN] RED until J-31d lands. The FALSE clause and the TRUE one are asserted in
    the same node, so the repair cannot delete the true half by accident."""
    lines = TABLE.read_text(encoding="utf-8").splitlines()
    l306 = lines[305]
    true_clause = "zero cells of BED-K's shape have ever been run"
    true_at = [i for i, ln in enumerate(lines, 1) if true_clause in ln]
    callers = []
    for p in sorted((ROOT / "scripts").rglob("*.py")):
        text = p.read_text(encoding="utf-8", errors="replace")
        for i, ln in enumerate(text.splitlines(), 1):
            if re.search(r"^\s*\w+\s*=\s*bed_k\.build_delay\(", ln):
                callers.append((p.name, i, ln.strip()))
    assert len(callers) == 2, callers
    assert all("n=500, d=4, seed=7" in c[2] for c in callers), callers
    assert true_at == [152], (
        "the TRUE statement of the same fact -- MERCURY cited it as `:151`, it is "
        "at %s -- must survive M-30b unchanged; a table edit that moves it has "
        "deleted the half that was right" % (true_at,))
    assert "zero callers under `scripts/`" not in l306, (
        "M-30b: `:306` asserts zero callers under `scripts/`; there are two, %s, "
        "both at the registered kwargs. THE GRADE DOES NOT MOVE -- both consume the "
        "kernel and never the bed's data, so 'one cell of BED-K's shape actually "
        "run' is still unsatisfied. Repair the EVIDENCE clause only: 'two callers "
        "under `scripts/`, both reading the kernel only, so zero cells of BED-K's "
        "shape have ever been run (`:151`)'" % ([(c[0], c[1]) for c in callers],))


# ------------------------------------------------------- J-32c, the structural cut

def test_PLANTED_NEGATIVE_the_index_row_is_what_the_cut_removes() -> None:
    """[RUN] The cut is load-bearing, and this is the line it removes. Without the
    exclusion the C37 row -- the record OF the withdrawal -- is scored as an
    occurrence of the withdrawn claim, which is J-32c's instance seven. The
    unexcluded scan is rebuilt inline so this node fails if the exclusion is
    deleted from `substring_hits()` AND fails if the row ever stops being there."""
    body = JOURNAL.read_text(encoding="utf-8").splitlines()
    unexcluded = [i for i, ln in enumerate(body, 1) if "denominator 62" in ln]
    excluded = substring_hits("denominator 62")
    removed = sorted(set(unexcluded) - set(excluded))
    assert removed, (
        "the cut removed nothing: either the C37 row has gone from the index or "
        "`substring_hits()` has lost its exclusion")
    assert all(INDEX_ROW_RE.match(body[i - 1]) for i in removed), removed
    assert all(body[i - 1].startswith("| C37 |") for i in removed), removed
    assert section_of(removed[0], body) == "(above the first section heading)", (
        "the index row is supposed to sit ABOVE every `## it.N` heading -- that is "
        "why a CHRONOLOGICAL prefix cut cannot exclude it and a structural one can")


def test_a_chronological_cut_cannot_freeze_a_count_over_a_table_that_grows_at_the_top() -> None:
    """[RUN] J-32c, stated as a measurement rather than as an opinion. Every index
    row lands before the first `## it.N` heading, so `frozen_prefix_count` files
    all 38 of them inside the frozen prefix of EVERY chronological cut, at every
    cut. That is the wrong dimension, and it is why the prefix reading moved from
    1 to 2 on a row this office asked for."""
    body = JOURNAL.read_text(encoding="utf-8").splitlines()
    rows = [i for i, ln in enumerate(body, 1) if INDEX_ROW_RE.match(ln)]
    assert len(rows) >= 38, len(rows)
    for cut in ("## it.1", "## it.29", "## it.30"):
        before, whole = frozen_prefix_count(rows, cut)
        assert before == whole == len(rows), (cut, before, whole)
