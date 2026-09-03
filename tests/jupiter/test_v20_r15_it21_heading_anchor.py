"""it.21 -- the 13 refusals RE-ANCHORED BY HEADING, and an instrument that scores a heading.

it.20 opened all 129 citation occurrences in `V20_R15_THEORY_TABLE.md` and scored
116. The other 13 were REFUSED under RULING J-20b: every one points into a file
this round is still writing (`V20_R15_LEAP_LEDGER.md` x10, `V20_R15_JOURNAL.md`
x2, `house-events.jsonl` x1). Each anchor was opened and verified correct at that
census. They were refused for their FILE, not their content.

J-20b named their repair and could not perform it, and said so:

    "Option 1 (cite prose by heading, `P-6`) is recorded as the correct repair for
    each of the 13, but is not the instrument's rule, BECAUSE A HEADING POINTER
    HAS NO LINE TO CHECK."

This file is the missing half: an instrument that gives a heading pointer a LINE.

  RULING J-21a -- A HEADING ANCHOR IS RESOLVED, THEN LANDED.
  An anchor is `(path, section_key, want)`, not `(path, lineno, want)`.
    1. RESOLVE. `section_key` must match EXACTLY ONE heading line in the file.
       Zero matches (section renamed or deleted) or two (section duplicated) is a
       REFUSAL, not a pass -- an unresolvable anchor is scored no better than a
       wrong one.
    2. SCOPE. The section runs from that heading to the next heading of the same
       or shallower level, or EOF. A file with no headings (`house-events.jsonl`)
       has `section_key = None` and the section is the whole file.
    3. LAND. `want` must appear on EXACTLY ONE line inside that section. That line
       number is the resolution, and the it.20 landing check is then applied to it
       verbatim: `want in line_at(path, lineno)`.
  Uniqueness at both steps is the whole of the teeth. A substring resolver that
  took the FIRST match would silently follow a duplicated heading or a repeated
  row into the wrong section and report a line number with full confidence.

  RULING J-21b -- THE 13 ARE RE-ISSUED, NOT REPAIRED.
  Under J-20a, changing a citation's `want` is a WITHDRAWAL plus a NEW CITATION.
  Three of these ten needed a longer `want` to be unique inside their section
  (`H10`, `H18`, `H117`), so they are issued as NEW cids `H*`, each with its own
  frozen want under `HEADING_SEAL`, and `REANCHORS` records which withdrawn cid
  each one replaces. No `C*` in the it.20 census is edited by this file. The it.20
  freeze is imported, not copied, so a later edit to it is RED here too.

  RULING J-21c -- AN ANCHOR CERTIFIES LOCATION, NOT ARGUMENT. RETIRED, NOT FIXED.
  Stated as a limit at it.20 and left untouched by the round. It is stated here as
  a RULING so it stops being a footnote: the landing check, the want freeze, and
  this heading resolver ALL certify that a cited line carries certain words. NONE
  of them certifies that those words SUPPORT the claim the table cell makes from
  them. `C110` is the standing example and it is GREEN under every instrument the
  round owns: the cell says a row commits "the failure of selecting a threshold
  after seeing the data it judges (`MISTAKES.md:451`)", and `MISTAKES.md:451`
  carries "A threshold refitted to the data it judges". The anchor certifies that
  the FAILURE HAS A NAME in the taxonomy. It says nothing about whether THIS ROW
  commits it -- which is the entire claim. No test in this repo can close that,
  because closing it means reading the cited passage and the claim and judging
  entailment, and that is a reader. RETIRE is the honest disposition; what an
  instrument can do is refuse to be mistaken for one, which is why this ruling is
  in the instrument and not only in the report.

COVERAGE REACHED: **129 of 129** occurrences scoreable -- 116 by line under it.20,
13 by heading under this file, 0 unscored. 0 failing in either population.
"""

from __future__ import annotations

import hashlib

from tests.jupiter.test_v20_r15_it20_citation_freeze import (
    CENSUS,
    LIVE_FILES,
    ROOT,
    WITHDRAWN,
    lands,
    line_at,
    table_citations,
)

#: new cid -> (path, section_key or None, want). J-21b: these are NEW citations,
#: not edits of the withdrawn ones. `want` is FROZEN under `HEADING_SEAL`.
HEADING_CENSUS: dict[str, tuple[str, str | None, str]] = {
    "H6": ("house-events.jsonl", None, '"event": "finding_contract"'),
    "H9": ("V20_R15_LEAP_LEDGER.md", "## it.7", "F4** | unattempted"),
    "H10": ("V20_R15_LEAP_LEDGER.md", "## it.7", "**L-4** | **`Lean #21 [S]`**"),
    "H11": ("V20_R15_LEAP_LEDGER.md", "## it.7", "machine-true and domain-empty"),
    "H15": ("V20_R15_JOURNAL.md", "### WHAT it.4 OWES", "measured on the corrected gate"),
    "H17": ("V20_R15_JOURNAL.md", "## it.4", "THE FREEZE"),
    "H18": ("V20_R15_LEAP_LEDGER.md", "## it.9 — JUPITER", "L-14** | **Q6 / W3"),
    "H84": ("V20_R15_LEAP_LEDGER.md", "## it.8", "F1 + const on both wings, but on the wrong object"),
    "H116": ("V20_R15_LEAP_LEDGER.md", "### Format compliance", "TERMINAL rows name no field"),
    "H117": ("V20_R15_LEAP_LEDGER.md", "## it.9 — JUPITER", "calibration / probabilistic forecasting (CRPS, pinball loss)"),
}

#: new cid -> the it.20 cid it replaces. J-21b bookkeeping: the withdrawn cid stays
#: withdrawn forever; this records what took its place.
REANCHORS: dict[str, int] = {f"H{c}": c for c in WITHDRAWN}

#: J-20a, applied to this office's own new pointers. Editing a `want` above without
#: touching this is RED.
HEADING_SEAL = "d8305c39a21b116008d08f459838018448b3bd635ce591432b7686886a4969ef"


def _lines(path: str) -> list[str]:
    return (ROOT / path).read_text(encoding="utf-8", errors="replace").splitlines()


def _level(line: str) -> int:
    s = line.lstrip()
    return len(s) - len(s.lstrip("#")) if s.startswith("#") else 0


def _levels(src: list[str]) -> list[int]:
    """`_level` per line, with every line inside a ``` fence forced to 0.

    RULING J-23a. `_level` counts leading `#` on ANY line, so a `# comment`
    inside a fenced block is a level-1 heading to it and a quoted `## it.7` is a
    second copy of the section. Both directions were struck (MARS, it.22).
    """
    out, fence = [], False
    for l in src:
        if l.lstrip().startswith("```"):
            fence = not fence
            out.append(0)
        else:
            out.append(0 if fence else _level(l))
    return out


def resolve(
    path: str, key: str | None, want: str, lines: list[str] | None = None
) -> int | str:
    """RULING J-21a. A heading anchor's CURRENT line, or a refusal naming why.

    Returns a 1-based line number, or a string beginning `REFUSED` -- never a
    best guess. `lines` overrides the file's content, which is how the append
    negative below simulates growth without writing to a file the round owns.
    """
    src = _lines(path) if lines is None else lines
    if not src:
        return f"REFUSED: {path} is unreadable or empty"
    if key is None:
        start, end = 1, len(src)
    else:
        lvl_of = _levels(src)
        heads = [i + 1 for i, l in enumerate(src) if lvl_of[i] and key in l]
        if len(heads) != 1:
            return f"REFUSED: section key {key!r} matches {len(heads)} headings in {path}, not 1"
        start = heads[0]
        lvl = lvl_of[start - 1]
        end = len(src)
        for i in range(start, len(src)):
            if 0 < lvl_of[i] <= lvl:
                end = i
                break
    hits = [i for i in range(start, end + 1) if want in src[i - 1]]
    if len(hits) != 1:
        return f"REFUSED: {want!r} appears on {len(hits)} lines in section {key!r} of {path}, not 1"
    return hits[0]


def not_landing_by_heading(
    manifest: dict[str, tuple[str, str | None, str]],
    lines: dict[str, list[str]] | None = None,
) -> list[str]:
    """Every heading anchor that does not resolve, or resolves and does not land."""
    out = []
    for cid, (path, key, want) in manifest.items():
        src = (lines or {}).get(path)
        got = resolve(path, key, want, src)
        if isinstance(got, str):
            out.append(f"{cid}: {got}")
            continue
        at = src[got - 1] if src is not None else line_at(path, got)
        if not lands(want, at):
            out.append(f"{cid}: resolved to {path}:{got} which does not carry {want!r}")
    return out


def test_all_thirteen_refusals_resolve_by_heading_and_land() -> None:
    """The it.20 refusals, scored. 13 of 13 occurrences, 0 failing."""
    assert not_landing_by_heading(HEADING_CENSUS) == []


def test_coverage_is_131_of_131_with_nothing_left_unscored() -> None:
    """116 by line + 13 by heading. State the number and do not round it."""
    by_line = {f"{p}:{n}" for c, (p, n, _) in CENSUS.items() if c not in WITHDRAWN}
    occ = table_citations()
    scored_by_line = sum(1 for p, n in occ if f"{p}:{n}" in by_line)
    scored_by_heading = sum(
        1 for p, n in occ if f"{p}:{n}" in {f"{CENSUS[c][0]}:{CENSUS[c][1]}" for c in WITHDRAWN}
    )
    assert scored_by_line == 118  # it.27: +2, M-25b and M-25c each ADD a citation
    assert scored_by_heading == 13
    assert scored_by_line + scored_by_heading == len(occ) == 131


def test_every_withdrawn_cid_has_exactly_one_re_anchor() -> None:
    """J-21b. Nothing is silently dropped and nothing is quietly edited."""
    assert set(REANCHORS) == set(HEADING_CENSUS)
    assert sorted(REANCHORS.values()) == sorted(WITHDRAWN)
    for new, old in REANCHORS.items():
        assert HEADING_CENSUS[new][0] == CENSUS[old][0], f"{new} re-anchored into a different file"
        assert HEADING_CENSUS[new][0] in LIVE_FILES


def test_the_heading_seal_is_intact() -> None:
    """J-20a applied to this office's own new wants."""
    seal = hashlib.sha256(
        "\n".join(
            f"{c}|{HEADING_CENSUS[c][1]}|{HEADING_CENSUS[c][2]}" for c in sorted(HEADING_CENSUS)
        ).encode("utf-8")
    ).hexdigest()
    assert seal == HEADING_SEAL, f"a frozen heading want was edited; recomputed {seal}"


def test_the_heading_anchor_SURVIVES_AN_APPEND_THAT_BREAKS_THE_DIGIT() -> None:
    """PLANTED NEGATIVE for J-21a, and it writes to no file in the tree.

    This is the exact event the INSPECTOR measured (`15:59` green, `16:11` RED,
    test file byte-identical) and the exact event J-20b refused rather than score.
    Simulate it: prepend 40 lines to `V20_R15_JOURNAL.md` in memory, as an
    iteration's append at the head does.

      * the it.20 DIGIT anchor for C15 and C17 stops carrying its want -- the
        failure mode that made them unscoreable;
      * the HEADING anchor for H15 and H17 resolves 40 lines lower and lands.

    The heading anchor is not merely safer than `:650`. It is the one of the two
    that can still be CHECKED after the append, which is the whole claim.
    """
    real = _lines("V20_R15_JOURNAL.md")
    grown = ["APPENDED LINE"] * 40 + real

    before = {c: resolve("V20_R15_JOURNAL.md", *HEADING_CENSUS[c][1:], real) for c in ("H15", "H17")}
    after = {c: resolve("V20_R15_JOURNAL.md", *HEADING_CENSUS[c][1:], grown) for c in ("H15", "H17")}
    assert all(isinstance(v, int) for v in before.values()), before
    assert after == {c: v + 40 for c, v in before.items()}, "the heading did not follow the append"
    assert not_landing_by_heading(
        {c: HEADING_CENSUS[c] for c in ("H15", "H17")},
        {"V20_R15_JOURNAL.md": grown},
    ) == [], "the heading anchor must still land after the append"

    for old in (15, 17):
        path, lineno, want = CENSUS[old]
        assert want not in grown[lineno - 1], (
            f"C{old}'s digit anchor must be BROKEN by the append, or this negative "
            f"proves nothing about what the heading bought"
        )


def test_an_unresolvable_or_ambiguous_heading_is_REFUSED_not_guessed() -> None:
    """PLANTED NEGATIVE for the uniqueness rule at both steps.

    A resolver that took the first match would report a confident line number for
    all three of these.
    """
    real = _lines("V20_R15_LEAP_LEDGER.md")
    dup = list(real) + ["## it.7 — a duplicated section heading"]

    gone = resolve("V20_R15_LEAP_LEDGER.md", "## it.99 — never written", "anything", real)
    assert gone == (
        "REFUSED: section key '## it.99 — never written' matches 0 headings in "
        "V20_R15_LEAP_LEDGER.md, not 1"
    ), gone

    ambiguous = resolve("V20_R15_LEAP_LEDGER.md", "## it.7", "F4", real)
    assert isinstance(ambiguous, str) and ambiguous.startswith("REFUSED:"), (
        f"a want appearing many times in its section must be refused, got {ambiguous}"
    )

    duplicated = resolve("V20_R15_LEAP_LEDGER.md", "## it.7", "F4** | unattempted", dup)
    assert duplicated == (
        "REFUSED: section key '## it.7' matches 2 headings in V20_R15_LEAP_LEDGER.md, not 1"
    ), duplicated
    assert resolve("V20_R15_LEAP_LEDGER.md", "## it.7", "F4** | unattempted", real) == 28


def test_the_instrument_cannot_see_the_ARGUMENT_and_says_so() -> None:
    """RULING J-21c, as an assertion rather than a footnote.

    C110 lands under every instrument the round owns. What it certifies is that
    the named failure EXISTS IN THE TAXONOMY -- not that the cell's row commits
    it, which is the cell's actual claim. The instrument is green and the claim
    is unexamined; both facts are asserted here so neither can be read as the
    other.
    """
    path, lineno, want = CENSUS[110]
    assert (path, want) == ("MISTAKES.md", "A threshold refitted to the data it judges")
    assert want in line_at(path, lineno), "the location check passes"
    cell = (ROOT / "V20_R15_THEORY_TABLE.md").read_text(encoding="utf-8")
    assert (
        "failure of selecting a threshold after seeing the data it judges "
        "(`MISTAKES.md:451`)." in cell
    ), "the cell moved; re-take this ruling against its new wording"
    assert want not in cell, (
        "the cell does not quote the cited line -- it PARAPHRASES the rule and "
        "APPLIES it to a choice this office has not made. The anchor certifies "
        "that the rule sits in the taxonomy at that line. Nothing here, and "
        "nothing in the round, certifies the application. J-21c: RETIRED."
    )
