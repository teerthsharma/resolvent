"""it.12 SATURN — three instruments, one file.

A.  THE GOVERNING LAW SEARCH.  `CEQ_V20_R15_CONTRACT.md:58` names
    `L-GRADE (F0-F4 + HOW-BAD gap)` among laws "all standing", so it should
    predate this round.  The searcher below looks for a DEFINITION of a named
    law, and it is calibrated on both sides before its zero is read as
    absence: it must FIND `L-FLOOR` and `L-CERT` (defined at
    `CEQ_V20_R15_CONTRACT.md:60,64`), it must find a rubric PLANTED into a
    temp file, and only then does its zero for `L-GRADE` mean anything.
    `V20_R15_IT1_SATURN.md:309` and `MISTAKES.md` already carry this class:
    a search structurally incapable of finding the thing, returning zero,
    read as evidence of absence.

B.  THE LEDGER LINE-LEVEL BASELINE.  `V20_R15_LEAP_LEDGER.md` was edited in
    place at constant 197 lines during the it.10 audit and a whole-file digest
    "cannot say which row".  Per-row digests make the next in-place edit
    LOCATABLE, not merely detectable.  Same instrument as the it.4 wing
    manifest (`tests/saturn/test_v20_r15_freeze_manifest.py`).

C.  THE FIELD RULING, AS A TEST.  Granted at it.9: a LEAPABLE grade naming a
    THEOREM or a BED rather than a FIELD is inadmissible.  A ruling with no
    test is a ruling that decays, and this one has to survive to it.35.

D.  THE `[RUN]` CENSUS AND ITS FORWARD-ONLY GATE.  A `[RUN]` marker that
    cannot name its command is the same class as an event with no `t`
    (`scale/ledger.py:208`), so it gets the same treatment: refused by
    construction on the way in, counted on the way back.
"""
from __future__ import annotations

import hashlib
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEDGER = ROOT / "V20_R15_LEAP_LEDGER.md"
CONTRACT = ROOT / "CEQ_V20_R15_CONTRACT.md"


# --------------------------------------------------------------------------
# A.  THE LAW SEARCH
# --------------------------------------------------------------------------

#: Where a standing law would be written down if it were written down.
LAW_CORPUS = (
    "CONTRACT.md", "CEQ_V16_CONTRACT.md", "CEQ_V15_CONTRACT.md",
    "CEQ_V15_1_DELTA.md", "CEQ_V15_2_DELTA.md", "CEQ_V15_3_DELTA.md",
    "CEQ_V20_R15_CONTRACT.md", "MISTAKES.md", "workdonenew.md",
    "workdonenewseal.md", "V15_LEDGER.md", "V17K_RULINGS.md",
    "LOOP_PROMPT.md", "LOOP_PROMPT_ROUND2_ARCHIVE.md",
    "LOOP_PROMPT_ROUND3_ARCHIVE.md", "LOOP_PROMPT_ROUND4_ARCHIVE.md",
    "LOOP_PROMPT_ROUND5_ARCHIVE.md", "LOOP_PROMPT_ROUND6_ARCHIVE.md",
    "LOOP_PROMPT_ROUND7_ARCHIVE.md", "STRUCK.md", "DONE.md", "BOARD.md",
    "LOOP.md", "AUDIT.md", "CHECKLIST.md",
)


def law_definition_sites(law: str, texts: dict[str, str]) -> list[str]:
    """Lines that DEFINE `law`, not lines that merely cite it.

    A definition is a mention of the law's name followed, on the same line,
    by prose that is not just another law name -- i.e. the name is the
    subject of the line rather than an item in a list.  This is the
    discriminator that separates `CEQ_V20_R15_CONTRACT.md:60` (` L-FLOOR
    every capability number ships...`) from `:58` (`... L-FIND, L-GRADE
    (F0-F4 ...), L-LEAP,`), and it is the whole search.
    """
    hits: list[str] = []
    # The law name is the SUBJECT of its defining line: it opens the line,
    # after markdown furniture only.  `CEQ_V20_R15_CONTRACT.md:60` reads
    # " L-FLOOR  every capability number ships..."; `:58` reads
    # "... L-FIND, L-GRADE (F0-F4 + HOW-BAD gap), L-LEAP," -- mid-list, and a
    # parenthetical gloss inside a list of names is a citation, not a rule.
    head = re.compile(r"^[\s*|>`#-]*" + re.escape(law) + r"\**\s*(?P<rest>.*)$")
    for name, text in texts.items():
        for i, line in enumerate(text.splitlines(), 1):
            m = head.match(line)
            if not m:
                continue
            rest = m.group("rest").strip(" \t*|:=-")
            if len(rest.split()) < 4:
                continue
            hits.append(f"{name}:{i}")
    return hits


@pytest.fixture(scope="module")
def corpus() -> dict[str, str]:
    out = {}
    for name in LAW_CORPUS:
        p = ROOT / name
        if p.is_file():
            out[name] = p.read_text(encoding="utf-8", errors="replace")
    return out


def test_the_searcher_finds_laws_that_exist(corpus):
    """POSITIVE CONTROL. Zero is only evidence from a searcher that can find."""
    for law in ("L-FLOOR", "L-CERT"):
        sites = law_definition_sites(law, corpus)
        assert sites, (
            f"{law} is defined at CEQ_V20_R15_CONTRACT.md and the searcher "
            f"missed it; its zero for any other law is therefore meaningless")
    assert any(s.startswith("CEQ_V20_R15_CONTRACT.md") for s in
               law_definition_sites("L-FLOOR", corpus))


def test_the_searcher_finds_a_planted_rubric(tmp_path):
    """PLANTED POSITIVE. A rubric written down IS found by this searcher."""
    planted = tmp_path / "PLANTED.md"
    planted.write_text(
        "LAWS: all standing + L-INST, L-GRADE, L-LEAP\n"
        "L-GRADE  every theory exit carries F0 exact, F1 bounded with the\n"
        "         constant, F2 partial, F3 failed instance, F4 unattempted.\n",
        encoding="utf-8")
    texts = {"PLANTED.md": planted.read_text(encoding="utf-8")}
    sites = law_definition_sites("L-GRADE", texts)
    assert sites == ["PLANTED.md:2"], sites


def test_L_GRADE_has_no_definition_in_the_law_corpus(corpus):
    """THE FINDING.  L-GRADE is cited as standing law and never written down."""
    sites = law_definition_sites("L-GRADE", corpus)
    assert sites == [], (
        "L-GRADE now has a definition site -- update V20_R15_IT12_SATURN.md "
        f"and the reconstruction: {sites}")
    # And the bare citation IS present, so the law is invoked-but-undefined
    # rather than simply absent from the round's vocabulary.
    assert "L-GRADE" in corpus["CEQ_V20_R15_CONTRACT.md"]


def test_L_GRADE_occurs_exactly_once_in_the_whole_tree():
    """The strongest form of the finding, and it needs no discriminator at
    all: the string `L-GRADE` occurs ONCE anywhere under the repo root, and
    that once is the citation at `CEQ_V20_R15_CONTRACT.md:58`.  A definition
    cannot be hiding in a file the law corpus above does not list, because
    there is no second occurrence to hide in."""
    # This iteration's own artifacts are commentary ABOUT the absence; counting
    # them would make the test self-defeating from the moment it is written.
    mine = {"test_v20_r15_it12_saturn.py", "V20_R15_IT12_SATURN.md",
            "house-events.jsonl"}
    skip = {".git", "__pycache__", ".venv", "node_modules"}
    sites = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {
                ".md", ".py", ".txt", ".json", ".jsonl", ".lean", ".sh", ".yml", ".ini"}:
            continue
        # `V20_R15_IT12_*.md` is this iteration's cohort: JUPITER filed his
        # it.12 report while this test was being written and it discusses the
        # absence too. Excluding the cohort by prefix is disclosed in
        # V20_R15_IT12_SATURN.md rather than left as a silent filter.
        if skip & set(p.parts) or p.name in mine or p.name.startswith("V20_R15_IT12_"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if "L-GRADE" in line:
                sites.append(f"{p.relative_to(ROOT).as_posix()}:{i}")
    # it.14 REPAIR. This node was written as `sites == [the one citation]` and
    # went red two iterations later on `V20_R15_IT13_*.md` -- reports DISCUSSING
    # the absence. A raw occurrence count is not the property claimed; the claim
    # is that no line DEFINES the law. Re-stated over the discriminator, applied
    # tree-wide, so a later report citing `L-GRADE` cannot break it and a later
    # report DEFINING it must.
    assert "CEQ_V20_R15_CONTRACT.md:58" in sites, (
        f"the one known citation of L-GRADE has moved: {sites[:5]}")
    # Occurrences in a LAW-BEARING file are the ones that could be a rule. Every
    # other occurrence is a round report or the journal discussing the absence,
    # and those multiplied from 0 to 15 in two iterations without a definition
    # appearing. The filter is by file class, disclosed here, not by name.
    in_law_files = [s for s in sites
                    if pathlib.Path(s.rsplit(":", 1)[0]).name in set(LAW_CORPUS)]
    assert in_law_files == ["CEQ_V20_R15_CONTRACT.md:58"], in_law_files
    defining = law_definition_sites("L-GRADE", {
        n: (ROOT / n).read_text(encoding="utf-8", errors="replace")
        for n in {s.rsplit(":", 1)[0] for s in in_law_files}})
    assert defining == [], f"L-GRADE is defined after all, at {defining}"


def test_F0_through_F4_are_never_characterised_together(corpus):
    """The scale itself is undefined: no line in the corpus glosses F2 or F3."""
    prose = {n: t for n, t in corpus.items()}
    for grade in ("F2", "F3"):
        pat = re.compile(rf"\b{grade}\b\s*(?:means|=|:=|is defined|stands for)")
        hits = [f"{n}:{i}" for n, t in prose.items()
                for i, ln in enumerate(t.splitlines(), 1) if pat.search(ln)]
        assert hits == [], f"{grade} now has a gloss: {hits}"


# --------------------------------------------------------------------------
# B.  THE LEDGER PER-ROW BASELINE
# --------------------------------------------------------------------------

#: `| **L-9** | ... |` -- a graded row in one of the ledger's tables.
ROW_RE = re.compile(r"^\|\s*\*\*(L-[A-Z]?\d+)\*\*\s*\|(.*)\|\s*$")
#: The it.7 VENUS row is a key/value block, not a table row; it is graded all
#: the same and the Inspector counts it among the inadmissible four.
KV_FIELD_RE = re.compile(r"^\|\s*\*\*the field \(never the theorem\)\*\*\s*\|(.*)\|\s*$")
KV_GRADE_RE = re.compile(r"^\|\s*\*\*grade\*\*\s*\|\s*\*\*(\w+)\*\*\s*\|\s*$")

DIGEST_RE = re.compile(r"^(L-[A-Z]?\d+|V-it7)\s+([0-9a-f]{16})\s*$", re.M)
DIGEST_BLOCK = re.compile(r"```ledger-rows\n(.*?)```", re.DOTALL)


def ledger_rows(text: str | None = None) -> dict[str, str]:
    """Row id -> normalised row text.  Whitespace-normalised so a reflow is
    not reported as a content edit; everything else is covered."""
    text = LEDGER.read_text(encoding="utf-8", errors="replace") if text is None else text
    rows: dict[str, str] = {}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = ROW_RE.match(line)
        if m:
            rows[m.group(1)] = " ".join(line.split())
            continue
        m = KV_FIELD_RE.match(line)
        if m:
            # Pair it with the nearest preceding grade line: one graded unit.
            grade = ""
            for back in range(i - 1, max(i - 12, -1), -1):
                g = KV_GRADE_RE.match(lines[back])
                if g:
                    grade = g.group(1)
                    break
            rows["V-it7"] = " ".join((grade + " " + line).split())
    return rows


def row_digest(row: str) -> str:
    return hashlib.sha256(row.encode("utf-8")).hexdigest()[:16]


def declared_digests() -> dict[str, str]:
    m = DIGEST_BLOCK.search(LEDGER.read_text(encoding="utf-8", errors="replace"))
    assert m, (
        "V20_R15_LEAP_LEDGER.md carries no ```ledger-rows digest block. The "
        "it.10 Inspector's finding stands: an in-place edit is detectable by "
        "a whole-file hash and not locatable to a row.")
    return dict(DIGEST_RE.findall(m.group(1)))


def test_every_ledger_row_matches_its_declared_digest():
    declared, actual = declared_digests(), ledger_rows()
    assert set(declared) == set(actual), (
        f"rows added/removed since the freeze: "
        f"added={sorted(set(actual) - set(declared))} "
        f"removed={sorted(set(declared) - set(actual))}")
    moved = [k for k in sorted(actual) if row_digest(actual[k]) != declared[k]]
    assert moved == [], f"rows edited in place since the freeze: {moved}"


def test_the_row_digest_names_the_row_that_moved():
    """PLANTED NEGATIVE.  The it.10 complaint was 'a digest cannot say which
    row'.  Mutate one row in memory; the instrument must name L-9 and only
    L-9 -- detection is not the bar, location is."""
    text = LEDGER.read_text(encoding="utf-8", errors="replace")
    mutated = text.replace("**L-9** | **Q4 / W1 + W3**",
                           "**L-9** | **Q4 / W1 + W2**", 1)
    assert mutated != text, "the planted edit did not apply; the test is vacuous"
    declared, actual = declared_digests(), ledger_rows(mutated)
    moved = [k for k in sorted(actual) if row_digest(actual[k]) != declared[k]]
    assert moved == ["L-9"], moved


def test_the_declared_digest_covers_every_row_in_the_file():
    """A freeze over a subset is not a freeze.  21 table rows + the it.7 VENUS
    key/value row = 22 graded units; the count is derived, never asserted as a
    literal, because a hardcoded denominator is what got it.3 struck."""
    rows = ledger_rows()
    # it.14: the line that stood here ended in a disjunct that is true whenever
    # the ledger has any row at all -- the `or True` class this office's own
    # it.12 census counted in two other suites, shipped inside the census's own
    # file. Removed rather than repaired: the real coverage claim is the
    # assertion below, and the left-hand comparison counted rows against
    # themselves.
    assert set(declared_digests()) == set(rows), (
        f"the digest block does not cover every graded row: "
        f"undigested={sorted(set(rows) - set(declared_digests()))} "
        f"stale={sorted(set(declared_digests()) - set(rows))}")


# --------------------------------------------------------------------------
# C.  THE FIELD RULING
# --------------------------------------------------------------------------

#: This round's own nouns.  The it.9 ruling, in the Inspector's words: "Naming
#: a field and then saying what you want from it is compliant; naming the want
#: and calling it a field is not."  A field head noun built out of the round's
#: own vocabulary is the want wearing the field's name.
ROUND_NOUNS = ("gate", "corner descent", "frac_gate", "lambda_hat", "wing",
               "arm_", "bed-", " bed", "cell", "floor", "seed", "nrmse")


def field_cell(row: str) -> str:
    cells = [c.strip() for c in row.strip("| ").split("|")]
    return cells[-1] if cells else ""


def verdict_cell(row: str) -> str:
    cells = [c.strip() for c in row.strip("| ").split("|")]
    return cells[-2] if len(cells) >= 2 else ""


def field_head(field: str) -> str:
    """The head noun: everything before the first em-dash, which is where the
    compliant rows stop naming the discipline and start stating the want."""
    field = field.strip()
    if field.startswith("**"):
        end = field.find("**", 2)
        if end != -1:
            field = field[2:end]
    return re.split(r"—|--", field, 1)[0].strip(" *`").lower()


def inadmissible_leapable_rows() -> dict[str, str]:
    out = {}
    for rid, row in ledger_rows().items():
        if rid == "V-it7":
            verdict, field = "LEAPABLE", field_cell(row)
        else:
            verdict, field = verdict_cell(row), field_cell(row)
        if "LEAPABLE" not in verdict:
            continue
        head = field_head(field)
        if head.startswith("none") or not head:
            out[rid] = f"LEAPABLE over a `none` field: {head[:60]!r}"
        elif any(n in head for n in ROUND_NOUNS):
            out[rid] = f"field head is this round's own vocabulary: {head[:60]!r}"
    return out


#: Frozen at it.12.  The Inspector counts four; this list is the census, and
#: the test fails on a NEW violation *and* on an unrecorded repair.
#: The Inspector counts four (V-it7, L-2, L-9, L-14).  The detector finds
#: those four and L-13, which carries the identical shape as L-14 -- verdict
#: "TERMINAL as stated; LEAPABLE only after a bed change", field cell opening
#: "none for the metric".  Five, not four, and the fifth is recorded rather
#: than rounded away.
KNOWN_INADMISSIBLE = {"V-it7", "L-2", "L-9", "L-13", "L-14"}


def test_the_FIELD_ruling_is_enforced_row_by_row():
    found = inadmissible_leapable_rows()
    assert set(found) == KNOWN_INADMISSIBLE, (
        f"the inadmissible set moved. new={sorted(set(found) - KNOWN_INADMISSIBLE)} "
        f"repaired={sorted(KNOWN_INADMISSIBLE - set(found))}; details={found}")


def test_the_FIELD_detector_fires_on_a_planted_violation():
    """PLANTED NEGATIVE.  A compliant row rewritten to name a theorem must be
    caught; otherwise the four above are found by accident."""
    compliant = ("| **L-99** | x | **F1** | y | **LEAPABLE** | "
                 "**realization theory for linear time-invariant systems** — what is wanted |")
    assert not any(n in field_head(field_cell(compliant)) for n in ROUND_NOUNS)
    planted = compliant.replace(
        "**realization theory for linear time-invariant systems**",
        "**the theorem that the gate lands on the corner**")
    head = field_head(field_cell(planted))
    assert any(n in head for n in ROUND_NOUNS), head


# --------------------------------------------------------------------------
# D.  THE `[RUN]` GATE
# --------------------------------------------------------------------------

def test_ledger_append_refuses_a_run_claim_with_no_command(tmp_path):
    """FORWARD-ONLY.  `scale/ledger.append` already refuses an event with no
    `t` by construction (`scale/ledger.py:231-240`).  A `[RUN]` claim with no
    runnable command is the same class, and gets the same refusal."""
    from scale import ledger

    p = tmp_path / "e.jsonl"
    with pytest.raises(ValueError, match="run"):
        ledger.append({"t": "finding", "run": "[RUN] the annex comparison"}, p)
    assert not p.exists() or p.read_text(encoding="utf-8") == ""

    line = ledger.append(
        {"t": "finding", "run": "pytest tests/saturn/test_v20_r15_it12_saturn.py"}, p)
    assert "pytest" in line


def test_run_census_is_reproducible():
    """The census is a script, not a paragraph."""
    from scripts import saturn_run_census as c

    rep = c.census(ROOT)
    assert rep["markers"] > 0 and rep["runnable"] + rep["unrunnable"] == rep["markers"]
    # SATURN is not exempt: this office's own files are in the denominator.
    assert any(o.startswith("SATURN") for o in rep["by_office"]), rep["by_office"]
