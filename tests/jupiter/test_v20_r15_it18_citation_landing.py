"""it.18 -- the three citations the it.18 census found that do not LAND.

it.17 censused 123 citations, found 28, repaired all 28, and shipped an
instrument whose MANIFEST holds those 29 repaired pointers. This file exists
because that manifest is the wrong population twice over:

  * the table now holds **128** citations, not 123 -- two it.17 repairs
    replaced one citation with four and one with two;
  * the manifest covers the citations it.17 REPAIRED, not the citations it.17
    WROTE, and two of the three failures below were introduced by those repairs.

Each entry is `(path, lineno, substring)` -- hand-verified against the file at
HEAD, deliberately NOT generated from the table, because a manifest derived from
the thing it checks asserts nothing.

RED until the three repairs are made in V20_R15_THEORY_TABLE.md. Under RULING
J-17e they are made IN PLACE: digits changed inside the line that already holds
them, table length unchanged at 443, so no external pointer into the table moves.

Calibration: `test_the_checker_fires_on_a_wrong_line_inside_a_file_that_exists`
shifts a citation that DOES land by +10 -- it still resolves, the it.14 checker
stays green on it, and this checker must still name it. Without that, green here
would mean only that the parser found nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "V20_R15_THEORY_TABLE.md"

#: the true location each failing citation must be repaired TO, with a verbatim
#: substring of the line it must land on.
REPAIRS = {
    "C17": ("V20_R15_JOURNAL.md", 650, "THE FREEZE"),
    "C64": ("ceq/arm_smprime.py", 409, 'm["smp_values"] = values'),
    "C103": ("V20_R15_IT13_MERCURY.md", 189, "print(list(kdata.BED_SPECS))"),
}

#: the wrong pointer each one currently carries. Every one of these RESOLVES --
#: that is the entire point; resolvability never saw any of them.
WITHDRAWN = {
    "C17": "V20_R15_JOURNAL.md:648",
    "C64": "scripts/v15_r1.py:801",
    "C103": "V20_R15_IT89_INSPECTOR.md:63",
}

# it.24: one regex for the round, imported rather than restated -- the same move
# J-23c made for the landing predicate. it.24 taught it two notations (`:A-B` and
# `:*`), and a copy here would have read the population as 120 and called it 129.
CITE_RE = re.compile(r"`([^`]+?):(\d+(?:-\d+)?|\*)`")
EXT_RE = re.compile(r"\.(py|md|lean|jsonl|json|txt|toml|hs|ipynb|pgn)$")


def line_at(path: str, lineno: int) -> str:
    lines = (ROOT / path).read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[lineno - 1] if 1 <= lineno <= len(lines) else ""


def not_landing(text: str, manifest: dict[str, tuple[str, int, str]]) -> list[str]:
    """Every manifest entry whose cited line does not carry its substring."""
    bad = []
    for cid, (path, lineno, want) in manifest.items():
        if f"`{path}:{lineno}`" not in text:
            bad.append(f"{cid}: {path}:{lineno} is not cited in the table")
        elif want not in line_at(path, lineno):
            bad.append(f"{cid}: {path}:{lineno} does not carry {want!r}")
    return bad


#: The census population, RE-MEASURED at HEAD after this iteration's repairs.
#: It has moved twice for the same reason: a citation repaired by SPLITTING is
#: a citation added. it.17 took 123 -> 128 (one pointer into four, one into
#: two); C64's repair here took 128 -> 129, because the ROUTE needed a second
#: citation for a second claim. This constant carries the iteration that
#: measured it precisely because it is not a constant of the table.
POPULATION_AT_IT18 = 129


def test_the_population_is_129_and_the_number_is_dated():
    """123 named the pre-it.17 table. Nothing about the table pins this count."""
    text = TABLE.read_text(encoding="utf-8")
    cites = [(p, n) for p, n in CITE_RE.findall(text) if EXT_RE.search(p)]
    assert len(cites) == POPULATION_AT_IT18, f"census population moved: {len(cites)}"


def test_every_true_location_carries_what_the_table_claims():
    """The targets were right before the table was touched. GREEN now."""
    for cid, (path, lineno, want) in REPAIRS.items():
        assert want in line_at(path, lineno), f"{cid}: {path}:{lineno} lacks {want!r}"


def test_the_three_repaired_pointers_are_present_in_the_table():
    """RED until the repairs land."""
    text = TABLE.read_text(encoding="utf-8")
    assert not_landing(text, REPAIRS) == []


def test_no_withdrawn_pointer_survives_in_the_table():
    """RED until the repairs land. A pointer can be added and the old one left."""
    text = TABLE.read_text(encoding="utf-8")
    assert [c for c, s in WITHDRAWN.items() if f"`{s}`" in text] == []


def test_the_wrong_file_pointers_name_a_file_that_lacks_the_symbol_entirely():
    """C64 and C103 are mechanism 5, not line slips: the cited FILE has nothing.

    This is what separates them from an off-by-a-row and it is why renumbering
    the digits cannot repair either one.
    """
    assert "smp_values" not in (ROOT / "scripts/v15_r1.py").read_text(encoding="utf-8")
    assert "BED_SPECS" not in (ROOT / "V20_R15_IT89_INSPECTOR.md").read_text(encoding="utf-8")


def test_the_checker_fires_on_a_wrong_line_inside_a_file_that_exists():
    """PLANTED NEGATIVE. Not a missing file -- the failure it.14 already caught.

    Shift a citation that LANDS by +10. It still resolves, so the it.14 checker
    stays green on it. This checker must name it.
    """
    good = {"PN": ("CEQ_V20_R15_CONTRACT.md", 123, "state distribution exists")}
    text = TABLE.read_text(encoding="utf-8")
    assert not_landing(text, good) == [], "the control citation must land unmutated"

    shifted = {"PN": ("CEQ_V20_R15_CONTRACT.md", 133, "state distribution exists")}
    mutated = text.replace("`CEQ_V20_R15_CONTRACT.md:123`", "`CEQ_V20_R15_CONTRACT.md:133`")
    assert mutated != text, "the mutation did not apply"
    assert (ROOT / "CEQ_V20_R15_CONTRACT.md").read_text(
        encoding="utf-8").count("\n") >= 133, "the shifted line must still RESOLVE"
    assert not_landing(mutated, shifted) != [], "the checker missed a wrong line"
