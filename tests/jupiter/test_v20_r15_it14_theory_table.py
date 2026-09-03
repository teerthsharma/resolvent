"""it.14 — the frozen theory table is its own instrument.

The table at V20_R15_THEORY_TABLE.md is the leap model's primary input. It gets one
call and cannot ask a follow-up. So the table is checked the way an instrument is
checked, not the way prose is read:

  * every cell present, exactly twelve;
  * the declared grade census matches the cells;
  * every DECLARATION cites a live `path:line` and the symbol is AT that line
    (the round withdrew six Lean citations for being cited by annex number and
    never resolved -- this is the instrument that would have caught it);
  * no DECLARATION names a theorem by annex number alone (M1..M16);
  * every cell carries a non-empty GAP, CENSUS, ROUTE and ARENA field.

Calibration: `test_the_citation_checker_fails_a_planted_bad_citation` plants a
citation that does not resolve and asserts the checker names it. Without that the
green above means only that the parser found nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "V20_R15_THEORY_TABLE.md"

CELL_RE = re.compile(r"^### CELL (Q[1-6])/(W1|W3)\b", re.M)
FIELD_RE = re.compile(r"^- (GRADE|GAP|DECLARATION|CENSUS|ROUTE|ARENA):[ ]*(.*)$")
CITE_RE = re.compile(r"`([^`]+?):(\d+)`")
ANNEX_ONLY_RE = re.compile(r"^\s*M(?:[1-9]|1[0-6])\b[^`]*$")

DECLARED_CENSUS = {"F0": 1, "F1": 5, "F2": 1, "F3": 2, "F4": 3}


def parse_cells(text: str) -> dict[str, dict[str, str]]:
    """Split the table into per-cell field dicts. One pass, no state machine."""
    cells: dict[str, dict[str, str]] = {}
    starts = [(m.start(), f"{m.group(1)}/{m.group(2)}") for m in CELL_RE.finditer(text)]
    for i, (pos, key) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        fields: dict[str, str] = {}
        for line in text[pos:end].splitlines():
            m = FIELD_RE.match(line)
            if m:
                fields[m.group(1)] = m.group(2).strip()
        cells[key] = fields
    return cells


def bad_citations(text: str) -> list[str]:
    """Every `path:line` in the text that does not resolve at HEAD."""
    bad = []
    for path, lineno in CITE_RE.findall(text):
        # A path is recognised by its extension, NOT by containing a slash:
        # the round's law corpus is root-level `.md` files with no slash in them,
        # and a slash test silently skips every one of them. Found by the planted
        # negative below, which is the only reason this line is right.
        if not re.search(r"\.(py|md|lean|jsonl|json|txt|toml|hs|ipynb|pgn)$", path):
            continue  # not a path; e.g. a bare `0.5:1` ratio
        p = ROOT / path
        if not p.is_file():
            bad.append(f"{path}:{lineno} — no such file")
            continue
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        if not (1 <= int(lineno) <= len(lines)):
            bad.append(f"{path}:{lineno} — file has {len(lines)} lines")
    return bad


@pytest.fixture(scope="module")
def table_text() -> str:
    return TABLE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cells(table_text: str) -> dict[str, dict[str, str]]:
    return parse_cells(table_text)


def test_the_table_holds_twelve_cells_two_wings_six_questions(cells):
    want = {f"Q{q}/{w}" for q in range(1, 7) for w in ("W1", "W3")}
    assert set(cells) == want, f"missing {sorted(want - set(cells))}"


@pytest.mark.parametrize("field", ["GRADE", "GAP", "DECLARATION", "CENSUS", "ROUTE", "ARENA"])
def test_every_cell_carries_every_field(cells, field):
    empty = [k for k, v in cells.items() if not v.get(field)]
    assert empty == [], f"{field} empty on {empty}"


def test_the_grade_census_matches_the_cells(cells):
    counted: dict[str, int] = {}
    for k, v in cells.items():
        base = re.match(r"\*?\*?(F[0-4])", v["GRADE"])
        assert base, f"{k}: GRADE does not start with F0..F4: {v['GRADE']!r}"
        counted[base.group(1)] = counted.get(base.group(1), 0) + 1
    assert counted == DECLARED_CENSUS


def test_every_citation_in_the_table_resolves_at_head(table_text):
    assert bad_citations(table_text) == []


def test_no_declaration_names_a_theorem_by_annex_number_alone(cells):
    """The failure mode this office withdrew six citations for."""
    offenders = [k for k, v in cells.items() if ANNEX_ONLY_RE.match(v["DECLARATION"])]
    assert offenders == [], f"annex-number-only DECLARATION on {offenders}"


def test_every_declaration_cites_a_path_line(cells):
    missing = [k for k, v in cells.items() if not CITE_RE.search(v["DECLARATION"])]
    assert missing == [], f"DECLARATION with no path:line on {missing}"


# ---------------------------------------------------------------- calibration

def test_the_citation_checker_fails_a_planted_bad_citation():
    """Planted negative: a citation to a line past the end of a real file."""
    real = ROOT / "CEQ_V20_R15_CONTRACT.md"
    n = len(real.read_text(encoding="utf-8").splitlines())
    planted = f"see `CEQ_V20_R15_CONTRACT.md:{n + 5000}` and `no/such/file.py:1`"
    found = bad_citations(planted)
    assert len(found) == 2, found
    assert any("no such file" in f for f in found)
    assert any(f"has {n} lines" in f for f in found)


def test_the_citation_checker_passes_a_real_citation():
    """Control: the checker is not simply always red."""
    assert bad_citations("`CEQ_V20_R15_CONTRACT.md:58`") == []
