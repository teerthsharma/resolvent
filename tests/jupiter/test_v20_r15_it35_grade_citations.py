"""it.35 -- the node that binds JUPITER's EVALUATION-FOR-LEAP filing.

Two properties, both of them things the filing would otherwise merely assert:

  * every `path:line` citation in V20_R15_IT35_JUPITER.md LANDS -- the path
    exists and the line is in range;
  * every cell in section 2 carries a verdict token that is EXACTLY `LEAPABLE`
    or `TERMINAL` with no qualifier -- RULING J-14b's property, which the round
    applied to the ledger at V20_R15_THEORY_TABLE.md:353-356 and did NOT apply
    to its own gate table at :326-330.

RED provenance: this node cannot pass without V20_R15_IT35_JUPITER.md, and it
did not exist when the node was written. The standing control is the planted
negative below -- a citation shifted past end-of-file that the checker must
name. Without it, green here would mean only that the parser found nothing.

The regex is the round's, restated rather than imported so this node has no
dependency on another iteration's test module surviving.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILING = ROOT / "V20_R15_IT35_JUPITER.md"

CITE_RE = re.compile(r"`([^`]+?):(\d+(?:-\d+)?|\*)`")
EXT_RE = re.compile(r"\.(py|md|lean|jsonl|json|txt|toml|hs|ipynb|pgn)$")

#: the eight F1/F2/F3 cells the contract sends to the gate. Q1/W1 is F0 and is
#: NOT here; the F4 trio is NOT-PUT under J-14 and is NOT here either.
GATE_CELLS = (
    "Q1/W3", "Q2/W3", "Q3/W1", "Q3/W3",
    "Q4/W1", "Q4/W3", "Q5/W1", "Q5/W3",
)


def citations(text: str) -> list[tuple[str, str]]:
    return [(p, ln) for p, ln in CITE_RE.findall(text) if EXT_RE.search(p)]


def lands(path: str, lineno: str) -> bool:
    f = ROOT / path
    if not f.is_file():
        return False
    if lineno == "*":
        return True
    first = int(lineno.split("-")[0])
    last = int(lineno.split("-")[-1])
    n = len(f.read_text(encoding="utf-8", errors="replace").splitlines())
    return 1 <= first <= n and 1 <= last <= n


def test_every_citation_in_the_it35_filing_lands() -> None:
    text = FILING.read_text(encoding="utf-8")
    cites = citations(text)
    assert len(cites) >= 20, f"parser found only {len(cites)} citations"
    bad = [f"{p}:{ln}" for p, ln in cites if not lands(p, ln)]
    assert bad == [], f"citations that do not land: {bad}"


def test_the_checker_fires_on_a_citation_past_end_of_file() -> None:
    """Planted negative. Without this, green above means nothing."""
    assert not lands("V20_R15_THEORY_TABLE.md", "99999")
    assert not lands("no/such/file.py", "1")
    assert lands("V20_R15_THEORY_TABLE.md", "443")


def test_every_gate_cell_carries_one_unqualified_verdict_token() -> None:
    """RULING J-14b: a hybrid verdict is not a verdict."""
    rows = [
        ln for ln in FILING.read_text(encoding="utf-8").splitlines()
        if ln.startswith("|") and any(f"**{c}**" in ln for c in GATE_CELLS)
    ]
    assert len(rows) == 8, f"expected 8 graded cells, found {len(rows)}"
    for ln in rows:
        cells = [c.strip() for c in ln.split("|")]
        verdicts = [c for c in cells if c in ("**LEAPABLE**", "**TERMINAL**")]
        assert len(verdicts) == 1, f"not exactly one bare verdict token: {ln[:90]}"
    joined = "\n".join(rows)
    for banned in ("TERMINAL as a leap target", "LEAPABLE by", "LEAPABLE, but"):
        assert banned not in joined, f"hybrid verdict form present: {banned}"


def test_the_f4_cells_are_named_ungradeable_and_not_graded() -> None:
    text = FILING.read_text(encoding="utf-8")
    for cell in ("Q2/W1", "Q6/W1", "Q6/W3"):
        assert cell in text, f"F4 cell {cell} not named"
    assert "NOT-PUT" in text
    # the F4 trio must not appear in the graded table of section 2
    graded = [
        ln for ln in text.splitlines()
        if ln.startswith("|") and ("**LEAPABLE**" in ln or "**TERMINAL**" in ln)
    ]
    for ln in graded:
        for cell in ("**Q2/W1**", "**Q6/W1**", "**Q6/W3**"):
            assert cell not in ln, f"F4 cell graded: {ln[:90]}"
