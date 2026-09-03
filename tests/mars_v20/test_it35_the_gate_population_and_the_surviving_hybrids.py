"""MARS it.35 — two RED assertions against the UNMUTATED table.

Both read V20_R15_THEORY_TABLE.md as it stands at HEAD. Neither mutates anything.

A. The gate's population is the F1/F2/F3 cells. Section 4.2 is headed "THE NINE
   CELLS THAT DO REACH THE GATE" and lists nine rows, but one of them (Q1/W1) is
   F0 and is annotated "not a failure" in its own row. The gate takes EIGHT.

B. RULING J-14b killed hybrid verdicts in the VERDICT column. Two rows moved the
   hybrid into the GATE-CLASS / FIELD columns, where the ruling does not reach:
   Q3/W3 "LEAPABLE, but ~6 GPU-s buys it outright" and Q5/W3 "LEAPABLE — and it
   is a SCORING RULE, not a theorem". A verdict that names its own non-theorem
   repair in the same cell is the same fork J-14b forbade.
"""
import re
from pathlib import Path

TABLE = Path(__file__).resolve().parents[2] / "V20_R15_THEORY_TABLE.md"

# The at-a-glance grid, section 1. cell -> grade token.
GLANCE = {
    ("Q1", "W1"): "F0", ("Q1", "W3"): "F1",
    ("Q2", "W1"): "F4", ("Q2", "W3"): "F1",
    ("Q3", "W1"): "F2", ("Q3", "W3"): "F1",
    ("Q4", "W1"): "F3", ("Q4", "W3"): "F3",
    ("Q5", "W1"): "F1", ("Q5", "W3"): "F1",
    ("Q6", "W1"): "F4", ("Q6", "W3"): "F4",
}


def _gate_rows(text):
    """Rows of the 4.2 table: (cell, grade, gate class, field)."""
    rows = []
    for line in text.splitlines():
        m = re.match(r"^\|\s*(Q\d/W\d)\s*\|([^|]*)\|([^|]*)\|([^|]*)\|", line)
        if m:
            rows.append(tuple(g.strip() for g in m.groups()))
    return rows


def test_the_gate_population_is_eight_not_nine():
    text = TABLE.read_text(encoding="utf-8")
    in_scope = [c for c, g in GLANCE.items() if g in ("F1", "F2", "F3")]
    assert len(in_scope) == 8, in_scope
    heading = [l for l in text.splitlines() if "CELLS THAT DO REACH THE GATE" in l]
    assert heading, "section 4.2 heading not found"
    # RED: the heading counts nine; the gate's contract takes F1/F2/F3 only.
    assert "NINE" not in heading[0].upper(), (
        f"4.2 is headed {heading[0]!r} but only {len(in_scope)} cells are F1/F2/F3; "
        "Q1/W1 is F0 and its own row says 'not a failure'. The header count and the "
        "gate's population differ by one."
    )


def test_no_gate_row_negates_its_own_verdict_in_the_field_column():
    text = TABLE.read_text(encoding="utf-8")
    rows = _gate_rows(text)
    assert len(rows) >= 8, f"only parsed {len(rows)} gate rows"
    hybrids = []
    for cell, grade, verdict, field in rows:
        blob = f"{verdict} {field}"
        if "LEAPABLE" in verdict and re.search(
            r"not a theorem|buys it outright|GPU-s buys|SCORING RULE", blob
        ):
            hybrids.append((cell, verdict))
    # RED: J-14b resolved hybrids in the verdict column; these survived in the others.
    assert not hybrids, (
        "RULING J-14b forbids a verdict that is both. These rows are LEAPABLE and "
        f"name their own non-theorem repair in the same row: {hybrids}"
    )
