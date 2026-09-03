"""it.33 MERCURY -- the band-only ruling at its own citation target, and CLAUSE D.

RED FIRST, against unmutated code. The one new finding of this iteration is not a
number this office produced; it is a POINTER TARGET. `V20_R15_THEORY_TABLE.md:184`
and `:354` both cite `V20_R15_IT13_MERCURY.md:146` as the authority for
**"206-537 GPU-s, band only -- NO POINT"**. The cited line is this office's own
it.13 RETIRE row, and it reads `**206 - 537** (point 309.0)`. The pointer LANDS and
the line it lands on ASSERTS THE THING THE CITING LINE WITHDRAWS -- at one decimal,
which is JUPITER's it.32 detail: `309.015` and `309.047` both round to `309.0`, so
the rounded point does not even carry the distinction M-30a drew. The withdrawal
was applied in the citing office and not at the address the citation resolves to.

CLAUSE D of the landings census follows: table lines `335-380`, the leap-row block,
disjoint from clause C (`179-192`).
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
TABLE = ROOT / "V20_R15_THEORY_TABLE.md"
IT13 = ROOT / "V20_R15_IT13_MERCURY.md"


def _line(path: pathlib.Path, n: int) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    assert n <= len(lines), "%s has %d lines, pointer names :%d" % (path.name, len(lines), n)
    return lines[n - 1]


# --------------------------------------------------------------------------
# M-33a. RED. The citation target contradicts the citing claim.
# --------------------------------------------------------------------------


def test_m33a_the_band_only_claim_cites_a_line_that_quotes_the_point() -> None:
    """RED. Both table addresses that carry the it.32 ruling cite it.13:146."""
    citing = [184, 354]
    for n in citing:
        ln = _line(TABLE, n)
        assert "V20_R15_IT13_MERCURY.md:146" in ln, "the citing line moved: %d" % n
        assert "no point" in ln.lower(), "the citing line no longer claims no-point: %d" % n
    target = _line(IT13, 146)
    assert "206" in target and "537" in target, "the target is not the band row: %r" % target
    assert "point 309.0" not in target, (
        "the address cited for `band only, NO POINT` is the one line still quoting a "
        "point: V20_R15_IT13_MERCURY.md:146 -> %r. The withdrawal was applied in the "
        "citing office and not at the target it resolves to." % target
    )


def test_m33a_control_the_pointer_resolves_and_the_reader_is_capable() -> None:
    """GREEN control. Discriminates `the target is clean` from `the reader is blind`."""
    assert "point 626.1" in _line(IT13, 148), (
        "the reader cannot see a parenthesised point at all -- M-33a is void"
    )
    assert IT13.exists() and TABLE.exists()


# --------------------------------------------------------------------------
# CLAUSE D of the landings census. Count, not a rate. No extrapolation.
# --------------------------------------------------------------------------

# Table lines 335-380, the leap-row block. Disjoint from clause C (179-192), whose
# 14 pointers are NOT re-counted here. Extracted by opening every cited line.
CLAUSE_D = (
    (335, "V20_R15_IT567_INSPECTOR.md:482", "as the head noun, which is what the clause requires"),
    (339, "tests/saturn/test_v20_r15_it12_saturn.py:319", "def inadmissible_leapable_rows()"),
    (346, "tests/saturn/test_v20_r15_it12_saturn.py:329", 'if head.startswith("none")'),
    (347, "V20_R15_LEAP_LEDGER.md:99", "TERMINAL rows name no field, which is the rule."),
    (354, "V20_R15_IT13_MERCURY.md:146", "206"),
    (368, "V20_R15_LEAP_LEDGER.md:130", "**L-13**"),
    (380, "tests/saturn/test_v20_r15_it12_saturn.py:343", "KNOWN_INADMISSIBLE = {"),
)


def test_clause_d_every_pointer_in_the_leap_row_block_lands() -> None:
    """`7 of 7` LAND. `53 -> 60 of 129`. One of the seven is M-33a's: it resolves
    to a line that contradicts the claim citing it, and it is counted as a LANDING
    because the pointer resolves -- the defect is in the target's text, not in the
    address, and this census scores addresses."""
    failed = []
    for table_line, ptr, expect in CLAUSE_D:
        rel, n = ptr.rsplit(":", 1)
        p = ROOT / rel
        if not p.exists():
            failed.append((table_line, ptr, "missing file"))
            continue
        text = _line(p, int(n))
        if expect not in text:
            failed.append((table_line, ptr, text[:80]))
    assert failed == [], "clause D pointers that do not land: %s" % (failed,)
    assert len(CLAUSE_D) == 7
    assert len({t for t, _, _ in CLAUSE_D} & set(range(179, 193))) == 0, (
        "clause D overlaps clause C -- the 14 would be double-counted"
    )


def test_clause_d_control_the_reader_can_fail() -> None:
    """GREEN control. A wrong line number under a real file must not pass."""
    text = _line(ROOT / "V20_R15_LEAP_LEDGER.md", 100)
    assert "TERMINAL rows name no field, which is the rule." not in text, (
        "the reader returns the same text for :99 and :100 -- it is not reading lines"
    )
