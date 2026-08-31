"""SATURN P0.1 -- the census must not ATTIC an instrument whose RED is its deliverable.

THE FAILURE THIS EXISTS TO PREVENT, with its cost. Presumption P1' classifies a control
as vacuous-by-scope when its planted input does not enter at the production front door.
Applied to the round-10 census it condemned 27 test files, and FIVE of those are
refutation instruments -- files that exist to REFUTE a claim, whose failing IS the result.

`tests/foreman/test_signedness_is_not_new.py` is the sharpest case. It reimplements SimA
(Koohpayegani & Pirsiavash, arXiv 2206.08898) and Differential Transformer (Ye et al.,
arXiv 2410.05258) FROM THEIR DEFINING EQUATIONS and races them on this module's own
instrument, to answer "is SIGNEDNESS alone enough to be new?". It is the prior-art
defence of the whole contribution. The census marked it vacuous and ATTIC.

WHY P1' MISFIRES HERE, which is the part that generalises. A refutation instrument
constructs its own input BY NECESSITY: the mechanism it races is a competitor's, and a
competitor has no production batch path in this repository. Building SimA from its paper
is the entire method, not a shortcut around one. P1' keys on where the input enters; for
these files the relevant scope is the CLAIM, not the batch. This is the third presumption
in this census to fail by keying on a surface feature instead of the instrument's role --
after P1 (keyed on who built the input, killed by MARS) and P3 (keyed on a literal path,
killed by tests/saturn/test_journal_path_is_discoverable.py).

RED FIRST. Against AUDIT.md as it stands this test FAILS, naming all five.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUDIT = ROOT / "AUDIT.md"

#: A refutation instrument, identified by the repo's own convention rather than by taste:
#: a `test_claim_*` function asserts the optimistic claim, so its FAILURE is the finding.
#: DONE_ARCHIVE_ROUND1.md:1013 records exactly these names under a leading "RED:".
CLAIM_DEF = re.compile(r"^def test_claim_", re.M)


def _refutation_instruments() -> list[str]:
    out = []
    for p in sorted((ROOT / "tests").rglob("test_*.py")):
        rel = p.relative_to(ROOT).as_posix()
        if CLAIM_DEF.search(p.read_text(encoding="utf-8", errors="replace")):
            out.append(rel)
    return out


def _audit_rows() -> dict[str, tuple[str, str]]:
    """path -> (class, KEEP/ATTIC), parsed from the shipped sheet."""
    rows = {}
    for line in AUDIT.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 7:
            continue
        rows[cells[1].strip("`")] = (cells[4], cells[5])
    return rows


def test_the_audit_sheet_is_parseable_and_populated():
    """Premise bind. A test over an empty parse is its own vacuous control."""
    rows = _audit_rows()
    assert len(rows) >= 300, f"AUDIT.md parsed to only {len(rows)} rows; the parser drifted"


def test_at_least_one_refutation_instrument_exists():
    """Must-fire arm. A rule over an empty set proves nothing."""
    found = _refutation_instruments()
    assert found, "no test_claim_* files found; the assertion below would range over nothing"


@pytest.mark.parametrize("path", _refutation_instruments())
def test_a_refutation_instrument_is_never_atticked(path: str):
    """RED. An instrument whose RED is its deliverable cannot be retired for being red.

    Its input is constructed because the mechanism it races is a competitor's, and a
    competitor has no production path here. That is the method, not a defect in scope.
    """
    rows = _audit_rows()
    assert path in rows, f"{path} is a refutation instrument absent from AUDIT.md entirely"
    cls, ka = rows[path]
    assert ka != "ATTIC", (
        f"{path} is classed {cls}/{ka} in AUDIT.md. It defines test_claim_* functions, so "
        f"its FAILING is its finding, not evidence that it is vacuous. P1' condemned it for "
        f"building its own input; a refutation instrument must build its own input, because "
        f"the mechanism it races is a competitor's and has no production batch path in this "
        f"repository. Reclassify by the instrument's ROLE, not by where its tensor came from.")
