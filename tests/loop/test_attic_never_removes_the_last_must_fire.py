"""A module's last rejection-region calibration cannot be retired silently.

THE ROW. `AUDIT.md:446` classes `tests/cameron/test_hankel_mustfire.py` vacuous /
ATTIC. Its printed reason is:

    "PRESUMED vacuous-by-scope (L-SCOPE): builds its own tensor/graph instead of
     drawing from the production batch path"

That sentence is presumption P1, and P1 was killed in round 10 iteration 1. MARS
measured that it keys on WHO built the input while the repo's own V-14 rule keys
on WHAT PATH the input takes, and that every planted positive is constructed --
so P1 fires on cures as readily as diseases. It was replaced by P1', then P1''.
The withdrawal never reached this row's reason cell.

THE CONFIRMATION HAS THE SAME HOLE. The iteration-2 spot-check re-measured the row
and confirmed it dead against a three-part conjunction, one leg of which is
"0 `test_claim_*` functions". MERCURY independently measured that exact predicate
blind in the same iteration: 13 `tests/foreman` files carry an in-band
`# CLAIM AS WRITTEN -- RED` banner and only 10 use the `test_claim_*` prefix, so
the classifier "cannot see a banner, only a name". This file names its instruments
`test_mustfire_*`. It has five of them, plus a `test_selftest_*` that checks the
Nerode instrument reports a gap where there is none. A must-fire battery was
scored as having no refutation instrument because it did not spell them the one
way the rule can read.

WHAT IS ACTUALLY LOST. `ceq.hankel` is imported by five tracked files, two of them
production: `scale/eprocess.py` and `scale/negation_scope.py`. Of the three test
files that import it, only this one fires the instrument in BOTH directions --
`test_mustfire_no_gap_on_*` and `test_mustfire_gap_on_*`. The other two carry one
count-tracking assertion and two bar-fires nodes, and the bar is not the Hankel
instrument. Retiring this row leaves `ceq.hankel` with no measured rejection region
while production code still calls it.

WHAT THIS TEST CLAIMS, AND WHAT IT DOES NOT. It does not claim the ATTIC verdict is
wrong -- the conjunction may well hold on a predicate that can see must-fires. It
claims a kill shipped without a replacement route: nothing in the sheet names where
`ceq.hankel`'s rejection region is measured after this file is retired. Under the
round's standing rule, a kill with no route is an incomplete report, and the three
legal shapes are reroute (name the surviving calibration), reprice (state what it
costs to build one), or retire (state the measured reason the calibration is not
needed). Any of the three turns this green. Doing nothing does not.

SCOPE. Measured blast radius is one row of the 29 ATTIC `.py` rows, so this is a
targeted guard, not a sweep. It is written generally anyway because the predicate
that produced it is general.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUDIT = ROOT / "AUDIT.md"

#: A node that drives an instrument to its rejection region. Deliberately NOT the
#: `test_claim_*` prefix -- that predicate is the one measured blind this round.
MUST_FIRE_NODE = re.compile(
    r"^\s*def (test_mustfire\w*|test_\w*must_fire\w*|test_selftest\w*)\s*\(", re.M
)

#: The module whose calibration the drawn row carries.
CALIBRATED_MODULE = "ceq.hankel"


def audit_rows() -> list[tuple[str, str, str]]:
    """(path, class, disposition) for every classed row; the 3-cell journals are skipped."""
    rows = []
    for line in AUDIT.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*`([^`]+)`\s*\|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 7:
            continue
        rows.append((m.group(1), cells[4], cells[5]))
    return rows


def attic_py_rows() -> list[str]:
    return [p for p, _c, d in audit_rows() if d == "ATTIC" and p.endswith(".py")]


#: Where iteration 4's P0.3 move puts a retired row. The sheet keeps naming the row
#: by its ORIGINAL path, so a guard that resolves only that path stops seeing the
#: file the instant the move it guards is executed.
ATTIC_PREFIX = "attic"


def resolve(rel: str) -> pathlib.Path | None:
    """The file at its sheet path, or where the attic move would have put it."""
    for candidate in (ROOT / rel, ROOT / ATTIC_PREFIX / rel):
        if candidate.exists():
            return candidate
    return None


def must_fire_nodes(rel: str) -> list[str]:
    """Must-fire nodes in a row's file, wherever the file currently lives.

    The first draft returned `[]` for a path that does not resolve. Measured before
    iteration 4 ran: after the attic move, `tests/cameron/test_hankel_mustfire.py`
    becomes `attic/tests/cameron/test_hankel_mustfire.py`, `must_fire_nodes` returns
    `[]`, and this guard PASSES -- silently, on every moved row, at exactly the
    moment the kill it objects to is carried out.

    That is a control with no rejection region against the only event it exists to
    catch: MISTAKES.md's V-class, committed inside a guard written against the
    opposite failure. `resolve()` looks in both places, and a row whose file is in
    neither now RAISES rather than reporting nothing, because a sheet row naming a
    file that exists nowhere is itself a defect and must not read as clean.
    """
    path = resolve(rel)
    if path is None:
        raise FileNotFoundError(
            f"AUDIT.md row {rel!r} resolves to no file, at its own path or under "
            f"{ATTIC_PREFIX}/. A row naming a file that exists nowhere cannot be "
            "checked and must not be reported as having nothing to lose."
        )
    return MUST_FIRE_NODE.findall(path.read_text(encoding="utf-8"))


def importers_of(module: str) -> list[str]:
    """Tracked files that import the module, by git grep -- not by a literal scan.

    The first draft of this used one over-built alternation and returned the empty
    set, which `test_the_calibrated_module_still_has_production_importers` caught
    on its first run. That is the guard doing its job on its own author: an
    absence from a search that cannot find what it seeks is not evidence of
    absence (MISTAKES.md V-7). The pattern below is the one measured to return
    the five known importers.
    """
    pkg, _, leaf = module.rpartition(".")
    patterns = [module, rf"from {re.escape(pkg)} import .*\b{re.escape(leaf)}\b"]
    out = subprocess.run(
        ["git", "grep", "-l", "-E", "|".join(patterns),
         "--", "tests/*.py", "scale/*.py", "ceq/*.py"],
        cwd=ROOT, capture_output=True, text=True,
    )
    return sorted(out.stdout.split())


def test_the_node_pattern_finds_the_instruments_it_searches_for():
    """Must-fire for this file's own reader. A pattern that matches nothing would
    pass every assertion below by vacuity -- the MISTAKES.md V-7 shape."""
    nodes = must_fire_nodes("tests/cameron/test_hankel_mustfire.py")
    assert len(nodes) >= 5, (
        f"the reader found {len(nodes)} must-fire nodes in a file named "
        "test_hankel_mustfire.py; the pattern is broken, not the sheet"
    )


def test_the_node_pattern_does_not_match_an_ordinary_test():
    """Must-not-fire. Without this, a pattern that matched every `def test_` would
    make the verdict below meaningless."""
    assert not MUST_FIRE_NODE.findall(
        "def test_two_dof_lemma_holds():\n    pass\n"
    ), "the pattern matches ordinary tests, so a hit carries no information"


def test_the_calibrated_module_still_has_production_importers():
    """Premise. If nothing imports ceq.hankel any more, retiring its calibration
    costs nothing and this guard should be deleted rather than satisfied."""
    importers = importers_of(CALIBRATED_MODULE)
    production = [f for f in importers if f.startswith(("scale/", "ceq/"))]
    assert production, (
        f"{CALIBRATED_MODULE} has no production importer; this guard is obsolete"
    )


@pytest.mark.parametrize("rel", attic_py_rows())
def test_an_atticked_file_is_not_the_last_must_fire_for_a_live_module(rel: str):
    """THE DEFECT. One case per ATTIC row so a repair reads as a partial pass."""
    nodes = must_fire_nodes(rel)
    if not nodes:
        return  # nothing to lose; the overwhelming majority of ATTIC rows
    reason = next((r for p, _c, _d in audit_rows() for r in [p] if p == rel), rel)
    assert False, (
        f"{rel} is classed ATTIC and carries {len(nodes)} must-fire nodes "
        f"({', '.join(nodes[:3])}...). Its printed reason is presumption P1, "
        "withdrawn in iteration 1, and the conjunction that confirmed it uses the "
        "`test_claim_*` prefix, measured blind in iteration 2. A kill must ship a "
        "replacement route: name the surviving calibration for the module this file "
        "fires (reroute), price building one (reprice), or state the measured reason "
        "none is needed (retire). Any of the three turns this green. "
        f"(row: {reason})"
    )


#: The exact wording of presumption P1, withdrawn at round 10 iteration 1. MARS
#: measured that it keys on WHO built the input while the repo's own V-14 rule keys
#: on WHAT PATH the input takes -- and every planted positive is constructed, so P1
#: has no discriminating power and fires on cures as readily as diseases. It was
#: replaced by P1', then by P1''.
WITHDRAWN_P1 = "builds its own tensor/graph instead of drawing from the production batch path"


def test_the_withdrawn_wording_is_findable_in_the_sheet_at_all():
    """Must-fire for the check below. If the phrase never matched, the assertion
    would pass by vacuity and report a clean sheet it had not read."""
    text = AUDIT.read_text(encoding="utf-8")
    assert WITHDRAWN_P1 in text or "P1'" in text, (
        "neither the withdrawn wording nor its replacement appears in AUDIT.md; "
        "this check is reading the wrong file or the wrong sheet"
    )


def test_no_attic_row_is_justified_by_a_withdrawn_presumption():
    """THE DEFECT. A row may be correctly ATTIC and still cite a retracted rule.

    Measured at the pinned revision f823b02: 8 of 33 ATTIC rows print P1 verbatim as
    their reason. P1 was withdrawn two iterations earlier. The classifications may
    survive -- the iteration-3 spot-check re-measured 5 ATTIC rows DEAD against their
    stated conjunctions -- but the conjunctions carry blind legs of their own (the
    hankel row's included "0 test_claim_*", which MERCURY measured blind in the same
    iteration), and iteration 4 moves all 33 rows on these printed grounds.

    This is the sheet keyed on "the reason text as written" standing in for "the rule
    currently in force". Repair is to re-state each row's ground in the rule that
    actually governs it, not to delete the sentence.
    """
    offenders = [p for p, _c, d in audit_rows() if d == "ATTIC"
                 and WITHDRAWN_P1 in _reason_for(p)]
    assert not offenders, (
        f"{len(offenders)} ATTIC rows cite presumption P1, withdrawn at iteration 1, "
        f"as their reason: {offenders}. Re-state the ground in the governing rule; "
        "a row moved to attic/ on a retracted justification cannot be reviewed later "
        "by reading its own row."
    )


def _reason_for(path: str) -> str:
    """The reason cell for a row, from the shipped sheet."""
    for line in AUDIT.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*`([^`]+)`\s*\|", line)
        if m and m.group(1) == path:
            cells = [c.strip() for c in line.split("|")]
            if len(cells) >= 7:
                return cells[6]
    return ""
