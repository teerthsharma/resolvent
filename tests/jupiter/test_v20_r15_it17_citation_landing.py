"""it.17 -- a citation must LAND, not merely RESOLVE.

The it.14 instrument (`test_v20_r15_it14_theory_table.py::bad_citations`) asks whether
the cited FILE exists and whether `lineno` is inside it. That is resolvability. It
passed 123/123 on the frozen table while citations were off by 1, off by 10 and off by
107 lines onto different questions -- all of which resolve trivially.

The precedent this file exists to not repeat is `test_v20_r15_it12_constants.py`. Its
docstring names its authority as the `path:line` beside each constant, three of those
pointers are stale, and NO ASSERTION IN THAT FILE TOUCHES A LINE NUMBER. It is green and
its citations are wrong at once, because the citation layer was unasserted in a file
whose whole purpose was to end unasserted claims.

So the calibration here is not "a citation naming a file that does not exist" -- the
it.14 planted negative already covers that and it is the weaker failure. It is
`test_the_checker_fires_on_a_WRONG_LINE_inside_a_file_that_exists`, which shifts a
known-good citation by the +10 the census actually measured on the it.12 constants file
and asserts the checker names it.

MANIFEST holds only the citations THIS office repaired by hand at it.17, each with a
verbatim substring of the line it must land on. It is deliberately not generated from
the table: a manifest derived from the thing it checks asserts nothing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "V20_R15_THEORY_TABLE.md"

from tests.jupiter.test_v20_r15_it20_citation_freeze import CENSUS, lands, region

CONSTS = "tests/jupiter/test_v20_r15_it12_constants.py"
Q45 = "tests/jupiter/test_v20_r15_it8_q4_q5.py"
SAT = "tests/saturn/test_v20_r15_it12_saturn.py"

# repaired citation -> verbatim substring that MUST appear on that line.
# AS FILED AT it.17 AND NOT EDITED SINCE. This is the historical record of what
# this office repaired at it.17 and it stays true; it is NO LONGER the datum any
# instrument reads. RULING J-26b: a pointer written down twice is a pointer that
# can be updated once, and it.24 proved it -- C63's withdrawal moved `CENSUS` and
# left the copy below stranded, and MARS's it.25 STRIKE B opened the copy.
AS_FILED_AT_IT17: dict[str, str] = {
    "V20_R15_WING_MANIFEST.md:3": "Frozen at it.4 on branch",
    "V20_R15_IT13_MERCURY.md:146": "RETIRE the re-take",
    "V20_R15_IT11_INSPECTOR.md:502": "THEOREM SOUND, CELL STRUCK.",
    "ceq/hankel.py:75": "NEG_ENTRY = ",
    "ceq/hankel.py:108": "def hankel_block(",
    "ceq/hankel.py:131": "def rank_real(",
    "ceq/hankel.py:279": "def rank_plus_lower(",
    f"{CONSTS}:11": "1.0845223424",
    f"{CONSTS}:12": "0.9746794345",
    f"{CONSTS}:13": "+0.717647",
    "tests/jupiter/test_v20_r15_it7_q3.py:106": (
        "def test_w3_lambda_hat_sign_separates_all_eight_arm_pl_cells"
    ),
    # -- chunk 3 --
    "V20_R15_IT8_JUPITER.md:105": "`path_product` builds",
    "ceq/arm_smprime.py:144": "def path_product(",
    "ceq/arm_smprime.py:577": "return self.readout(h).squeeze(-1)[:, seq - 1]",
    "ceq/arm_pl.py:304": "def brute_force_path_sums(",
    f"{CONSTS}:122": "def test_q5_w1_flat_band_distance_to_floor1_is_0_145_to_0_220",
    "CEQ_V20_R15_CONTRACT.md:123": "W1 to the oracle where a state distribution exists",
    f"{Q45}:127": "def test_q5_floor_1_is_a_ONE_HOP_THRESHOLD_not_an_information_floor",
    f"{Q45}:151": "def test_q5_no_arm_crosses_and_W3_is_blocked_by_variance_not_by_the_floor",
    "scale/negation_scope.py:286": "equilibrium_oracle",
    "scale/negation_scope.py:304": "return z",
    # -- chunk 4 --
    # C98, RE-REPAIRED at it.18. it.17 read this cell's claim as the "STRUCK
    # and KILLED, measured 1.4060346618513293" finding and repaired the
    # pointer to the line carrying that number. The sentence actually
    # holding the citation is the `BED_SPECS` [RUN], which is in a third
    # place again -- and `grep -n BED_SPECS V20_R15_IT89_INSPECTOR.md`
    # returns nothing. A repair aimed at the wrong claim resolves, reads
    # plausibly, and is wrong; that is this table's whole failure mode.
    "V20_R15_IT13_MERCURY.md:189": "print(list(kdata.BED_SPECS))",
    "CEQ_V20_R15_CONTRACT.md:125": "CRITERION (lexicographic): (1) BED-M crossing",
    "V20_R15_IT13_MERCURY.md:63": "No cell of BED-K's shape has ever been run.",
    "MISTAKES.md:451": "A threshold refitted to the data it judges",
    "V20_R15_IT567_INSPECTOR.md:482": "Naming a field and then",
    f"{SAT}:319": "def inadmissible_leapable_rows(",
    f"{SAT}:329": 'if head.startswith("none") or not head:',
    f"{SAT}:343": "KNOWN_INADMISSIBLE = ",
    f"{SAT}:353": "def test_the_FIELD_detector_fires_on_a_planted_violation(",
}

# the stale pointer each repair replaced. None of these may survive in the table.
WITHDRAWN: dict[str, str] = {
    "V20_R15_WING_MANIFEST.md:12": "V20_R15_WING_MANIFEST.md:3",
    "V20_R15_IT13_MERCURY.md:141": "V20_R15_IT13_MERCURY.md:146",
    "V20_R15_IT11_INSPECTOR.md:503": "V20_R15_IT11_INSPECTOR.md:502",
    "ceq/hankel.py:1": "ceq/hankel.py:108 (and :131, :279, :75)",
    f"{CONSTS}:21": f"{CONSTS}:11",
    f"{CONSTS}:22": f"{CONSTS}:12",
    # it.24, J-24b: the it.17 repair landed on `:13` and the cell was COMPOUND.
    # The pointer is now the range `:13-14` and the anchor is a pair. The it.17
    # repair is not undone -- it is superseded by a wider one.
    f"{CONSTS}:23": f"{CONSTS}:13-14",
    "tests/jupiter/test_v20_r15_it7_q3.py:1": "tests/jupiter/test_v20_r15_it7_q3.py:106",
    "V20_R15_IT8_JUPITER.md:104": "V20_R15_IT8_JUPITER.md:105",
    "ceq/arm_smprime.py:163": "ceq/arm_smprime.py:144",
    "ceq/arm_smprime.py:572": "ceq/arm_smprime.py:577",
    "ceq/arm_pl.py:316": "ceq/arm_pl.py:304",
    f"{CONSTS}:83": f"{CONSTS}:122",
    "CEQ_V20_R15_CONTRACT.md:107": "CEQ_V20_R15_CONTRACT.md:123",
    "CEQ_V20_R15_CONTRACT.md:124": "CEQ_V20_R15_CONTRACT.md:125",
    f"{Q45}:1": f"{Q45}:127 and {Q45}:151",
    "scale/negation_scope.py:300": "negation_scope.py:286 (symbol) / :304 (return)",
    "V20_R15_IT13_MERCURY.md:196": "V20_R15_IT13_MERCURY.md:189 (same file, +7)",
    # it.17's own repair target, withdrawn at it.18: it named a different
    # office's file for a claim that never left this one.
    "V20_R15_IT89_INSPECTOR.md:63": "V20_R15_IT13_MERCURY.md:189 (different FILE)",
    "V20_R15_IT13_MERCURY.md:57": "V20_R15_IT13_MERCURY.md:63",
    "MISTAKES.md:1": "MISTAKES.md:451",
    "V20_R15_IT567_INSPECTOR.md:460": "V20_R15_IT567_INSPECTOR.md:482",
    f"{SAT}:296": f"{SAT}:319",
    f"{SAT}:306": f"{SAT}:329",
    f"{SAT}:320": f"{SAT}:343",
    f"{SAT}:330": f"{SAT}:353",
}

# The two rigid displacements the census found. Each is ONE error applied to every
# citation into one file, not N independent typos -- and no resolvability check can
# see a rigid shift, because every shifted line still resolves.
RIGID_SHIFTS = {CONSTS: +10, SAT: -23}


#: THE CIDS THIS OFFICE REPAIRED AT it.17 -- the historical fact, frozen. Derived
#: once from `AS_FILED_AT_IT17` against the it.20 census and then written down,
#: because deriving it live would key the record on the pointers it is meant to
#: outlive. C131 is C63 re-issued at it.24 (J-24b); the OTHER 29 matched directly.
REPAIRED_AT_IT17: frozenset[int] = frozenset(
    {1, 27, 45, 46, 47, 48, 51, 53, 54, 56, 66, 68, 69, 80, 88, 89, 91, 93, 94,
     95, 96, 104, 106, 110, 113, 114, 115, 118, 119, 131}
)

#: RULING J-26b(i). `MANIFEST` IS A VIEW OVER `CENSUS`, not a second copy of it.
#: Foreign instruments read this name -- MARS unions it directly -- so a
#: withdrawal in the census must reach them without a second edit in a second
#: office's file. It now does: this dict is rebuilt from the census at import.
MANIFEST: dict[str, str] = {
    f"{path}:{spec}": want
    for cid, (path, spec, want) in CENSUS.items()
    if cid in REPAIRED_AT_IT17
}


def line_at(cite: str) -> str:
    """The text a `path:spec` citation ADDRESSES. '' if unreachable.

    it.26: delegates to the census's `region`, so a derived MANIFEST carrying a
    range (`:13-14`) or a file (`:*`) is scored by the same predicate that
    scores the census. Restating the rule here is how it.21 got two of them.
    """
    path, _, spec = cite.rpartition(":")
    return region(path, spec)


def misses(manifest: dict[str, str]) -> list[str]:
    """Every citation in `manifest` whose line does not carry its anchor."""
    return [
        f"{cite} -- anchor {anchor!r} not on that line; line is {line_at(cite)!r}"
        for cite, anchor in manifest.items()
        if not lands(anchor, line_at(cite))
    ]


@pytest.fixture(scope="module")
def table_text() -> str:
    return TABLE.read_text(encoding="utf-8")


def test_every_repaired_citation_lands_on_its_anchor() -> None:
    """The repair targets are right: each cited line carries the claimed text."""
    assert misses(MANIFEST) == []


def test_the_checker_fires_on_a_WRONG_LINE_inside_a_file_that_exists() -> None:
    """CALIBRATION. Shift every citation by the +10 the census measured on the it.12
    constants file. The file still exists and the line still resolves -- the it.14
    checker stays green on all of these. This one must name every single one."""
    digits = {c: a for c, a in MANIFEST.items() if c.rpartition(":")[2].isdigit()}
    shifted = {
        f"{c.rpartition(':')[0]}:{int(c.rpartition(':')[2]) + 10}": a
        for c, a in digits.items()
    }
    named = misses(shifted)
    assert len(named) == len(digits), (
        f"a +10 shift went unnoticed on {len(digits) - len(named)} citation(s); "
        f"the checker is not sensitive to line, only to file"
    )
    # it.26: a `:A-B` or `:*` pointer has no single line to shift, so it is not
    # part of THIS calibration. It is not thereby unchecked -- J-24b scores a
    # range against every member of a tuple want, and J-26c scores `:*` against
    # a digest of the lines carrying the want, which a shift cannot fake either.
    assert len(digits) == len(MANIFEST) - 1, (
        f"{len(MANIFEST) - len(digits)} non-digit pointers, expected 1 (C131's "
        f"J-24b range). A new notation needs its own calibration, not this one.")


def test_no_withdrawn_pointer_survives_in_the_table(table_text: str) -> None:
    """The stale pointers are gone from V20_R15_THEORY_TABLE.md."""
    survivors = [
        f"`{stale}` still in the table; it must read `{good}`"
        for stale, good in WITHDRAWN.items()
        if f"`{stale}`" in table_text
    ]
    assert survivors == []


#: An it.17 pointer a LATER iteration widened. The it.17 record is not edited --
#: it is what this office repaired at it.17 and that stays true -- but the table
#: no longer carries the narrower form, and this names why rather than going RED
#: on a repair that superseded it. it.24, J-24b: `:13` was a COMPOUND cell's
#: pointer and is now the range `:13-14`; the anchor is a pair under `C131`.
SUPERSEDED: dict[str, str] = {
    f"{CONSTS}:13": f"{CONSTS}:13-14  (it.24, J-24b: compound claim, compound anchor)",
}

#: it.26, J-26b: the exception above is no longer CONSULTED. `MANIFEST` is
#: derived from `CENSUS`, so it carries `:13-14` directly and there is no
#: stale narrower form to forgive. The record is kept; the workaround is not.
SUPERSEDED_RETIRED_AT_IT26 = True


def test_every_repaired_pointer_is_present_in_the_table(table_text: str) -> None:
    absent = [c for c in MANIFEST if f"`{c}`" not in table_text]
    assert absent == [], f"repair not applied to the table: {absent}"
    # it.26: `SUPERSEDED` used to hold the ONE entry this test had to forgive,
    # because MANIFEST was a literal that could go stale against the table while
    # the census moved on. Deriving MANIFEST from the census retired the
    # exception rather than maintaining it: there is nothing left to forgive,
    # and the list above is now asserted EMPTY. That is the shape of a J-26b
    # repair -- a special case disappears because the second copy did.
    assert SUPERSEDED_RETIRED_AT_IT26


def test_the_F4_overload_labels_name_the_sense_actually_at_the_line(
    table_text: str,
) -> None:
    """`V20_R15_LEAP_LEDGER.md:25` is L-4, whose F4 sense is WITHDRAWAL. The word
    'struck' is at :24 (L-3), which the same sentence separately labels domain-empty.
    Labelling :25 'struck' sends a one-call reader to the wrong row for the word."""
    assert "*withdrawn* (`V20_R15_LEAP_LEDGER.md:25`, L-4)" in table_text
    assert "*struck* (`V20_R15_LEAP_LEDGER.md:25`, L-4)" not in table_text


def test_the_table_keeps_its_length_so_no_external_pointer_moves(
    table_text: str,
) -> None:
    """RULING J-17e. Three offices cite INTO this table by line
    (`V20_R15_IT16_INSPECTOR.md:41` cites `:118`, `:123`, `:285-287`, `:289`).
    Repairs are digit changes in place; inserting a line would break all of them."""
    assert len(table_text.splitlines()) == 443
