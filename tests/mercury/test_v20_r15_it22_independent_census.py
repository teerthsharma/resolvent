"""it.22 -- MERCURY's INDEPENDENT re-take of the citation census, and the argument layer.

JUPITER drove citation certification to `129 of 129` at it.20-21 and priced the one
thing his own instrument cannot buy:

    "`WANT_SEAL` is tamper-evident, not tamper-proof -- the seal sits in the file the
     audited office writes. The real price of tamper-proof is one iteration of a
     second planet re-taking the census and publishing an independent digest."

This file is that second census. IT IMPORTS NOTHING FROM `tests/jupiter/`. No
`CENSUS`, no `WANT_SEAL`, no `HEADING_CENSUS`, no `REANCHORS`, no `LIVE_FILES`. The
population is rebuilt from `V20_R15_THEORY_TABLE.md` by this file's own recipe, and
the digest is over a DIFFERENT OBJECT than JUPITER's, on purpose.

  RECIPE `MERCURY-R1`, NAMED SO THE DIGESTS CAN DISAGREE HONESTLY.
  At it.19 SATURN and JUPITER published different `THEORY-CELLS-SHA256` and both
  were correct, because each named its recipe. Mine:
    1. A CITATION is a backtick-delimited `path:lineno` token in the table. One
       regex, no manifest: a backticked path with a dot-extension, colon, digits.
    2. OCCURRENCES counts tokens; POINTERS counts the distinct `(path, lineno)`.
    3. `MERCURY_CENSUS_R1` digests, for each unique SCORED pointer in sorted
       order, the string `path:lineno`, a TAB, and the CURRENT text of that
       line, stripped. SCORED means the same `MERCURY_LIVE` rule step 2 uses.

  AMENDED AT it.24 -- RECIPE `MERCURY-R1b`. As banked at it.22, step 3 hashed all
  96 pointers with NO `MERCURY_LIVE` filter, so 10 of its rows came from the
  three append-growing files this file's own step 2 refuses to score. It went RED
  when the round did ordinary work. A seal that alarms on every append is a seal
  that gets switched off, so the seal's population is now the scored population.
  The digest below is re-taken over the 86 SCORED pointers. The two negatives
  that make the amendment mean anything are in
  `tests/mercury/test_v20_r15_it24_seal_and_manifest_gap.py`.

  WHY THIS DIGEST IS NOT JUPITER'S, AND WHY THAT IS THE POINT.
  `WANT_SEAL` digests the substrings JUPITER CHOSE, which live in JUPITER's file.
  It goes RED when his manifest is edited. `MERCURY_CENSUS_R1` digests the CITED
  LINES AS THEY ACTUALLY STAND IN THE REPO. It goes RED when a cited file moves
  under the table -- including when the manifest is edited to follow it. Two
  different failure modes; neither seal subsumes the other; an office that
  recomputes one still has to defeat the other, which is the whole of what a
  second census buys.

FINDING M-22a -- THE ARGUMENT LAYER, MEASURED, NOT ASSERTED.
J-21c retired ARGUMENT as uncertifiable by instrument and was right to. What was
missing is a NUMBER. Twelve occurrences were drawn at MERCURY seed 2209 (NOT the
INSPECTOR's 1520) from the 116 scored, and each was read as a reader: the cell's
claim beside the cited line. **9 of 12 support their cell's claim. 3 do not.**
ALL THREE ARE GREEN UNDER EVERY LOCATION INSTRUMENT THE ROUND OWNS. The three are
asserted individually below, each RED against unmutated code, each a different
mechanism:

  * `THREE_LINES_LOW`  -- the cell attributes a GRADE to a line that carries only
    the ITEM HEADER; the grade is three lines further down.
  * `COMPOUND_HALF`    -- the cell says two constants are "both bound" at one
    line; the line carries one of them.
  * `LINE_ONE_IDIOM`   -- the cell attributes MEASURED VALUES to a `<file>:1`
    pointer, a module-docstring opener that carries none of them. A file pointer
    wearing a line number.

LIMITS.
  * 12 is a sample of 116 and `9/12` is a SAMPLED RATE. It is not extrapolated to
    129 and must not be. The INSPECTOR's `8-in-20` was withdrawn at it.20 partly
    for being presented as one thing and used as another.
  * This file certifies nothing about the 13 heading-anchored occurrences; they
    are excluded from the sampling frame, not judged.
"""

from __future__ import annotations

import hashlib
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
TABLE = ROOT / "V20_R15_THEORY_TABLE.md"

#: RECIPE MERCURY-R1 step 1.
TOKEN = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9]+):(\d+)`")

#: Files this round appends to every iteration. DERIVED HERE, NOT IMPORTED.
#: J-20b refuses pointers into them; this office reaches the same three from its
#: own extraction rather than taking the list.
MERCURY_LIVE = ("V20_R15_LEAP_LEDGER.md", "V20_R15_JOURNAL.md", "house-events.jsonl")

MERCURY_OCCURRENCES = 120
MERCURY_POINTERS = 96
MERCURY_SCORED = 107
MERCURY_REFUSED = 13
#: Distinct pointers, split by the `MERCURY_LIVE` rule. 86 + 10 = 96.
MERCURY_SCORED_POINTERS = 86
MERCURY_LIVE_POINTERS = 10
#: Recipe MERCURY-R1b step 3. DATED READING, re-taken 11:56Z at it.24 over the
#: SCORED pointers only. The table was edited by another office at 11:51:49Z
#: DURING it.24 (occurrences 129 -> 120, length still 443), so every count in
#: this block is an it.24 re-take, not the it.22 census.
MERCURY_CENSUS_R1 = "bf2ab827aff619f8552b7980109658a34a613ec2876c8abb7d22373d7225faf4"


def occurrences() -> list[tuple[int, str, int]]:
    out = []
    for i, line in enumerate(TABLE.read_text(encoding="utf-8").splitlines(), 1):
        for m in TOKEN.finditer(line):
            out.append((i, m.group(1), int(m.group(2))))
    return out


def line_at(path: str, lineno: int) -> str:
    lines = (ROOT / path).read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[lineno - 1] if 1 <= lineno <= len(lines) else ""


def scored_pointers() -> list[tuple[str, int]]:
    """The seal's population. THE SAME `MERCURY_LIVE` RULE STEP 2 SCORES BY."""
    return sorted({(p, n) for _, p, n in occurrences() if p not in MERCURY_LIVE})


def census_digest(override=None) -> str:
    """RECIPE MERCURY-R1b. `override` replaces a file's lines IN MEMORY ONLY, so a
    negative can move a file under the seal without touching the repo."""
    over = override or {}

    def at(path: str, lineno: int) -> str:
        if path not in over:
            return line_at(path, lineno)
        lines = over[path]
        return lines[lineno - 1] if 1 <= lineno <= len(lines) else ""

    rows = [f"{p}:{n}\t{at(p, n).strip()}" for p, n in scored_pointers()]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def test_mercury_recount_agrees_with_jupiter_on_totals() -> None:
    """The independent recount. Agreement here is EVIDENCE; it was not assumed."""
    occ = occurrences()
    uniq = {(p, n) for _, p, n in occ}
    assert len(occ) == MERCURY_OCCURRENCES, f"occurrences {len(occ)}"
    assert len(uniq) == MERCURY_POINTERS, f"unique pointers {len(uniq)}"
    scored = [o for o in occ if o[1] not in MERCURY_LIVE]
    assert len(scored) == MERCURY_SCORED
    assert len(occ) - len(scored) == MERCURY_REFUSED


def test_every_pointer_resolves() -> None:
    """The it.14 check, re-run by a second office. Weak, and named as weak."""
    for path, lineno in sorted({(p, n) for _, p, n in occurrences()}):
        f = ROOT / path
        assert f.exists(), f"{path} missing"
        n = len(f.read_text(encoding="utf-8", errors="replace").splitlines())
        assert 1 <= lineno <= n, f"{path}:{lineno} out of range (file has {n})"


def test_the_seal_population_is_the_scored_population() -> None:
    """The it.24 amendment, asserted rather than described. 86 + 10 = 96."""
    ptrs = {(p, n) for _, p, n in occurrences()}
    assert len(scored_pointers()) == MERCURY_SCORED_POINTERS
    assert len(ptrs) - len(scored_pointers()) == MERCURY_LIVE_POINTERS
    assert not {p for p, _ in scored_pointers()} & set(MERCURY_LIVE)


def test_mercury_census_digest_r1() -> None:
    """RECIPE MERCURY-R1b. RED if a SCORED cited line's content moves. GREEN on an
    append: the three growing files are not in the population at all."""
    got = census_digest()
    assert got == MERCURY_CENSUS_R1, f"MERCURY-CENSUS-R1 moved: {got}"


# --------------------------------------------------------------------------
# FINDING M-22a. Three argument-layer defects, GREEN under location, RED here.
# Each asserts the proposition THE CELL MAKES at that citation, of the cited
# line. Not an instrument -- three literal readings, banked.
# --------------------------------------------------------------------------


def test_m22a_three_lines_low_contract_239_does_not_carry_f3() -> None:
    """The table says the contract grades `M14` `F3` at `:239`. It does not.

    `V20_R15_THEORY_TABLE.md:52` -- "`F3` in the contract
    (`CEQ_V20_R15_CONTRACT.md:239`)". `:239` is the M14 ITEM HEADER
    (`M14 CHEEGER STRATIFICATION...`); the token `F3` first appears at `:242`.
    A landing check anchored on `M14 CHEEGER` passes. The claim is three lines low.
    """
    assert "`CEQ_V20_R15_CONTRACT.md:239`" not in TABLE.read_text(encoding="utf-8"), (
        "M-22a/THREE_LINES_LOW: the citation is back in the table. It attributes grade "
        "F3 to CEQ_V20_R15_CONTRACT.md:239, which carries only the M14 ITEM HEADER; "
        f"the line reads {line_at('CEQ_V20_R15_CONTRACT.md', 239)!r} and F3 is at :242"
    )


def test_m22a_compound_half_constants_13_does_not_carry_beta() -> None:
    """The cell said `+0.717647` and `-0.032353` were BOTH bound at `:13`.

    REPAIRED AT it.25 -- THE MARKER, NOT THE FINDING. As shipped at it.22 this
    asserted `"-0.032353" in line_at(constants, 13)`. That demanded the FILE move
    when the defect was that the CITATION did not match the file, so it was RED
    before the repair and RED after it: a defect marker must assert against the
    thing the repair will change. JUPITER struck it at it.24 and was right.
    C63 was withdrawn and re-issued as C131 with the range `:13-14`. This now
    asserts the CITATION -- the bare `:13` compound is gone, and the range that
    replaced it actually carries both halves.
    """
    table = TABLE.read_text(encoding="utf-8")
    const = "tests/jupiter/test_v20_r15_it12_constants.py"
    assert f"`{const}:13`" not in table, (
        "M-22a/COMPOUND_HALF: the bare `:13` compound citation is back. It claims "
        "+0.717647 and -0.032353 are BOTH bound at :13, which reads "
        f"{line_at(const, 13)!r}; beta is at :14"
    )
    assert f"`{const}:13-14`" in table, "C131's range citation left the table"
    span = line_at(const, 13) + line_at(const, 14)
    for want in ("0.717647", "0.032353"):
        assert want in span, f"C131 range :13-14 does not carry {want}: {span!r}"


def test_m22a_line_one_idiom_arm_pl_1_carries_no_measurement() -> None:
    """The table attributes a MEASURED INSTANCE to `ceq/arm_pl.py:1`.

    The Q2/W3 cell reads "200 draws, on `ceq/arm_pl.py:1`: `t_1 = -2.483118`,
    hull [...], excess `1.0845223424`". `:1` is the module docstring's opening
    line and carries none of those numbers. `:1` here means THE FILE; the table's
    own convention gives it a line number, and every line-landing instrument in
    the round scores it as a line.
    """
    assert "`ceq/arm_pl.py:1`" not in TABLE.read_text(encoding="utf-8"), (
        "M-22a/LINE_ONE_IDIOM: the file-pointer-wearing-a-line-number is back. The "
        "cell attributes t_1 = -2.483118 to ceq/arm_pl.py:1, which reads "
        f"{line_at('ceq/arm_pl.py', 1)!r}"
    )


def test_m22a_line_one_idiom_is_a_class_not_an_instance() -> None:
    """`:1` was used as a file pointer 8 times across 6 files, not once.

    RETIRED AT it.24, GREEN FOR A NEW REASON. The class is EMPTY: another office
    removed every `:1` pointer from the table between 11:46Z and 11:51:49Z of this
    iteration (occurrences 129 -> 120; 8 of the 9 lost are these). This now goes
    RED if a file-pointer-wearing-a-line-number comes back.
    """
    ones = [(p, n) for _, p, n in occurrences() if n == 1]
    assert len(ones) == 0, f"`:1` occurrences: {len(ones)}"
    assert len({p for p, _ in ones}) == 0, f"`:1` files: {sorted({p for p, _ in ones})}"
