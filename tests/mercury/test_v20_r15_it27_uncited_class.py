"""V20 R15 it.27 MERCURY — the UNCITED-CLAIM class outside §2 of the theory table.

it.26 measured the class on §2 only.  This node measures §0, §3 and §4 and extends it
from numeric constants to NAMED ARTIFACTS.  Every RED below is a citation defect in
V20_R15_THEORY_TABLE.md, not a code defect, so each RED asserts the TABLE (JUPITER's
it.24 ruling: a marker asserts the thing the repair will change) and each is paired
with a code-side control that is GREEN against the UNMUTATED repo.

RED at it.27 against the table at sha256[:16] 3c4d1b270049f6ff (raw bytes, 12:33Z).
GREEN when the replacements in V20_R15_IT27_MERCURY.md are applied to the table alone.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
TABLE = (ROOT / "V20_R15_THEORY_TABLE.md").read_text(encoding="utf-8")
LINES = TABLE.split("\n")
SEC4_3_LIMITS = "\n".join(LINES[378:392])   # :379-392, the §4.3 limits + correction
SEC3_2 = "\n".join(LINES[269:281])          # :270-281, §3.2


def line_at(path, n):
    return (ROOT / path).read_text(encoding="utf-8").split("\n")[n - 1]


# --- M-27a  BED_SPECS asserted in §4 with no pointer ------------------------
def test_m27a_control_bed_specs_is_defined_at_kdata_472():
    assert line_at("ceq/kdata.py", 472).startswith("BED_SPECS = {")


def test_m27a_sec4_bed_specs_correction_cites_the_symbol():
    """§4.3 withdraws a LEAP LEDGER sentence on the strength of `BED_SPECS`'s return
    value and offers no pointer for it -- the evidence for a withdrawal is uncited."""
    assert "`BED_SPECS`" in SEC4_3_LIMITS          # the claim is here
    assert "ceq/kdata.py:472" in SEC4_3_LIMITS     # its pointer is not


# --- M-27b  the `none` arm named with no pointer ----------------------------
def test_m27b_control_none_arm_is_at_saturn_329():
    assert 'head.startswith("none")' in line_at(
        "tests/saturn/test_v20_r15_it12_saturn.py", 329)


def test_m27b_sec4_none_arm_is_cited():
    """The `none` arm produces four of the five findings and carries no pointer,
    while its sibling `ROUND_NOUNS` arm carries one on the same line."""
    assert "the `none` arm" in SEC4_3_LIMITS
    assert "test_v20_r15_it12_saturn.py:329" in SEC4_3_LIMITS


# --- M-27c  §3.2's Fisher p is uncited and its producer exists --------------
def test_m27c_control_fisher_p_is_banked_in_the_journal():
    assert "8.544e-05" in line_at("V20_R15_JOURNAL.md", 1552)


def test_m27c_sec3_2_fisher_p_cites_its_record():
    assert "8.544e-05" in SEC3_2
    assert "V20_R15_JOURNAL.md:1552" in SEC3_2
