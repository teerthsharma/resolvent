"""V20 R15 it.26 MERCURY — the three it.25 landing defects, bound to the CITATION.

JUPITER's it.24 ruling, adopted: a defect marker must assert against the thing
the repair will change.  All three defects are wrong CITATIONS in
V20_R15_THEORY_TABLE.md, not wrong CODE, so every RED below asserts the table's
text.  The code-side controls assert the unmutated repo and are GREEN now; if a
control ever goes RED the defect was "repaired" by moving the code, which is not
the repair.

RED at it.26 against the unmutated table (sha256[:16] 3c4d1b270049f6ff, raw bytes).
GREEN when JUPITER applies the replacements named in V20_R15_IT26_MERCURY.md §2.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
TABLE = (ROOT / "V20_R15_THEORY_TABLE.md").read_text(encoding="utf-8")


def line_at(path, n):
    return (ROOT / path).read_text(encoding="utf-8").split("\n")[n - 1]


# --- M-25c  REGISTRY_FOR_ARTEFACT -----------------------------------------
def test_m25c_control_475_is_a_registry_entry_and_236_is_the_generator():
    assert '"generator": "ceq.beds.bed_k.build_delay"' in line_at("ceq/kdata.py", 475)
    assert line_at("ceq/beds/bed_k.py", 236).startswith("def build_delay(")


def test_m25c_admission_condition_cites_the_generator_not_the_registry():
    """The F4 admission condition must send the leap to the def, not to a dict key."""
    assert "ceq/beds/bed_k.py:236" in TABLE
    assert "the generator exists at `ceq/kdata.py:475`" not in TABLE


# --- M-25a  SPAN_HEAD ------------------------------------------------------
def test_m25a_control_547_carries_one_flag_and_the_nine_span_547_558():
    body = (ROOT / "scripts/v15_r1.py").read_text(encoding="utf-8").split("\n")
    assert body[546].count("add_argument") == 1
    assert sum("add_argument" in l for l in body[546:558]) == 9


def test_m25a_nine_flags_are_cited_as_a_range():
    """C131's `:A-B` notation exists; a nine-line span may not land on its head."""
    assert "scripts/v15_r1.py:547-558" in TABLE
    assert "argparse flags** at `scripts/v15_r1.py:547`" not in TABLE


# --- M-25b  COMPOUND_HALF --------------------------------------------------
def test_m25b_control_586_computes_floor1_and_the_value_is_nowhere_in_the_file():
    src = (ROOT / "scripts/v15_r1.py").read_text(encoding="utf-8")
    assert "floor1 = math.sqrt((T_STAR - 1) / T_STAR)" in line_at("scripts/v15_r1.py", 586)
    assert "0.7071067811865476" not in src          # computed, never written
    assert line_at("scripts/v15_r1.py", 138) == "T_STAR = 2"


def test_m25b_computed_constant_cites_its_formula_and_its_input():
    """A computed constant may be cited to the formula AND the input it is computed
    from, never to a line that does not contain it."""
    assert "scripts/v15_r1.py:138" in TABLE                 # t* = 2, the input
    assert "= 0.7071067811865476`\n(`scripts/v15_r1.py:586`" not in TABLE
