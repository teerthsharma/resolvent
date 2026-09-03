"""IT30 MERCURY. Two questions the theory table leaves open.

(1) V20_R15_THEORY_TABLE.md:306, Q2/W1's ADMISSION CONDITION, asserts build_delay
    "has zero callers under scripts/". It has two. The open question is not whether
    that clause is false -- it.29 measured that -- but whether the two call sites
    SATISFY the condition, whose operative text is "one cell of BED-K's shape
    actually run". They do not: both consume b["K"] and nothing else.

(2) The same file quotes ~275 GPU-s at :354 and says at :184 that the number does
    not evaluate to 275. Settled here against the formula, which is runnable.

RED FIRST, against unmutated code: test_red_* assert the table's text and fail.
"""
import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CHEEGER = ROOT / "scripts" / "v20_m14_cheeger.py"
TABLE = ROOT / "V20_R15_THEORY_TABLE.md"

# a cell of BED-K's shape needs the bed's DATA, not just its operator.
DATA_KEYS = {"b", "z", "pos", "plant"}


def _call_sites():
    """Every build_delay call under scripts/, with the bed keys its result feeds."""
    out = []
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        src = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name != "build_delay":
                continue
            kw = {k.arg: getattr(k.value, "value", None) for k in node.keywords}
            # the name the result is bound to, then every subscript of that name
            bind = None
            for anc in ast.walk(tree):
                if isinstance(anc, ast.Assign) and anc.value is node:
                    if isinstance(anc.targets[0], ast.Name):
                        bind = anc.targets[0].id
            keys = set()
            if bind:
                for anc in ast.walk(tree):
                    if (isinstance(anc, ast.Subscript)
                            and isinstance(anc.value, ast.Name)
                            and anc.value.id == bind
                            and isinstance(anc.slice, ast.Constant)):
                        keys.add(anc.slice.value)
            out.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                        "line": node.lineno, "kwargs": kw, "bind": bind,
                        "keys_consumed": sorted(keys),
                        "src": src[node.lineno - 1].strip()})
    return out


# ---------------------------------------------------------------- RED
@pytest.mark.xfail(strict=True, reason="THEORY_TABLE:306 as written; RED is the finding")
def test_red_table_306_says_zero_callers_under_scripts():
    assert len(_call_sites()) == 0


@pytest.mark.xfail(strict=True, reason="THEORY_TABLE:354 as written; RED is the finding")
def test_red_table_354_price_275_matches_its_own_formula():
    # V20_R15_IT8_JUPITER.md:216 -- 3 x (0.25 + 1 + 4) x (16.16 + 1.78 + 1.68)
    assert round(3 * (0.25 + 1 + 4) * (16.16 + 1.78 + 1.68), 3) == 275


# -------------------------------------------------------------- GREEN
def test_two_call_sites_at_exactly_the_registered_kwargs():
    sites = _call_sites()
    assert len(sites) == 2, sites
    assert {(s["path"], s["line"]) for s in sites} == {
        ("scripts/v20_m14_cheeger.py", 347), ("scripts/v20_m14_cheeger.py", 462)}
    for s in sites:
        assert s["kwargs"] == {"n": 500, "d": 4, "seed": 7}, s   # kdata.py:476


def test_neither_call_site_runs_a_cell_of_bed_ks_shape():
    """The operative clause. A cell needs an arm run on the bed's data; both sites
    read only the kernel operator K and import no arm."""
    for s in _call_sites():
        assert s["keys_consumed"] == ["K"], s
        assert not (set(s["keys_consumed"]) & DATA_KEYS), s
    text = CHEEGER.read_text(encoding="utf-8", errors="replace")
    for arm in ("arm_smprime", "arm_pl", "hankel", "rank_real", "rank_plus_lower"):
        assert arm not in text, arm


def test_neither_275_nor_a_point_on_the_band_is_the_it8_formula():
    """TWO objects and a label, and the point that used to name one of them is
    withdrawn at it.33. IT8:216's literal constants give 309.015 -- the formula AS
    WRITTEN, and it is NOT on the band; the band is `_sweep(e)` over it.13's
    REMEASURED means, a different object; 275 is neither. The band's midpoint is
    withdrawn under MARS-31-C and the it.32 ruling, and JUPITER's closing detail
    is why it cannot come back rounded: at ONE DECIMAL the literal and the
    midpoint are the same number, so a quoted point reinstates exactly the
    distinction it was made to draw."""
    from tests.mercury.phase_c_price import jupiter_it8_formula, s_exponent_band
    literal = 3 * (0.25 + 1 + 4) * (16.16 + 1.78 + 1.68)
    assert round(literal, 3) == 309.015                          # the formula AS WRITTEN
    assert round(literal, 3) != round(jupiter_it8_formula(), 3)  # two objects at 3 dp
    assert round(literal, 1) == round(jupiter_it8_formula(), 1)  # one number at 1 dp
    lo, mid, hi = s_exponent_band()
    assert lo < mid < hi, "the family is not monotone in e"
    assert (round(lo, 1), round(hi, 1)) == (206.0, 537.2)        # BAND ONLY, NO POINT
    assert lo < 275 < hi and 275 not in (round(lo, 3), round(hi, 3))


def test_the_275_at_354_landed_as_the_band_with_no_point():
    """WAS RED at it.30: `:354` was the load-bearing address of a price the round
    ruled is produced by nothing, and MARS-31-C2 re-struck it. It LANDED -- the
    edit is JUPITER's, at it.32, and this office re-reads it rather than citing
    it. The replacement is the band and it carries `band only, no point`, so the
    landing is the it.32 ruling itself and not a substituted point."""
    line = TABLE.read_text(encoding="utf-8", errors="replace").splitlines()[353]
    assert "~275 GPU-s" not in line, line
    assert "206" in line and "537 GPU-s" in line, line
    assert "band only, no point" in line, line
    assert "309" not in line, "a point came back at the address the band replaced"
