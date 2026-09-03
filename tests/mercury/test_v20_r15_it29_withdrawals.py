"""MERCURY it.29. Three published MERCURY numbers, each with the node that settles it.

    python -m pytest tests/mercury/test_v20_r15_it29_withdrawals.py -q

Every `m29*` node is RED against the unmutated tree and goes GREEN on a
DOCUMENT edit alone. Every `*_control` node is GREEN against the unmutated
tree; a control going RED means the subject moved, not that it was repaired.
"""
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from screen_v20_r15_it29_numerics import screen  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
TABLE = ROOT / "V20_R15_THEORY_TABLE.md"
DIGEST = "942e4208893444cd"
SEC2 = (126, 228)
SEC4 = (282, 392)


def _text(name):
    return (ROOT / name).read_text(encoding="utf-8")


# --- 1. `0 of 22 files under tests/mercury are known to git` -----------------

def test_m29a_the_two_reports_do_not_carry_the_0_of_22_literal():
    """RED. it.26:16 and it.27:18 attribute MARS's `tests/mars_v20` figure to
    `tests/mercury/`. The measured literal is `9 of 25`."""
    for name in ("V20_R15_IT26_MERCURY.md", "V20_R15_IT27_MERCURY.md"):
        body = _text(name)
        hit = [ln for ln in body.splitlines() if "0 of 22" in ln and "tests/mercury" in ln]
        assert hit == [], f"{name} still carries the carried-over figure: {hit}"


def test_m29a_control_nine_of_twentyfive_mercury_files_are_tracked():
    """GREEN. The replacement literal, measured. `9 of 25` is the 13:03:08Z
    reading, BEFORE this office wrote its two instruments; `9 of 27` is the
    same reading after. The tracked half does not move either way, and it is
    the half the withdrawn figure got wrong. Goes RED the moment anyone commits
    a file under tests/mercury -- which is the correct signal."""
    tracked = subprocess.run(
        ["git", "ls-files", "tests/mercury"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    on_disk = sorted(p.name for p in (ROOT / "tests" / "mercury").glob("*.py"))
    assert len(tracked) == 9
    assert len(on_disk) == 27  # 25 at 13:03:08Z + this office's screen and marker node
    assert all(pathlib.Path(t).name.startswith(("test_r9_", "test_r10_")) for t in tracked)


# --- 2. sec.4 asserts integer-shaped quantities, and does not address them ---

def test_m29b_sec4_prices_carry_a_pointer():
    """RED. `~275 GPU-s` at :354 and `~6 GPU-s` at :326 are prices the leap acts
    on. Neither line carries a `path:line`. :354 is worse: sec.2 at :184 of the
    same file records that `~275` does not evaluate to 275, and :354 does not
    point at that correction."""
    lines = TABLE.read_text(encoding="utf-8").splitlines()
    for n in (326, 354):
        assert ".py:" in lines[n - 1] or ".md:" in lines[n - 1], (
            f"V20_R15_THEORY_TABLE.md:{n} asserts a price with no pointer: {lines[n - 1][:90]}"
        )


def test_m29b_control_sec4_screens_five_floats_and_112_integers():
    """GREEN. The population behind the hand read, fixed by a committed screen."""
    digest, floats, ints = screen(str(TABLE), *SEC4)
    assert digest == DIGEST
    assert [t for _, t in floats] == ["4.1", "4.2", "0.3", "4.3", "0.3"]
    assert len(ints) == 112


# --- 3. it.26's `62` ---------------------------------------------------------

def test_m29c_it26_publishes_the_screen_count_the_screen_returns():
    """RED. it.26:138 publishes `-> 63 tokens` for the sec.2 float screen. Two
    independent re-implementations and this one return 64."""
    line = [ln for ln in _text("V20_R15_IT26_MERCURY.md").splitlines() if "float-shaped tokens only" in ln]
    assert len(line) == 1
    assert "64" in line[0], f"it.26 still publishes a screen count the screen does not return: {line[0].strip()}"


def test_m29c_control_sec2_screens_64_floats_of_which_62_are_distinct():
    """GREEN, and it is the whole reconciliation. 64 raw tokens; 62 distinct
    (line, token) pairs; 63 after the ONE rejection it.26 names (`0.3` at :180).
    `62` is reachable only by dropping the two same-line repeats AND not
    applying the named rejection -- the one arithmetic it.26's prose forbids."""
    digest, floats, _ = screen(str(TABLE), *SEC2)
    assert digest == DIGEST
    assert len(floats) == 64
    assert len(set(floats)) == 62
    repeats = sorted({t for t in floats if floats.count(t) > 1})
    assert repeats == [(141, "1e-6"), (221, "0.0")]
    assert len([t for t in floats if t != (180, "0.3")]) == 63


# --- 4. M-29a: the Q2/W1 admission condition asserts a false fact ------------

def test_m29d_sec4_1_zero_callers_claim_is_true():
    """RED. V20_R15_THEORY_TABLE.md:306 -- the ADMISSION CONDITION for the F4
    cell Q2/W1 -- asserts `build_delay` has "zero callers under `scripts/`".
    It has two."""
    calls = []
    for f in sorted((ROOT / "scripts").glob("*.py")):
        for i, ln in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if "bed_k.build_delay(" in ln and '"' not in ln:
                calls.append(f"{f.name}:{i}")
    assert calls == [], f"the admission condition names zero callers; there are {len(calls)}: {calls}"


def test_m29d_control_the_claim_is_where_this_office_says_it_is():
    """GREEN. The sentence under test is at :306 and says what is quoted."""
    line = TABLE.read_text(encoding="utf-8").splitlines()[305]
    assert "zero callers under" in line and "Q2/W1" in line
