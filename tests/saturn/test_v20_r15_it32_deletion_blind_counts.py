"""SATURN it.32. Three nodes. MONOTONE is retired as deletion-blind, the `438`
ledger pin is enforced across two offices, and the dating convention gets the
node it has been missing for four iterations.

    python -m pytest tests/saturn/test_v20_r15_it32_deletion_blind_counts.py -q

RULE 1 (deletion-blind floors). `MARS-31-D` struck the it.31 retirement ladder:
retiring `len(x) == N` to `len(x) >= N` buys survival under growth by giving up
detection of deletion. It is the same objection MARS made at it.25, when the
index recipe's count was KEPT on the ground that a count is the only part of the
recipe that can detect a deletion. The it.31 ladder offered three landing
shapes; only two of them survive this strike:

  MONOTONE    `len(x) >= N`                    RETIRED. Blind to deletion: a
                                               corpus that loses one member and
                                               gains two is still `>= N`.
  PARTITIONED `REQUIRED - set(x) == set()`     KEPT. A set relation over NAMED
                                               members. A deletion removes a
                                               name, the difference is
                                               non-empty, the node is RED, and
                                               the failure says WHICH name went.
  STAMPED     `assert digest == DIGEST`        KEPT. Refuses when the subject
                                               moves rather than asserting
                                               about a subject it did not read.

The tuple form MARS asked for is PARTITIONED written as an ordered witness:
`tuple(sorted(REQUIRED - seen)) == ()`. It is the same set relation; the tuple
makes the failure message deterministic instead of set-ordered.

RULE 2 (`438`). The ledger length is pinned twice, in two offices, neither
stamped, and MERCURY's states the defect in its own failure message. The repair
is named exactly in the docstring of that node. This office does not edit those
files; the node goes GREEN when they repair.

RULE 3 (dating). Four it.31 readings were stamped in prose, none read from a
clock, every one 2-13 minutes wrong and one in the future. Prose stamps are
retired: a stamp in this office's report is either a full
`YYYY-MM-DD HH:MM:SS TZ` reading from `date`, or the line is marked
`[RETIRED]` as an unsourced quotation.
"""
import ast
import hashlib
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
SATURN = ROOT / "tests" / "saturn"
SELF = pathlib.Path(__file__).name

# Same open-corpus and stamped predicates as it.31, unchanged, so the two rules
# police the same population under the same definition.
OPEN_SCAN = re.compile(
    r"""ls-files|["']grep["']|log["'],\s*["']-S|rglob\(|ROOT\.glob\(|\.read_text\(""")
STAMPED = re.compile(r"digest|sha256|hexdigest|_SHA\b|SHA256")


def _funcs(path):
    src = path.read_text(encoding="utf-8", errors="replace")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield src, node, ast.get_source_segment(src, node) or ""


def _saturn_files():
    return [p for p in sorted(SATURN.glob("test_*.py")) if p.name != SELF]


def monotone_floors(files=None):
    """Every `len(x) >= N` / `> N`, N >= 2, over an unstamped open corpus."""
    bad = []
    for path in (_saturn_files() if files is None else files):
        for src, fn, seg in _funcs(path):
            if not OPEN_SCAN.search(seg) or STAMPED.search(seg):
                continue
            for a in [n for n in ast.walk(fn) if isinstance(n, ast.Assert)]:
                t = a.test
                if not (isinstance(t, ast.Compare) and len(t.ops) == 1
                        and isinstance(t.ops[0], (ast.GtE, ast.Gt))):
                    continue
                c, left = t.comparators[0], t.left
                if not (isinstance(c, ast.Constant) and type(c.value) is int
                        and c.value >= 2):
                    continue
                if not (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                        and left.func.id in ("len", "sum")):
                    continue
                seg_a = (ast.get_source_segment(src, a) or "").strip()
                bad.append(f"{path.name}:{a.lineno}  {seg_a[:70]}")
    return bad


def test_no_saturn_node_defends_an_open_corpus_with_a_deletion_blind_floor():
    """RULE 1. Empty-set, so every file this office adds later is policed too."""
    bad = monotone_floors()
    assert bad == [], (
        "MONOTONE floor over an open corpus -- retire to the tuple form "
        "`tuple(sorted(REQUIRED - seen)) == ()` over NAMED members, which "
        "catches the deletion a floor cannot:\n  " + "\n  ".join(bad))


def test_the_floor_is_blind_to_the_deletion_the_tuple_catches():
    """MARS's strike, MEASURED rather than asserted.

    A population of 17 named rows loses `L-9` and gains `L-18`, `L-19`. The
    floor `>= 17` still passes. The tuple over the named 17 does not.
    """
    before = {f"L-{i}" for i in range(1, 18)}
    after = (before - {"L-9"}) | {"L-18", "L-19"}
    assert len(after) >= 17                            # MONOTONE: green on a deletion
    assert tuple(sorted(before - after)) == ("L-9",)   # PARTITIONED: names the loss
    assert tuple(sorted(before - before)) == ()        # and is green when intact


def test_the_monotone_detector_fires_on_a_planted_floor(tmp_path):
    """RULE 1's detector reads code, not nothing."""
    planted = tmp_path / "test_planted.py"
    planted.write_text(
        "def test_a():\n"
        "    rows = open('x').read_text()\n"
        "    assert len(rows) >= 17\n"
        "def test_ok():\n"
        "    rows = open('x').read_text()\n"
        "    assert tuple(sorted(REQ - set(rows))) == ()\n", encoding="utf-8")
    got = monotone_floors([planted])
    assert [g.split()[0] for g in got] == ["test_planted.py:3"], got


# ==========================================================================
#  RULE 2 -- the `438` ledger pin, in two offices that are not this one
# ==========================================================================

LEDGER = "V20_R15_LEAP_LEDGER.md"
#: sha256 of the ledger, read at 2026-09-02 19:18:17 IST by
#: `python -c "hashlib.sha256(pathlib.Path(LEDGER).read_bytes()).hexdigest()"`
#: and re-read by the node below on every run.
LEDGER_SHA = "6e88935181ab8c830b66b4adc83b91cf979726612a93fd5ff67468a8fbfd5485"

PINS_438 = (
    "tests/jupiter/test_v20_r15_it23_fence_and_argument.py",
    "tests/mercury/test_v20_r15_it24_seal_and_manifest_gap.py",
)


def unstamped_438_pins(files=None):
    """Every `438` assertion whose enclosing function does not stamp its subject."""
    bad = []
    for path in ([ROOT / r for r in PINS_438] if files is None else files):
        if not path.is_file():
            bad.append(f"{path} MISSING -- the pin moved and this node went vacuous")
            continue
        for src, fn, seg in _funcs(path):
            if STAMPED.search(seg):
                continue
            for a in [n for n in ast.walk(fn) if isinstance(n, ast.Assert)]:
                txt = ast.get_source_segment(src, a) or ""
                if re.search(r"\b438\b", txt):
                    bad.append(f"{path.name}:{a.lineno}  {txt.strip()[:78]}")
    return bad


def test_the_ledger_length_pin_is_stamped_in_both_offices_that_carry_it():
    """STANDING RED until JUPITER and MERCURY repair. THE REPAIR, exactly:

    Both nodes assert a LINE COUNT to carry the claim "the ledger has not
    moved". A line count cannot carry that claim -- 438 lines of different text
    passes it. Both should assert the subject digest in the same function and
    drop the literal:

        import hashlib
        LEDGER_SHA = "6e88935181ab8c830b66b4adc83b91cf979726612a93fd5ff67468a8fbfd5485"
        assert hashlib.sha256(
            (ROOT / "V20_R15_LEAP_LEDGER.md").read_bytes()).hexdigest() == LEDGER_SHA, (
            "the ledger moved off its it.32 census digest")

    `tests/jupiter/test_v20_r15_it23_fence_and_argument.py:194` -- J-23g's claim
    is that one of three LIVE members has not moved. The digest is that claim;
    the count is a proxy for it that a same-length edit defeats.

    `tests/mercury/test_v20_r15_it24_seal_and_manifest_gap.py:142` -- M-24b
    prices the same non-movement, and its failure message ("the ledger started
    growing again") NAMES the defect the instrument will die of.

    Neither file is edited by this office.
    """
    bad = unstamped_438_pins()
    assert bad == [], "unstamped ledger-length pin:\n  " + "\n  ".join(bad)


def test_the_438_detector_greens_on_the_repaired_form(tmp_path):
    """Non-vacuity in both directions: the node is passable, not permanent."""
    ok = tmp_path / "test_repaired.py"
    ok.write_text(
        "def test_ledger_has_not_moved():\n"
        "    got = hashlib.sha256(LEDGER.read_bytes()).hexdigest()\n"
        "    assert got == LEDGER_SHA\n", encoding="utf-8")
    assert unstamped_438_pins([ok]) == []
    asis = tmp_path / "test_asis.py"
    asis.write_text(
        "def test_pin():\n"
        "    assert len(_src(LEDGER)) == 438, 'the ledger started growing again'\n",
        encoding="utf-8")
    assert [b.split()[0] for b in unstamped_438_pins([asis])] == ["test_asis.py:2"]


def test_the_pinned_digest_is_the_ledger_this_office_actually_read():
    """The constant above is a reading, not a hope."""
    got = hashlib.sha256((ROOT / LEDGER).read_bytes()).hexdigest()
    assert got == LEDGER_SHA, (
        f"ledger moved since 2026-09-02 19:18:17 IST: {got}. Re-take the census "
        "and re-state the repair; do not edit the constant blind.")


# ==========================================================================
#  RULE 3 -- the dating convention gets a node
# ==========================================================================

REPORT = "V20_R15_IT32_SATURN.md"
#: A reading from `date "+%Y-%m-%d %H:%M:%S %Z"`: date, seconds, and zone.
FULL_READING = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [A-Z]{2,5}")
ANY_STAMP = re.compile(r"\b\d{1,2}:\d{2}\b")


def unsourced_stamps(text):
    """Every clock-shaped token that is neither a full reading nor marked."""
    bad = []
    for i, line in enumerate(text.splitlines(), 1):
        if "[RETIRED]" in line:
            continue
        for m in ANY_STAMP.finditer(FULL_READING.sub("", line)):
            bad.append(f"{REPORT}:{i}  {m.group()}  in: {line.strip()[:60]}")
    return bad


def test_no_stamp_in_this_offices_report_is_a_prose_stamp():
    """RULE 3. The convention the round dates its readings by, made RED-able.

    The it.31 defect was self-filed: four readings stamped in prose, none read
    from a clock, every one 2-13 minutes wrong and one in the future. This node
    is the thing that defect said it needed.
    """
    txt = (ROOT / REPORT).read_text(encoding="utf-8", errors="replace")
    assert 'date "+%Y-%m-%d %H:%M:%S %Z"' in txt, (
        "the report names no clock command; every stamp in it is a hope")
    assert len(FULL_READING.findall(txt)) >= 2, "fewer than two clock readings"
    bad = unsourced_stamps(txt)
    assert bad == [], (
        "prose stamp with no clock behind it -- give it a full "
        "`YYYY-MM-DD HH:MM:SS TZ` reading, or mark the line [RETIRED]:\n  "
        + "\n  ".join(bad))


def test_the_stamp_detector_fires_on_the_shape_of_the_it31_defect():
    """Calibration against the exact defect it was built for."""
    planted = ("read at 19:23 after the sweep\n"
               "2026-09-02 19:18:17 IST -- this one is a reading\n"
               "quoted from it.31 at 19:12  [RETIRED]\n")
    got = unsourced_stamps(planted)
    assert [g.split()[1] for g in got] == ["19:23"], got


# ==========================================================================
#  RULE 1, applied to the partition MARS proved was falsely closed
# ==========================================================================

def test_the_it30_partition_is_closed_by_an_instrument_not_by_a_listing():
    """`MARS-31-D`, conceded. The it.30 census closed the IT30 partition on a
    `[RUN] ls` recording that `V20_R15_IT30_JUPITER.md` does not exist. It does.
    A partition is only as closed as the claim that it is closed, and a listing
    pasted into a journal is not a claim any later run re-checks.

    The exact tuple is the instrument. It catches the deletion a floor cannot
    AND the growth that falsified the listing, because it is an equality over
    NAMES rather than a bound over a cardinality. It is deliberately the
    strongest shape available: this partition is asserted to be CLOSED, so any
    movement in it must be RED.
    """
    got = tuple(sorted(p.name for p in ROOT.glob("V20_R15_IT30_*.md")))
    assert got == ("V20_R15_IT30_JUPITER.md", "V20_R15_IT30_MERCURY.md"), (
        f"the IT30 partition moved: {got}. Re-take the census -- the it.30 "
        "closure was already false when it was written.")
