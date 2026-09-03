"""it.34 -- P-33a THE CENSUS KEYED BY A DIGEST OVER THE CITATION SET, and
P-33b THE THEORY TABLE'S RADIUS, ENUMERATED.

RULING J-34a -- A REMEDY THAT ADDS A CITATION TO A FROZEN CENSUS IS A
  SELF-BREAKING REMEDY, AND THE FAULT IS IN THE KEY, NOT IN THE REMEDY.
  Seven of the eight it.32 breaks were this office's own two added pointers:
  `path:N` 120 -> 122, `md` 40 -> 41, `py` +1. `M-30a` says *no point without a
  pointer*; `M-30b` says *name the callers*. Both remedies ADD a citation. The
  it.20 census FREEZES the citation count. The provenance instrument and the
  freeze instrument count the same objects in opposite directions and neither
  names the other.
  THE REPAIR IS A KEY, NOT A NUMBER. `census_key()` digests the citation SET
  (the sorted multiset of `path:spec` occurrences). Every count is then LOOKED
  UP by that key in `STATES`, an append-only ledger of declared states. An added
  pointer is a NEW KEY -- a state this ledger has not yet declared -- and the
  instrument answers with `diff_occurrences()`, which NAMES the pointer that
  moved. A census that reports "the citation set changed, here is the diff" is
  an instrument. One that reports "122 != 120" is a tripwire.
  The negative is `test_an_added_pointer_is_a_NEW_STATE_and_the_diff_NAMES_it`:
  it adds one pointer in memory and asserts the key moves AND the diff names
  exactly that pointer. If the key ever stops covering the set, that test says so.

RULING J-34b -- A RADIUS IS A PROPERTY OF AN EDIT, NOT OF AN OFFICE.
  `RADIUS` (it.31) is *the 14 files that read `V20_R15_JOURNAL.md`*. The it.32
  edit wrote the journal AND `V20_R15_THEORY_TABLE.md`, and declared one radius
  over both. The table's readers had never been enumerated by any office.
  `TABLE_RADIUS` enumerates them: 23 test files, 9 jupiter / 2 mars / 8 mercury
  / 4 saturn -- this file included, since it reads the table too. It is frozen the way `RADIUS` is frozen AND, unlike `RADIUS`,
  RE-DERIVED at run time, so an office that adds a reader without adding it to
  the list is named by the instrument rather than by the next iteration's
  breakage. Nine files are in both radii; fourteen are in the table's alone, and
  those fourteen are exactly what the it.32 declaration did not cover.

RULING J-34c -- THREE POPULATIONS, ONE RULE AND TWO DATES.
  MERCURY's it.33 note: her extractor returns `122` occurrences over 71 lines
  against the round's `129` and `133`. They reconcile EXACTLY and no renaming is
  owed:
      133 = 122 plain `:N`  +  8 file `:*`  +  3 range `:A-B`
  MERCURY-R1 admits only digit notation; this office's `CITE_RE` also admits the
  `:*` (J-24a) and `:A-B` (J-24b) notations THIS OFFICE INVENTED after her recipe
  was banked. So `122` and `133` are ONE rule at two ADMISSION WIDTHS, and `129`
  is this office's own rule at an earlier DATE (it.24; 131 at it.27; 133 at
  it.32). Two of the three differ by notation, two by date, none by object.
  Asserted below, not argued: `test_the_three_populations_reconcile_exactly`.

  WHAT IS OWED IS A NAME, NOT A RECONCILIATION. A population is `(rule, date)`
  and a bare integer is neither. `STATES` keys by digest for exactly this
  reason: `133` is not a fact about the table, it is a fact about a rule applied
  to a state, and the digest names the state.

[RUN] baselines, this window, `date -u` on this box, branch `v17k-gate0`:
  * 2026-09-02T14:17:49Z -> 14:18:04Z, RADIUS re-taken on unmutated code:
    `15 RED / 91 GREEN` in 12.24s. The it.31 `RADIUS_BASELINE = (9, 97)` is a
    reading carried from it.31 in a timezone `date -u` cannot emit (MARS-33-D),
    so `16 - 9 = 7` was a difference of two readings. Against an in-window
    baseline the it.32 delta is `15 - 16 = -1`, and the three "unattributed"
    nodes of it.33 are inside that comparison artefact.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
from collections import Counter

from tests.jupiter.test_v20_r15_it20_citation_freeze import CITE_RE, TABLE, table_citations

ROOT = pathlib.Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ P-33a

def census_key(occ: "list[tuple[str, str]] | None" = None) -> str:
    """J-34a. sha256[:16] over the sorted citation MULTISET. A multiset, not a
    set: `V20_R15_IT13_MERCURY.md:146` is cited twice (`:184` and `:354`) and
    losing one of them must move the key."""
    occ = table_citations() if occ is None else occ
    blob = "\n".join(sorted("%s:%s" % pair for pair in occ))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def census_state(occ: "list[tuple[str, str]] | None" = None) -> dict:
    """Every count this office publishes about the table, derived in one place."""
    occ = table_citations() if occ is None else occ
    specs = [s for _, s in occ]
    return {
        "population": len(occ),
        "unique": len(set(occ)),
        "files": len({p for p, _ in occ}),
        "ext": dict(sorted(Counter(p.rsplit(".", 1)[1] for p, _ in occ).items())),
        "notation": (sum(s.isdigit() for s in specs),
                     sum("-" in s for s in specs),
                     sum(s == "*" for s in specs)),
    }


def diff_occurrences(before, after):
    """J-34a. What a keyed census answers with instead of `122 != 120`."""
    b, a = Counter(before), Counter(after)
    added = sorted("%s:%s" % k for k in (a - b).elements())
    removed = sorted("%s:%s" % k for k in (b - a).elements())
    return added, removed


#: APPEND-ONLY. A new state is a new row; no row is ever edited, which is the
#: whole difference between this and a frozen count. Declared at
#: `[RUN] 2026-09-02T14:23:09Z`, this box, AFTER the M-33a edit of this iteration.
#: The it.24 numbers (`131`, `{'md': 40, 'py': 71, ...}`, `(120, 3, 8)`) are NOT
#: deleted and NOT corrected -- they were the state at THEIR key and this is a
#: different state. That is the difference a key buys: nothing is ever wrong,
#: things are DATED.
STATES = {
    "4e935cc9db3dbd5a": {
        "population": 133,
        "unique": 106,
        "files": 39,
        "ext": {"jsonl": 1, "lean": 19, "md": 41, "py": 72},
        "notation": (122, 3, 8),
    },
}

#: The two pointers this office ADDED at it.32 as the `M-30a` / `M-30b`
#: remedies. Seven of the eight it.32 breaks are these two lines.
IT32_REMEDY_POINTERS = [
    ("V20_R15_IT13_MERCURY.md", "146"),      # M-30a, at table `:354`
    ("scripts/v20_m14_cheeger.py", "347"),   # M-30b, at table `:306`
]


def test_the_current_citation_set_is_a_DECLARED_STATE() -> None:
    """J-34a. The whole repair, as one assertion. This does not compare a count;
    it asks whether the SET is one this office has declared."""
    occ = table_citations()
    key = census_key(occ)
    assert key in STATES, (
        "the citation set is not a declared state (key %s). This is NOT a broken "
        "count -- run diff_occurrences() against the declared set and declare the "
        "new state. Declared keys: %s" % (key, sorted(STATES)))


def test_every_count_is_LOOKED_UP_BY_KEY_AND_NOT_FROZEN_IN_THE_ASSERTION() -> None:
    """The it.24 shape (`== 131`, `== {'md': 40, ...}`, `== (120, 3, 8)`) is what
    broke seven nodes on two added pointers. Here the numbers come from the row
    the key selects, so adding a pointer selects a DIFFERENT ROW."""
    state = census_state()
    assert STATES[census_key()] == state, state
    assert state["population"] == sum(state["ext"].values()) == sum(state["notation"])


def test_an_added_pointer_is_a_NEW_STATE_and_the_diff_NAMES_it() -> None:
    """PLANTED NEGATIVE for J-34a. Writes to no file in the tree.

    Simulates exactly the it.32 remedy: one pointer added to the table. Under
    the frozen census this is `+1` on a number and seven RED nodes. Under the
    key it is a state this ledger has not declared, and the answer names the
    line."""
    occ = table_citations()
    plus = occ + [("ceq/beds/bed_k.py", "999")]
    assert census_key(plus) != census_key(occ), (
        "the key does not cover the set -- an added pointer must move it")
    assert census_key(plus) not in STATES, "an undeclared state must not be declared"
    added, removed = diff_occurrences(occ, plus)
    assert added == ["ceq/beds/bed_k.py:999"] and removed == [], (added, removed)
    # and the count-shaped answer is the one that carries no information:
    assert census_state(plus)["population"] == census_state(occ)["population"] + 1


def test_the_SEVEN_it32_breaks_reduce_to_a_TWO_LINE_DIFF() -> None:
    """J-34a's evidence. Reconstructs the pre-it.32 set by removing the two
    remedy pointers and asserts the diff is exactly those two -- the finding
    content of seven RED nodes, stated as the two lines it always was."""
    occ = table_citations()
    for pointer in IT32_REMEDY_POINTERS:
        assert pointer in occ, "premise gone: %s:%s is not cited" % pointer
    before = list(occ)
    for pointer in IT32_REMEDY_POINTERS:
        before.remove(pointer)
    added, removed = diff_occurrences(before, occ)
    assert removed == [] and added == sorted(
        "%s:%s" % p for p in IT32_REMEDY_POINTERS), (added, removed)
    assert census_state(before)["population"] + 2 == census_state(occ)["population"]


def test_the_three_populations_reconcile_exactly() -> None:
    """J-34c. MERCURY's `122`, and this office's `133`, are ONE rule at two
    admission widths. Her recipe (a backticked dotted path, colon, DIGITS) is
    this office's `CITE_RE` minus the `:*` and `:A-B` notations invented after
    her recipe was banked."""
    txt = TABLE.read_text(encoding="utf-8")
    mercury_r1 = re.compile(r"`([A-Za-z0-9_./-]+\.[A-Za-z]+):(\d+)`").findall(txt)
    plain, rng, star = census_state()["notation"]
    assert len(mercury_r1) == plain == 122, (len(mercury_r1), plain)
    assert (plain, rng, star) == (122, 3, 8)
    assert plain + rng + star == census_state()["population"] == 133
    # her 71 lines: the same occurrences, counted by LINE rather than by token.
    lines = [i for i, ln in enumerate(txt.splitlines(), 1)
             if re.search(r"`[A-Za-z0-9_./-]+\.[A-Za-z]+:\d+`", ln)]
    assert len(lines) == 71, len(lines)
    # the control: the two rules differ ONLY on notation, never on a path.
    assert {p for p, _ in CITE_RE.findall(txt)} >= {p for p, _ in mercury_r1}


# ------------------------------------------------------------------ P-33b

#: J-34b. The 23 test files that read `V20_R15_THEORY_TABLE.md`. Frozen the way
#: `RADIUS` is frozen, and RE-DERIVED by `test_the_table_radius_is_complete...`.
TABLE_RADIUS = [
    "tests/jupiter/test_v20_r15_it14_theory_table.py",
    "tests/jupiter/test_v20_r15_it17_citation_landing.py",
    "tests/jupiter/test_v20_r15_it18_citation_landing.py",
    "tests/jupiter/test_v20_r15_it19_q2_rows_and_m14.py",
    "tests/jupiter/test_v20_r15_it20_citation_freeze.py",
    "tests/jupiter/test_v20_r15_it21_heading_anchor.py",
    "tests/jupiter/test_v20_r15_it23_fence_and_argument.py",
    "tests/jupiter/test_v20_r15_it31_c37_and_table_edits.py",
    # J-31a one level up: this file reads the table, so it is IN its own radius.
    # A radius that excludes the instrument declaring it is a radius that cannot
    # see its own edit.
    "tests/jupiter/test_v20_r15_it34_keyed_census.py",
    "tests/mars_v20/test_it20_the_census_still_certifies_its_own_coverage.py",
    "tests/mars_v20/test_v20_r15_it31_the_repairs_of_it29_it30.py",
    "tests/mercury/test_v20_r15_it22_independent_census.py",
    "tests/mercury/test_v20_r15_it24_seal_and_manifest_gap.py",
    "tests/mercury/test_v20_r15_it26_landing_repairs.py",
    "tests/mercury/test_v20_r15_it27_uncited_class.py",
    "tests/mercury/test_v20_r15_it28_named_artifacts.py",
    "tests/mercury/test_v20_r15_it29_withdrawals.py",
    "tests/mercury/test_v20_r15_it30_admission.py",
    "tests/mercury/test_v20_r15_it33_band_only_and_clause_d.py",
    "tests/saturn/test_v20_r15_it14_saturn.py",
    "tests/saturn/test_v20_r15_it19_theory_digest.py",
    "tests/saturn/test_v20_r15_it27_subject_provenance.py",
    "tests/saturn/test_v20_r15_it31_open_corpus_counts.py",
]

#: `[RUN]` PAIRED, both inside this window, same command, `python -m pytest
#: $(TABLE_RADIUS) -q`, this box, branch `v17k-gate0`:
#:   before the M-33a edit, 2026-09-02T14:24:18Z -> :27Z -- 33 RED / 134 GREEN / 2 xfail
#:   after  the M-33a edit, 2026-09-02T14:23:35Z -> :44Z -- 32 RED / 135 GREEN / 2 xfail
#: `-1 RED`, and the node is named:
#: `tests/mercury/...it33...::test_m33a_the_band_only_claim_cites_a_line_that_quotes_the_point`.
#: Unlike `RADIUS_BASELINE = (9, 97)` this pair is two readings of ONE command in
#: ONE window, so the difference is a DEVIATION and not a difference of readings.
TABLE_RADIUS_BASELINE = (32, 135)


def test_the_table_radius_is_complete_and_every_member_reads_the_table() -> None:
    """J-34b. `RADIUS` is a list nobody re-derives; this one is. Two failures,
    both named: a member that stopped reading the table, and a reader that was
    never added."""
    derived = sorted(
        p.relative_to(ROOT).as_posix()
        for p in ROOT.glob("tests/*/test_*.py")
        if "V20_R15_THEORY_TABLE" in p.read_text(encoding="utf-8", errors="replace"))
    assert derived == sorted(TABLE_RADIUS), (
        "unlisted readers: %s / listed non-readers: %s"
        % (sorted(set(derived) - set(TABLE_RADIUS)),
           sorted(set(TABLE_RADIUS) - set(derived))))
    assert len(TABLE_RADIUS) == 23


def test_a_radius_is_a_property_of_an_EDIT_not_of_an_OFFICE() -> None:
    """J-34b, as a measurement. The it.32 edit wrote both corpora and declared
    one radius. Fourteen files read the table and NOT the journal; they are what
    that declaration did not cover."""
    from tests.jupiter.test_v20_r15_it31_c37_and_table_edits import RADIUS

    both = set(RADIUS) & set(TABLE_RADIUS)
    table_only = set(TABLE_RADIUS) - set(RADIUS)
    assert len(both) == 9 and len(table_only) == 14, (len(both), len(table_only))
    assert len(set(RADIUS) | set(TABLE_RADIUS)) == 28
    # the union is the radius of the it.32 edit; NEITHER list alone was.
    assert set(RADIUS) != set(TABLE_RADIUS)
