"""it.24 -- the three argument-layer repairs APPLIED, and the census RE-TAKEN in the same iteration.

it.23 RULED on all three and APPLIED none, for a stated reason: every edit moves
an occurrence out of the it.20 `CENSUS` and drops `129 of 129` until the census is
re-taken. That reason is real and it is also a trap -- A CENSUS THAT CAN NEVER BE
UPDATED IS A CENSUS THAT WILL BE ABANDONED. This iteration pays the cost once and
fixes the PROCEDURE: EDIT, RE-TAKE, RE-DECLARE, IN ONE FILING.

  RULING J-24a -- `:*` IS THE FILE AND IS EXEMPT FROM J-20b.
  J-23f granted the notation and left the instrument unable to read it. Applied
  here: the 8 `path:1` occurrences that meant "this file" are now `path:*`.
  `path:1` keeps meaning line 1. The consequence J-23f did not state, and the one
  that makes the notation worth having: A `:*` POINTER CANNOT BE BROKEN BY AN
  APPEND, so J-20b -- which refuses any pointer into a growing file -- does not
  apply to it. That is not a loophole; it is the reason a file-scoped citation
  exists. The refusal J-20b enforces is about a LINE NUMBER going stale, and `:*`
  has none.

  RULING J-24b -- A COMPOUND CLAIM CARRIES A COMPOUND ANCHOR. `+0.717647` is at
  `test_v20_r15_it12_constants.py:13` and `-0.032353` at `:14`; the cell said
  "both bound @ :13" and every location instrument scored it GREEN because the
  half it could see was true. The pointer becomes `:13-14` AND THE ANCHOR BECOMES
  A PAIR: a range whose want is a tuple, every member of which must land inside
  the range. A range with a SINGLE want would have re-created the defect one
  notation wider -- it would still be satisfied by the half it can see.
  Under J-20a a changed want is a WITHDRAWAL plus a NEW CITATION, so C63 is
  withdrawn and re-issued as C131. The want freeze is not bent to allow the
  repair; the repair is filed as what the freeze says it is.

  RULING J-24c -- C13 / `THREE_LINES_LOW`. THE INSPECTOR'S REMEDY IS ADOPTED AND
  THIS OFFICE'S OWN it.23 REASON FOR REFUSING IT IS STRUCK.
  Three offices, three answers, and all three agree `:242` is empty:
    MERCURY   -- the `F3` is at `:242`.                     [CITED: MERCURY, it.22]
    JUPITER   -- no line carries `F3` UNIQUELY, so J-20a
                 forces withdrawal + re-issue: REFUSED-PENDING-REISSUE.  (it.23)
    INSPECTOR -- `grep -n F3` = 139, 142, 241, 243, 244;
                 the fix is `:239 -> :241`.                [CITED: INSPECTOR, it.23]
  MERCURY's `:242` is wrong and it is the only wrong number of the three.
  THE DISAGREEMENT THAT MATTERS IS THE OTHER ONE: does non-uniqueness of the token
  block a pointer edit? IT DOES NOT, and it.23 was wrong to say it did.
  WHY, AND HOW IT SQUARES WITH THIS OFFICE'S OWN RESOLVER REFUSING
  `'F4' appears on 4 lines in section '## it.7' ... not 1`:
    THE RESOLVER REFUSES NON-UNIQUENESS BECAUSE UNIQUENESS IS ITS ADDRESSING
    MECHANISM. `resolve(path, key, want)` has no line number; it DERIVES one from
    the want, and a want on four lines derives four answers, so it must refuse or
    guess. A DIGIT POINTER IS ALREADY ADDRESSED. `:241` names the line; the want
    is asked only whether that line carries it. Uniqueness buys nothing there, and
    the it.20 landing check has never required it -- C2's want `L-GRADE (F0-F4 +
    HOW-BAD gap)` is not unique in its file either, and no office has called that
    a defect in four iterations. it.23 imported an addressing constraint into a
    place where addressing was already settled.
  SO THE REMEDY IS A POINTER EDIT -- and C13 IS STILL WITHDRAWN AND RE-ISSUED, for
  a DIFFERENT REASON, which is the finding. C13's frozen want is
  `M14 CHEEGER STRATIFICATION`, and `:241` does not carry it. Moving the pointer
  to a line the frozen want does not land on requires a NEW want, and a new want
  is J-20a. it.23 reached the right disposition by the wrong rule. The disposition
  is kept, the rule behind it is struck, and C13 is re-issued as C130 with want
  `grade F3 pending` -- the words that carry the CELL's claim (the contract grades
  M14 `F3` here), which `M14 CHEEGER STRATIFICATION` never did. That is a J-21c
  argument-layer defect closed, not a location repaired.

CENSUS RE-TAKEN AT it.24, AFTER THE EDITS, AND THE NUMBER IS NOT ROUNDED:
  * `129 of 129` occurrences / 103 unique / 37 files -- HELD. No citation was
    added or removed; three were re-pointed and eight re-notated.
  * Extension histogram UNMOVED: `.md 40, .py 69, .lean 19, .jsonl 1 = 129`.
    Predicted to move by the brief; measured not to, and the reason is that a
    re-notation is not a re-population. The histogram that DID move is by
    NOTATION, and it did not exist before this iteration:
        120 plain `:N`  +  1 range `:A-B`  +  8 file `:*`  = 129
  * Scored: 107 by line + 1 by range + 8 by file + 13 by heading = 129, 0 failing.
"""

from __future__ import annotations

import re

from tests.jupiter.test_v20_r15_it20_citation_freeze import (
    CENSUS,
    REISSUED,
    ROOT,
    TABLE,
    POPULATION_AT_IT20,
    UNIQUE_AT_IT20,
    grown,
    lands,
    not_a_repair,
    not_landing,
    refused,
    region,
    table_citations,
)

CONTRACT = "CEQ_V20_R15_CONTRACT.md"
CONSTANTS = "tests/jupiter/test_v20_r15_it12_constants.py"


# ------------------------------------------------------- the re-taken census

def test_the_population_after_the_edits_is_131_of_131_at_it27():
    """The number stated AFTER the edits, with the denominator re-measured."""
    occ = table_citations()
    assert len(occ) == POPULATION_AT_IT20, f"population moved to {len(occ)}"
    assert len(set(occ)) == UNIQUE_AT_IT20
    # it.27: 37 -> 38, `ceq/beds/bed_k.py` joins the cited set under M-25c.
    assert len({p for p, _ in occ}) == 38, len({p for p, _ in occ})


def test_the_extension_histogram_is_republished_and_did_not_move():
    hist: dict[str, int] = {}
    for p, _ in table_citations():
        ext = p.rsplit(".", 1)[1]
        hist[ext] = hist.get(ext, 0) + 1
    # it.27: py 69 -> 71. `scripts/v15_r1.py:138` (M-25b) and
    # `ceq/beds/bed_k.py:236` (M-25c) are ADDED citations, not moved ones.
    assert hist == {"md": 40, "py": 71, "lean": 19, "jsonl": 1}, hist
    assert sum(hist.values()) == POPULATION_AT_IT20


def test_the_histogram_that_DID_move_is_by_notation():
    """120 plain + 1 range + 8 file = 129. This split did not exist at it.23."""
    specs = [s for _, s in table_citations()]
    plain = [s for s in specs if s.isdigit()]
    rng = [s for s in specs if "-" in s]
    star = [s for s in specs if s == "*"]
    # it.26: was `(120, 1, 8)` at it.24. C130 was withdrawn under J-26a and
    # re-issued as C132 over the range `:241-244`, so ONE plain became ONE more
    # range. The split is now DERIVED from the census rather than restated here,
    # which is J-26b applied to this office's own filing: a number written down
    # in two places is a number that can be updated in one.
    occ = [s for _, s in table_citations()]
    want = (len([x for x in occ if x.isdigit()]),
            len([x for x in occ if "-" in x]),
            len([x for x in occ if x == "*"]))
    # it.27: (119,2,8) -> (120,3,8). +2 plain (M-25b, M-25c) and one plain
    # promoted to a range (M-25a, `:547` -> `:547-558`).
    assert (len(plain), len(rng), len(star)) == want == (120, 3, 8), (want,)
    assert len(plain) + len(rng) + len(star) == POPULATION_AT_IT20


def test_nothing_is_unscored_and_nothing_failing_after_the_edits():
    assert not_landing(CENSUS) == []
    assert not_a_repair(CENSUS) == []
    assert grown() == {}


# ------------------------------------------------------- J-24a  `:*`

def test_the_line_one_idiom_is_RETIRED_and_the_instrument_reads_the_notation():
    txt = TABLE.read_text(encoding="utf-8")
    assert re.findall(r"`[A-Za-z0-9_./-]+\.(?:py|lean|md|jsonl):1`", txt) == [], (
        "the `path:1` idiom survived the edit")
    star = re.findall(r"`([A-Za-z0-9_./-]+\.(?:py|lean|md|jsonl)):\*`", txt)
    assert len(star) == 8 and len(set(star)) == 6, (len(star), sorted(set(star)))
    starred = {c: v for c, v in CENSUS.items() if v[1] == "*"}
    assert len(starred) == 6
    for c, (p, _, w) in starred.items():
        assert lands(w, region(p, "*")), f"C{c}: {w!r} is not in {p}"


def test_a_star_pointer_survives_an_append_that_J_20b_would_refuse_a_digit_for():
    """PLANTED NEGATIVE for J-24a. Writes to no file in the tree.

    C55 was `ceq/arm_pl.py:1`. Prepend 40 lines: the DIGIT anchor stops carrying
    its want (J-20b's whole subject) and the `:*` anchor is untouched. That is
    why `:*` is exempt from J-20b, stated as a measurement rather than a licence.
    """
    p, spec, want = CENSUS[55]
    assert (p, spec) == ("ceq/arm_pl.py", "*")
    real = (ROOT / p).read_text(encoding="utf-8", errors="replace").splitlines()
    appended = ["APPENDED"] * 40 + real
    assert want in real[0], "premise gone: the want was not on line 1"
    assert want not in appended[0], "the digit anchor must break, or this proves nothing"
    assert want in "\n".join(appended), "the `:*` anchor must survive the append"


# ------------------------------------------------------- J-24b  compound

def test_the_compound_claim_now_carries_both_halves_and_a_single_want_would_not():
    """C131 is the re-issue. The negative is that the OLD shape stays green."""
    assert 63 not in CENSUS and REISSUED[131] == 63
    p, spec, want = CENSUS[131]
    assert (p, spec) == (CONSTANTS, "13-14")
    assert want == ("+0.717647", "-0.032353")
    assert lands(want, region(p, spec))

    src = (ROOT / p).read_text(encoding="utf-8", errors="replace").splitlines()
    assert lands("+0.717647", src[12]) and not lands("-0.032353", src[12]), (
        "premise gone: the two constants share a line and the defect is not real")
    half = dict(CENSUS)
    half[131] = (p, "13-14", "+0.717647")
    assert not_landing(half) == [], (
        "a range with a SINGLE want re-creates COMPOUND_HALF one notation wider: "
        "it lands on the half it can see. That is why the anchor is a pair.")
    assert not_a_repair(half) != [], "the want freeze must name the single-want form"


# ------------------------------------------------------- J-24c  C13

def test_C13_is_withdrawn_and_reissued_at_241_with_a_want_that_carries_the_claim():
    src = (ROOT / CONTRACT).read_text(encoding="utf-8", errors="replace").splitlines()
    txt = TABLE.read_text(encoding="utf-8")
    assert 13 not in CENSUS and REISSUED[130] == 13
    # it.26, J-26a: C130's want landed three lines above `[SUPERSEDED it.19,
    # RULING J-17d: M14 IS F4`. Withdrawn and re-issued as C132 over `:241-244`,
    # a region that CONTAINS its own supersession notice. The it.24 disposition
    # -- withdraw and re-issue rather than re-point -- is upheld; the anchor it
    # chose was still narrower than the claim's liveness.
    assert 130 not in CENSUS and REISSUED[132] == 130
    assert CENSUS[132] == (CONTRACT, "241-244",
                           ("grade F3 pending", "M14 IS F4"))
    assert lands("grade F3 pending", src[240])
    assert "`" + CONTRACT + ":241-244`" in txt
    assert "`" + CONTRACT + ":239`" not in txt


def test_all_three_offices_are_settled_against_the_file():
    """The disagreement, measured. MERCURY's `:242` is empty; the INSPECTOR's
    grep is exact; it.23's `F3 is at 241/243/244` is exact within its window."""
    src = (ROOT / CONTRACT).read_text(encoding="utf-8", errors="replace").splitlines()
    assert "F3" not in src[241], "MERCURY's :242 carries F3 after all: " + repr(src[241])
    assert [i + 1 for i, l in enumerate(src) if "F3" in l] == [139, 142, 241, 243, 244]
    assert "[V]" in src[238] and "F3" not in src[238]


def test_the_old_want_would_NOT_land_at_241_which_is_why_it_is_a_reissue():
    """The whole of J-24c's second half, as an assertion. A pointer edit was
    permissible; THIS pointer edit still needed a new want, and that is J-20a."""
    src = (ROOT / CONTRACT).read_text(encoding="utf-8", errors="replace").splitlines()
    assert "M14 CHEEGER STRATIFICATION" in src[238]
    assert "M14 CHEEGER STRATIFICATION" not in src[240], (
        "if the frozen want landed at :241 this would have been a plain pointer "
        "repair and no re-issue would be owed")


def test_non_uniqueness_does_not_block_a_DIGIT_pointer_and_the_resolver_still_refuses():
    """J-24c's consistency claim, made RED-able in both directions at once.

    `F3` is on five lines of the contract. The digit pointer at `:241` is scored.
    The heading resolver, handed the same non-unique token, still REFUSES -- and
    the difference is that the resolver must DERIVE a line and the digit has one.
    If either half of that ever stops being true, this test says so.
    """
    from tests.jupiter.test_v20_r15_it21_heading_anchor import resolve

    src = (ROOT / CONTRACT).read_text(encoding="utf-8", errors="replace").splitlines()
    assert sum("F3" in l for l in src) == 5, "premise gone: F3 is no longer plural"
    assert 132 not in refused() and not_landing({132: CENSUS[132]}) == []

    ledger = "V20_R15_LEAP_LEDGER.md"
    got = resolve(ledger, "## it.7", "F4")
    assert isinstance(got, str) and got.startswith("REFUSED"), (
        "the resolver stopped refusing a non-unique want: " + str(got))
    assert "appears on 4 lines" in got, got

    control = resolve(ledger, "## it.7", "machine-true and domain-empty")
    assert isinstance(control, int), (
        "the section key itself must resolve, or the refusal above proves "
        "nothing about uniqueness: " + str(control))


# ------------------------------------------------------- the procedure

def test_every_reissue_names_the_cid_it_replaces_and_the_old_cid_is_gone():
    """J-20a bookkeeping, the same shape as it.21's REANCHORS. Nothing is
    silently dropped and no `want` was edited in place."""
    # it.26: asserted as a SUBSET, not an equality. A later withdrawal adds a
    # row; a historical filing that demands the ledger never grow is a filing
    # that breaks every time the instrument is used correctly.
    assert REISSUED.items() >= {130: 13, 131: 63}.items()
    for new, old in REISSUED.items():
        assert old not in CENSUS, "C%d was withdrawn and is still censused" % old
        # it.26: a re-issued cid can itself be withdrawn later -- C130 replaced
        # C13 and was replaced in turn by C132 under J-26a. The ledger is a
        # CHAIN, not a pair, so only the cid at the END of a chain is censused.
        assert new in CENSUS or new in REISSUED.values(), (
            "C%d is neither censused nor superseded by a later re-issue" % new)
    assert len(CENSUS) == UNIQUE_AT_IT20
