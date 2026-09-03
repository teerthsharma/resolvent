"""it.29 -- THE ENFORCEMENT NODE THAT COULD NOT FAIL, AND WHAT EARNING THE FIELD MEANS.

CORRECTION 35, against this office. `live_hits()` as shipped at it.27 opened with

    if any(literal in lits for lits in dead_literals().values()):
        return []

and every literal the enforcement node iterates is drawn from `dead_literals()`,
so the guard was always true, the return was always `[]`, and
`assert live_hits(lit) == []` could not fail for any input the node can build.
The node reported `8 passed` and the coordinator published that as evidence the
field works. `V-16`: the instrument that cannot measure, reporting a pass.

This file is the check the it.27 file could not perform on itself. It reconstructs
the old guard and DEMONSTRATES its vacuity against the narrowed one, rather than
asserting it in prose.

RULING J-29a -- THE SHORT-CIRCUIT IS NARROWED TO THE DECLARING ROW. Repaired in
  `live_hits` itself. The index-row exclusion is already a superset of "the row
  currently under test", so the narrowing IS the deletion of the blanket guard; a
  row parameter would be a no-op and is not added.

RULING J-29b -- `none` IS A SENTINEL, NOT A SEARCH KEY. C21, C30 and C33 declare
  ``overturns: `none```. The it.27 extraction read the sentinel through the same
  backtick grammar as a literal and handed the grep the word `none`, live 25 times
  in the body. This is the ENTIRE gap between the coordinator's `26 of 41` and the
  nurse's `29 of 44` -- one measurement under two grammars, `44 - 3 = 41`,
  `29 - 3 = 26`. The `62/63/64` phenomenon with the denominator resolved.

RULING J-29c -- DEAD AS A CLAIM vs LIVE AS A STRING. A row may over-claim what it
  retires: C1 withdraws *"no arm crosses"* and declares `0.7071`, but `0.7071` is
  still the live VALUE of BED-M's floor1 in prose that no longer draws the
  withdrawn conclusion from it. `overturns:` names SEARCH KEYS, and a search key
  that is also a live identifier, symbol, value or filename cannot be retired by
  declaration. The grammar narrows: a declared literal must be a PHRASE THAT
  ASSERTS -- long enough that its occurrence is the claim itself. The four rows
  whose only live hits precede their own settling iteration are a separate class
  and are NOT over-claims: the journal is chronological, and a hit in an entry
  written before the correction landed is the record, not a restatement.
"""

from __future__ import annotations

import hashlib
import pathlib
import re

from tests.jupiter.test_v20_r15_it27_star_lands_and_overturns import (
    JOURNAL,
    NONE,
    OVERTURNS_RE,
    dead_literals,
    hits_in,
    index_rows,
    live_hits,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
SETTLED_RE = re.compile(r"\|\s*it\.(\d+)\s*[—-]")


def declared_pairs() -> list[tuple[int, str]]:
    return [(n, l) for n, lits in sorted(dead_literals().items()) for l in lits]


def declaration_digest() -> str:
    """RULING J-30a. A digest over the (row, literal) DECLARATIONS only -- not over
    the index rows -- so it is invariant to row prose and moves exactly when a
    declaration moves. The nodes below are declared against it."""
    blob = chr(10).join("%d|%s" % p for p in sorted(declared_pairs()))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


#: RULING J-30a -- A READING IS DECLARED AGAINST THE DECLARATION SET IT WAS TAKEN ON.
#: it.29 froze `26 of 41` and `15 of 41` naming no state, so the coordinator's it.30
#: grammar edit turned three nodes RED and the RED said nothing about whether the
#: edit was right. `(declarations, unearned, earned)` per declaration digest, both
#: states measured 2026-09-02T18:56Z against journal `11b77f9c0e96b2da`:
READINGS: dict[str, tuple[int, int, int]] = {
    "6817b208c2304e9d": (41, 26, 15),   # the index as it stands, it.29 and it.30
    "9ce562ee317520b5": (31, 7, 24),    # after J-30b's 34 specified literals land
}


def reading() -> tuple[int, int, int]:
    d = declaration_digest()
    assert d in READINGS, (
        "the CORRECTIONS INDEX declarations moved to a set no office has re-taken "
        "(declaration digest %s). That is the blast radius of an index edit and it "
        "is not repaired by re-freezing a number: re-take the census and add the "
        "row to READINGS, per J-30a." % d)
    return READINGS[d]


def _old_guard_live_hits(literal: str) -> list[int]:
    """`live_hits` EXACTLY as it.27 shipped it, reconstructed so its vacuity is
    measured and not asserted. Deleted from production by J-29a."""
    if any(literal in lits for lits in dead_literals().values()):
        return []
    body = JOURNAL.read_text(encoding="utf-8").splitlines()
    indexed = {ln for _, ln in index_rows()}
    return [i for i, ln in enumerate(body, 1) if literal in ln and ln not in indexed]


# ------------------------------------------------- the node could not fail (V-16)

def test_the_it27_guard_returned_empty_for_every_input_the_node_can_construct() -> None:
    """[RUN] THE DEFECT, MEASURED. Not one of the 41 declared literals could ever
    have reached a non-empty return, so `8 passed` measured nothing."""
    pairs = declared_pairs()
    assert pairs, "no row carries `overturns:`; the premise of the defect is gone"
    vacuous = [(n, l) for n, l in pairs if _old_guard_live_hits(l) == []]
    assert len(vacuous) == len(pairs), (
        "the it.27 guard was NOT vacuous for %d of %d -- re-derive Correction 35"
        % (len(pairs) - len(vacuous), len(pairs)))


def test_the_narrowed_node_can_observe_a_violation() -> None:
    """[RUN] The same 41 inputs against the narrowed predicate. RED is not the
    finding here; the finding is that RED is REACHABLE at all."""
    pairs = declared_pairs()
    n_pairs, n_unearned, _ = reading()
    unearned = [(n, l) for n, l in pairs if live_hits(l)]
    assert (len(unearned), len(pairs)) == (n_unearned, n_pairs), (
        "the declared reading for this index is %d of %d unearned; measured %d of "
        "%d. Movement is expected as rows are repaired -- re-take and re-declare "
        "in READINGS, do not silence the node."
        % (n_unearned, n_pairs, len(unearned), len(pairs)))


def test_the_narrowed_node_is_not_red_for_everything() -> None:
    """THE CONTROL that separates a working instrument from a broken one. 15 of the
    41 declarations ARE honest: the literal has zero live hits outside the index."""
    n_pairs, _, n_earned = reading()
    earned = [(n, l) for n, l in declared_pairs() if not live_hits(l)]
    assert len(earned) == n_earned, (
        "%d of %d declarations return empty against a declared %d; the node is "
        "meant to discriminate, not to condemn every row"
        % (len(earned), n_pairs, n_earned))


def test_hits_in_returns_a_hit_on_a_synthetic_body() -> None:
    """THE SMALLEST CHECK THAT THE ASSERTION CAN SEE. The predicate is put on three
    lines of injected text where the answer is known by construction."""
    body = ["| C1 | quotes `X` |", "the body still says X", "unrelated"]
    indexed = {body[0]}
    assert hits_in("X", body, indexed) == [2]
    assert hits_in("X", body, set()) == [1, 2], "the index exclusion is inverted"
    assert hits_in("Y", body, indexed) == []


# ------------------------------------------------------------- J-29b, the sentinel

def test_the_none_sentinel_is_the_whole_26_of_41_vs_29_of_44_gap() -> None:
    """[RUN] The coordinator and the nurse disagreed on the denominator and agreed
    on the defect. This is why: three rows, one grammar apart."""
    raw = []
    for n, ln in index_rows():
        m = OVERTURNS_RE.search(ln)
        if m:
            raw.extend((n, l) for l in re.findall(r"`([^`]+)`", m.group(1)))
    sentinel = [(n, l) for n, l in raw if l == NONE]
    assert [n for n, _ in sentinel] == [21, 30, 33], sentinel
    assert len(raw) == 44 and len(declared_pairs()) == 41
    body = JOURNAL.read_text(encoding="utf-8").splitlines()
    it29 = min(i for i, ln in enumerate(body, 1) if ln.startswith("## it.29"))
    before = [h for h in live_hits(NONE) if h < it29]
    assert len(before) == 25, (
        "the word `none` was live %d times in the body written before it.29; the "
        "it.27 grammar handed that to the grep as a retired claim" % len(before))
    assert len(live_hits(NONE)) > len(before), (
        "SELF-REFERENCE, J-30d: entries written after this ruling quote the "
        "sentinel while stating it, so a whole-body count of `none` cannot stay "
        "frozen. The cut is what makes the reading re-runnable.")


# ------------------------------------- J-29c, dead as a claim vs live as a string

def _settled_at(row: str) -> int | None:
    found = SETTLED_RE.findall(row)
    return int(found[-1]) if found else None


def test_four_of_the_26_are_record_and_not_restatement() -> None:
    """RETIRED AT it.30 IN FAVOUR OF `test_v20_r15_it30_phrase_grammar.
    py::test_the_specified_literals_are_not_restated_after_their_row_settled`.

    The finding stands and the CUT moved (J-30c): this node split at the START of
    the settling section, which counts the correction's own quotation of the claim
    it retires as the body still asserting it -- C24 and C26 are exactly that. The
    successor splits at the END, is measured for BOTH declaration sets, and so does
    not have to be re-frozen every time a row is narrowed. Kept as a stub so the
    supersession is recorded in the file that shipped the defect."""
    from tests.jupiter.test_v20_r15_it30_phrase_grammar import restated
    assert restated(24, "Off by one and by **fifty-eight**") == []


# ------------------------------------------------- J-26b, STAR_DIGESTS' one home

def test_the_star_digests_have_exactly_one_home() -> None:
    """J-26b's ruling, applied to J-26c's datum. The it.20 dict is the home because
    `lands()` consumes it; the it.26 dict is a re-keyed VIEW. Observed by looking
    for a second hard-coded copy, so re-inlining one turns this RED."""
    from tests.jupiter.test_v20_r15_it20_citation_freeze import STAR_DIGESTS as HOME
    from tests.jupiter.test_v20_r15_it26_live_claim import STAR_DIGESTS as VIEW

    assert len(HOME) == 6 and len(VIEW) == 6
    assert set(VIEW.values()) == {d[:16] for d in HOME.values()}
    src = (ROOT / "tests/jupiter/test_v20_r15_it26_live_claim.py").read_text("utf-8")
    assert not re.search(r'"[0-9a-f]{16}"', src), (
        "a second hard-coded copy of the `:*` digests is back in the it.26 file; "
        "the two agreed at it.26 only because the prefixes happened to match")
