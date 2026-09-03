"""it.30 -- THE PHRASE GRAMMAR, APPLIED TO ALL 34 ROWS, WITH ITS BLAST RADIUS.

J-29c specified the grammar and did not enforce it. The coordinator then applied a
MECHANICAL first-N-words extraction to all 34 CORRECTIONS INDEX rows and four it.29
nodes went RED, because those nodes freeze specific literals and their measured
membership. The edit was reverted; the index is back at declaration digest
`6817b208c2304e9d`.

RULING J-30a -- A READING IS DECLARED AGAINST THE DECLARATION SET IT WAS TAKEN ON.
  The it.29 nodes froze `26 of 41` and `15 of 41` against today's index and named no
  state, so ANY row edit turns them RED and the RED says nothing about whether the
  edit was good. They are re-declared in `READINGS` (in the it.29 file, which owns
  them), keyed by `declaration_digest()` -- a digest over the (row, literal) pairs
  only, so it is invariant to row prose and moves exactly when a declaration moves.
  Two states are declared: today's, and the one this file specifies. A third state
  is RED with the blast-radius instruction, which is the correct answer to an
  un-re-taken edit.

RULING J-30b -- THE 34 LITERALS ARE SPECIFIED HERE, EXACTLY, IN `SPEC`. A
  first-N-words rule is not the grammar. The literal is the ASSERTING PHRASE, taken
  from the row's own claim column by judgment, one per row. 20 rows already carry a
  phrase and are repeated VERBATIM so the coordinator's edit touches 13 cells and
  not 34. Three rows keep the `none` sentinel.

  The shape rule, enforced in `test_every_declared_literal_on_the_index_is_phrase_
  shaped`: a space, >= 12 characters, and no backtick (the field's own delimiter)
  and no pipe (the table's). RED against today's index -- 21 of 41 declarations are
  token-shaped -- and GREEN when `SPEC` lands.

RULING J-30c -- THE SETTLING ENTRY'S OWN QUOTE IS THE WITHDRAWAL, NOT A RESTATEMENT.
  The it.29 chronology cut was the START of the settling `## it.N` section, which
  counts the correction's own quotation of the claim it retires as the body still
  asserting it. C24 and C26 are exactly that: the it.16 entry quotes it.14's
  sentence in order to withdraw it. The cut moves to the END of the settling section
  (the first heading of a LATER iteration). `restated()` is that predicate.

WHAT THE GRAMMAR BUYS, MEASURED (2026-09-02T19:05Z, journal `2bdcfc57a332bbff`):
  today  41 declarations, 21 token-shaped, 26 unearned, 13 restated after settling;
  SPEC   31 declarations,  0 token-shaped,  7 unearned,  3 restated after settling.
  11 of today's 13 restatements are token-shaped cells. The defect is the token.
"""

from __future__ import annotations

import re

from tests.jupiter.test_v20_r15_it27_star_lands_and_overturns import (
    JOURNAL,
    NONE,
    dead_literals,
    hits_in,
    index_rows,
)
from tests.jupiter.test_v20_r15_it29_overturns_can_fail import (
    READINGS,
    declaration_digest,
    declared_pairs,
)

SETTLED_RE = re.compile(r"\|\s*it\.(\d+)\s*[—-]")

#: J-30b. One asserting phrase per CORRECTIONS INDEX row. Rows whose current cell is
#: already phrase-shaped are repeated verbatim; the 13 marked NARROWED are the edit.
SPEC: dict[int, str] = {
    1: "no arm crosses",                                                    # NARROWED from `0.7071`
    2: "the eval draw's own **sign census**",                               # NARROWED from `frac_gate_annihilated`
    3: "an **8x arena cost**",                                              # NARROWED from `N=65`
    4: "**EXIT B costs zero and is already met**",
    5: "SATURN and MERCURY **independently** refuted the determinism blocker",
    6: "JUPITER and SATURN reached N=2 **independently, from opposite directions**",
    7: "two wings the record produced, **one only prose did**",
    8: "K6 shrank — two tokens **gained producers**",
    9: "a **pole precisely on the unit circle**, marginal stability, an almost-all-pass filter",
    10: "live-band decay **under 3% per position**",
    11: "W1's fifteen failures are **M1's predicted descent to the corner**",
    12: "leap ledger open with **five MARS rows",                           # NARROWED from `L-M1..L-M5`
    13: "**12 of 16**",
    14: "7 of 8 fresh vs",                                                  # NARROWED from `softmax`, `p = 6.730e-04`
    15: "used as an **information floor**",                                 # NARROWED from `floor_1`
    16: "and that separation survives everything found this iteration",     # NARROWED from `12 of 16`, `softmax`, `p = 6.730e-04`
    17: "every GPU-second and cost ratio quoted from it.3 onward",          # NARROWED from `0.3455`, `10.17x`, `41.9x`
    18: "**N = 1 primitive**, not 3",
    19: "every pair separates by",                                          # NARROWED from `>= 0.30`
    20: "Annex: M14 at **F1, HOW-BAD 108x**, death re-attributed to V-25",
    21: NONE,
    22: "the Q2/W1 pointers",                                               # NARROWED from two file:line pointers and `Q2/W1`
    23: "**the check confirms him**",
    24: "Off by one and by **fifty-eight**",
    25: "this office quoted **JUPITER's unprompted count** and did not check it",
    26: "repeated in **three places each**",
    27: "**both branches fired in test**",
    28: "five :466 sites",                                                  # NARROWED from `:466`
    29: "premises, read as file sizes",                                     # NARROWED from `6,601`, `10,147`
    30: NONE,
    31: "the INSPECTOR is **still auditing**",
    32: "Correction 31's own numbers",                                      # NARROWED from `36,786`, `14m45s`, `15m10s`
    33: NONE,
    34: "Occurrences 129 -> 120, pointers 103 -> 96",
}

#: The declaration digest `SPEC` produces once the coordinator applies it. Computed,
#: not guessed: it depends only on the (row, literal) pairs, so it is the same
#: whatever prose surrounds the cell.
SPEC_DIGEST = "9ce562ee317520b5"


def phrase_shaped(literal: str) -> bool:
    """J-30b. A phrase that asserts: a space, >= 12 characters, and none of the two
    characters the field and the table use as delimiters."""
    return (" " in literal and len(literal) >= 12
            and "`" not in literal and "|" not in literal)


def _body() -> list[str]:
    return JOURNAL.read_text(encoding="utf-8").splitlines()


def _section_starts() -> list[tuple[int, int]]:
    out = []
    for i, ln in enumerate(_body(), 1):
        m = re.match(r"^## it\.(\d+)\b", ln)
        if m:
            out.append((int(m.group(1)), i))
    return out


def settled_at(row_text: str) -> int | None:
    found = SETTLED_RE.findall(row_text)
    return int(found[-1]) if found else None


def restated(row: int, literal: str) -> list[int]:
    """J-30c. Live hits at or after the END of the settling iteration's section --
    the body asserting a withdrawn claim, as opposed to the record of it having been
    asserted, or the correction quoting it in order to withdraw it."""
    rows = dict(index_rows())
    n = settled_at(rows[row]) if row in rows else None
    if n is None:
        return []
    heads = _section_starts()
    later = [i for m, i in heads if m > n]
    cut = min(later) if later else len(_body()) + 1
    indexed = {ln for _, ln in index_rows()}
    return [h for h in hits_in(literal, _body(), indexed) if h >= cut]


# ------------------------------------------------------- J-30b, the enforcement node

def test_every_declared_literal_on_the_index_is_phrase_shaped() -> None:
    """RED until the coordinator applies `SPEC`. THIS IS THE ENFORCEMENT NODE for
    the grammar J-29c specified and did not enforce."""
    tokens = [(n, l) for n, l in declared_pairs() if not phrase_shaped(l)]
    assert not tokens, (
        "%d of %d declarations are TOKEN-shaped: the cell names the string the row "
        "retires instead of the phrase that asserts the retired claim, so the "
        "declaration retires a constant that outlives the claim. Replace each with "
        "the literal specified in SPEC (this file). token-shaped:\n%s" % (
            len(tokens), len(declared_pairs()),
            "\n".join("  C%-3d %r" % t for t in tokens)))


def test_the_specified_replacements_satisfy_the_grammar_they_demand() -> None:
    """The demand is not unsatisfiable: every one of the 34 specified cells passes
    the rule the node above enforces, and the three sentinels are exempt by name."""
    assert sorted(SPEC) == [n for n, _ in index_rows()], (
        "SPEC does not cover the index one-for-one: %d rows, %d specified"
        % (len(index_rows()), len(SPEC)))
    bad = [(n, l) for n, l in SPEC.items() if l != NONE and not phrase_shaped(l)]
    assert not bad, bad
    assert sorted(n for n, l in SPEC.items() if l == NONE) == [21, 30, 33]


def test_the_grammar_rejects_the_cells_it_is_meant_to_reject() -> None:
    """THE CONTROL. `phrase_shaped` is not true for everything and not false for
    everything, and it rejects the coordinator's four mechanical extractions."""
    assert phrase_shaped("no arm crosses") and phrase_shaped(SPEC[9])
    for bad in ("0.7071", "softmax", ":466", "N=65", "12 of 16", "14m45s",
                "a `b` c", "a | b"):
        assert not phrase_shaped(bad), bad
    assert phrase_shaped("the eval draw's") and phrase_shaped("EXIT B costs"), (
        "MEASURED LIMIT OF THE SHAPE RULE: two of the four mechanical truncations "
        "the coordinator produced PASS it. Shape is necessary and not sufficient, "
        "which is exactly why the 34 literals are specified by judgment in SPEC and "
        "frozen there rather than derived by a rule.")


# ---------------------------------------------------------- J-30c, restatement

def test_the_specified_literals_are_not_restated_after_their_row_settled() -> None:
    """[RUN] What the grammar BUYS, frozen with exact membership. Today's 41
    declarations are restated after settlement in 13 places; the 31 in SPEC in 3.
    Both exceptions are named, so this moves if either is repaired."""
    today = [(n, l) for n, l in declared_pairs() if restated(n, l)]
    assert len(today) == 13, (
        "today's index restates %d declared literals after their settling section; "
        "it.30 measured 13 (re-taken 19:05Z) -- re-date the reading: %s" % (len(today), today))
    assert len([1 for n, l in today if not phrase_shaped(l)]) == 11, today
    spec = [(n, l) for n, l in SPEC.items() if l != NONE and restated(n, l)]
    assert spec == [(1, "no arm crosses"),
                    (10, "live-band decay **under 3% per position**"),
                    (19, "every pair separates by")], spec


def test_restated_is_narrower_than_live_and_both_can_observe_a_violation() -> None:
    """CONTROL for J-30c. The chronology cut must SUBTRACT hits and not all of
    them, or the ruling is either a no-op or an amnesty."""
    pairs = declared_pairs()
    live = [(n, l) for n, l in pairs if hits_in(l, _body(), {ln for _, ln in index_rows()})]
    rest = [(n, l) for n, l in pairs if restated(n, l)]
    assert set(rest) < set(live), "the cut is not a proper narrowing of live hits"
    assert rest, "the cut retired every hit; a predicate true of nothing is J-29a again"
    assert restated(24, "Off by one and by **fifty-eight**") == [], (
        "C24's only live hit is the it.16 correction quoting it.14 in order to "
        "withdraw it; if that counts as a restatement the cut is still J-29c's")


# ----------------------------------------------- J-30a, the blast radius, declared

def test_the_blast_radius_of_the_spec_edit_is_declared_before_it_lands() -> None:
    """The four it.29 nodes the coordinator's edit broke read `READINGS` now. Both
    states are declared HERE, before the edit, which is what computing the blast
    radius means: today's index and the index SPEC produces."""
    assert declaration_digest() in READINGS, declaration_digest()
    assert SPEC_DIGEST in READINGS, (
        "the it.29 readings carry no entry for the declaration set SPEC produces, "
        "so applying SPEC would turn the it.29 count nodes RED on a change this "
        "office asked for")
    assert READINGS[SPEC_DIGEST][0] == 31 and READINGS[SPEC_DIGEST][1] == 7, (
        "SPEC leaves 31 declarations of which 7 keep a live hit somewhere in the "
        "body; that is the re-taken reading, not a repair to green")
    assert dead_literals(), "the premise is gone: no row carries `overturns:`"
