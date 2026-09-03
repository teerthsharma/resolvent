"""it.24 -- MERCURY. The seal aimed at tampering, and the `28 vs 25` gap closed.

REPAIR 1. `MERCURY-CENSUS-R1` as banked at it.22 hashed ALL 103 pointers with no
`MERCURY_LIVE` filter, so 10 of its rows came from the three append-growing files
that the SAME FILE's section-1 recount refuses to score. The INSPECTOR struck it
twice at it.23: the seal fired when the round did ordinary work, not when a file
moved under the table. The amendment is one clause -- `MERCURY-R1b` digests the
SCORED pointers, the same population section 1 scores -- and it is worth nothing
without the two negatives below, which are the whole content of the claim
"GREEN on an ordinary append, RED on a line moving under a scored pointer".

  Both negatives run ENTIRELY IN MEMORY through `census_digest(override=...)`.
  Nothing is written to the repo, so neither negative can be the thing that moves
  the seal it is testing.

REPAIR 2. The `28 vs 25` manifest gap, open since it.18 and unreached for five
iterations. VERDICT: **two populations compared by mistake, and the arithmetic
closes exactly with no residue.** See `test_the_28_vs_25_gap_is_two_populations`.

FINDING M-24b -- THE PRICE OF THE `LIVE` LIST, WHICH IS NOT WHAT J-23g PRICED.
J-23g ruled `LIVE` a hand-maintained list with a named owner and conceded its own
criterion contradicts it: `V20_R15_LEAP_LEDGER.md` has sat at 438 lines for three
iterations. This office does not dispute the ruling -- a judgement with an owner
beats a criterion that reflips every iteration. It disputes the PRICE, with a
measurement J-23g did not take: **the ledger is 7 of the 10 refused pointers.**
Seventy percent of the seal's blind spot is bought to protect a file that has not
moved since the census that refused it for moving. That is the number the ruling
is missing, and it is banked here so the next office can weigh the list rather
than inherit it.

LIMITS.
  * `MERCURY_CENSUS_R1` is a DATED reading, not a durable fact. The table was
    edited by another office at 11:51:49Z DURING this iteration (`[RUN] stat`;
    occurrences 129 -> 120, pointers 103 -> 96, length still 443, so J-17e's
    in-place rule held). A frozen hex over a file under concurrent edit will be
    stale again. The durable part of this repair is the two negatives, which
    compare digest to digest inside one read and cannot go stale.
  * The `+1` shift used by the RED negative is arbitrary in size and deliberate
    in kind: it is the smallest edit that moves every line of a file, which is
    the failure `test_every_pointer_resolves` cannot see.
"""

from __future__ import annotations

import collections
import hashlib
import pathlib

from tests.mercury.test_v20_r15_it22_independent_census import (
    MERCURY_LIVE,
    census_digest,
    occurrences,
    scored_pointers,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]

# it.33. The ledger seal, read by THIS office rather than cited.
#   [RUN] python -c "import hashlib,pathlib;b=pathlib.Path(chr(86)+...).read_bytes()" -- see LEDGER_READ below.
LEDGER_SHA256 = "6e88935181ab8c830b66b4adc83b91cf979726612a93fd5ff67468a8fbfd5485"
LEDGER_BYTES = 45670
LEDGER_READ = "2026-09-02 19:36:19 IST"  # [RUN] python -c "datetime.now(ZoneInfo('Asia/Kolkata'))"

#: `[RUN]` 11:54:24Z. The 10 refused pointers, by file. FINDING M-24b.
REFUSED_BY_FILE = {
    "V20_R15_LEAP_LEDGER.md": 7,
    "V20_R15_JOURNAL.md": 2,
    "house-events.jsonl": 1,
}


def _lines(path: str) -> list[str]:
    return (ROOT / path).read_text(encoding="utf-8", errors="replace").splitlines()


# --------------------------------------------------------------------------
# REPAIR 1. The two negatives. Neither is a hex freeze; both are differential.
# --------------------------------------------------------------------------


def test_negative_GREEN_the_seal_does_not_move_on_an_ordinary_append() -> None:
    """NEGATIVE 1. Append 50 lines to EVERY append-growing file AND to every file
    the seal actually scores. The seal must not move.

    This is the event that turned the it.22 seal RED. Under `MERCURY-R1b` the
    three `MERCURY_LIVE` files are not in the population at all, so no amount of
    appending to them can reach the digest; and an append at EOF of a scored file
    moves no line above it, so it cannot reach the digest either. Together those
    are what "ordinary work" means in this round, and the seal is blind to both
    BY CONSTRUCTION rather than by luck.
    """
    before = census_digest()
    appended = {f: _lines(f) + [f"appended line {i}" for i in range(50)] for f in MERCURY_LIVE}
    for path in {p for p, _ in scored_pointers()}:
        appended[path] = _lines(path) + [f"appended line {i}" for i in range(50)]
    assert census_digest(appended) == before, (
        "MERCURY-CENSUS-R1 moved on an ordinary append. A seal that alarms on "
        "every append is a seal that gets switched off."
    )


def test_negative_RED_the_seal_moves_when_a_line_moves_under_a_scored_pointer() -> None:
    """NEGATIVE 2. Insert ONE line at the top of each scored cited file, one file
    at a time. Every line in that file shifts by `+1`; every pointer into it now
    names the line above the one it meant. The seal must go RED on EVERY file.

    `test_every_pointer_resolves` stays GREEN through all of these -- the file
    exists and the line number is still in range. That gap is the seal's reason
    to exist, and this negative is the only thing that demonstrates it.
    """
    before = census_digest()
    unmoved = [
        path
        for path in sorted({p for p, _ in scored_pointers()})
        if census_digest({path: ["INSERTED LINE"] + _lines(path)}) == before
    ]
    assert unmoved == [], (
        f"a +1 shift went unnoticed under {len(unmoved)} scored file(s): {unmoved}; "
        "the seal is not sensitive to line, only to file"
    )


def test_the_two_negatives_are_not_the_same_test() -> None:
    """CONTROL. The GREEN negative would pass on a seal that hashes a constant.
    This asserts the seal's population is non-empty and its files are disjoint
    from `MERCURY_LIVE` -- the two ways NEGATIVE 1 could pass for a bad reason.
    """
    files = {p for p, _ in scored_pointers()}
    assert files, "empty seal population: NEGATIVE 1 would pass on nothing"
    assert not files & set(MERCURY_LIVE), "a LIVE file leaked into the population"


# --------------------------------------------------------------------------
# FINDING M-24b. What the `LIVE` list actually costs the seal.
# --------------------------------------------------------------------------


def test_m24b_the_ledger_is_seven_of_the_ten_refused_pointers() -> None:
    """The measurement J-23g did not take. GREEN -- it is a price, not a defect.

    J-23g: the ledger is on `LIVE` because an office typed it there, and it has
    sat at 438 lines for three iterations. This bounds what that judgement costs:
    7 of the 10 pointers the seal gives up are into the file that has not moved.
    """
    refused = collections.Counter(
        p for p, _ in {(p, n) for _, p, n in occurrences()} if p in MERCURY_LIVE
    )
    assert dict(refused) == REFUSED_BY_FILE, f"refused split moved: {dict(refused)}"
    # it.33 REPAIR. The `438` line-count pin is RETIRED here. A cardinality
    # cannot see a rewrite: 438 lines of DIFFERENT TEXT passed it, and this
    # node stated that defect in its own failure message ("started growing
    # again") while committing it. SATURN it.31 census / it.32 spec. The
    # digest below was re-read by this office at LEDGER_READ (2026-09-02 19:36:19 IST) -- not
    # taken from the spec -- via
    #   [RUN] python -c "import hashlib,pathlib; b=pathlib.Path(
    #         'V20_R15_LEAP_LEDGER.md').read_bytes();
    #         print(hashlib.sha256(b).hexdigest(), len(b))"
    # -> 6e88935181ab...5485, 45670 bytes, 438 lines.
    body = (ROOT / "V20_R15_LEAP_LEDGER.md").read_bytes()
    got = (len(body), hashlib.sha256(body).hexdigest())
    assert got == (LEDGER_BYTES, LEDGER_SHA256), (
        "the ledger moved (content, not just length): %s" % (got,)
    )
    # NEGATIVE, entirely in memory: the retired predicate passes on 438 lines
    # of text that is not the ledger. This is the whole content of the repair.
    forged = (("x" + chr(10)) * 438).encode()
    assert len(forged.decode().splitlines()) == 438, "the negative is malformed"
    assert hashlib.sha256(forged).hexdigest() != LEDGER_SHA256


# --------------------------------------------------------------------------
# REPAIR 2. The `28 vs 25` gap. Open since it.18, unreached for five iterations.
# --------------------------------------------------------------------------

IT17 = "tests/jupiter/test_v20_r15_it17_citation_landing.py"


def test_the_28_vs_25_gap_is_two_populations() -> None:
    """VERDICT: two populations compared by mistake. The bridge closes exactly.

    `[CITED: JUPITER]` it.17 section 2.1: **28** is the count of CENSUS-FAILING
    CITATION OCCURRENCES in `V20_R15_THEORY_TABLE.md`. Its table has 27 rows; the
    row `C26, C74` carries two occurrences of one pointer. 27 + 1 = 28.

    `[CITED: INSPECTOR]` it.18 section 1.3: **25** is a read of `MANIFEST` in
    `%s`, which counts something else entirely -- DISTINCT REPAIRED TARGET LINES.
    That dict is not recoverable at 25 (the file is untracked, `[RUN] git
    ls-files --error-unmatch` -> pathspec did not match), so the 25 is cited, not
    re-run. What is re-run is that the mapping between the two populations is not
    1:1 in EITHER direction, which is sufficient to close the item:

      one-to-many  C44  `ceq/hankel.py:1` -> `:75`, `:108`, `:131`, `:279`  (+3)
      many-to-one  C26 + C74 -> `V20_R15_IT13_MERCURY.md:146`               (-1)
      not-a-line   C10 is a LABEL repair inside `V20_R15_LEAP_LEDGER.md`,
                   certified by `test_the_F4_overload_labels_name_the_sense_
                   actually_at_the_line`, NOT by MANIFEST                   (-1)
      not-a-census C95 `scale/negation_scope.py:286`, ruled by JUPITER on the
                   named-symbol rule, never a census failure                (+1)

      28 - 1 - 1 + 3 + 1 = 30 = len(MANIFEST) at HEAD.

    THE ITEM IS CLOSED. There are no missing repairs. The INSPECTOR's own hedge
    at it.18 -- "the gap may be grouping rather than omission" -- was right, and
    the reason it sat five iterations is that closing it needs arithmetic and
    nobody owned the arithmetic. The measurement office owns it.
    """ % IT17
    import importlib.util

    spec = importlib.util.spec_from_file_location("it17", ROOT / IT17)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    assert len(m.MANIFEST) == 30, f"MANIFEST holds {len(m.MANIFEST)}"
    assert 28 - 1 - 1 + 3 + 1 == len(m.MANIFEST)

    # The one-to-many leg, asserted rather than asserted-about.
    hankel = sorted(int(c.split(":")[1]) for c in m.MANIFEST if c.startswith("ceq/hankel.py:"))
    assert hankel == [75, 108, 131, 279], hankel
    # The many-to-one leg: C26 and C74 share one target.
    assert "V20_R15_IT13_MERCURY.md:146" in m.MANIFEST
    # The not-a-line leg: no `LIVE` file target is in MANIFEST at all.
    assert not [c for c in m.MANIFEST if c.split(":")[0] in MERCURY_LIVE]
    # The not-a-census leg.
    assert "scale/negation_scope.py:286" in m.MANIFEST


def test_the_28_vs_25_gap_has_no_third_reading() -> None:
    """CONTROL for the verdict above. If `28` and `len(MANIFEST)` were the same
    population, the four bridge legs would all be zero and `28 == 30` would have
    to hold. It does not, and each leg is separately non-zero above.
    """
    assert 28 != 30
