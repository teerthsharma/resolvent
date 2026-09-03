"""it.27 -- J-26c ROUTED INTO THE PRODUCTION PREDICATE, and MARS's `overturns:` half.

it.26 built the right device and left it in the wrong file. `STAR_DIGESTS` lived in
`test_v20_r15_it26_live_claim.py`; `lands()` -- the predicate every office imports --
was unmodified, so MARS's STRIKE C stayed legitimately RED. This file is RED first
against that unmodified predicate, and the repair is made in `lands()` itself.

RULING J-27a -- `:*` IS SCORED AGAINST THE LINES THAT CARRY THE WANT, IN `lands()`.
  `region(path,'*')` returns the whole file, so `want in text` is a grep, and a grep
  cannot tell a want from its own obituary: delete the asserting line, append a line
  quoting the want, and presence is unchanged. A frozen COUNT does not close it
  either -- all six `:*` wants occur exactly once, so the obituary reads 1 against a
  frozen 1 (measured at it.26, re-measured here). What moves under the obituary is
  the CONTENT of the carrying lines, so that is what is frozen.

  The predicate is keyed on the `want`, not on the cid, because `lands(want, text)`
  is the signature four offices already call and widening it would restate the rule
  in every caller -- the defect it.21 shipped twice. A `:*` want is unique to its
  citation; `test_the_star_wants_are_unique_so_keying_on_them_is_sound` asserts it
  rather than assuming it.

  WHAT IT AUTHENTICATES, against the INSPECTOR's it.25 ruling that `[RUN]` provenance
  authenticates a reading and not a subject: this device authenticates THE SUBJECT.
  The frozen digest is over the bytes of the lines that carry the claim, so it moves
  when the claim's own text moves and stays still when anything else does. WHAT IT
  STILL CANNOT SEE: a line that keeps its bytes and loses its meaning -- the same
  silent-supersession boundary J-26a moved and did not retire -- and a want whose
  carrying line was already wrong when the census froze it.

RULING J-27b -- `overturns:` ON CORRECTIONS INDEX ROWS. MARS's it.25 route, second
  half. The index is the coordinator's file and is NOT edited here; the field is
  specified and the node that enforces it is shipped, so the row change lands against
  a test that already exists.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tests.jupiter.test_v20_r15_it20_citation_freeze import (  # noqa: E402
    CENSUS,
    NEWLINE,
    lands,
    region,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "V20_R15_JOURNAL.md"

STAR_CIDS = [c for c, (_, s, _) in sorted(CENSUS.items()) if s == "*"]


def _obituary(text: str, want: str) -> str:
    """MARS's it.25 mutation, verbatim in shape: the want deleted from every line
    it appears on, and ONE line appended recording the deletion. It is the shape
    every repair record in this round has."""
    kept = [ln for ln in text.splitlines() if want not in ln]
    return NEWLINE.join(kept + ['# the want "' + want + '" was removed at it.25'])


def _deleted(text: str, want: str) -> str:
    return NEWLINE.join(ln for ln in text.splitlines() if want not in ln)


def _append(text: str) -> str:
    """An UNRELATED append -- J-24a's whole reason for exempting `:*` from J-20b."""
    return text + NEWLINE + "# an unrelated line appended at it.27"


# ------------------------------------------------------------- J-27a, the repair

def test_the_star_wants_are_unique_so_keying_on_them_is_sound() -> None:
    """PREMISE, asserted not assumed. The predicate looks a want up by its text."""
    assert len(STAR_CIDS) == 6, "the `:*` population moved: " + repr(STAR_CIDS)
    star_wants = [CENSUS[c][2] for c in STAR_CIDS]
    assert len(set(star_wants)) == 6, "two `:*` citations share a want"
    others = [w for c, (_, _, w) in CENSUS.items() if c not in STAR_CIDS]
    assert not (set(star_wants) & {w for w in others if isinstance(w, str)}), (
        "a `:*` want is also a line-pointer want; keying on the want would "
        "score a single line against a file-scoped digest")


def test_the_control_a_star_anchor_lands_unmutated_and_survives_an_append() -> None:
    """CONTROL. The probe is not RED for everything, and the repair does not
    retract J-24a: an append still cannot break a `:*` anchor."""
    for c in STAR_CIDS:
        path, _, want = CENSUS[c]
        text = region(path, "*")
        assert lands(want, text), "C%d: premise gone, the want no longer lands" % c
        assert lands(want, _append(text)), (
            "C%d: an unrelated append broke a `:*` anchor. That retracts J-24a "
            "and re-opens every `:*` pointer to J-20b's refusal" % c)


def test_the_control_a_star_anchor_breaks_on_a_silent_deletion() -> None:
    """CONTROL, MARS's own. `:*` is not true for everything."""
    for c in STAR_CIDS:
        path, _, want = CENSUS[c]
        assert not lands(want, _deleted(region(path, "*"), want)), (
            "C%d: a silent deletion is not caught, so the obituary probe is "
            "false for everything" % c)


def test_a_frozen_COUNT_would_be_GREEN_on_the_obituary_for_all_six() -> None:
    """PLANTED NEGATIVE AGAINST MARS'S OWN ROUTE, re-measured at it.27.

    His route was `WANT_COUNT[cid] = text.count(want)`, scored `>= frozen`. Every
    `:*` want occurs exactly once and the obituary line QUOTES the want, so the
    count reads 1 against a frozen 1 and passes on the exact mutation it was built
    to catch. Filed against this office's remedy, not as a point against the strike.
    """
    for c in STAR_CIDS:
        path, _, want = CENSUS[c]
        text = region(path, "*")
        before, after = text.count(want), _obituary(text, want).count(want)
        assert before == 1, "C%d: count is %d; the negative assumed 1" % (c, before)
        assert after >= before, (
            "C%d: the count freeze would have CAUGHT the obituary after all -- "
            "re-derive J-27a, the digest may be buying nothing here" % c)


def test_lands_can_tell_a_want_from_its_own_obituary() -> None:
    """RED against the it.20 predicate as it stood at it.26. MARS's STRIKE C, at
    the place it was actually opened: `lands`, not a device in a JUPITER filing."""
    missed = []
    for c in STAR_CIDS:
        path, _, want = CENSUS[c]
        if lands(want, _obituary(region(path, "*"), want)):
            missed.append("C%d: %s:*" % (c, path))
    assert not missed, (
        "`lands()` still LANDS on a file where the want was deleted and only the "
        "line recording its removal remains, for %d of %d `:*` pointers. J-26c "
        "froze the carrying lines in the it.26 instrument and left the PRODUCTION "
        "predicate unmodified, so every office that imports `lands` still greps. "
        "missed: %s" % (len(missed), len(STAR_CIDS), missed))


def test_the_star_digests_are_over_the_carrying_lines_and_are_re_derivable() -> None:
    """The freeze is not a magic constant: the recipe is here and is re-run."""
    from tests.jupiter.test_v20_r15_it20_citation_freeze import STAR_DIGESTS

    assert set(STAR_DIGESTS) == {CENSUS[c][2] for c in STAR_CIDS}
    for c in STAR_CIDS:
        path, _, want = CENSUS[c]
        carrying = [ln for ln in region(path, "*").split(NEWLINE) if want in ln]
        assert carrying, "C%d: nothing carries the want" % c
        digest = hashlib.sha256(NEWLINE.join(carrying).encode("utf-8")).hexdigest()
        assert digest == STAR_DIGESTS[want], (
            "C%d: the lines carrying %r in %s have MOVED since census. That is "
            "J-27a firing, not a broken test: re-open the pointer." % (c, want, path))


# --------------------------------------------------- J-27b, the `overturns:` field

#: THE FIELD, SPECIFIED. One trailing cell on each CORRECTIONS INDEX row, holding
#: the literals the row retires, comma-separated inside backticks:
#:
#:     | C1 | "no arm crosses ..." | it.1 | **6 of 24 cross.** | it.2 | overturns: `no arm crosses`
#:
#: The literals are the SEARCH KEYS a later reader would grep for, not a prose
#: restatement. A row with no machine-checkable literal declares `overturns: none`,
#: which is a claim the round can audit; an ABSENT field is not.
#:
#: WHAT A GREP-BASED INSTRUMENT MUST DO TO HONOUR IT, in `live_hits` below:
#:   1. read the field off every index row -- `dead_literals`;
#:   2. return NOTHING for a literal any row declares dead;
#:   3. never count a hit that is itself an index row.
#: (3) alone is not enough and that is the whole finding: a body entry may restate
#: the withdrawn claim, so excluding the index does not make the grep sound; only
#: the declaration does.
OVERTURNS_RE = re.compile(r"overturns:\s*(.+?)\s*$")
INDEX_ROW_RE = re.compile(r"^\|\s*C(\d+)\s*\|")

#: RULING J-29b -- `none` IS A SENTINEL, NOT A SEARCH KEY. The field's own
#: specification above says a row with no machine-checkable literal declares
#: `overturns: none`; the it.27 extraction read it through the same backtick
#: grammar as every other cell and handed the grep the four-letter WORD, which
#: occurs 25 times in the journal body. Three rows -- C21, C30, C33 -- carry it.
#: This is the whole gap between the coordinator's `26 of 41` and the nurse's
#: `29 of 44`: the same measurement under two grammars, differing by exactly the
#: three sentinels. Reserved here so a row can decline to name a literal.
NONE = "none"


def hits_in(literal: str, body: list[str], indexed: set[str]) -> list[int]:
    """Line numbers in `body` carrying `literal`, excluding index rows. Split out
    of `live_hits` so the it.29 node can put it on a synthetic body and SEE it
    return a hit -- the check that the assertion can observe a violation."""
    return [i for i, ln in enumerate(body, 1) if literal in ln and ln not in indexed]


def index_rows() -> list[tuple[int, str]]:
    """Every CORRECTIONS INDEX row, by row number. The index is READ here, never
    written: it is the coordinator's file."""
    out = []
    for ln in JOURNAL.read_text(encoding="utf-8").splitlines():
        m = INDEX_ROW_RE.match(ln)
        if m:
            out.append((int(m.group(1)), ln))
    return out


def dead_literals() -> dict[int, list[str]]:
    """What an `overturns:`-aware instrument reads. Empty until the rows carry it."""
    out = {}
    for n, ln in index_rows():
        m = OVERTURNS_RE.search(ln)
        if m:
            lits = re.findall(r"`([^`]+)`", m.group(1))
            out[n] = [] if lits == [NONE] else lits
    return out


def live_hits(literal: str) -> list[int]:
    """A stale-claim grep that HONOURS the field: hits outside the index.

    RULING J-29a, REPAIRED. As shipped at it.27 this opened with

        if any(literal in lits for lits in dead_literals().values()):
            return []

    and every literal the enforcement node below can construct is drawn from
    `dead_literals()`, so the guard was ALWAYS true, the return was ALWAYS `[]`,
    and `assert live_hits(lit) == []` could not fail for any input the node can
    reach. `V-16`: the instrument that cannot measure, reporting a pass. The
    short-circuit is narrowed to the DECLARING ROW, which the index-row exclusion
    already covers as a superset -- a literal C1 retires is searched everywhere
    except the index itself. A declaration now has to be EARNED, not asserted."""
    body = JOURNAL.read_text(encoding="utf-8").splitlines()
    return hits_in(literal, body, {ln for _, ln in index_rows()})


def test_the_stale_claim_grep_is_100_percent_false_positive_across_the_index() -> None:
    """THE MEASUREMENT that makes the field worth a field, re-taken at it.27.

    The INSPECTOR measured 33 of 33. Every index row QUOTES a literal from the claim
    it overturns, so a reader grepping for a withdrawn claim finds it on the very row
    that retires it. The index is the round's guard against stale claims and it is
    itself the densest source of stale-claim hits."""
    rows = index_rows()
    assert len(rows) >= 33, "the index shrank to %d rows -- re-derive" % len(rows)
    quoting = [n for n, ln in rows if re.search(r"[`\"][^`\"]{4,}[`\"]", ln)]
    assert len(quoting) == len(rows), (
        "%d of %d index rows quote a literal from the claim they overturn; the "
        "measurement assumed all of them" % (len(quoting), len(rows)))


def test_an_overturns_aware_grep_returns_nothing_for_a_declared_dead_literal() -> None:
    """RED until the coordinator adds the field. THIS IS THE ENFORCEMENT NODE.

    C1 retires *"no arm crosses BED-M's floor1 `0.7071`"*. Once its row declares
    ``overturns: `no arm crosses` ``, `live_hits` returns empty for that literal and
    a grep-based instrument stops reporting a claim withdrawn 25 iterations ago. The
    row change is one cell; the node it lands against is this one."""
    declared = dead_literals()
    assert declared, (
        "no CORRECTIONS INDEX row carries an `overturns:` field, so a stale-claim "
        "grep has nothing to subtract and every one of its hits on the index is a "
        "false positive. Add the trailing cell specified above OVERTURNS_RE -- "
        "`overturns: `literal`, `literal`` -- starting with C1's `no arm crosses`.")
    unparsed = [n for n, ln in index_rows()
                if OVERTURNS_RE.search(ln)
                and not re.findall(r"`([^`]+)`", OVERTURNS_RE.search(ln).group(1))]
    assert not unparsed, (
        "C%s declares `overturns:` and names neither a backticked literal nor the "
        "`none` sentinel" % unparsed)
    pairs = [(n, l) for n, lits in sorted(declared.items()) for l in lits]
    unearned = [(n, l, live_hits(l)) for n, l in pairs if live_hits(l)]
    assert not unearned, (
        "%d of %d declared literals are UNEARNED: the row says the claim is "
        "retired and the journal body still asserts the literal outside the "
        "index. A row either withdraws what the body still says, or narrows its "
        "`overturns:` cell to a literal that is dead AS A CLAIM rather than a "
        "string still in live use (J-29c). unearned:\n%s" % (
            len(unearned), len(pairs),
            "\n".join("  C%-3d %-46r %d hits, first at %d"
                       % (n, l, len(h), h[0]) for n, l, h in unearned)))
