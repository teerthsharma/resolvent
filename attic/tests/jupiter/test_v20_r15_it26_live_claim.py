"""it.26 -- THREE DEVICES THAT AUTHENTICATE THE SUBJECT, NOT THE READING.

The INSPECTOR's it.25 ruling, which shapes all three:

    `[RUN]` provenance is necessary and not sufficient. Every anti-fabrication
    device this round built authenticates that a reading HAPPENED. None
    authenticates that the SUBJECT HAD A VALUE.

`WANT_SEAL`, the census digests and `FILE_LINES_AT_CENSUS` all certify that a
reading happened at a place. Each device below certifies something about the
SUBJECT: that the claim is still LIVE (J-26a), that the datum has ONE HOME
(J-26b), and that the lines asserting a want are the SAME LINES (J-26c).

  RULING J-26a -- A WANT THAT CAN BE SUPERSEDED IS ANCHORED OVER ITS OWN
  SUPERSESSION, OR IT IS REFUSED.
  it.24 withdrew C13 and re-issued it as C130 because the old want certified the
  item's NAME while the cell asserted its GRADE. The re-issued want
  `grade F3 pending` lands at `CEQ_V20_R15_CONTRACT.md:241` -- and `:244` of the
  same file reads `[SUPERSEDED it.19, RULING J-17d: M14 IS F4`. Locationally
  perfect, semantically void: the repair built to close a J-21c argument-layer
  defect reproduced the class three lines lower.
  THE INSPECTOR IS RIGHT THAT IT IS A DEFECT AND WRONG ABOUT WHICH ONE. The cell
  does not assert M14 is F3. It reads, verbatim:
      **`M14` carries three grades in three files for one item** -- `F3` in the
      contract (`CEQ_V20_R15_CONTRACT.md:241`) ... **RESOLVED at it.17 --
      `M14 = F4`.**
  The cell's claim is a THREE-GRADE CONFLICT and its resolution. `:241` certifies
  the F3 half. Nothing in the citation certifies that the cell knows the F3 is
  dead, so a reader who follows the pointer reads a live-looking F3 and the
  instrument scores it GREEN. THE DEFECT IS THAT THE ANCHOR IS NARROWER THAN THE
  CLAIM'S LIVENESS, which is J-24b's compound-anchor finding arriving for a
  second reason: not a claim split across two constants, but a claim whose
  CONTENT and whose STANDING sit on different lines.
  So the remedy is not a third pointer edit. It is the rule that a landing check
  reads one line and one line cannot say whether the claim is live -- made
  mechanical: SCAN FORWARD FROM THE ADDRESSED REGION TO THE END OF THE ENCLOSING
  BLOCK, AND REFUSE ANY POINTER WHOSE SUPERSESSION NOTICE LIES OUTSIDE ITS OWN
  REGION. Widen the region until it contains what would falsify it, and the
  refusal lifts because the reader now cannot miss the notice either.

  WHAT THIS DOES AND DOES NOT REPAIR -- the question the round put directly.
  IT IS REPAIRABLE FOR MARKED SUPERSESSION AND NOT FOR SILENT SUPERSESSION, so
  it MOVES the it.21 reader-only boundary rather than retiring it. A document
  that records its own overturns in a fixed vocabulary can be audited by machine;
  a document that merely stops meaning what it said cannot. The vocabulary is the
  load-bearing assumption and it is stated as one, not smuggled in: this round's
  corpus announces supersession, and the probe below is measured firing on
  1 OF 93 SCORED POINTERS -- C130 and nothing else. A probe that reddened the
  census would be measuring the vocabulary, not the defect.

  RULING J-26b -- A WITHDRAWAL REWRITES THE DATUM ONCE, OR IT DOES NOT REWRITE IT.
  MARS's STRIKE B: it.24 withdrew C63 from `CENSUS` and re-issued it widened, and
  it.17's `MANIFEST` -- a hand-written literal, stated by it.24 as 'not edited' --
  still carries the OLD pointer. `MANIFEST` is what foreign instruments read.
  THE PROCEDURE, NOT THE DATUM, IS THE DEFECT, and the correction has two halves:
    (i)  DERIVE. `MANIFEST` becomes a VIEW over `CENSUS`, keyed by the cids it.17
         repaired. The it.17 record of WHICH citations that office repaired is a
         historical fact and is kept verbatim; the POINTER and the WANT are read
         live from the census, so a withdrawal moves one datum and every reader
         of it moves with it. A second copy of a pointer is a second place to be
         wrong, and it.24 proved it by being wrong there.
    (ii) SCOPE THE RE-TAKE BY IMPORTERS, NOT BY OFFICE. `EDIT / RE-TAKE /
         RE-DECLARE` re-took `tests/jupiter` only, which is the editing office
         grading its own blast radius. The blast radius of an edit to a module is
         the set of files that IMPORT it, and that set is computable -- so the
         instrument computes it and refuses a re-take narrower than it.

  RULING J-26c -- `:*` IS SCORED AGAINST THE LINES, NOT AGAINST PRESENCE, AND
  NOT AGAINST A COUNT EITHER.
  MARS's STRIKE C is upheld: `region(path,'*')` is the whole file, so a `:*`
  anchor is a grep, and a grep cannot tell a want from a line recording the
  want's removal -- the CORRECTIONS-INDEX property imported into the citation
  instrument.
  HIS PROPOSED ROUTE DOES NOT HOLD, AND THE MEASUREMENT IS THE REASON.
  "Score `:*` against a COUNT of occurrences frozen at census" fails on ALL SIX
  of this round's `:*` pointers, because every one of them has COUNT 1 [RUN,
  below]: his own obituary mutation deletes the one asserting line and appends a
  line that quotes the want, so the count goes 1 -> 1 and the freeze is GREEN on
  the mutation it was built to catch. A count is a weaker digest of the same
  reading; it authenticates the reading again.
  WHAT AUTHENTICATES THE SUBJECT IS THE LINES THEMSELVES. `star_digest` freezes a
  digest over the SET OF LINES CARRYING THE WANT. An append of unrelated text
  does not change that set, so J-24a's whole reason for exempting `:*` from J-20b
  survives intact; a deletion empties it; an obituary REPLACES the asserting line
  with a different line and the digest moves. Strictly stronger than the count at
  the same cost, and it is the same instrument the round already trusts -- a
  content digest -- pointed at the subject instead of at the act of reading.

RED FIRST, AGAINST UNMUTATED CODE. Every test below was written and run before
any repair was applied. The failures are recorded verbatim in
V20_R15_IT26_JUPITER.md with their reading time.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from tests.jupiter.test_v20_r15_it20_citation_freeze import (
    CENSUS,
    NEWLINE,
    REISSUED,
    ROOT,
    lands,
    refused,
    region,
)
from tests.jupiter.test_v20_r15_it20_citation_freeze import (
    STAR_DIGESTS as CARRYING_DIGESTS,
)

#: The vocabulary a document uses to announce that it has overturned itself.
#: LOAD-BEARING AND STATED: silent supersession is out of reach (J-26a LIMITS).
SUPERSESSION = re.compile(
    r"SUPERSEDED|SUPERSEDES|STRUCK|WITHDRAWN|OVERTURNED|RETIRED|NO LONGER"
)

#: ponytail: the enclosing block is "forward to the next column-0 line, capped at
#: 20". That is the shape of this corpus's entries (`M14 ...` at column 0, its
#: body indented), not a general parser. A citation into a language whose blocks
#: are not indentation-delimited needs a real one; nothing in this census is.
BLOCK_CAP = 20


def block_after(path: str, spec: object) -> list[tuple[int, str]]:
    """Lines from the end of the addressed region to the end of its block."""
    p = ROOT / path
    if not p.is_file() or spec == "*":
        return []
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    last = int(str(spec).split("-")[-1])
    out = []
    for i in range(last, min(last + BLOCK_CAP, len(lines))):
        if lines[i].strip() and not lines[i][:1].isspace():
            break
        out.append((i + 1, lines[i]))
    return out


def superseded(manifest: dict) -> list[str]:
    """RULING J-26a. Every SCORED pointer whose supersession is outside it."""
    skip = refused()
    out = []
    for cid, (path, spec, want) in sorted(manifest.items()):
        if cid in skip:
            continue
        inside = region(path, spec)
        for lineno, text in block_after(path, spec):
            if SUPERSESSION.search(text) and text not in inside:
                out.append(
                    f"C{cid}: {path}:{spec} lands {want!r}, but :{lineno} of the "
                    f"same block reads {text.strip()[:60]!r} -- the citation "
                    f"certifies a claim the cited document marks dead, and the "
                    f"notice is outside the anchored region"
                )
                break
    return out


def star_lines(path: str, want: object) -> list[str]:
    """The lines a `:*` pointer's want actually appears on. The SUBJECT."""
    return [ln for ln in region(path, "*").split(NEWLINE) if lands(want, ln)]


def star_digest(path: str, want: object) -> str:
    """A digest over those lines. Append-stable, deletion- and obituary-fragile."""
    return hashlib.sha256(
        NEWLINE.join(star_lines(path, want)).encode("utf-8")
    ).hexdigest()[:16]


# ============================================================ REPAIR 1 -- J-26a

def test_the_control_the_supersession_probe_is_not_red_for_everything() -> None:
    """Control: 92 of 93 scored pointers have NO downstream supersession."""
    scored = len([c for c in CENSUS if c not in refused()])
    assert len(superseded(CENSUS)) < scored / 2, (
        "control gone: the probe reddens half the census, so it is measuring the "
        "supersession vocabulary and not the defect -- re-derive it")


def test_no_scored_citation_certifies_a_grade_its_own_document_supersedes() -> None:
    """RED at it.26: C130's want lands three lines above its own obituary."""
    assert superseded(CENSUS) == [], NEWLINE.join(superseded(CENSUS))


def test_widening_the_anchor_over_the_notice_lifts_the_refusal() -> None:
    """The remedy is checkable, not advisory: the notice inside the region passes."""
    wide = {130: ("CEQ_V20_R15_CONTRACT.md", "241-244",
                  ("grade F3 pending", "M14 IS F4"))}
    assert superseded(wide) == [], superseded(wide)
    path, spec, want = wide[130]
    assert lands(want, region(path, spec)), "the widened anchor must still land"


# ============================================================ REPAIR 2 -- J-26b

def test_no_withdrawal_strands_a_pointer_a_foreign_instrument_reads() -> None:
    """RED at it.26: MARS STRIKE B, as the editing office's own assertion."""
    from tests.jupiter import test_v20_r15_it17_citation_landing as IT17

    censused = {f"{p}:{s}" for p, s, _ in CENSUS.values()}
    stranded = sorted(set(IT17.MANIFEST) - censused)
    assert not stranded, (
        f"it.24 withdrew cids {sorted(REISSUED.values())} and re-issued them "
        f"widened; it.17's MANIFEST is a hand-written literal and still carries "
        f"the old pointer. Two copies of one datum, one updated. stranded: "
        f"{stranded}")


def importers(module: str) -> set[str]:
    """Every test file that imports `module`, tracked or not. J-26b(ii)."""
    stem = Path(module).stem
    out = subprocess.run(
        ["git", "grep", "-l", "--untracked", stem, "--", "*.py"],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout.split()
    return {f for f in out if "/test_" in f and Path(f).stem != stem}


#: THE CORRECTED PROCEDURE, AS A DECLARATION THE NEXT EDIT MUST SATISFY.
#: it.24's was EDIT / RE-TAKE / RE-DECLARE and it left the re-take unscoped, so
#: the editing office re-took the editing office. The correction is one clause:
#:
#:     EDIT / COMPUTE THE BLAST RADIUS / RE-TAKE ALL OF IT / RE-DECLARE
#:
#: where the blast radius of an edit to module M is `importers(M)` -- every test
#: file that names M, tracked OR untracked, because `git grep` without
#: `--untracked` is blind to exactly the files this round writes. The set is
#: computed, not nominated, so the next office cannot scope it to its own suite
#: by forgetting. RE-TAKEN AT it.26 is asserted below to EQUAL that set.
RETAKEN_AT_IT26: frozenset[str] = frozenset({
    "tests/jupiter/test_v20_r15_it17_citation_landing.py",
    "tests/jupiter/test_v20_r15_it21_heading_anchor.py",
    "tests/jupiter/test_v20_r15_it23_fence_and_argument.py",
    "tests/jupiter/test_v20_r15_it24_census_retake.py",
    "tests/jupiter/test_v20_r15_it26_live_claim.py",
    "tests/mars_v20/test_it25_the_repair_record_reads_as_the_defect.py",
    "tests/jupiter/test_v20_r15_it27_star_lands_and_overturns.py",
    "tests/jupiter/test_v20_r15_it29_overturns_can_fail.py",
})


def test_the_retake_covers_the_blast_radius_and_the_blast_radius_is_computed():
    """RULING J-26b(ii). The re-take is scoped by IMPORTER, not by OFFICE.

    RED at it.26 in its first form, which asserted the radius was inside
    `tests/jupiter`: it is not -- `tests/mars_v20/test_it25_...py` imports the
    census, and that ONE foreign importer is the +1 between JUPITER's `31
    failed` and MARS's `32 failed` at it.24. The finding is kept as the
    standing check it should have been: the radius is measured, and the re-take
    must cover it.
    """
    hit = importers("test_v20_r15_it20_citation_freeze")
    assert hit, "the importer probe found nothing -- git grep is not running"
    missed = sorted(hit - RETAKEN_AT_IT26)
    assert not missed, (
        f"editing the census touches {len(hit)} importers; {len(missed)} were "
        f"not re-taken: {missed}. This is it.24's defect exactly -- re-taking "
        f"the office that owns the file instead of the files that read it.")
    foreign = sorted(f for f in RETAKEN_AT_IT26 if not f.startswith("tests/jupiter/"))
    assert foreign, (
        "the re-take names no file outside tests/jupiter, so this check cannot "
        "distinguish 'scoped by importer' from 'scoped by office' -- the exact "
        "ambiguity it exists to remove")


# ============================================================ REPAIR 3 -- J-26c

STARS = {c: v for c, v in CENSUS.items() if v[1] == "*"}


def test_the_measurement_that_kills_the_count_freeze() -> None:
    """[RUN] Every `:*` want occurs exactly ONCE, so a count cannot fall."""
    counts = {c: len(star_lines(p, w)) for c, (p, _, w) in STARS.items()}
    assert set(counts.values()) == {1}, counts
    assert len(counts) == 6, counts


def _obituary(text: str, want: str) -> str:
    """MARS's mutation: the want deleted from every line, one line saying so."""
    kept = [ln for ln in text.splitlines() if want not in ln]
    return NEWLINE.join(kept + [f'# the want "{want}" was removed at it.25'])


def test_a_frozen_COUNT_is_green_on_the_mutation_it_was_built_to_catch() -> None:
    """RED at it.26 against MARS's own proposed route, not against his strike.

    His STRIKE C is upheld. The COUNT remedy he names is scored here and fails:
    the obituary line quotes the want, so 1 occurrence becomes 1 occurrence.
    """
    path, _, want = CENSUS[55]
    before = len(star_lines(path, want))
    # it.27: PRESENCE, not `lands`. J-27a made `lands` digest-scored for `:*`
    # wants, and this negative is about MARS's COUNT route, which is a count of
    # PRESENCE. Scoring it through the repaired predicate would measure the
    # repair instead of the route it retires.
    after = len([ln for ln in _obituary(region(path, "*"), want).split(NEWLINE)
                 if want in ln])
    assert before == after == 1, (
        f"PLANTED NEGATIVE for J-26c, and it is MARS's own proposed remedy. A "
        f"count frozen at census reads {before} before the obituary and {after} "
        f"after it, so the count freeze is GREEN on the exact mutation STRIKE C "
        f"was built from -- for all 6 `:*` pointers, because each has count 1 "
        f"and the obituary line quotes the want. If this ever goes RED because "
        f"a `:*` want acquired a second occurrence, the count route becomes "
        f"viable for THAT pointer and stays dead for the others; the line "
        f"digest below is what does not depend on the population.")


def test_a_line_digest_catches_the_obituary_AND_survives_an_append() -> None:
    """The device that replaces it, with both of J-24a's properties intact."""
    path, _, want = CENSUS[55]
    text = region(path, "*")

    def dig(t: str) -> str:
        return hashlib.sha256(
            NEWLINE.join(l for l in t.split(NEWLINE) if lands(want, l)).encode()
        ).hexdigest()[:16]

    live = dig(text)
    assert dig(_obituary(text, want)) != live, "obituary must move the digest"
    kept = [ln for ln in text.splitlines() if want not in ln]
    assert dig(NEWLINE.join(kept)) != live, "silent deletion must move it too"
    assert dig(text + NEWLINE + "# an unrelated append at it.26") == live, (
        "an append must NOT move it -- that is J-24a's whole reason for "
        "exempting `:*` from J-20b, and it survives the strengthening")


#: RULING J-26c, DE-DUPLICATED AT it.29. This datum had TWO homes -- here, keyed
#: by cid at 16 hex, and in `test_v20_r15_it20_citation_freeze.py` keyed by want at
#: full hex -- which is `J-26b`'s own one-home ruling violated by `J-26c`'s own
#: datum, both written by this office. The it.20 dict is the home, because `lands()`
#: consumes it; this is a VIEW of it, re-keyed cid -> 16 hex. Not a tautology: the
#: node below still scores the FILE against the frozen constant. The two copies
#: agreed today only because the prefixes matched, which no node asserted.
STAR_DIGESTS: dict[int, str] = {
    c: CARRYING_DIGESTS[w][:16] for c, (_, _, w) in sorted(STARS.items())
}


def test_every_star_pointer_matches_its_frozen_line_digest() -> None:
    """All 6 `:*` pointers scored against their lines, not against presence."""
    now = {c: star_digest(p, w) for c, (p, _, w) in STARS.items()}
    assert now == STAR_DIGESTS, (
        f"a `:*` pointer's asserting line changed. Under J-26c that is not an "
        f"append and is not scored as a landing: open the file and decide "
        f"whether the want was repaired, moved, or given an obituary. now={now}")
