"""THE FREEZE (it.4): the wing list is frozen, tamper-evident, and re-resolved.

Three things this file binds, and nothing else.

1.  FOUND, NOT NAMED. The round's binding kill is "a wing named rather than
    found is struck". Read mechanically, the discriminator is clause (b): a
    wing is FOUND iff `results/` holds at least one journalled record whose
    `kind` is the wing's arm name. A name that appears only in source and prose
    is a NAMED wing. The discriminator is calibrated on both sides -- the two
    frozen wings must be found, and `arm_phase` must read exactly zero, so the
    counter is seen to be able to return zero before zero is used as a verdict
    (MISTAKES.md:804, V-15, the condemning rule with no planted negative).

2.  TAMPER-EVIDENT. The manifest declares a sha256 over its own citation rows.
    A list that can be edited after it is frozen was never frozen. The planted
    negative moves ONE citation's line number by one and asserts the hash test
    fires -- and that the moved citation stops resolving.

    WHAT THE HASH COVERS: the wing rows, exactly the twelve-field
    `wing|clause|path:line|anchor` tuples, normalised and sorted.
    WHAT IT DOES NOT COVER: the CONTENT of the cited files (that is the
    anchor check, node 3, which must be re-run at every HEAD), the prose of
    the manifest, the struck-wing section, and the prices in it. Editing a
    cited FILE does not move this hash; editing the LIST does.

3.  STILL RESOLVING AT HEAD. Every citation is re-resolved here, not trusted
    from it.1. `spotcheck_draw.py` read a sheet live from the working tree and
    a pre-registered draw stopped reproducing when the sheet was amended; the
    same shape applied to a frozen list is `MISTAKES.md:364` P-6,
    line-reference drift. The resolver is imported from the it.1 rubric rather
    than reimplemented, so there is one checker in this round, not two.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

import pytest

from tests.saturn.test_v20_r15_wing_rubric import resolve

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "V20_R15_WING_MANIFEST.md"

CLAUSES = ("a", "b", "c", "d")

#: ``W<n>|<clause>|<path>:<line>|<anchor>`` rows, one per clause per wing.
ROW_BLOCK = re.compile(r"```freeze-manifest\n(.*?)```", re.DOTALL)
#: The declared digest, on its own line.
SHA_LINE = re.compile(r"^FREEZE-SHA256\s*=\s*([0-9a-f]{64})\s*$", re.M)
#: The declared count, on its own line.
N_LINE = re.compile(r"^FROZEN-N\s*=\s*(\d+)\s*$", re.M)
#: Wings retired to UNPRODUCED, with the price that makes the retirement real.
RETIRED_LINE = re.compile(r"^RETIRED\s*=\s*(\S+)\s+PRICE\s+([0-9.]+)\s*s\s*$", re.M)


def _text() -> str:
    if not MANIFEST.is_file():
        raise AssertionError(
            f"{MANIFEST.name} does not exist. The list is not frozen, so N is "
            f"undefined and every wing in this round is a named wing.")
    return MANIFEST.read_text(encoding="utf-8", errors="replace")


def rows() -> list[tuple[str, str, str, str]]:
    m = ROW_BLOCK.search(_text())
    assert m, "no ```freeze-manifest block in the manifest: nothing to freeze"
    out = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|", 3)
        assert len(parts) == 4, f"malformed manifest row: {line!r}"
        out.append(tuple(p.strip() for p in parts))
    return out


def digest(rs) -> str:
    """sha256 over the normalised, sorted citation tuples. Rows only."""
    body = "\n".join("|".join(r) for r in sorted(rs))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def journalled_cells(kind: str) -> int:
    """Records in `results/` whose `kind` field is exactly `kind`.

    This is clause (b) made mechanical. It reads the ledger, not the source
    tree: a module that exists but has never produced a journalled cell scores
    zero here, which is precisely the named-not-found condition.
    """
    n = 0
    for p in sorted((ROOT / "results").rglob("*.jsonl")):
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            ln = ln.strip()
            if not ln.startswith("{"):
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            if isinstance(rec, dict) and rec.get("kind") == kind:
                n += 1
    return n


# ======================================================================
# 1. FOUND, NOT NAMED -- with both sides of the discriminator exercised
# ======================================================================

def test_the_found_wing_counter_reads_zero_for_the_named_wing():
    """CALIBRATION. `arm_phase` is the planted negative for the discriminator.

    It is a real module with a real class and a real gate function, and it has
    never produced a journalled cell. If this ever reads non-zero, W2's strike
    is void and the manifest must be re-opened.
    """
    assert journalled_cells("arm_phase") == 0, (
        "arm_phase now has journalled cells; the it.4 strike is void")


def test_the_found_wing_counter_is_not_always_zero():
    """CALIBRATION, other side. A counter that cannot count is not evidence."""
    assert journalled_cells("softmax") > 0


#: Parametrized on WING IDS, not arm names. Parametrized on arm names it took no
#: wing argument at all, so it asked "does this arm exist?" and never "is this arm
#: THIS wing's?" -- it could not witness a wing/arm mismatch in either direction
#: (MARS it.18 STRIKE 2). The arm is now derived from the wing under test.
@pytest.mark.parametrize("wing", sorted({r[0] for r in rows()}))
def test_every_frozen_wing_is_found_and_not_merely_named(wing):
    arm = _arm_of(wing)
    assert journalled_cells(arm) > 0, (
        f"{wing} is on the frozen list carrying arm {arm!r} with zero journalled "
        f"cells: it is a named wing, and the round's binding kill strikes it")


def test_the_frozen_list_names_only_found_wings():
    named = {r[0] for r in rows()}
    arms = {}
    for w, clause, cite, _anchor in rows():
        if clause == "b":
            arms[w] = cite
    assert set(arms) == named, "some wing carries no clause-(b) citation"
    for w, cite in arms.items():
        assert cite.startswith("results/"), (
            f"{w} clause (b) cites {cite}, which is not under results/. After "
            f"the it.4 strike, ledger evidence means a journalled cell.")


# ======================================================================
# 2. THE FREEZE -- structure, count, and the tamper-evident digest
# ======================================================================

def test_every_frozen_wing_cites_all_four_clauses():
    per = {}
    for w, clause, cite, anchor in rows():
        per.setdefault(w, {})[clause] = (cite, anchor)
    assert per, "the frozen list is empty"
    for w, got in sorted(per.items()):
        assert set(got) == set(CLAUSES), (
            f"{w} cites {sorted(got)}, not all four of {CLAUSES}")


def test_the_declared_N_equals_the_number_of_wing_rows():
    m = N_LINE.search(_text())
    assert m, "the manifest declares no FROZEN-N line"
    assert int(m.group(1)) == len({r[0] for r in rows()})


def test_no_wing_cites_the_same_line_for_two_different_clauses():
    seen = {}
    for w, clause, cite, _ in rows():
        key = (w, cite)
        assert key not in seen, (
            f"{w} cites {cite} for both clause {seen[key]} and clause {clause}: "
            f"one line cannot be four kinds of evidence")
        seen[key] = clause


def test_the_declared_hash_matches_the_rows():
    m = SHA_LINE.search(_text())
    assert m, "the manifest declares no FREEZE-SHA256 line: it is not frozen"
    assert m.group(1) == digest(rows()), (
        "the manifest rows do not hash to the declared digest -- the list was "
        "edited after it was frozen, or the digest was never recomputed")


def test_moving_one_citation_by_one_line_fires_both_binds():
    """THE PLANTED NEGATIVE. Perturb ONE line number by +1 and watch it fire.

    Two failures are demanded, because the two binds are independent: the
    digest moves (the list changed), and the moved citation stops resolving
    (the content is no longer at the cited line). A tamper that only moved one
    of the two would show which check is asleep.
    """
    rs = rows()
    victim = next(r for r in rs if r[1] == "a")
    path_s, _, line_s = victim[2].rpartition(":")
    bumped = (victim[0], victim[1], f"{path_s}:{int(line_s) + 1}", victim[3])
    tampered = [bumped if r == victim else r for r in rs]

    assert digest(tampered) != digest(rs), "the digest is blind to a row edit"
    with pytest.raises(AssertionError):
        resolve(bumped[2], bumped[3])


# ======================================================================
# 3. STILL RESOLVING AT HEAD
# ======================================================================

@pytest.mark.parametrize("clause", CLAUSES)
def test_every_frozen_citation_resolves_at_head(clause):
    for w, c, cite, anchor in rows():
        if c != clause:
            continue
        resolve(cite, anchor)   # raises, naming which of the three checks failed


# ======================================================================
# 4. THE STRIKE SHIPS A PRICE
# ======================================================================

def test_the_struck_wing_is_retired_with_a_number():
    m = RETIRED_LINE.search(_text())
    assert m, (
        "no RETIRED = <wing> PRICE <n>s line. A wing struck without a "
        "replacement route is an unfinished report (STATE.md:12).")
    assert m.group(1) == "arm_phase"
    assert float(m.group(2)) > 0.0


# ======================================================================
# 5. THE LOG DEFECT C14, REPAIRED MECHANICALLY
# ======================================================================
#
# it.2's C14 strike was a LOG defect: one event at `house-events.jsonl:11719`
# with no `t` field at all, so RED and verdict arrived in one atom and nothing
# in the log established RED-first. The repair is a bind, not a resolution.
#
# Scope is deliberately this office's it.4 events, not the whole file: the log
# is append-only, and 116 of its 11,764 lines predate any `t` convention (they
# carry `kind` or `event` instead -- e.g. `:399`, `:993` use `kind`; `:11744`,
# appended THIS round by another node, uses `event`). Binding the history would
# be a red that can never go green because the fix is forbidden.

EVENTS = ROOT / "house-events.jsonl"
_T_ORDER = {"red": 0, "finding": 1, "done": 2}


def _my_it4_events():
    out = []
    for i, ln in enumerate(EVENTS.read_text(encoding="utf-8",
                                            errors="replace").splitlines(), 1):
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        if (str(rec.get("agent", "")).lower() == "saturn"
                and rec.get("iteration") == 4):
            out.append((i, rec))
    return out


def test_the_missing_t_detector_actually_detects():
    """CALIBRATION for the node below: a t-less event must be visible as one."""
    assert "t" not in {"agent": "Saturn", "iteration": 4}


def test_every_it4_saturn_event_carries_a_t_field():
    evs = _my_it4_events()
    assert evs, "this office logged nothing at it.4"
    missing = [i for i, r in evs if "t" not in r]
    assert not missing, f"it.4 SATURN events with no t field: {missing} (C14)"


def test_the_it4_red_is_its_own_event_and_precedes_every_finding():
    evs = _my_it4_events()
    kinds = [(i, r["t"]) for i, r in evs if r["t"] in _T_ORDER]
    assert kinds[0][1] == "red", (
        f"the first it.4 SATURN event is {kinds[0]}, not a red: RED-first is "
        f"not established by the log (C14)")
    first_red = kinds[0][0]
    findings = [i for i, t in kinds if t == "finding"]
    assert findings and min(findings) > first_red


# ======================================================================
# 5. THE JOINT -- added it.14 under VENUS's `V-26`
#
# Nodes 1-3 above are TWO MARGINALS. `journalled_cells(wing) > 0` is a
# function of `results/` alone; `resolve(cite, anchor)` is a function of the
# cited file's text alone. Neither ranges over the PAIR, and clause (b)'s
# claim is a pair: *this wing's evidence is the record on this line*.
#
# Exchanging the `kind` field of the two clause-(b) records preserves every
# per-kind count and moves no anchor, so every node above stays green while
# the manifest certifies W1 by an arm_pl record and W3 by an arm_smprime one.
# Demonstrated at
# tests/saturn/test_v20_r15_it14_saturn.py::test_the_it4_freeze_is_two_marginals_and_stays_green_under_the_joint_break
# ======================================================================

def _norm(s: str) -> str:
    """Lowercased, with every run of non-alphanumerics folded to one underscore, so
    `V16_ARM_SMPRIME.md:529` and `ARM PL with the parity claim RETIRED` both spell
    their arm the way clause (a)'s module stem does."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower())


def _arm_of(wing: str) -> str:
    """The wing's arm name, DERIVED from its own clause-(a) module path rather
    than written down here -- a hardcoded map would be a second place for the
    manifest to be wrong."""
    for w, clause, cite, _anchor in rows():
        if w == wing and clause == "a":
            return pathlib.Path(cite.rpartition(":")[0]).stem
    raise AssertionError(f"{wing} cites no clause (a); its arm name is underivable")


def test_every_clause_b_line_journals_that_wings_own_arm():
    """THE JOINT. The cited line must be a journalled cell OF THAT WING."""
    for w, clause, cite, _anchor in rows():
        if clause != "b":
            continue
        path_s, _, line_s = cite.rpartition(":")
        text = (ROOT / path_s).read_text(encoding="utf-8", errors="replace").splitlines()
        lineno = int(line_s)
        assert 1 <= lineno <= len(text), f"{w} clause (b) cites {cite}, past end of file"
        try:
            rec = json.loads(text[lineno - 1])
        except Exception as exc:                       # pragma: no cover - diagnostic
            raise AssertionError(f"{w} clause (b) line {cite} is not a JSON record: {exc}")
        arm = _arm_of(w)
        assert isinstance(rec, dict) and rec.get("kind") == arm, (
            f"{w} clause (b) cites {cite}, which journals kind="
            f"{rec.get('kind')!r}, not {arm!r}. The list and the ledger are "
            f"paired in prose and were never paired in a node.")


def test_every_wings_arm_is_corroborated_outside_clause_a():
    """THE SECOND WITNESS, and the reason clause (b) alone was not enough.

    `_arm_of` reads clause (a). `test_every_clause_b_line_journals_that_wings_own_arm`
    then checks clause (b) AGAINST clause (a) -- and clause (a) against nothing.
    Replacing the hardcoded wing->arm map with a derivation removed the second place
    the manifest could disagree with itself, which was the only place it could be
    CAUGHT being wrong. Exchanging W1's and W3's clause-(a) and clause-(b) citations
    together preserved every per-collection statistic and left all seven semantic
    nodes green while W1 -- whose (c) and (d) still cite V16_ARM_SMPRIME.md and the
    15.970 smprime price -- was certified `arm_pl`.

    Clause (c) is the wing's ACCEPTED KILL. It is cited independently of the module
    path clause (a) names, so it is the outside witness: the derived arm must be
    named by clause (c)'s path or by the anchor that must occur on its exact line.
    Already satisfied by the true rows; fires under the exchange.
    """
    for wing in sorted({r[0] for r in rows()}):
        arm = _arm_of(wing)
        kill = [(cite, anchor) for w, c, cite, anchor in rows() if w == wing and c == "c"]
        assert kill, f"{wing} cites no clause (c); its arm has no second witness"
        cite, anchor = kill[0]
        witness = _norm(f"{cite} {anchor}")
        assert _norm(arm) in witness, (
            f"{wing} derives arm {arm!r} from its clause-(a) module path, but its "
            f"clause (c) -- the kill it accepted -- names neither: {cite} / {anchor!r}. "
            "Clause (a) and clause (c) disagree about which wing this is, and only "
            "one of them can be right.")


# ======================================================================
# 6. THE THIRD WITNESS AND THE GROUND PIN -- added it.21 under MARS's it.20
#    STRIKE 1
#
# The it.19 second witness was priced wrong in this office's own Limits: it read
# "(a)+(c) together survives". It does not -- `_arm_of` reads clause (a) and
# clause (b) is checked against it, so the two-clause move is caught by the joint
# node, not by the new witness. What survives is (a)+(b)+(c) moved TOGETHER, and
# it survived every node the freeze file shipped: W1 certified `arm_pl` while its
# clause (d) still cited V17_R4_RETAKE_PRICE.md:194, the 15.970 smprime price.
# tests/mars_v20/test_it20_two_clauses_is_not_two_witnesses.py is that RED.
#
# Two nodes answer it, and they are answering different questions.
#
#   * The witness below reads clause (d)'s cited LINE. Every witness before it is
#     a function of the manifest's own rows -- clause (c) is a third seat in the
#     same closed loop, because a tampering editor moves rows. This one reads a
#     file the editor does not own: the price table names its arm in the same row
#     as the number, so the manifest cannot claim W1's cost without naming W1's arm.
#   * The pin below reads a map held OUTSIDE the citation set entirely. Every
#     clause is a citation; move all four together and the manifest is internally
#     perfect and describes the other wing.
# ======================================================================

def test_every_wings_arm_is_named_by_the_price_row_it_cites():
    """THE THIRD WITNESS -- the first that reads a cited FILE rather than the list.

    `V17_R4_RETAKE_PRICE.md:194` is ``| `arm_smprime` | **15.970** | ...`` and
    `:195` is ``| `arm_pl` | **1.614** | ...``. The arm name and the price are on
    ONE line, so clause (d) cannot be moved to another wing without moving a line
    that names the arm it came from. The anchor check (node 3) already binds the
    NUMBER to the line; this binds the WING to it, which is the pair the manifest
    asserts in prose and never asserted in a node.
    """
    for wing in sorted({r[0] for r in rows()}):
        arm = _arm_of(wing)
        cost = [(cite, anchor) for w, c, cite, anchor in rows()
                if w == wing and c == "d"]
        assert cost, f"{wing} cites no clause (d); its cost has no witness"
        cite, anchor = cost[0]
        path_s, _, line_s = cite.rpartition(":")
        text = (ROOT / path_s).read_text(encoding="utf-8",
                                         errors="replace").splitlines()
        lineno = int(line_s)
        assert 1 <= lineno <= len(text), (
            f"{wing} clause (d) cites {cite}, past the end of the file")
        line = text[lineno - 1]
        assert _norm(arm) in _norm(line), (
            f"{wing} derives arm {arm!r} from its clause-(a) module path and prices "
            f"itself at {cite} ({anchor!r}) -- a line that reads {line.strip()!r} and "
            f"names a different arm. The cost footprint belongs to the other wing. "
            f"Clauses (a), (b) and (c) can all be moved together and stay mutually "
            f"consistent; this line lives outside the manifest and does not move.")


def test_the_wing_ids_are_pinned_to_their_arms_outside_the_citation_set():
    """THE GROUND PIN, and the ruling on re-importing what it.14 struck.

    it.14 deleted a hardcoded `wing -> arm` map from `_arm_of` and was right to:
    used as a SOURCE, a written-down copy of a value clause (a) already determines
    is a second place for the manifest to be wrong, and the two can diverge quietly.

    Re-imported here it is not a source. `_arm_of` still derives; the map is the
    ASSERTION TARGET, and divergence is now the loudest thing in the file rather
    than the quietest. The difference that licenses it is the one MARS's V-26
    mutation makes visible: every clause of this manifest is a CITATION, and an
    editor who moves all four together leaves a document that is internally perfect
    and describes the other wing. A pin that no clause can move is the only thing
    the four-clause move cannot satisfy. A map removed for being a second place to
    be wrong was also the only second place to be RIGHT.

    Imported inside the node because `test_v20_r15_it14_saturn` imports this module
    at load; a module-level import here would be a cycle.
    """
    from tests.saturn.test_v20_r15_it14_saturn import WING_ARM

    derived = {w: _arm_of(w) for w in sorted({r[0] for r in rows()})}
    assert set(derived) == set(WING_ARM), (
        f"the frozen list names wings {sorted(derived)} and the pin holds "
        f"{sorted(WING_ARM)}. A wing added or struck without moving the pin means "
        f"the identity of the list is no longer stated anywhere outside itself.")
    for wing, arm in sorted(derived.items()):
        assert arm == WING_ARM[wing], (
            f"{wing} derives arm {arm!r} from its own citations, but {wing} IS "
            f"{WING_ARM[wing]!r} (tests/saturn/test_v20_r15_it14_saturn.py:39). "
            f"Every clause agrees with every other clause and all of them are "
            f"about the wrong wing.")


# ==========================================================================
#  it.22 -- THE PIN'S WITNESS
#
#  it.21 shipped the pin above and filed its own limit against it: "wing
#  identity now rests on a two-entry map that nothing corroborates." Two
#  hand-written entries, and the whole three-clause-swap defence hangs on them.
#  A pin no clause can move is still a pin an EDITOR can move.
#
#  The witness is deliberately not a fifth clause, and not another document
#  citing a line. It is the round's own filed record: every office that has
#  written about a wing has written its id beside its arm, in prose, in its own
#  report, across twenty-two iterations. To move the pin now an editor must also
#  rewrite reports authored by MARS, VENUS, JUPITER, MERCURY and the INSPECTOR --
#  documents this office does not own, which the JOURNAL quotes, and which the
#  round's append-only law forbids amending. That is a change of class, not
#  another seat in the same loop.
# ==========================================================================

_WING_ID = re.compile(r"\bW\d+\b")
_ARM_NAME = re.compile(r"\barm_[a-z][a-z0-9_]*\b")
#: The office that authored a round record, from its filename.
_OFFICE = re.compile(r"^V20_R15_IT\d+[A-Z]*_([A-Z0-9]+)\.md$")
#: This office's report for THIS iteration. A witness must not be authored by the
#: same office in the same breath as the thing it certifies.
_SELF = "V20_R15_IT22_SATURN.md"


def _corpus_votes() -> dict:
    """{wing: {arm: {filename, ...}}} over the round's filed reports.

    One vote per FILE, so a single document cannot carry a wing by repetition,
    and each wing id is paired with the NEAREST arm name on its own line so that
    a row naming both wings (`W3 arm_pl - W1 arm_smprime`) votes for both
    correctly rather than for everything.

    ponytail: nearest-token-on-a-line, not a parse. Its ceiling is a sentence
    that names a wing and mentions the other wing's arm closer to it; the
    margin assertions below are what price that ceiling, and the corpus in fact
    contains contrary votes (see the calibration node) which it survives.
    """
    votes: dict = {}
    for f in sorted(MANIFEST.parent.glob("V20_R15_IT*.md")):
        if f.name == _SELF:
            continue
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            wings = list(_WING_ID.finditer(line))
            arms = list(_ARM_NAME.finditer(line))
            if not wings or not arms:
                continue
            for w in wings:
                near = min(arms, key=lambda a: abs(a.start() - w.start()))
                votes.setdefault(w.group(0), {}).setdefault(
                    near.group(0), set()).add(f.name)
    return votes


def _offices(files) -> set:
    return {m.group(1) for m in (_OFFICE.match(n) for n in files) if m}


def test_the_pin_is_corroborated_by_the_rounds_own_filed_record():
    """The corroboration it.21 said the pin lacked."""
    from tests.saturn.test_v20_r15_it14_saturn import WING_ARM

    votes = _corpus_votes()
    for wing, arm in sorted(WING_ARM.items()):
        tally = sorted(((len(v), a) for a, v in votes.get(wing, {}).items()),
                       reverse=True)
        assert tally, f"{wing} is named beside no arm anywhere in the round's record"
        top_n, top_arm = tally[0]
        runner = tally[1][0] if len(tally) > 1 else 0
        who = _offices(votes[wing][top_arm])
        assert top_arm == arm, (
            f"the pin holds {wing} = {arm!r} and the round's own record says "
            f"{top_arm!r} in {top_n} files against {runner}. The pin is the only "
            f"statement of wing identity outside the manifest and it disagrees "
            f"with every office that has written about this wing.")
        assert top_n >= 5 and top_n >= 2 * runner, (
            f"{wing} = {arm!r} is carried by {top_n} files against {runner} for the "
            f"runner-up. That is not a corroboration, it is a coin toss.")
        assert len(who) >= 3, (
            f"{wing} = {arm!r} is stated only by {sorted(who)}. A witness one office "
            f"can author alone is the pin again with more words.")


def test_the_corroboration_can_return_false():
    """A probe that cannot return false is not a probe.

    The corpus is not unanimous and that is the point: the it.18 and it.21
    mutation transcripts print `arm_of W1 = arm_pl` verbatim, so the SWAPPED
    pairing is physically present in the round's record. The method has to
    resolve against it rather than merely find no contradiction.
    """
    from tests.saturn.test_v20_r15_it14_saturn import WING_ARM

    votes = _corpus_votes()
    arms = set(WING_ARM.values())
    for wing, arm in sorted(WING_ARM.items()):
        wrong = sorted(arms - {arm})
        contrary = {a: len(votes.get(wing, {}).get(a, ())) for a in wrong}
        assert any(contrary.values()), (
            f"no file in the round's record ever pairs {wing} with {wrong}, so this "
            f"method has never been asked to choose. It agrees with the pin because "
            f"it cannot disagree with anything.")
        right = len(votes[wing][arm])
        assert right > max(contrary.values()), (
            f"{wing}: {right} files say {arm!r} and {contrary} say otherwise.")
