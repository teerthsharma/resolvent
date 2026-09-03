"""L-FIND: every wing listed carries four RESOLVABLE citations, one per clause.

THE BINDING KILL THIS ENFORCES. "A wing named rather than found => struck."
A wing list is worthless if its citations are decorative -- `path:line` pairs
that point at a file that moved, a line that drifted, or a claim that was never
there. This project has already struck numbers for exactly that shape
(`STRUCK.md:24` -- an interval "in no .py, .json, .jsonl or .txt in the tree"),
and `MISTAKES.md:364` names line-reference drift as its own failure class P-6.

So a citation is checked THREE ways, and the third is the one that matters:

    1. the file exists
    2. the line number is inside the file
    3. an ANCHOR string declared beside the citation actually occurs on that
       exact line

Check 3 is what makes this a VALUE comparison rather than a structure-by-regex
comparison. `tests/loop/test_arms_distinct.py` records that four
structure-by-regex instruments in this repository have broken, and that the two
that never broke both compare values. This one compares values.

CALIBRATED BEFORE USE, BOTH HALVES. `MISTAKES.md:804` (V-15, "the condemning
rule with no planted negative") and `MISTAKES.md:117` (V-7, "a search
structurally incapable of finding anything, read as absence") are the two
entries this file is written against, so:

    * the citation checker is fed a PLANTED BAD citation and must raise;
    * the clause-(c) absence search is fed a term whose kill is KNOWN to be in
      the record and must FIND it -- otherwise "no kill found" means only that
      the search cannot find kills.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORT = ROOT / "V20_R15_IT1_SATURN.md"

#: The four clauses of the L-FIND rubric. A wing must cite one line per clause.
CLAUSES = ("a", "b", "c", "d")

#: Fenced block in the report carrying the machine-checkable citations, one per
#: line: ``W<n>|<clause>|<path>:<line>|<anchor>``
CITATION_BLOCK = re.compile(r"```l-find-citations\n(.*?)```", re.DOTALL)

#: Fenced block carrying the clause-(c) absence claims, one per line:
#: ``<near-miss id>|<regex the record is searched for>``
ABSENCE_BLOCK = re.compile(r"```l-find-absence\n(.*?)```", re.DOTALL)

#: Words that mark a line as recording a kill.
KILL_WORDS = ("kill", "struck", "dead", "killed", "retired", "withdrawn",
              "must-fire", "refuted")


# ==========================================================================
# the checker, as a pure function so it can be run against a KNOWN-BAD input
# ==========================================================================

def resolve(cite: str, anchor: str) -> str:
    """Return the cited line, or RAISE naming which of the three checks failed."""
    if ":" not in cite:
        raise AssertionError(f"citation {cite!r} is not of the form path:line")
    path_s, _, line_s = cite.rpartition(":")
    if not line_s.isdigit():
        raise AssertionError(f"citation {cite!r} has no line number")
    path = ROOT / path_s
    if not path.is_file():
        raise AssertionError(
            f"citation {cite!r}: no such file under the repo root. This is the "
            f"STRUCK.md:24 shape -- a reference to a producer not in the tree.")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    n = int(line_s)
    if not 1 <= n <= len(lines):
        raise AssertionError(
            f"citation {cite!r}: file has {len(lines)} lines, so line {n} does "
            f"not exist (MISTAKES.md:364, line-reference drift)")
    got = lines[n - 1]
    if anchor not in got:
        raise AssertionError(
            f"citation {cite!r}: anchor {anchor!r} is NOT on that line.\n"
            f"  line {n} actually reads: {got.strip()[:200]!r}\n"
            f"This is line-reference drift (MISTAKES.md:364): the citation "
            f"resolves to a real line that does not carry the claim.")
    return got


def kill_hits(pattern: str) -> list[str]:
    """Every root-level .md line matching `pattern` AND carrying a kill word.

    Scoped to root documents deliberately: `kaggle/snapshot/` is a byte copy of
    the tree and would double every hit, and `attic/` is archive.

    THE REPORT ITSELF IS EXCLUDED. It names each near-miss beside the word
    "kill" in the very table that claims no kill exists, so leaving it in makes
    every absence claim refute itself the moment it is written down. A document
    cannot be evidence for its own claim.
    """
    rx = re.compile(pattern, re.IGNORECASE)
    out = []
    for md in sorted(ROOT.glob("*.md")):
        if md.name == REPORT.name:
            continue
        text = md.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            if rx.search(line) and any(w in low for w in KILL_WORDS):
                out.append(f"{md.name}:{i}: {line.strip()[:160]}")
    return out


# ==========================================================================
# CALIBRATION -- both instruments must be SEEN to fire before any green counts
# ==========================================================================

def test_the_citation_checker_rejects_a_file_that_does_not_exist():
    with pytest.raises(AssertionError, match="no such file"):
        resolve("ceq/an_arm_that_was_never_written.py:1", "anything")


def test_the_citation_checker_rejects_a_line_past_the_end_of_the_file():
    with pytest.raises(AssertionError, match="does not exist"):
        resolve("STRUCK.md:999999", "anything")


def test_the_citation_checker_rejects_an_anchor_that_is_not_on_the_line():
    """The check that makes this a VALUE comparison. Planted negative."""
    with pytest.raises(AssertionError, match="is NOT on that line"):
        resolve("STRUCK.md:1", "this string is not on line 1 of STRUCK.md")


def test_the_citation_checker_accepts_a_citation_that_is_true():
    """The other half: it must not cry wolf."""
    assert "STRUCK" in resolve("STRUCK.md:1", "# STRUCK")


def test_the_absence_search_can_find_a_kill_that_is_known_to_be_in_the_record():
    """V-7's bind. An absence search that cannot find a present kill is void.

    `ARSENAL.md:303-304` records the DeltaFloor-by-eviction selector dying with
    both pre-registered kills fired. If the search cannot see that, then every
    "no kill found" this file reports means nothing.
    """
    hits = kill_hits(r"ΔFloor|DeltaFloor")
    assert hits, (
        "the clause-(c) absence search found NO kill for the DeltaFloor "
        "selector, whose kill IS in the record at ARSENAL.md:303. The search "
        "is structurally incapable of finding a kill (MISTAKES.md:117, V-7) "
        "and every absence it reports is void.")


# ==========================================================================
# THE BIND ITSELF
# ==========================================================================

def _report_text() -> str:
    if not REPORT.is_file():
        pytest.fail(
            f"{REPORT.name} does not exist. The wing list has not been filed, "
            f"so no wing in this round carries four citations and L-FIND is "
            f"unsatisfied by default.")
    return REPORT.read_text(encoding="utf-8", errors="replace")


def _citations() -> dict:
    m = CITATION_BLOCK.search(_report_text())
    if m is None:
        pytest.fail(
            "no ```l-find-citations``` block in the report. A wing list whose "
            "citations cannot be machine-resolved is a wing list NAMED rather "
            "than FOUND.")
    wings: dict = {}
    for raw in m.group(1).splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split("|")
        assert len(parts) == 4, (
            f"malformed citation row {raw!r}; "
            f"want W<n>|<clause>|<path:line>|<anchor>")
        wing, clause, cite, anchor = (p.strip() for p in parts)
        assert clause in CLAUSES, f"unknown clause {clause!r} in {raw!r}"
        wings.setdefault(wing, {})[clause] = (cite, anchor)
    assert wings, "the citation block is empty: N = 0 wings"
    return wings


def test_every_wing_cites_all_four_clauses():
    wings = _citations()
    incomplete = {w: sorted(set(CLAUSES) - set(cl))
                  for w, cl in wings.items() if set(cl) != set(CLAUSES)}
    assert not incomplete, (
        f"wings missing a clause citation: {incomplete}. A wing short of any "
        f"clause is a NEAR-MISS and belongs in the near-miss table, not the "
        f"wing list.")


@pytest.mark.parametrize("clause", CLAUSES)
def test_every_citation_resolves_to_a_line_that_carries_its_anchor(clause):
    wings = _citations()
    failures = []
    for wing in sorted(wings):
        cite, anchor = wings[wing][clause]
        try:
            resolve(cite, anchor)
        except AssertionError as exc:
            failures.append(f"{wing} clause ({clause}): {exc}")
    assert not failures, "\n".join(failures)


def test_no_wing_cites_the_same_line_for_two_different_clauses():
    """Four citations that are one citation four times is one citation."""
    wings = _citations()
    bad = {}
    for wing, cl in wings.items():
        seen = [c for c, _ in cl.values()]
        if len(set(seen)) != len(seen):
            bad[wing] = seen
    assert not bad, (
        f"wings reusing one line across clauses: {bad}. Four clauses need four "
        f"independent pieces of evidence.")


def test_each_near_miss_claiming_no_kill_really_has_none_in_the_record():
    """The near-miss table's clause-(c) claims, checked against the record."""
    m = ABSENCE_BLOCK.search(_report_text())
    if m is None:
        pytest.fail(
            "no ```l-find-absence``` block in the report. Every near-miss "
            "declared to be missing clause (c) must name the search that found "
            "no kill, so the search can be re-run.")
    rows = [r.strip() for r in m.group(1).splitlines()
            if r.strip() and not r.strip().startswith("#")]
    assert rows, "the absence block is empty"
    found = {}
    for raw in rows:
        nm, _, pattern = raw.partition("|")
        hits = kill_hits(pattern.strip())
        if hits:
            found[nm.strip()] = hits[:5]
    assert not found, (
        f"near-misses declared to have NO kill in the record, for which a kill "
        f"WAS found: {found}. Each is a wing, not a near-miss, or the declared "
        f"search pattern is wrong.")


# ==========================================================================
# CLAUSE (b) SECOND BIND -- the annex's own [RUN] instances
# ==========================================================================
# L-FIND clause (b) admits "an X-item with a run instance". The Mathematics
# Annex in `CEQ_V20_R15_CONTRACT.md` marks sixteen items `[RUN: ...]`. Under
# `MISTAKES.md:289` (P-1, "a number with no live producer") a `[RUN]` figure is
# admissible only while something in the tree can emit it. These four tokens are
# distinctive enough that a substring search cannot collide with an unrelated
# float, which is why they and not the short decimals are the ones bound.

ANNEX_RUN_TOKENS = {
    "M9-F1  Cantelli/Boole cutoff": "d=65",
    "M2     gate-landscape theta": "1.3e-3",
}

#: Annex `[RUN]` items whose figure CANNOT be bound by this instrument, and the
#: reason. Not "no producer exists" -- "this search cannot decide".
#:
#: THE MECHANISM, and it is this file's own defect (`MISTAKES.md` V-7, "a search
#: structurally incapable of finding anything, read as absence"). Iteration 1
#: registered four tokens here, two of them ENGLISH PHRASES. `producers_for` is
#: a literal substring search and its calibration control,
#: `test_the_producer_search_finds_a_number_that_does_have_one`, is a NUMBER
#: (`0.7071067811865476`) -- so the search is certified for compact literals
#: only. No `.py` emits "d=20 at the same"; the zero-hit reading was structural,
#: not evidential, and both phrase tokens were struck against Wilson §7 M13,
#: which names live producers (`scale/foreman_consequence.py:12`,
#: `scale/foreman_signfloor.py:382`) that this file's own filter does not
#: exclude. V-7 was cited by name in this module's own docstring and enforced on
#: the clause-(c) ABSENCE search while the producer search went unguarded --
#: one rule, half an instrument.
UNBINDABLE_BY_SUBSTRING = {
    "M9-F1' Azuma cutoff": "d=20 at the same",
    "M13    W1 vs KL": "0.492 vs KL 0.519",
}

#: Extensions that can EMIT a number, as opposed to quote one.
PRODUCER_SUFFIXES = (".py", ".json", ".jsonl", ".txt", ".log")


#: A token this search is CERTIFIED for: a compact literal carrying a digit and
#: no whitespace. The calibration control is a number, so the certified domain
#: is numbers and number-shaped literals -- never an English phrase, which no
#: `.py` emits and which therefore reads zero hits for a structural reason
#: rather than an evidential one.
ADMISSIBLE_TOKEN = re.compile(r"^\S*\d\S*$")


def producers_for(token: str) -> list[str]:
    """Files that could have produced `token`, excluding prose and archives.

    RAISES on a token outside the certified domain rather than returning `[]`.
    A zero-hit reading from an unfindable token is the V-7 defect
    (`MISTAKES.md:117`) and is exactly how two of iteration 1's four K6
    instances were struck: they were English phrases, and no producer emits
    prose. Rejecting at REGISTRATION is what makes the remaining zero-hit
    readings mean "no producer" instead of "wrong question".

    THIS FILE IS EXCLUDED, and that exclusion is the whole check. The first
    version of this search matched its own `ANNEX_RUN_TOKENS` literals and went
    GREEN by quoting itself -- `MISTAKES.md:72` (V-3, "the assertion is an
    algebraic identity of your own construction"). A search whose only hit is
    the searcher has found nothing.
    """
    if not ADMISSIBLE_TOKEN.match(token):
        raise ValueError(
            f"token {token!r} is outside this search's certified domain. "
            f"`producers_for` is a literal substring search calibrated on a "
            f"NUMBER; a token containing whitespace is prose, no .py emits it, "
            f"and its zero-hit reading would be structural rather than "
            f"evidential (MISTAKES.md:117, V-7).")
    out = []
    for path in ROOT.rglob("*"):
        if path.suffix not in PRODUCER_SUFFIXES or not path.is_file():
            continue
        if path.resolve() == pathlib.Path(__file__).resolve():
            continue
        # `house-events*.jsonl` is the agents' append-only EVENT LOG. It records
        # what was said, including contract text pasted verbatim, so it will
        # contain any number the contract contains. Quoting a number is not
        # emitting it -- treating the transcript as a producer is P-2,
        # `MISTAKES.md:300` ("a number whose only home is a commit message").
        if path.name.startswith("house-events"):
            continue
        parts = path.parts
        if any(p in (".git", "kaggle", "attic", "__pycache__") for p in parts):
            continue
        try:
            if token in path.read_text(encoding="utf-8", errors="replace"):
                out.append(str(path.relative_to(ROOT)))
        except OSError:
            continue
    return out


def test_the_producer_search_finds_a_number_that_does_have_one():
    """Calibration (V-7). A search that finds nothing must first find something.

    `floor_1` is journalled as a full-precision double in several `results/`
    cells, so a working search MUST return them. The first control tried here
    was `0.371583`, MISTAKES.md:2153's re-run zero-step gate-R2 -- and it FAILED,
    because that number lives only in prose. That failure is the reason this
    control is a journalled one: a control that is itself unproduced cannot
    calibrate a producer search.
    """
    hits = producers_for("0.7071067811865476")
    assert hits, (
        "the producer search found no file for a number that HAS a producer; "
        "the search is void (MISTAKES.md:117, V-7)")
    assert any(h.replace("\\", "/").startswith("results/") for h in hits), (
        f"the control resolved only to non-journal files {hits}; a producer "
        f"search calibrated on prose is still void")


def test_every_annex_run_instance_has_a_producer_in_the_tree():
    """Each `[RUN:]` figure in the annex must be emittable by something here."""
    orphans = {name: tok for name, tok in ANNEX_RUN_TOKENS.items()
               if not producers_for(tok)}
    assert not orphans, (
        f"annex items marked [RUN] whose figure NO .py/.json/.jsonl/.txt/.log "
        f"in this tree can emit: {orphans}. Under MISTAKES.md:289 (P-1, a "
        f"number with no live producer) these are not reproducible here, so "
        f"they cannot yet serve as L-FIND clause (b) ledger evidence.")


# ==========================================================================
# THE PRODUCER SEARCH'S OWN DOMAIN -- calibration of the REPAIR (K6 it.2)
# ==========================================================================

def test_the_producer_search_rejects_a_phrase_instead_of_reading_zero_hits():
    """RED-first for the repair. The struck token must RAISE, not return [].

    `"d=20 at the same"` is annex PROSE. Iteration 1 registered it, got zero
    hits, and read that as "no producer" -- the V-7 shape. The repaired search
    must refuse the question.
    """
    with pytest.raises(ValueError, match="certified domain"):
        producers_for("d=20 at the same")


def test_the_producer_search_rejects_the_other_struck_token_too():
    with pytest.raises(ValueError, match="certified domain"):
        producers_for("0.492 vs KL 0.519")


def test_every_registered_token_is_inside_the_certified_domain():
    """Registration gate. What the search cannot decide cannot be registered."""
    bad = {n: t for n, t in ANNEX_RUN_TOKENS.items()
           if not ADMISSIBLE_TOKEN.match(t)}
    assert not bad, (
        f"tokens registered that this search is not certified for: {bad}. "
        f"Move them to UNBINDABLE_BY_SUBSTRING with the reason.")


def test_the_calibration_control_is_itself_inside_the_domain():
    """The rule must cover the control, or it is a rule for other people."""
    assert ADMISSIBLE_TOKEN.match("0.7071067811865476")


def test_the_withdrawn_tokens_are_recorded_rather_than_deleted():
    """A struck instance is withdrawn in writing, never quietly dropped."""
    assert set(UNBINDABLE_BY_SUBSTRING) == {"M9-F1' Azuma cutoff",
                                            "M13    W1 vs KL"}
    assert not (set(UNBINDABLE_BY_SUBSTRING) & set(ANNEX_RUN_TOKENS))


# ==========================================================================
# K5 -- the arena's criterion (3) has NO computable value for any wing
# ==========================================================================
# `V20_R15_IT1_SATURN.md` argued K5 ("GPU-seconds-to-floor is unmeasurable for
# every wing") from arithmetic with citations and never bound it; the Inspector
# struck it to Open as unbound. This is the node. It is a MEASUREMENT over the
# journal, not a re-citation: seconds-to-floor is defined only for a cell that
# REACHES the floor, so the criterion has a value iff some cell crosses.

RETAKE = ROOT / "results" / "v17k_r4_retake.jsonl"


def cells(path=RETAKE) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if '"t": "cell"' in l]


def seconds_to_floor(rows, floor: float):
    """Total GPU-seconds up to the first cell at or below `floor`, per arm.

    `None` for an arm no cell of which reaches the floor: the quantity is then
    UNDEFINED, not large. Pure function of its rows so it can be run against a
    PLANTED CROSSING, which is the only way to show it can return a number.
    """
    out = {}
    for r in rows:
        kind, secs = r["kind"], r.get("secs", 0.0)
        acc, hit = out.get(kind, (0.0, None))
        acc += secs
        if hit is None and r["eval_nrmse"] <= floor:
            hit = acc
        out[kind] = (acc, hit)
    return {k: hit for k, (acc, hit) in out.items()}


def test_the_seconds_to_floor_reader_returns_a_number_when_a_cell_crosses():
    """Calibration (V-15). A reader that always says 'undefined' proves nothing.

    Planted positive: one synthetic cell BELOW the floor must yield a value.
    """
    planted = [{"kind": "planted", "secs": 2.5, "eval_nrmse": 0.1}]
    got = seconds_to_floor(planted, 0.7071067811865476)
    assert got == {"planted": 2.5}, got


def test_criterion_three_is_computable_for_w1_and_w3_and_absent_for_w2():
    """K5 WITHDRAWN, by the node written to bind it.

    K5 claimed criterion (3) `lowest GPU-seconds-to-floor` is unmeasurable for
    EVERY wing. The measurement refutes it. `results/v17k_r4_retake.jsonl` has
    24 cells; six sit at or below `floor_1 = 0.7071067811865476` -- arm_pl at
    seeds 0/1/4/5/6 and arm_smprime at seed 2 -- and `dist_to_floor` is NEGATIVE
    on each, which is the journal's own sign convention for below-floor. So the
    criterion has a value for W3 (arm_pl, 1.884 GPU-s to first crossing) and for
    W1 (arm_smprime, 47.048 GPU-s), and it ranks them.

    What survives is strictly narrower and is the REROUTE: W2 (`arm_phase`) has
    NO cell in the retake journal at all, so criterion (3) is undefined for W2
    -- for want of a run, not for want of a crossing. That is the claim this
    test now binds.

    The it.1 citations were not wrong about their own subjects: `V15_R1.md:250`
    reads `crosses? NO` for the v15 CPU arm_pl round and `V17_R4_RETAKE.md:194`
    records zero sign-flips in 16. Neither is a statement about seconds-to-floor
    in the retake journal, and generalising them to "every wing" is the defect.
    """
    rows = cells()
    floor = json.loads(RETAKE.read_text(encoding="utf-8").splitlines()[0])["floor_1"]
    got = seconds_to_floor(rows, floor)
    crossers = [(r["kind"], r["seed"], r["eval_nrmse"], r["secs"])
                for r in rows if r["eval_nrmse"] <= floor]
    print("\n  K5 WITHDRAWN. floor_1 = {!r}; {}/{} cells cross".format(
        floor, len(crossers), len(rows)))
    print(f"     GPU-seconds-to-first-crossing = {got}")
    assert got.get("arm_pl") is not None and got.get("arm_smprime") is not None, (
        f"criterion (3) read as undefined for W1/W3, which would revive K5: {got}")
    assert "arm_phase" not in got, (
        "arm_phase now has cells in the retake journal; criterion (3) is no "
        "longer undefined for W2 and this reroute needs re-measuring")
    assert got["arm_pl"] < got["arm_smprime"], (
        f"criterion (3) no longer ranks W3 ahead of W1: {got}")
