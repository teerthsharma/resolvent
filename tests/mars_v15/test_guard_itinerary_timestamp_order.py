"""MARS standing attack #4 -- guards chosen after seeing itineraries.

FILED AT it.0, bound by Saturn, against `CEQ_V15_CONTRACT.md` PART III (S-G):
"The itinerary of isocommittor crossings (`q = 1/2`, TST's dividing surface, the
proved committor object) IS the action vocabulary; next best action = next
symbol" -- and PART I's `BED-1`, which "labels by SPLITTING PROBABILITY
(committor)... Guards = `q = 1/2` surfaces; itinerary = actions."

THE MECHANISM. A guard is a surface in state space; an itinerary is the sequence
of guard crossings a trajectory produces. The S-G claim -- that the guards form
a GENERATING PARTITION, measured by the Pesin deficit ("`DEFICIT = lambda-hat -
h_sym`, approximately 0 iff the guards are a generating partition") -- is only a
claim about the WORLD if the guards were fixed before anyone looked at the
itineraries they are then scored against. If a guard surface was tuned, moved,
or picked AFTER an itinerary was already computed, the "generating partition"
result stops being a discovery and becomes a definition chosen to fit its own
answer key: guards can always be gerrymandered to make SOME symbol sequence look
generating, and the Pesin deficit would read near zero by construction rather
than by the guards actually carving the dynamics at its transitions. THE NUMBER
THAT WOULD BE WRONG: the Pesin deficit itself (contract predicts "0.0003 at the
right guard, 0.156 at a wrong one") -- a small deficit would be evidence of
circularity, not of admissibility, if its guard postdates its itinerary.

CLASS. MISTAKES.md V (vacuous control). Closest existing entries are M-7 ("a
pre-registration with a hole" -- M-7's own fix was disclosing the row's
timestamp against the data's) and D-4 ("registration without admission"); this
attack is closer to a NEW MECHANISM than a restatement of either: M-7 is about
missing an outcome branch, D-4 is about silent registry degradation, and neither
checks ORDERING between two dependent artifacts (a guard and the itinerary it is
scored against). The instrument here -- journal timestamp ordering as the
falsifier for hindsight bias in a labelling scheme -- has no existing entry.

STATUS, run at time of filing. `scale/ledger.py` (`house-events.jsonl`, parsed,
never grepped, per the standing rule) carries zero events matching "itinerary"
and zero matching BED-1/isocommittor/`q = 1/2`, measured directly against the
live file: `guard`-word hits 23, `itinerary`-word hits 0, both-in-one-event 0,
BED-1/isocommittor/`q=1/2` hits 0. BED-1 has not been built --
`V15_LEDGER.md` NEXT is `it.3-4`; BED-1 heads land `it.18-21` per PART V. The
real test therefore SKIPS loudly rather than passing on an empty search
(MISTAKES.md V-7). The pairing/violation logic is exercised and must-fired on
synthetic ledger-shaped events below.
"""
from __future__ import annotations

import datetime
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scale import ledger  # noqa: E402

GUARD_RE = re.compile(r"\bguard\b", re.I)
ITINERARY_RE = re.compile(r"\bitinerary\b", re.I)
BED1_RE = re.compile(r"bed-?1\b|isocommittor|q\s*=\s*1\s*/\s*2", re.I)
TEXT_KEYS = ("name", "id", "claim", "finding", "detail", "task", "note",
             "mechanism", "tasks")


def _text(event: dict) -> str:
    parts = []
    for k in TEXT_KEYS:
        v = event.get(k)
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts.extend(str(x) for x in v)
    return " ".join(parts)


def _ts(event: dict) -> datetime.datetime | None:
    """The event's wall-clock timestamp, parsed from `ts` (ISO 8601, e.g.
    `2026-08-25T07:14:59`, the format measured in the live ledger). `None` if
    absent or unparseable -- an event this attack cannot order is excluded
    from the check rather than treated as arbitrarily early or late."""
    raw = event.get("ts")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.datetime.fromisoformat(raw)
    except ValueError:
        return None


# ================================================================ THE CENSUS
def bed1_guard_and_itinerary_events(events) -> tuple[list[dict], list[dict]]:
    """Split a stream of ledger events into (guard-definition events,
    itinerary events) for BED-1 specifically -- scoped with the BED-1 /
    isocommittor / `q=1/2` pattern so an unrelated R10-era `guard` (23 hits in
    the live ledger, none of them BED-1's) does not false-positive this
    census. `itinerary` alone needs no BED-1 scoping: it is 0 hits in the live
    ledger under any scoping, so widening would not change today's verdict,
    but the guard side does need it and both sides use the same rule for
    symmetry.
    """
    guards, itineraries = [], []
    for e in events:
        t = _text(e)
        if not BED1_RE.search(t):
            continue
        if GUARD_RE.search(t):
            guards.append(e)
        if ITINERARY_RE.search(t):
            itineraries.append(e)
    return guards, itineraries


def _pair_key(event: dict) -> str | None:
    """A shared identifier two events are 'about the same thing' by -- the
    text after 'for'/'on'/'at' in the name/id field, or the id/name itself if
    short. Falls back to None (unpaired) rather than guessing."""
    for k in ("id", "name"):
        v = event.get(k)
        if isinstance(v, str) and v:
            return v.lower().strip()
    return None


def paired_violations(guards: list[dict], itineraries: list[dict]) -> list[dict]:
    """Every (guard, itinerary) pair sharing a key where the guard's `ts`
    POSTDATES the itinerary's -- the guard was defined after the itinerary it
    is scored against already existed. Pairs where either side has no
    parseable `ts`, or no shared key, are skipped rather than guessed at.
    """
    by_key: dict[str, list[dict]] = {}
    for it in itineraries:
        k = _pair_key(it)
        if k:
            by_key.setdefault(k, []).append(it)
    violations = []
    for g in guards:
        k = _pair_key(g)
        if not k or k not in by_key:
            continue
        gt = _ts(g)
        if gt is None:
            continue
        for it in by_key[k]:
            it_t = _ts(it)
            if it_t is None:
                continue
            if gt > it_t:
                violations.append({"guard": g, "itinerary": it,
                                    "guard_ts": gt, "itinerary_ts": it_t})
    return violations


def assert_no_hindsight_guards(violations: list[dict]) -> None:
    assert not violations, (
        f"{len(violations)} guard-definition artifact(s) postdate the itinerary "
        f"they are scored against: "
        + "; ".join(
            f"guard {v['guard'].get('id', v['guard'].get('name'))!r} at "
            f"{v['guard_ts']} > itinerary {v['itinerary'].get('id', v['itinerary'].get('name'))!r} "
            f"at {v['itinerary_ts']}" for v in violations[:3])
        + (" ..." if len(violations) > 3 else "")
        + ". The guard was picked after its answer key existed; the S-G "
          "generating-partition claim it supports is circular.")


# =================================================================== TESTS
def test_ledger_carries_zero_itinerary_events_today():
    """Grounds the SKIP below in a measurement rather than an assumption.
    Measured directly against the live `house-events.jsonl` via `scale.ledger`
    (never grepped): 10,810 total events, 23 `guard`-word hits, 0
    `itinerary`-word hits, 0 events matching both, 0 BED-1/isocommittor/
    `q=1/2` hits. BED-1 has not been built."""
    events = list(ledger.read())
    assert events, "the ledger parsed to nothing; scale.ledger itself is broken"
    itinerary_hits = sum(1 for e in events if ITINERARY_RE.search(_text(e)))
    bed1_hits = sum(1 for e in events if BED1_RE.search(_text(e)))
    assert itinerary_hits == 0, (
        f"found {itinerary_hits} itinerary-word event(s); BED-1 may have "
        f"started -- re-run test_bed1_guards_do_not_postdate_their_itinerary, "
        f"it should no longer skip")
    assert bed1_hits == 0, bed1_hits


def test_bed1_guards_do_not_postdate_their_itinerary():
    """THE REAL ATTACK. SKIPS -- confirmed by the census test above, BED-1 has
    no guard-definition or itinerary events in the journal yet
    (`V15_LEDGER.md` NEXT=it.3-4; BED-1 heads land it.18-21 per
    `CEQ_V15_CONTRACT.md` PART V). This is not a promise of a future PASS: the
    must-fire tests below show `paired_violations` catching a real hindsight
    ordering on synthetic ledger-shaped events, so this fires the day BED-1's
    guards and itineraries are journalled, without being rewritten.
    """
    events = list(ledger.read())
    guards, itineraries = bed1_guard_and_itinerary_events(events)
    if not guards or not itineraries:
        pytest.skip(
            f"BED-1 not built: {len(guards)} guard-definition event(s), "
            f"{len(itineraries)} itinerary event(s) in house-events.jsonl. "
            f"V15_LEDGER.md NEXT=it.3-4, BED-1 heads scheduled it.18-21. See "
            f"test_paired_violations_MUST_FIRE_* below for proof the ordering "
            f"check catches a real hindsight-guard case.")
    violations = paired_violations(guards, itineraries)
    assert_no_hindsight_guards(violations)


# ============================================================== MUST-FIRE
def test_paired_violations_MUST_FIRE_on_a_guard_that_postdates_its_itinerary():
    """FIRES. A guard event timestamped AFTER the itinerary it is paired with
    by shared id -- exactly the shape of picking a `q=1/2` surface once the
    symbol sequence it should predict is already sitting in a journal."""
    guard = {"t": "finding", "agent": "Saturn",
              "id": "bed1_guard_channelA", "name": "bed1 guard q=1/2 channelA",
              "ts": "2026-09-05T10:00:00"}
    itinerary = {"t": "finding", "agent": "Saturn",
                 "id": "bed1_guard_channelA",
                 "name": "bed1 itinerary channelA",
                 "ts": "2026-09-05T09:00:00"}   # one hour BEFORE the guard
    violations = paired_violations([guard], [itinerary])
    assert len(violations) == 1, violations
    with pytest.raises(AssertionError, match="postdate the itinerary"):
        assert_no_hindsight_guards(violations)


def test_paired_violations_clean_case_does_not_fire():
    """The clean mirror of the must-fire above: same pair, guard BEFORE
    itinerary. Exists so the must-fire is shown to depend on the ordering and
    not merely on the pair existing."""
    guard = {"id": "bed1_guard_channelB", "name": "bed1 guard q=1/2 channelB",
              "ts": "2026-09-05T08:00:00"}
    itinerary = {"id": "bed1_guard_channelB", "name": "bed1 itinerary channelB",
                 "ts": "2026-09-05T09:00:00"}   # one hour AFTER the guard
    violations = paired_violations([guard], [itinerary])
    assert violations == []
    assert_no_hindsight_guards(violations)   # must not raise


def test_bed1_guard_and_itinerary_events_MUST_FIRE_scoping_finds_a_planted_pair():
    """Demonstrates `bed1_guard_and_itinerary_events` itself -- not just the
    downstream ordering check -- actually finds a planted BED-1 guard/
    itinerary pair inside a synthetic stream, so the SKIP above is shown to be
    about absence in the real ledger, not about the finder being broken."""
    events = [
        {"t": "finding", "id": "x", "name": "unrelated guard rail check",  # 23-style noise
         "ts": "2026-08-25T07:00:00"},
        {"t": "finding", "id": "bed1_guard_channelC",
         "name": "BED-1 guard q=1/2 surface channelC", "ts": "2026-09-06T12:00:00"},
        {"t": "finding", "id": "bed1_guard_channelC",
         "name": "BED-1 itinerary channelC", "ts": "2026-09-06T11:00:00"},
    ]
    guards, itineraries = bed1_guard_and_itinerary_events(events)
    assert len(guards) == 1 and len(itineraries) == 1, (guards, itineraries)
    violations = paired_violations(guards, itineraries)
    assert len(violations) == 1
    with pytest.raises(AssertionError):
        assert_no_hindsight_guards(violations)


def demo() -> None:
    events = list(ledger.read())
    guards, itineraries = bed1_guard_and_itinerary_events(events)
    print(f"live ledger: {len(events)} events, {len(guards)} BED-1 guard "
          f"event(s), {len(itineraries)} BED-1 itinerary event(s) -- "
          f"{'SKIP (BED-1 not built)' if not guards or not itineraries else 'checking'}")
    guard = {"id": "k", "ts": "2026-09-05T10:00:00"}
    itinerary = {"id": "k", "ts": "2026-09-05T09:00:00"}
    v = paired_violations([guard], [itinerary])
    assert len(v) == 1
    try:
        assert_no_hindsight_guards(v)
    except AssertionError:
        pass
    else:
        raise AssertionError("must-fire fixture did not fire")
    print("demo OK: planted guard-postdates-itinerary pair correctly raised")


if __name__ == "__main__":
    demo()
