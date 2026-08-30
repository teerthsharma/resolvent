"""Read house-events.jsonl by parsing it. Nobody greps the ledger again.

WHY THIS EXISTS, three costs in two iterations.

MERCURY encoded every red as `{"state": "RED"}`. NEPTUNE encoded them as
`{"status": "RED"}`. The convention is `{"status": "red"}`. The Health Inspector's
binding audit keys on the convention, so it saw NONE of their reds and reported
their findings unbound; he recorded it as a ledger hazard rather than a strike
because the reds existed, just not in a form any reader could find.

HOUSE then read the Inspector's warning, logged 16 reds correctly to the
convention, and verified them with

    grep -c '"agent":"HOUSE","status":"red"' house-events.jsonl   ->  0

`json.dumps` writes `{"t": "test", "agent": "HOUSE", ...}` with a space after each
colon. The events were all there. The grep was a literal-string proxy for an event
and the proxy did not match the serialisation -- the same defect, committed by the
reader of the warning about it, one iteration later.

That is three agents misled by three different hand-rolled parsers of one file.
The file is JSON. Parse it.

TOLERANCE, DELIBERATE. 4 lines of the ledger do not parse -- 1899, 2937, 2938,
5871, all `Invalid \\escape` from LaTeX backslashes in historical `finding` text,
all predating round 10. `house.py:202` already catches `JSONDecodeError` and the
board serves fine; `inspector.py` does not read this file at all. So the bad lines
have no victim today, and rewriting an append-only evidence log to fix them would
cost more than it buys. `read()` skips them and `unparseable()` counts them, so a
caller can see the loss instead of inheriting it silently.

The ledger is append-only evidence. Nothing here writes to it.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Iterator

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / "house-events.jsonl"


def read(path: pathlib.Path | None = None) -> Iterator[dict[str, Any]]:
    """Every parseable event, in file order. Malformed lines are skipped."""
    for line in (path or LEDGER).open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def unparseable(path: pathlib.Path | None = None) -> list[tuple[int, str]]:
    """(line number, first 120 chars) for every line `read()` had to skip.

    An absence a caller has not earned is the defect this whole module exists for,
    so the loss is countable rather than silent -- the rule `scale/journal_scan.py`
    established for the journals, applied to the ledger.
    """
    bad = []
    for i, line in enumerate((path or LEDGER).open(encoding="utf-8", errors="replace"), 1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            bad.append((i, line[:120].rstrip()))
    return bad


def _status(event: dict[str, Any]) -> str | None:
    """The event's status, normalised across every spelling in the file.

    Accepts `status` and `state`, and case-folds the value. This is NOT an
    endorsement of the variants -- new events use `{"status": "red"}` lowercase --
    but a reader that cannot see a historical red is the thing that caused this.
    """
    raw = event.get("status", event.get("state"))
    return raw.lower() if isinstance(raw, str) else None


def _agent(event: dict[str, Any]) -> str | None:
    """The event's agent, case-folded.

    Measured in the shipped ledger, round 10 iteration 3: `cameron` 880 events and
    `Cameron` 10; `Chase` 34 and `chase` 1; `Foreman` 27 and `FOREMAN` 10. An audit
    keyed on the capitalised spelling sees 10 of 890 Cameron events -- an 88x miss,
    in the field the Health Inspector's binding check filters on. That is the same
    defect as the `state`/`status` spelling split he recorded, one field over, and
    larger. 77 further events carry no agent at all and are unattributable to
    anyone; `read()` still yields them, `_agent` returns None, and they match no
    agent filter.
    """
    raw = event.get("agent")
    return raw.lower() if isinstance(raw, str) else None


def _iteration(event: dict[str, Any]) -> int | None:
    """The event's iteration as an INT, across both spellings in the file.

    Measured round 10 iteration 14: 10 events carry `iteration` as an int (1, 2,
    8, 9) and 55 carry it as the string form `"r10.it9"`. A reader filtering
    `e["iteration"] == 9` sees the first ten and none of the fifty-five; a reader
    matching `"r10.it9"` sees the reverse. That is the `state`/`status` split this
    module was written for and the `Cameron`/`cameron` split it measures, in a
    third field -- and this time the string form was introduced by HOUSE, who had
    read both of those findings before writing it.

    Returns None when no iteration is recorded or the string carries no digits,
    so an unparseable tag is distinguishable from an absent one by the caller
    passing an explicit `iteration=`; it never silently matches.
    """
    raw = event.get("iteration")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        # `rpartition` returns the WHOLE string in slot [2] when the separator is
        # absent, so harvesting digits unconditionally read "phase-0" as iteration
        # 0 -- a string carrying no iteration marker silently matching a real
        # iteration. Caught by this module's own must-fire on first run. Require
        # the marker to have been found before reading anything after it.
        head, marker, rest = raw.rpartition("it")
        if not marker:
            return None
        digits = "".join(c for c in rest if c.isdigit())
        return int(digits) if digits else None
    return None


def at_iteration(n: int) -> list[dict[str, Any]]:
    """Every event recorded at iteration `n`, whichever spelling it used."""
    return [e for e in read() if _iteration(e) == n]


def tests(agent: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
    """Test events, optionally filtered by normalised agent and normalised status."""
    want_status = status.lower() if status else None
    want_agent = agent.lower() if agent else None
    return [
        e for e in read()
        if e.get("t") == "test"
        and (want_agent is None or _agent(e) == want_agent)
        and (want_status is None or _status(e) == want_status)
    ]


def findings(agent: str | None = None) -> list[dict[str, Any]]:
    want = agent.lower() if agent else None
    return [e for e in read()
            if e.get("t") == "finding" and (want is None or _agent(e) == want)]


def audits(cites: str | None = None, verdict: str | None = None) -> list[dict[str, Any]]:
    """Adjudications only -- `t:audit` events that actually carry a verdict.

    Measured round 10 iteration 3: 91 `t:audit` events, of which 82 carry a verdict
    and 9 do not (MERCURY 8, SATURN 1). Those nine are seats logging their own
    measurements under the Inspector's event type, which has a defined shape --
    `cites` and `verdict`. Counting raw `t:audit` events as adjudications inflates
    the total by ~10%.

    This is the mild form of the round's recurring defect: an event TYPE standing in
    for an event's CONTENT. It costs nothing today because the only reader that
    matters is the Inspector, who filters. Making the filtered count the default
    accessor keeps it costing nothing.

    Verdicts in use, same measurement: clean 61, struck 16, clean-but-unchecked 5.
    """
    return [
        e for e in read()
        if e.get("t") == "audit" and e.get("verdict")
        and (cites is None or (isinstance(e.get("cites"), str)
                               and e["cites"].lower() == cites.lower()))
        and (verdict is None or e.get("verdict") == verdict)
    ]


def is_bound(node_name: str) -> bool:
    """True if some red event names this node. The binding check, spelled once."""
    return any(e.get("name") == node_name for e in tests(status="red"))


def demo() -> None:
    """Self-check: the reader finds what a grep for the same thing missed."""
    events = list(read())
    assert events, "the ledger parsed to nothing; the reader is broken"

    reds = tests(status="red")
    assert reds, "no red events at all, which the round's history says is impossible"

    # The exact failure this module was written for: HOUSE's 16 reds are present
    # to a parser and absent to the grep that was used to look for them.
    house = tests(agent="HOUSE", status="red")
    assert house, "HOUSE reds unreadable -- the defect this module exists to prevent"

    # Normalisation must reach the historical spellings, or the Inspector's
    # ledger-hazard finding is unfixed.
    spellings = {k for e in read() if e.get("t") == "test"
                 for k in ("status", "state") if k in e}
    assert _status({"state": "RED"}) == "red", "the reader cannot see MERCURY's spelling"
    assert _status({"status": "RED"}) == "red", "the reader cannot see NEPTUNE's spelling"
    assert _status({"status": "red"}) == "red", "the reader cannot see the convention"

    # The agent-field split, measured: an audit keyed on one spelling misses the rest.
    assert _agent({"agent": "Cameron"}) == _agent({"agent": "cameron"}),         "the reader cannot fold the Cameron/cameron split (880 vs 10 events)"
    assert _agent({"agent": "FOREMAN"}) == _agent({"agent": "Foreman"}),         "the reader cannot fold the Foreman/FOREMAN split"
    assert _agent({}) is None, "an agentless event must match no agent filter"
    folded = len(tests(agent="cameron", status="red"))
    assert folded >= 880, f"agent folding lost events: {folded}"

    # The iteration-field type split, measured: an int filter sees ten events and
    # a string filter sees fifty-five, and neither sees the other set.
    assert _iteration({"iteration": 9}) == 9, "the reader cannot fold the int form"
    assert _iteration({"iteration": "r10.it9"}) == 9, "the reader cannot fold the string form"
    assert _iteration({"iteration": "r10.it14"}) == 14, "multi-digit iterations are lost"
    assert _iteration({}) is None and _iteration({"iteration": "phase-0"}) is None, (
        "an absent or digit-free iteration must be None, never a silent match")
    folded = len(at_iteration(9))
    assert folded >= 2, f"iteration folding found only {folded} events at it.9"

    bad = unparseable()
    print(f"demo OK: {len(events):,} events, {len(reds)} red, "
          f"{len(house)} from HOUSE, status keys in use {sorted(spellings)}, "
          f"{len(bad)} unparseable lines skipped and counted")


if __name__ == "__main__":
    demo()
