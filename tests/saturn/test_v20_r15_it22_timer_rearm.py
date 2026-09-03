"""it.22 SATURN REPAIR 1 -- the cap re-armed under a running auditor, silently.

THE DEFECT AS THE INSPECTOR FILED IT (it.21, V20_R15_JOURNAL.md:4370):

    `check` reported `1m18s` at his open and `5m6s` at 16:38 for the same
    iteration -- the timer re-armed underneath a running agent. He therefore
    reports his wall-clock statements as WORK PERFORMED, not as instrument
    readings.

Two independent holes produce that symptom, and this file separates them.

  1. THE GUARD STOPS GUARDING AT THE MOMENT THE CAP FIRES. The it.6 `start`
     guard refuses a re-arm only while `_e -lt _b` -- while the iteration is
     INSIDE its budget. An OVERDUE iteration fails that test, so `start`
     falls straight through and re-arms it with no refusal and no distinct
     output. The one state in which a fresh clock does the most damage -- the
     room past cap, an auditor mid-audit reading OVERDUE -- is the one state
     the guard was written not to cover.

  2. A READING CANNOT BE DATED BY THE INSTRUMENT THAT PRODUCED IT. `check`
     prints elapsed and remaining and NOTHING that identifies which arm it is
     reading. Two readings minutes apart are indistinguishable from one
     monotonic clock, so a re-arm is invisible unless the reader happens to
     have written down the wall time of every call and done the subtraction
     by hand. That is what the INSPECTOR did, and it is why he alone saw it.
     `house-events.jsonl` cannot settle it either: the timer's own event lines
     carry no time, so the log records THAT an arm happened and never WHEN.

Route 2 is the load-bearing half, because it closes every re-arm route rather
than one: `--force`, a `stop`+`start` pair (legitimate at an iteration boundary,
and it erases `$CLOSED` -- the only breadcrumb -- on its way through), and
whatever the next one turns out to be. Route 1 without route 2 shuts one door
in a corridor; route 2 puts a light in the corridor.

CALIBRATION IS THE POINT OF THE LAST TWO NODES. "Refuse every re-arm" passes
node 1 and stops the loop dead, and the round forbids a kill without a
replacement route -- so `test_the_ordinary_iteration_boundary_still_arms` and
`test_the_it6_guard_still_refuses_a_live_iteration` demand both the new refusal
AND the two routes that must survive it.

The shipped script is copied unmodified into a throwaway ROOT (it derives ROOT
from BASH_SOURCE, so the copy is self-contained). The repo's own `.claude/` is
never touched -- the room is inside a live iteration while this runs.

Run:  python -m pytest tests/saturn/test_v20_r15_it22_timer_rearm.py -q
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "iteration_timer.sh"
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="no bash on PATH")


def _sandbox(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "root"
    (root / "scripts").mkdir(parents=True)
    (root / ".claude").mkdir()
    shutil.copy2(SCRIPT, root / "scripts" / "iteration_timer.sh")
    (root / ".claude" / "ralph-loop.local.md").write_text(
        "iteration: 27\n", encoding="utf-8")
    return root


def _run(root: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [BASH, str(root / "scripts" / "iteration_timer.sh"), *args],
        capture_output=True, text=True, timeout=60)


def _state(root: pathlib.Path) -> dict:
    txt = (root / ".claude" / "iteration.start").read_text(encoding="utf-8")
    return dict(l.split("=", 1) for l in txt.splitlines() if "=" in l)


def _age_the_clock(root: pathlib.Path, seconds: int) -> None:
    """Push `start=` back so the live iteration is `seconds` old. Nothing else moves."""
    p = root / ".claude" / "iteration.start"
    st = _state(root)
    st["start"] = str(int(st["start"]) - seconds)
    p.write_text("".join("%s=%s\n" % kv for kv in st.items()), encoding="utf-8")


# ==========================================================================
# 1.  THE HOLE -- the guard quits at exactly the moment the cap fires
# ==========================================================================

def test_start_refuses_to_rearm_an_iteration_that_is_already_overdue(tmp_path):
    root = _sandbox(tmp_path)
    assert _run(root, "start", "20").returncode == 0
    _age_the_clock(root, 1300)          # 21m40s against a 20m cap: OVERDUE
    armed_at = _state(root)["start"]    # read AFTER the ageing, so the node cannot
                                        # pass by comparing against a stale value
    over = _run(root, "check")
    assert over.returncode == 1 and "OVERDUE" in over.stdout, over.stdout

    r = _run(root, "start", "20")
    assert r.returncode == 2 and "REFUSED" in r.stderr, (
        "`start` re-armed an OVERDUE iteration and said nothing. The it.6 guard "
        "refuses only while `_e -lt _b`, so it stops guarding at the exact moment "
        "the cap fires -- the room past its budget, an auditor mid-audit holding an "
        "OVERDUE reading, and a fresh 20 minutes handed to everyone still inside "
        "the old one. A cap that resets is worse than no cap, because the room "
        "believes it.\nrc=%r\nstdout=%r\nstderr=%r" % (r.returncode, r.stdout, r.stderr))
    assert _state(root)["start"] == armed_at, (
        "`start` was refused and moved the clock anyway.")


# ==========================================================================
# 2.  THE CORRIDOR LIGHT -- a reading that names the arm it came from
# ==========================================================================

def test_two_check_readings_across_a_rearm_cannot_be_read_as_one_clock(tmp_path):
    """The INSPECTOR's exact position, reproduced: two `check` calls, a re-arm
    between them, and nothing in either reading that says so."""
    root = _sandbox(tmp_path)
    _run(root, "start", "20")
    first = _run(root, "check").stdout
    time.sleep(1.1)                      # so two arms cannot share a second
    _run(root, "stop")                   # the legitimate iteration boundary
    _run(root, "start", "20")            # ... under an agent still running
    second = _run(root, "check").stdout

    stamps = [re.search(r"armed\s+(\S+)", o) for o in (first, second)]
    assert all(stamps), (
        "neither reading names the arm it came from:\n"
        "  %r\n  %r\n"
        "The clock was re-armed between these two calls and both readings are "
        "consistent with one monotonic clock. The INSPECTOR could only catch it "
        "by writing down the wall time of every call and subtracting by hand, and "
        "an auditor who cannot date a reading cannot report it as an instrument "
        "reading at all." % (first.strip(), second.strip()))
    assert stamps[0].group(1) != stamps[1].group(1), (
        "both readings claim the same arm %r across a stop/start pair, so the "
        "stamp is not the arm's identity." % stamps[0].group(1))


def test_the_event_log_can_say_when_an_arm_happened(tmp_path):
    """`house-events.jsonl` is the only persistent record of arms, and it is the
    corroboration an auditor reaches for second. Its timer lines carry no time,
    so it records THAT the clock was armed and never WHEN -- it can neither
    confirm nor refute a re-arm, and the it.21 audit had nothing to appeal to."""
    root = _sandbox(tmp_path)
    _run(root, "start", "20")
    _run(root, "stop")
    lines = [json.loads(l) for l in
             (root / "house-events.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    timer = [r for r in lines if r.get("agent") == "timer"]
    assert timer, "the timer logged nothing"
    undated = [r for r in timer if not r.get("ts")]
    assert not undated, (
        "%d of %d timer events carry no timestamp, e.g. %r. Two arms of the same "
        "iteration are indistinguishable in the log from one arm, and their order "
        "says nothing about their spacing." % (len(undated), len(timer), undated[0]))


# ==========================================================================
# 3.  CALIBRATION -- what must NOT change
# ==========================================================================

def test_the_ordinary_iteration_boundary_still_arms(tmp_path):
    """The replacement route. `stop` then `start` is how every iteration begins,
    and `--force` is the deliberate override; a repair that refuses either has
    retired the cap rather than repaired it."""
    root = _sandbox(tmp_path)
    assert _run(root, "start", "20").returncode == 0
    assert _run(root, "stop").returncode == 0
    r = _run(root, "start", "20")
    assert r.returncode == 0, "a fresh iteration was refused after a clean stop: %r" % r.stderr

    _age_the_clock(root, 1300)
    f = _run(root, "start", "20", "--force")
    assert f.returncode == 0, "`--force` no longer overrides an overdue clock: %r" % f.stderr

    c = _run(root, "check")
    assert c.returncode == 0 and re.search(r"\d+m\d+s elapsed", c.stdout), (
        "`check` no longer reports elapsed/remaining on a live clock: %r" % c.stdout)


def test_the_it6_guard_still_refuses_a_live_iteration(tmp_path):
    """it.6's own case, unchanged: a re-arm inside the budget is still refused,
    with a message that names the iteration it is protecting."""
    root = _sandbox(tmp_path)
    _run(root, "start", "20")
    r = _run(root, "start", "20")
    assert r.returncode == 2 and "REFUSED" in r.stderr, (
        "the it.6 guard no longer holds: rc=%r %r" % (r.returncode, r.stderr))
    assert "still live" in r.stderr or "OVERDUE" in r.stderr, r.stderr
