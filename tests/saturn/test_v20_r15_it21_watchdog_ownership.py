"""it.21 SATURN REPAIR 1 -- the watchdog channel signals a pid it never verified.

MARS's it.20 STRIKE 2, taken. The it.19 repair retired `$PIDFILE` because a
channel with no writer launders "unknown" as "confirmed"; it left `$WATCHDOG`
(`.claude/iteration.watchdog`) standing, and that channel is the same defect with
the arms reversed:

  * `$PIDFILE` had NO writer and answered in the PERMISSIVE value -- a wrong number.
  * `$WATCHDOG` HAS a writer (`echo $! > "$WATCHDOG"`, the `start` branch) and
    answers in the DESTRUCTIVE value -- `kill "$wpid"` with no liveness, ownership
    or start-time probe. A wrong ACTION.

And `rm -f "$WATCHDOG"` occurred exactly once in the whole script, inside `disarm`,
so a watchdog that ran to completion left a file naming a dead pid. Dead pid plus
OS pid reuse means `stop` SIGTERMs a process this repo never started.

THE ROUTE SHIPPED (MARS proposed a `trap ... EXIT` plus a start-stamp corroborated
before signalling; this takes the trap, moves the stamp, and adds the probe that
actually establishes ownership):

  1. `trap "rm -f $8" EXIT` in the detached watchdog body -- the registration dies
     with the process that owns it, so the stale-file window closes for every exit
     the shell can see.
  2. `_owns_watchdog` before any signal: the pid's OWN record (`/proc/<pid>/cmdline`)
     must still name THIS iteration's `$STATE` path. A file we wrote cannot survive
     pid reuse as evidence -- the kernel's record of what that pid is running can.
     No probe, no signal: the destructive branch fails CLOSED.
  3. The start-stamp corroboration is moved from the killer to the killed. The
     watchdog fires only if `$STATE`'s `start=` is still the value it was armed
     against, so a watchdog that outlives its disarm is INERT rather than a reason
     to keep killing on weak evidence. Failing closed at (2) therefore costs nothing.

CALIBRATION IS THE POINT OF THIS FILE. "Never kill anything" passes the RED and
breaks the instrument, so `test_disarm_still_kills_the_watchdog_it_actually_armed`
arms a real watchdog through the shipped `start` and demands it be dead after `stop`.

MARS's own caveat is honoured: every process signalled here is spawned THROUGH bash
and reports its own shell pid. A `Popen` pid on Git for Windows is a Win32 pid that
bash's `kill` cannot address, and a test that measures the pid-namespace gap instead
of the defect is not a RED.

Run:  python -m pytest tests/saturn/test_v20_r15_it21_watchdog_ownership.py -q
"""
from __future__ import annotations

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
    """The SHIPPED script, copied unmodified. Nothing here is mutated."""
    (tmp_path / "scripts").mkdir()
    shutil.copy2(SCRIPT, tmp_path / "scripts" / "iteration_timer.sh")
    (tmp_path / ".claude").mkdir()
    return tmp_path


def _run(root: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, str(root / "scripts" / "iteration_timer.sh"), *args],
                          capture_output=True, text=True, timeout=60)


def _bash_spawn_sleep(seconds: int, pidfile: pathlib.Path) -> str:
    """A victim bash can actually signal, and its own shell pid."""
    subprocess.run(
        [BASH, "-c", f'sleep {seconds} </dev/null >/dev/null 2>&1 & '
                     f'echo $! > "{pidfile.as_posix()}"; disown'],
        capture_output=True, text=True, timeout=30)
    pid = pidfile.read_text().strip()
    assert pid.isdigit(), f"could not spawn a victim through bash: {pid!r}"
    return pid


def _alive(pid: str) -> bool:
    return subprocess.run([BASH, "-c", f"kill -0 {pid}"],
                          capture_output=True).returncode == 0


def _kill(pid: str) -> None:
    subprocess.run([BASH, "-c", f"kill -9 {pid} 2>/dev/null"], capture_output=True)


# ======================================================================
# 0. RETIRED at it.24 -- this node was a RACE, not a probe calibration
# ======================================================================
# `test_the_ownership_probe_can_read_a_live_process_and_can_return_false` spawned
# `bash -c "sleep 5" MARKER &` and looked for MARKER in /proc/$!/cmdline. That spawn
# shape is not the one `_owns_watchdog` reads, and on MSYS2 its cmdline returns at
# least four different things depending on where the read lands in the fork/exec
# sequence -- the child's argv, the parent's, '/usr/bin/bash' mid-exec, and 'sleep 5'
# once bash's last-command optimization has replaced the process. Measured over six
# runs of the node alone: 1 failed, 5 passed. it.21's `6 passed` and the it.23
# INSPECTOR's `5 passed, 1 failed` are both true readings of a coin flip.
#
# REPLACEMENT ROUTE, shipped before this kill:
#   tests/saturn/test_v20_r15_it24_probe_calibration.py
# calibrates the SHIPPED construct both ways -- the probe must own the watchdog the
# real `start` armed, and must decline a pid it did not arm. A probe returning false
# for everything fails the first; one returning true for everything fails the second.

# ======================================================================
# 1. THE RED -- a stale registration is not a licence to kill
# ======================================================================

def test_stop_does_not_signal_a_pid_the_watchdog_file_merely_names(tmp_path):
    """MARS it.20 STRIKE 2, reproduced against the shipped script.

    The file names a pid. It does not establish that the pid is still the watchdog
    that was armed, and before this repair the script never asked.
    """
    root = _sandbox(tmp_path)
    victim = _bash_spawn_sleep(25, root / "victim.pid")
    try:
        assert _alive(victim), "calibration: the victim was not running before `stop`"

        (root / ".claude" / "iteration.watchdog").write_text(f"{victim}\n")
        (root / ".claude" / "iteration.start").write_text(
            f"start={int(time.time())}\nbudget_s=1200\niteration=it21-red\n")

        _run(root, "stop")
        time.sleep(1.0)

        assert _alive(victim), (
            f"`stop` sent SIGTERM to pid {victim} on the strength of "
            f".claude/iteration.watchdog alone. The pid was never this iteration's "
            f"watchdog; nothing in the file says it is, and OS pid reuse makes the "
            f"number a stranger. it.19 retired $PIDFILE for reading a process fact "
            f"it could not source -- this channel acts on the same unsourced fact "
            f"with a signal.")
        assert not (root / ".claude" / "iteration.watchdog").is_file(), (
            "the unverified registration was left in place for the next caller "
            "to act on again")
    finally:
        _kill(victim)


# ======================================================================
# 2. THE PLANTED NEGATIVE -- the disarm that still disarms
# ======================================================================

def test_disarm_still_kills_the_watchdog_it_actually_armed(tmp_path):
    """THE CONTROL. `_owns_watchdog` returning false for everything would pass the
    RED above and silently retire the disarm, which is the `and False` repair in
    shell clothing. A real watchdog, armed by the shipped `start`, must be DEAD
    after `stop` -- otherwise a stale watchdog survives into the next iteration and
    raises OVERDUE against a budget it was never armed for.
    """
    root = _sandbox(tmp_path)
    out = _run(root, "start", "1")
    assert out.returncode == 0, f"`start` failed: {out.stdout}{out.stderr}"

    wfile = root / ".claude" / "iteration.watchdog"
    assert wfile.is_file(), "`start` registered no watchdog"
    wpid = wfile.read_text().strip()
    assert wpid.isdigit(), f"watchdog registration is not a pid: {wpid!r}"
    try:
        assert _alive(wpid), "calibration: the armed watchdog was not running"

        _run(root, "stop")
        time.sleep(1.5)

        assert not _alive(wpid), (
            f"`stop` left watchdog pid {wpid} running. The ownership probe declines "
            f"a signal it should have issued: this watchdog IS the one `start` armed, "
            f"and a surviving watchdog raises OVERDUE against the next iteration.")
        assert not wfile.is_file(), "`stop` left the registration behind"
    finally:
        _kill(wpid)


# ======================================================================
# 3. THE REGISTRATION DIES WITH THE PROCESS THAT OWNS IT
# ======================================================================

def _watchdog_body() -> str:
    """The SHIPPED detached body, lifted verbatim from the script text."""
    src = SCRIPT.read_text(encoding="utf-8")
    start = src.index("nohup bash -c '") + len("nohup bash -c '")
    end = src.index("' _ \"$((MINUTES*60))\"", start)
    return src[start:end]


def test_the_watchdog_body_clears_its_own_registration_on_exit(tmp_path):
    """MARS's route, taken. Before the repair `rm -f "$WATCHDOG"` occurred once in
    the whole script, inside `disarm`, so a watchdog that ran to completion left a
    file naming a dead pid -- the stale registration the RED above exploits.

    The shipped body is run directly with a one-second deadline rather than through
    `start 1`, because a sixty-second wait to observe a `trap` is a sixty-second
    wait; the text executed here is the text the script ships.
    """
    root = _sandbox(tmp_path)
    claude = root / ".claude"
    state, overdue = claude / "iteration.start", claude / "ITERATION_OVERDUE"
    wfile, pidf, events = claude / "iteration.watchdog", claude / "iteration.pids", root / "e.jsonl"
    stamp = str(int(time.time()))
    state.write_text(f"start={stamp}\nbudget_s=60\niteration=it21-trap\n")
    wfile.write_text("999999\n")

    r = subprocess.run(
        [BASH, "-c", _watchdog_body(), "_", "1", str(state), str(overdue),
         "it21-trap", "1", str(pidf), str(events), str(wfile), stamp],
        capture_output=True, text=True, timeout=60)

    assert overdue.is_file(), (
        f"the body did not fire on a live STATE with a matching stamp; the trap "
        f"cannot be credited for a body that did nothing: {r.stderr!r}")
    assert not wfile.is_file(), (
        "the watchdog ran to completion and left .claude/iteration.watchdog naming "
        "its own dead pid -- the exact stale registration the next `stop` signals")


def test_a_watchdog_outliving_its_iteration_is_inert(tmp_path):
    """The start-stamp corroboration, moved from the killer to the killed.

    `_owns_watchdog` fails CLOSED, so a box with no readable procfs never signals.
    That is only affordable because a surviving watchdog cannot do damage: it fires
    only against the `start=` stamp it was armed for, and `start` re-stamps STATE.
    Without this, failing closed would trade a wrong kill for a wrong OVERDUE.
    """
    root = _sandbox(tmp_path)
    claude = root / ".claude"
    state, overdue = claude / "iteration.start", claude / "ITERATION_OVERDUE"
    wfile, pidf, events = claude / "iteration.watchdog", claude / "iteration.pids", root / "e.jsonl"
    armed = int(time.time())
    state.write_text(f"start={armed}\nbudget_s=60\niteration=it21-stale\n")

    # The next iteration re-armed the clock under this watchdog while it slept.
    state.write_text(f"start={armed + 999}\nbudget_s=1200\niteration=it21-next\n")

    subprocess.run(
        [BASH, "-c", _watchdog_body(), "_", "1", str(state), str(overdue),
         "it21-stale", "1", str(pidf), str(events), str(wfile), str(armed)],
        capture_output=True, text=True, timeout=60)

    assert not overdue.is_file(), (
        "a watchdog armed for one iteration raised OVERDUE against the next one. "
        "The stamp it was armed with is not the stamp STATE now carries.")


# ======================================================================
# 4. THE SOURCE BINDS -- so the repair cannot be quietly walked back
# ======================================================================

def test_no_signal_is_issued_without_the_ownership_probe():
    """The bind MARS's strike turns on: every `kill` of the watchdog registration
    is guarded. Read as text because the guard is the ONLY thing standing between
    a recycled pid and a SIGTERM."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert "_owns_watchdog" in src, "the ownership probe is gone"
    disarm = src[src.index("disarm() {"):src.index("# Current loop iteration")]
    assert re.search(r'_owns_watchdog "\$wpid"[^\n]*\n\s*then\s*\n?|'
                     r'&&\s*_owns_watchdog "\$wpid"', disarm), (
        f"`disarm` signals without consulting the ownership probe:\n{disarm}")
    for m in re.finditer(r'^\s*kill "\$wpid"', src, re.M):
        seg = src[max(0, m.start() - 400):m.start()]
        assert "_owns_watchdog" in seg, (
            "an unguarded `kill \"$wpid\"` was reintroduced")
