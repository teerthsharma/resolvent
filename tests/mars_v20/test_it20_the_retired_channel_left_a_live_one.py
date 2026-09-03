"""MARS it.20 STRIKE 2 -- the pid retirement covered the channel with no writer
and left the channel that has one.

SATURN's it.19 REPAIR 1 retired `$PIDFILE` because "a check nobody feeds launders
unknown as confirmed", and the stated cost is that the timer now carries no liveness
verdict. Correct, and incomplete. `scripts/iteration_timer.sh` holds a SECOND pid
channel, `$WATCHDOG` = `.claude/iteration.watchdog`, and this one is different in
exactly the way that matters: it HAS a writer (`echo $! > "$WATCHDOG"`, the `start`
branch), so it is live in this repo today.

`disarm()` reads it and issues `kill "$wpid"` with no check that the pid is still
the watchdog it armed -- no liveness probe, no ownership probe, no start-time probe.
And the detached watchdog body NEVER removes its own file: `rm -f "$WATCHDOG"` occurs
once in the whole script, inside `disarm` itself. So a watchdog that runs to
completion leaves a file naming a dead pid, and the next `start` or `stop` kills
whatever the OS has since given that number to.

This is the it.18 defect with the arms reversed. There, a channel with no writer
answered in the permissive value. Here, a channel with a writer answers in the
DESTRUCTIVE value: an unsourceable process fact is acted on with a signal.

RED against unmutated code: the shipped script is copied unmodified into a scratch
tree, an innocent process is registered in place of a stale watchdog pid, and `stop`
is invoked. Nothing in this test is mutated.

Run:  python -m pytest tests/mars_v20/test_it20_the_retired_channel_left_a_live_one.py -q
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


def _sandbox(tmp_path):
    (tmp_path / "scripts").mkdir()
    shutil.copy2(SCRIPT, tmp_path / "scripts" / "iteration_timer.sh")
    (tmp_path / ".claude").mkdir()
    return tmp_path


def test_the_watchdog_channel_has_a_writer_and_the_retired_one_does_not():
    """The premise, asserted before the falsification is claimed."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert re.search(r'echo \$! > "\$WATCHDOG"', src), (
        "premise gone: nothing writes $WATCHDOG any more; re-derive the strike")
    assert not re.search(r'>>? *"?\$PIDFILE', src), (
        "premise gone: $PIDFILE now has a writer; REPAIR 1's route reopens")


def test_the_watchdog_file_outlives_the_process_it_names():
    """`rm -f "$WATCHDOG"` occurs exactly once, inside `disarm`. The detached
    watchdog body does not clear its own registration when it exits."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert src.count('rm -f "$WATCHDOG"') == 1
    body = src[src.index("nohup bash -c"):src.index('echo $! > "$WATCHDOG"')]
    assert "WATCHDOG" not in body and "$3" in body, (
        "the watchdog body now touches its own registration; re-derive the strike")


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_disarm_kills_a_pid_it_never_verified(tmp_path):
    """THE RED. A stale registration is a licence to kill an innocent process."""
    root = _sandbox(tmp_path)

    # The victim is spawned THROUGH bash and reports its own shell pid: on Git
    # for Windows a Popen pid is a Win32 pid, which bash's `kill` cannot address,
    # and a test that measures the pid-namespace gap instead of the defect is not
    # a RED. The pid written here is the one `disarm` will actually signal.
    pidf = root / "victim.pid"
    subprocess.run(
        [BASH, "-c", f'sleep 20 </dev/null >/dev/null 2>&1 & '
                     f'echo $! > "{pidf.as_posix()}"; disown'],
        capture_output=True, text=True, timeout=30)
    victim = pidf.read_text().strip()
    assert victim.isdigit(), f"could not spawn a victim: {victim!r}"

    def alive() -> bool:
        return subprocess.run([BASH, "-c", f"kill -0 {victim}"],
                              capture_output=True).returncode == 0

    try:
        assert alive(), "calibration: the victim was not running before `stop`"

        # Stand the victim's pid in for a watchdog that already exited. Nothing
        # about this file says the process it names is still that watchdog --
        # and the script never asks.
        (root / ".claude" / "iteration.watchdog").write_text(f"{victim}\n")
        (root / ".claude" / "iteration.start").write_text(
            f"start={int(time.time())}\nbudget_s=1200\niteration=mars-it20\n")

        subprocess.run([BASH, str(root / "scripts" / "iteration_timer.sh"), "stop"],
                       capture_output=True, text=True, timeout=60)
        time.sleep(1.0)

        assert alive(), (
            f"`stop` sent SIGTERM to pid {victim} on the strength of "
            f".claude/iteration.watchdog alone. The file names a pid; it does not "
            f"establish that the pid is still the watchdog that was armed, and "
            f"nothing clears it when the watchdog exits on its own. REPAIR 1 "
            f"retired $PIDFILE for reading a process fact it could not source. "
            f"$WATCHDOG reads the same unsourced fact and acts on it with a "
            f"signal, and unlike $PIDFILE it has a writer, so it is live today.")
    finally:
        subprocess.run([BASH, "-c", f"kill {victim} 2>/dev/null"], capture_output=True)
