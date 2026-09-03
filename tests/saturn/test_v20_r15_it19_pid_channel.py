"""SATURN it.19 REPAIR 1 -- the pid channel launders "unknown" as "confirmed".

MARS's it.18 STRIKE 1, premise verified by the coordinator: `.claude/iteration.pids`
does not exist and NOTHING in this repo writes it. `scripts/iteration_timer.sh check`
nevertheless branches on it, and the missing-file arm sets ANY_DEAD=1 -- the
PERMISSIVE value -- so the instrument answers "no evidence" and "evidence of death"
with the same sentence.

These four are RED against the unrepaired script and GREEN after.

  RED 1  absent pidfile and EMPTY pidfile are one evidentiary state (nothing
         registered). They must not produce opposite verdicts.

  RED 2  with nothing registered, `check` must not assert the process fact
         "NO live process" about processes it never inspected.

  RED 3  one LIVE pid and one dead pid must not read as death. ANY_DEAD is
         `exists dead`, never cleared, so one finished nurse of four licences the
         interruption verdict while three are still running.

  RED 4  a pid written without a trailing newline is dropped by `while read -r p`,
         so the watchdog's kill registry silently ignores its last entry. This is
         the SAME loop form and the watchdog is now its only reader.

Run:  python -m pytest tests/saturn/test_v20_r15_it19_pid_channel.py -x -q
"""
import os
import re
import shutil
import subprocess
import time
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
TIMER = REPO / "scripts" / "iteration_timer.sh"
BASH = shutil.which("bash") or r"C:\Program Files\Git\bin\bash.exe"

GAP_S = 600        # 10 min since the last `check` -- over the 300s threshold
ELAPSED_S = 600    # 10 min of a 20 min budget -- NOT overdue
BUDGET_S = 1200

pytestmark = pytest.mark.skipif(not os.path.exists(BASH), reason="bash unavailable")


def _sandbox(tmp_path, pidfile):
    """Stage a >300s check gap in a throwaway ROOT. The real repo is never touched."""
    root = tmp_path / "root"
    (root / "scripts").mkdir(parents=True)
    (root / ".claude").mkdir(parents=True)
    shutil.copy(TIMER, root / "scripts" / "iteration_timer.sh")
    now = int(time.time())
    (root / ".claude" / "iteration.start").write_text(
        f"start={now - ELAPSED_S}\nbudget_s={BUDGET_S}\niteration=19\n"
    )
    (root / ".claude" / "iteration.beat").write_text(str(now - GAP_S))
    if pidfile is not None:
        (root / ".claude" / "iteration.pids").write_text(pidfile)
    return root


def _check(root):
    p = subprocess.run(
        [BASH, str(root / "scripts" / "iteration_timer.sh"), "check"],
        capture_output=True, text=True,
    )
    return p.returncode, p.stdout + p.stderr


def _shape(out):
    """The verdict with the volatile fields normalised away.

    it.22 added `armed HH:MM:SSZ` to the `check` line -- the arm's own identity, so
    that two readings taken across a re-arm stop being indistinguishable. It is
    provenance, not verdict, and it is volatile by construction: these two sandboxes
    are built a fraction of a second apart and therefore carry different arms. It
    joins the elapsed/remaining seconds under the same filter. The CLAIM of both
    nodes below is untouched -- they still compare the whole verdict text.
    """
    out = re.sub(r"armed \d\d:\d\d:\d\dZ", "armed <ARM>", out)
    return re.sub(r"\d+m\d+s", "<T>", out).strip()


def test_absent_and_empty_pidfile_give_the_same_verdict(tmp_path):
    """RED 1: two spellings of 'nothing registered' must agree."""
    _, absent = _check(_sandbox(tmp_path / "a", pidfile=None))
    _, empty = _check(_sandbox(tmp_path / "b", pidfile=""))
    assert _shape(absent) == _shape(empty), (
        "absent and empty pidfile are the same evidentiary state and disagree:\n"
        f"--- absent ---\n{absent}\n--- empty ---\n{empty}"
    )


def test_check_asserts_no_process_fact_it_cannot_source(tmp_path):
    """RED 2: nothing registers pids, so no process claim is sourceable."""
    rc, out = _check(_sandbox(tmp_path, pidfile=None))
    assert rc == 0, out
    assert "NO live process" not in out, (
        "the instrument states a process fact with no writer feeding it:\n" + out
    )


def test_one_live_pid_does_not_read_as_death(tmp_path):
    """RED 3: ANY_DEAD is `exists dead`, so one finished nurse licences the verdict."""
    live = subprocess.Popen([BASH, "-c", "sleep 30"])
    try:
        dead = subprocess.run([BASH, "-c", "exit 0"])  # noqa: F841
        # A pid that is certainly not running: this shell's own child, reaped.
        gone = 999999
        _, mixed = _check(_sandbox(tmp_path / "m", pidfile=f"{live.pid}\n{gone}\n"))
        _, absent = _check(_sandbox(tmp_path / "n", pidfile=None))
        assert "NO live process" not in mixed, (
            f"pid {live.pid} was alive and the verdict still says nothing was:\n{mixed}"
        )
        assert _shape(mixed) == _shape(absent), (
            "pidfile contents still steer the verdict:\n"
            f"--- mixed ---\n{mixed}\n--- absent ---\n{absent}"
        )
    finally:
        live.kill()
        live.wait()


def test_last_pid_without_a_trailing_newline_is_registered(tmp_path):
    """RED 4: `while read -r p` drops a final line with no newline."""
    src = TIMER.read_text()
    loops = re.findall(r"while read -r p.*", src)
    assert loops, "the read loop moved; re-derive the strike"
    for loop in loops:
        assert "|| [" in loop, (
            "this loop drops a pid written without a trailing newline "
            f"(printf '1234' registers nothing): {loop!r}"
        )
