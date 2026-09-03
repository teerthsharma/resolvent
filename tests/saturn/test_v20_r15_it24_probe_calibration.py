"""it.24 SATURN -- the ownership probe, calibrated against the construct it reads.

THE RED, verbatim, at 207e7b9 against unmutated code:

    $ python -m pytest tests/saturn/test_v20_r15_it21_watchdog_ownership.py -q
    F.....
    E   AssertionError: probe blind to a live process: '/usr/bin/bash'
    E   assert 'it21-ownership-marker' in '/usr/bin/bash'
    1 failed, 5 passed in 6.58s

The INSPECTOR read that as the probe being broken and `kill "$wpid"` unreachable on
this box. Measured, it is the CALIBRATION that is broken, and the inference does not
survive: the failing node spawns

    bash -c "sleep 5" MARKER &          (call it SHAPE A)

and looks for MARKER in `/proc/$!/cmdline`. The shipped watchdog is

    nohup bash -c '<multi-line body>' _ 60 "$STATE" ... &    (SHAPE B)

SHAPE A's cmdline is a RACE on MSYS2 -- measured, the same read returns at least
four different things depending on where it lands in the fork/exec sequence:

    'bash -c sleep 5 it24MARK'   the child's true argv -- marker PRESENT
    the PARENT's cmdline         forked, not yet exec'd -- marker present only
                                 because the parent's command STRING embeds it
    '/usr/bin/bash'              mid-exec (this is the reading in the RED)
    'sleep 5'                    post-exec: bash's last-command optimization
                                 REPLACED the process, and MARKER was argv[0]
                                 of a bash that no longer exists

So the it.21 node has no stable verdict. Measured over six runs of that node alone:
1 failed, 5 passed. it.21's `6 passed` and the INSPECTOR's `5 passed, 1 failed` are
BOTH true readings of a node that is a coin flip -- neither is its verdict. No test
is shipped here asserting SHAPE A's outcome, because such a test would be the same
race with the arms reversed.

SHAPE B is never exec-optimized away -- the body is many commands -- so the pid stays
a live `bash -c` whose argv carries "$STATE" for the whole sleep. Measured on the real
`start`: probe TRUE on the watchdog it armed, FALSE on an unrelated bash-spawned pid,
and `stop` killed the watchdog it armed. The probe reads the pid's OWN record and it
works here.

This file replaces the retired it.21 node. Route shipped for the kill it removes:
nodes 2 and 3 below are the two-armed control -- a probe returning false for
everything fails node 2, a probe returning true for everything fails node 3.

Run:  python -m pytest tests/saturn/test_v20_r15_it24_probe_calibration.py -q
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "iteration_timer.sh"
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="no bash on PATH")


def _sh(script: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, "-c", script], capture_output=True, text=True,
                          timeout=timeout)


def _sandbox(tmp_path: pathlib.Path) -> pathlib.Path:
    """The SHIPPED script, copied unmodified. Nothing here is mutated."""
    (tmp_path / "scripts").mkdir()
    shutil.copy2(SCRIPT, tmp_path / "scripts" / "iteration_timer.sh")
    (tmp_path / ".claude").mkdir()
    return tmp_path


def _probe_fn() -> str:
    """`_owns_watchdog`, lifted verbatim from the shipped script text."""
    src = SCRIPT.read_text(encoding="utf-8")
    start = src.index("_owns_watchdog() {")
    return src[start:src.index("\n}\n", start) + 3]


# ======================================================================
# 2. THE CONTROL, POSITIVE ARM -- a probe returning false for everything dies here
# ======================================================================

def test_the_probe_owns_the_watchdog_the_shipped_start_armed(tmp_path):
    """SHAPE B, through the real `start`. The pid `start` registered must be a pid
    the SHIPPED `_owns_watchdog` claims, using the SHIPPED `$STATE` -- derived by
    bash's own `pwd`, because a Windows-form path never matches the POSIX-form path
    the script writes into the watchdog's argv, and comparing the two measures the
    harness rather than the probe.
    """
    root = _sandbox(tmp_path)
    r = _sh(f'set -e; cd "{root.as_posix()}"; ROOT="$(pwd)"; '
            f'STATE="$ROOT/.claude/iteration.start"; '
            f'bash scripts/iteration_timer.sh start 1 >/dev/null; '
            f'W="$(cat .claude/iteration.watchdog)"; '
            f'{_probe_fn()} '
            f'if _owns_watchdog "$W"; then echo PROBE_YES; else echo PROBE_NO; fi; '
            f'bash scripts/iteration_timer.sh stop >/dev/null; kill -9 "$W" 2>/dev/null; true')
    assert "PROBE_YES" in r.stdout, (
        f"the probe disowned the watchdog the shipped `start` had just armed. A probe "
        f"that answers false for everything passes every stale-registration RED and "
        f"silently retires the disarm: stdout={r.stdout!r} stderr={r.stderr[:300]!r}")


# ======================================================================
# 3. THE CONTROL, NEGATIVE ARM -- a probe returning true for everything dies here
# ======================================================================

def test_the_probe_declines_a_pid_that_is_not_this_iterations_watchdog(tmp_path):
    """The same probe, same STATE, against a bash-spawned process this iteration
    never armed. Without this arm the positive arm alone is satisfied by `return 0`.
    """
    root = _sandbox(tmp_path)
    r = _sh(f'set -e; cd "{root.as_posix()}"; ROOT="$(pwd)"; '
            f'STATE="$ROOT/.claude/iteration.start"; '
            f'printf "start=1\nbudget_s=60\niteration=it24\n" > "$STATE"; '
            f'bash -c \'sleep 20 </dev/null >/dev/null 2>&1 & echo $! > other.pid; disown\'; '
            f'sleep 0.5; O="$(cat other.pid)"; '
            f'{_probe_fn()} '
            f'if _owns_watchdog "$O"; then echo PROBE_YES; else echo PROBE_NO; fi; '
            f'kill -9 "$O" 2>/dev/null; true')
    assert "PROBE_NO" in r.stdout, (
        f"the probe claimed ownership of a pid this iteration never armed -- it is "
        f"not reading the process's own record: stdout={r.stdout!r}")
