"""MARS it.18 STRIKE 1 -- the heartbeat repair's corroboration is vacuous.

The it.17 strike: the beat measures `check` CALLS and reports PROCESSES, so every
overrun past cap+5m announced itself as "This MAY be an INTERRUPTION" with the
`--force` command underneath.

The it.18 repair (scripts/iteration_timer.sh) claims liveness now requires a
registered pid to be DEAD, corroborated against .claude/iteration.pids.

These tests hold the repair to that claim. Both are RED against unmutated code.

  RED 1  .claude/iteration.pids does not exist in this repo and nothing in the
         repo writes it, so the corroboration branch is never taken. On the
         missing-pidfile path the repair sets ANY_DEAD=1 -- the SAME permissive
         answer the strike named, reached by a second route -- and prints the
         flat factual claim "had NO live process" about processes it never
         looked at.

  RED 2  absent and EMPTY pidfile are the same evidentiary state (nothing
         registered) and must produce the same verdict. They produce opposite
         verdicts: absent => ANY_DEAD=1 (interruption), empty => ANY_DEAD=0
         (ordinary elapsed), because the `else` arm fires only on a missing
         FILE, not on a missing PID.

  RED 3  the repair removed the `--force` offer from the OVERDUE branch's text,
         but `start --force` still exists AND .claude/ralph-loop.local.md -- the
         standing per-iteration instruction, re-read every turn -- still says:
         "If check reports dead time with no live process, that is an
         INTERRUPTION not an overrun - re-arm with --force and continue."
         The licence the strike named was moved, not removed.

Run:  python -m pytest tests/mars_v20/test_it18_timer_liveness_defaults_permissive.py -x -q
"""
import os
import shutil
import subprocess
import time
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
TIMER = REPO / "scripts" / "iteration_timer.sh"
LOOPFILE = REPO / ".claude" / "ralph-loop.local.md"
BASH = shutil.which("bash") or r"C:\Program Files\Git\bin\bash.exe"

GAP_S = 600          # 10 min since the last `check` -- over the 300s threshold
ELAPSED_S = 600      # 10 min of a 20 min budget -- NOT overdue
BUDGET_S = 1200


def _sandbox(tmp_path, pidfile):
    """Copy the timer into a throwaway ROOT and stage a >300s check gap.

    The script derives ROOT from BASH_SOURCE, so the copy is self-contained and
    the real repo is never touched.
    """
    root = tmp_path / "root"
    (root / "scripts").mkdir(parents=True)
    (root / ".claude").mkdir(parents=True)
    shutil.copy(TIMER, root / "scripts" / "iteration_timer.sh")
    now = int(time.time())
    (root / ".claude" / "iteration.start").write_text(
        f"start={now - ELAPSED_S}\nbudget_s={BUDGET_S}\niteration=18\n"
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


@pytest.mark.skipif(not os.path.exists(BASH), reason="bash unavailable")
def test_missing_pidfile_does_not_assert_a_process_fact(tmp_path):
    """RED 1: with nothing registered, the timer must not state a process fact."""
    assert not (REPO / ".claude" / "iteration.pids").exists(), (
        "premise gone: a pidfile now exists, re-derive the strike"
    )
    rc, out = _check(_sandbox(tmp_path, pidfile=None))
    assert rc == 0, out
    assert "NO live process" not in out, (
        "the repair claims corroboration by a dead registered pid, but with no "
        "pidfile -- the only state this repo is ever in -- it defaults to "
        "ANY_DEAD=1 and asserts a fact about processes it never inspected:\n" + out
    )


@pytest.mark.skipif(not os.path.exists(BASH), reason="bash unavailable")
def test_absent_and_empty_pidfile_agree(tmp_path):
    """RED 2: two spellings of 'nothing registered' must not disagree."""
    _, absent = _check(_sandbox(tmp_path / "a", pidfile=None))
    _, empty = _check(_sandbox(tmp_path / "b", pidfile=""))
    a = "NO live process" in absent
    e = "NO live process" in empty
    assert a == e, (
        "absent and empty pidfile are the same evidence (nothing registered) "
        f"but give opposite verdicts: absent={a!r} empty={e!r}\n"
        f"--- absent ---\n{absent}\n--- empty ---\n{empty}"
    )


def test_force_rearm_licence_is_actually_gone():
    """RED 3: the offer was deleted from one file and left standing in another."""
    script = TIMER.read_text(encoding="utf-8", errors="replace")
    assert "--force" in script, "premise gone: start no longer accepts --force"
    loop = LOOPFILE.read_text(encoding="utf-8", errors="replace")
    hit = "re-arm with --force" in loop
    assert not hit, (
        "scripts/iteration_timer.sh still accepts `start --force`, and "
        ".claude/ralph-loop.local.md -- the standing instruction re-read every "
        "iteration -- still licences it off exactly the message the missing "
        "pidfile emits unconditionally. The cap is defeated by the struck route, "
        "one file over."
    )
