# V20 R15 it.24 — SATURN (WATSON, instruments)

HEAD `207e7b9`, branch `v17k-gate0`. Opened 11:46:04Z, filed 11:57Z, against a 20 min cap. No git writes,
nothing touched Kaggle. Every number below was produced by this office at this HEAD.

## VERDICT — the probe is REPAIRED, not retired. The calibration was the defect.

The INSPECTOR's reading was correct and his inference was not. `_owns_watchdog`
reads the pid's own record and it works on this box. What is broken is the node that
was supposed to calibrate it: it measures a spawn shape the script never uses.

    SHAPE A   bash -c "sleep 5" MARKER &                     <- the it.21 node
    SHAPE B   nohup bash -c '<multi-line body>' _ 60 "$STATE" ... &   <- what ships

SHAPE B is never exec-optimized away — the body is many commands — so the pid stays a
live `bash -c` carrying `$STATE` in its argv for the whole sleep. Measured through the
real `start`, with `$STATE` derived the way the script derives it: `[RUN]`

    script-derived STATE = /tmp/claude/.../probe2/.claude/iteration.start
    wpid=46540
    PROBE(real watchdog, script-form STATE) = TRUE
    PROBE(unrelated pid 46558) = FALSE  (correct)
    iteration ? closed at 0m2s
    wpid DEAD -- disarm signalled it

**`kill "$wpid"` is reachable on this box and `stop` does signal.** The claim that it
does not came from this office's first probe, which compared a Windows-form path against the
POSIX-form path the script writes into the watchdog's argv — that measures the harness,
not the probe. Recorded because it is the same defect class in this office's own instrument.

## THE PLATFORM FACT — SHAPE A is a race, and the it.21 node has no verdict

`/proc/<pid>/cmdline` for SHAPE A returns at least four different things on MSYS2
depending on where the read lands in the fork/exec sequence: `[RUN]`

    'bash -c sleep 5 it24MARK'   the child's true argv — marker PRESENT
    the PARENT's cmdline         forked, not yet exec'd (3/3 on an immediate read)
    '/usr/bin/bash'              mid-exec — this is the reading in the RED
    'sleep 5'                    post-exec, 3/3 after a 0.4s settle: bash's
                                 last-command optimization replaced the process,
                                 and MARKER was argv[0] of a bash that is gone

Six runs of that node alone: `[RUN]`

    run 1: 1 failed    run 2: 1 passed    run 3: 1 passed
    run 4: 1 passed    run 5: 1 passed    run 6: 1 passed

**it.21's `6 passed` and the INSPECTOR's `5 passed, 1 failed` are both true readings
of a coin flip.** The number in the it.21 report was not fabricated and not stale; it
was a real observation of a node with no stable verdict, which is worse, because a
`[RUN]` marker behind it is honest and still tells the reader nothing.

## WHAT SHIPPED

`tests/saturn/test_v20_r15_it24_probe_calibration.py` — the two-armed control on the
SHIPPED construct. A probe returning false for everything fails arm 1; one returning
true for everything fails arm 2. No `and False`, no shell equivalent.

The it.21 node is **retired in place**, with the replacement route named above it in
`tests/saturn/test_v20_r15_it21_watchdog_ownership.py`. Nothing else in that file was
touched. `[RUN]`

    python -m pytest tests/saturn/test_v20_r15_it24_probe_calibration.py \
                     tests/saturn/test_v20_r15_it21_watchdog_ownership.py -q
    7 passed in 7.60s

### The control is a control — mutation-tested both ways `[RUN]`

    MUTANT A  _owns_watchdog() { return 1; }   false for everything
              FAILED ...::test_the_probe_owns_the_watchdog_the_shipped_start_armed
              1 failed, 1 passed
    MUTANT B  _owns_watchdog() { return 0; }   true for everything
              FAILED ...::test_the_probe_declines_a_pid_that_is_not_this_iterations_watchdog
              1 failed, 1 passed
    UNMUTATED 2 passed
    sha256 scripts/iteration_timer.sh          67b049c6...88a5e931
    sha256 of a copy taken 11:47:10Z, before any mutation:  67b049c6...88a5e931
    diff: IDENTICAL

The restore was first "verified" with `git diff --stat -- scripts/iteration_timer.sh`
returning empty. **That was not evidence.** The script is UNTRACKED
(`git ls-files --error-unmatch` → `did not match any file(s) known to git`), so `git
diff` reports nothing about it no matter what it contains — a check that returns the
clean answer for every input. Replaced with a sha256 against a copy taken at 11:47:10Z,
before any mutation. Second instance this iteration of an instrument of mine reading a
channel that cannot carry the fact it is asked for, and the same class as the defect
under audit. Filed against this office.

Determinism, eight consecutive runs of the file: `2 passed` 8/8. `[RUN]`

### The first draft of this control did NOT catch MUTANT A

It printed `OWNED` / `NOT_OWNED` and asserted `"OWNED" in stdout`. `"OWNED"` is a
substring of `"NOT_OWNED"`, so the positive arm was satisfied by the probe declining
everything — the `and False` repair, in this office's own instrument, in the file whose whole
purpose is to catch it. It was found only by running the mutation, not by reading the
code. The tokens are now disjoint (`PROBE_YES` / `PROBE_NO`) and the mutation is the
evidence, not the reading. Filed against this office.

No test is shipped asserting SHAPE A's outcome. A first draft did, and it failed —
correctly — because the child's argv DOES carry the marker in one of the four windows.
Such a test is the same race with the arms reversed.

## REPAIR 2 — the `EXIT` trap, exercised through a real `start 1`

Scratch ROOT, the shipped script copied unmodified, a real 60-second wait. `[RUN]`

    === T0 11:47:10Z
    iteration ? armed: 1 min cap, deadline 11:48:10Z
    wpid=46097

    === NOW 11:47:46Z
    --- watchdog pid 46097: ALIVE
    --- ITERATION_OVERDUE: No such file or directory
    --- check: iteration ?: 0m37s elapsed, 0m23s left of 1m, armed 11:47:10Z   rc=0

    === NOW 11:48:25Z
    --- watchdog pid 46097: DEAD
    --- ITERATION_OVERDUE:
    2026-09-02T11:48:10Z
    iteration ? exceeded 1 min
    --- registration file: ls: cannot access '.claude/iteration.watchdog': No such file
    --- events:
    {"t":"dispatch","agent":"timer","ts":"2026-09-02T11:47:10Z","text":"iteration ? armed with a 1 min cap"}
    {"t":"finding","agent":"timer","ts":"2026-09-02T11:48:10Z","text":"ITERATION ? OVERDUE: exceeded 1 min wall clock"}
    --- check: OVERDUE - iteration ? ran 1m15s against a 1m cap, armed 11:47:10Z   rc=1

**The trap fires.** OVERDUE raised at the deadline to the second, the registration
removed by `trap ... EXIT` on a watchdog that ran to completion, the event logged, and
`check` exits 1. The MARS it.20 stale-`$WATCHDOG` condition does not reproduce. Owed
since it.21, declined twice, now settled on real evidence rather than body text.

## COUNTS RE-CHECKED — every suite number published in it.21 and it.22

All re-run at `207e7b9` by this office. `[RUN]`

| Published | Where | At HEAD now | Holds |
|---|---|---|---|
| `it22_timer_rearm.py` 5 passed | it.22 | 5 passed | YES |
| `it21_watchdog_ownership.py` 6 passed | it.21, it.22 | 6 passed alone; 1 failed 5 passed on another run | **NO — the node is a race, see above** |
| `it19_pid_channel.py` 4 passed | it.22 | 4 passed | YES |
| `freeze_manifest.py` 22 passed | it.21 | 24 passed | NO — superseded by it.22's `22 -> 24` |
| `freeze_manifest.py` 24 passed | it.22 | 24 passed | YES |
| mars `it18_timer_liveness_defaults_permissive.py` 3 passed | it.22 | 3 passed | YES |
| mars `it20_the_retired_channel_left_a_live_one.py` 2 passed, 1 failed | it.22 | 1 failed, 2 passed | YES |
| mars `it20_two_clauses_is_not_two_witnesses.py` 5 passed | it.21 | 5 passed | YES |
| `tests/saturn` 9 failed, 169 passed | it.21 | **8 failed, 177 passed** | NO |
| `tests/mars_v20` 39 failed | it.21 | **31 failed, 59 passed** | NO |

The two standing whole-suite counts the INSPECTOR flagged as neither confirmed nor
cleared are now **cleared, and both were wrong**. `tests/saturn` is +8 passed / −1
failed, of which it.22 added 5 (`it22_timer_rearm`) and 2 (freeze manifest); the
remaining +1/−1 is unaccounted for by this office and is a loose end, not a claim.
`tests/mars_v20` at 31 failed against a published 39 is an 8-failure gap this office
did not open and cannot source.

## Limits

The `9 failed`/`39 failed` gaps above are measured but not explained; nothing here
identifies which nodes moved or why, and the +1/−1 residue in `tests/saturn` is
unexplained. The probe's soundness rests on `/proc/<pid>/cmdline`
being the named pid's own record for SHAPE B; the pre-exec window measured above shows
that a read landing early can return the PARENT's line, so a `disarm` racing a fork
could in principle read a stranger — never observed, not tested, and the destructive
branch fails closed either way. Not reached: whether any it.21/it.22 count outside the
suite totals (hashes, manifest entries) has the same shape.
