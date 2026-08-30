"""Wait for the box to be idle, in a way that cannot mistake silence for idleness.

WHY THIS IS A FILE AND NOT THREE LINES IN A SCRIPT. Round 10 iteration 11 queued
two jobs behind ad-hoc idle checks and both checks were wrong, in two different
ways, within an hour:

    return int(out) if out.isdigit() else 0

FAIL-OPEN (instance 18, measured). An unreadable process query -- empty output
under memory pressure, a timeout, a transient CIM failure -- was mapped to 0, and
0 means "the box is idle, go ahead". So an absence of evidence was returned as
evidence of absence, in the field a scheduler uses to decide whether to allocate
7.9 GiB. It fired: a second 4.3 GiB seat launched while a 2.2 GiB seat was live.
`scale/vram_gate.py:47-50` already forbids exactly this -- it reports UNKNOWN
rather than GREEN when it cannot measure -- and the rule was applied one layer
too shallow, in the checker that decides whether to call that gate.

EDGE-TRIGGERED (instance 17, measured). A job that runs two subprocesses is
legitimately at zero processes for a second or two BETWEEN them. A wait that
fires on a single zero reading cannot distinguish that gap from the job ending,
and it did not: it woke mid-job and started a sibling.

Adding a retry loop to the fail-open version fixes NEITHER, because three
consecutive unreadable answers are three consecutive zeros. The two repairs are
independent and both are required: unknown must count as BUSY, and idleness must
PERSIST before it is believed.

WHAT THIS IS NOT. Not a scheduler, not a lock, not a queue. It answers one
question -- has the box been demonstrably quiet for long enough to believe it --
and it answers UNKNOWN honestly. Sequencing two jobs that must not overlap is
still done by depending on a durable fact the predecessor writes, never on
occupancy: a sibling that has not allocated yet is invisible to any occupancy
reading, which is the sentence vram_gate.py:19 already contains.
"""
from __future__ import annotations

import subprocess
import time
from typing import Callable

#: Returned by a probe that could not measure. Never 0 -- that is the whole point.
UNKNOWN = -1

#: Consecutive idle readings required, and the gap between them. Three at 20 s
#: spans 40 s of observed quiet, which comfortably exceeds the ~1-2 s gap between
#: two children of one job that defeated the edge-triggered version.
IDLE_CHECKS, CHECK_SECS = 3, 20


def process_probe(needle: str) -> Callable[[], int]:
    """A probe counting live python processes whose command line contains `needle`.

    Returns UNKNOWN, never 0, when the query fails or returns anything that is
    not a count.
    """
    cmd = ["powershell", "-NoProfile", "-Command",
           "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" |"
           f" Where-Object {{ $_.CommandLine -like '*{needle}*' }}).Count"]

    def probe() -> int:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=60).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return UNKNOWN
        return int(out) if out.isdigit() else UNKNOWN

    return probe


def is_idle(count: int) -> bool:
    """Only an actual measured zero is idle. UNKNOWN is busy."""
    return count == 0


def wait_until_idle(probe: Callable[[], int], *, checks: int = IDLE_CHECKS,
                    interval: float = CHECK_SECS, timeout: float | None = None,
                    log: Callable[[str], None] | None = None) -> bool:
    """Block until `checks` CONSECUTIVE readings are a measured zero.

    Returns True on sustained idle, False on timeout. A single non-idle reading
    -- including UNKNOWN -- resets the run to zero, so a transient gap between
    two children of one job cannot be mistaken for that job ending.
    """
    run = waited = 0
    while run < checks:
        count = probe()
        run = run + 1 if is_idle(count) else 0
        if run >= checks:
            break
        time.sleep(interval)
        waited += interval
        if timeout is not None and waited >= timeout:
            if log:
                log(f"timed out after {waited:.0f}s with {run}/{checks} consecutive idle")
            return False
        if log and waited % 600 == 0:
            log(f"waiting: last reading {count}, {run}/{checks} consecutive idle, "
                f"{waited / 60:.0f} min")
    return True


def demo() -> None:
    """Self-check: the two measured failures are each shown to be caught."""
    # MUST-FIRE 1, instance 18: an unreadable probe must never read as idle.
    assert not is_idle(UNKNOWN), "UNKNOWN counted as idle -- the fail-open defect"
    assert is_idle(0) and not is_idle(1)

    # MUST-FIRE 2, instance 17: a transient zero between two children of one job
    # must NOT release the wait. This sequence is the measured shape -- busy,
    # one-reading gap, busy again, then genuinely quiet.
    seq = [1, 1, 0, 1, 1, 0, 0, 0, 0]
    it = iter(seq)
    seen: list[int] = []

    def replay() -> int:
        v = next(it)
        seen.append(v)
        return v

    assert wait_until_idle(replay, checks=3, interval=0), "replay never settled"
    # The three consecutive zeros are at indices 5, 6, 7, so a correct wait
    # releases on the 8th reading and never consumes the 9th. An earlier draft of
    # this check asserted 9 and failed -- the assertion was wrong, not the loop,
    # and its message blamed the isolated zero at index 2, which would have shown
    # up as a release at 3 readings rather than 8. Kept as written because a
    # must-fire whose failure message misdiagnoses the failure is worse than none.
    assert len(seen) == 8, f"released after {len(seen)} readings, expected 8: {seen}"
    assert seen[-3:] == [0, 0, 0], f"released without three real zeros: {seen}"
    assert len(seen) > 3, (
        "released after 3 readings, so the isolated zero at index 2 was treated "
        "as idleness -- the edge-triggered defect")

    # MUST-FIRE 3: three consecutive UNKNOWNs must not release, or the retry loop
    # silently restores the fail-open behaviour it was added to fix.
    #
    # THIS CHECK WAS ITSELF VACUOUS AND THE INSPECTOR BROKE IT. The first version
    # was `assert wait_until_idle(lambda: next(unknowns), checks=3, interval=0)`,
    # which returns True whether or not the unknowns released the wait -- it never
    # looked at how many readings were consumed. Reinstating instance 18 at the
    # LOOP layer (leaving is_idle correct, so MUST-FIRE 1 still passed) gave:
    #     SHIPPED: True, 6 readings [-1, -1, -1, 0, 0, 0]
    #     BROKEN : True, 3 readings [-1, -1, -1]
    # and demo() exited 0 printing "consecutive unknowns do not release". The
    # check is now on the CONSUMED COUNT, the shape MUST-FIRE 2 already used --
    # a boolean return cannot distinguish these two behaviours and never could.
    reads: list[int] = []
    unknowns = iter([UNKNOWN] * 3 + [0, 0, 0])

    def probe_unknowns() -> int:
        v = next(unknowns)
        reads.append(v)
        return v

    assert wait_until_idle(probe_unknowns, checks=3, interval=0)
    assert len(reads) == 6, (
        f"released after {len(reads)} readings {reads}; three UNKNOWNs were "
        "treated as idle, so the retry loop has restored the fail-open defect "
        "this gate exists to prevent")

    # MUST-NOT-FIRE: a genuinely quiet box must be reported quiet, or this gate
    # blocks everything and is worse than no gate.
    assert wait_until_idle(lambda: 0, checks=3, interval=0), "a quiet box was refused"

    # MUST-FIRE 4: a box that never goes quiet must time out, not hang.
    assert not wait_until_idle(lambda: 1, checks=3, interval=0.01, timeout=0.05), \
        "a permanently busy box did not time out"

    print("demo OK: unknown counts as busy, a transient zero does not release, "
          "consecutive unknowns do not release, a quiet box passes, a busy box "
          "times out")


if __name__ == "__main__":
    demo()
