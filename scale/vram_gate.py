"""Price a job against what is free NOW, not against the card's total.

WHAT THE C-C GATE MISSES. The corpus admissibility battery's memory clause reads
"peak activation sized in writing against the card", and it exists because the
`impact` task requested 8192 MiB on an 8188 MiB card and died. That is a per-job
over-request and a per-job gate catches it.

Round 10 iteration 3 measured the other failure. NEPTUNE, sizing a large sgate run
at 1701.4 MiB allocated / 1810.0 MiB reserved against an 8188 MiB card, recorded
free memory across the iteration:

    7106 MiB   at probe, 1 process
    2034 MiB   sustained mid-run, 9 processes, 8 samples over 32 s, 100% util
     647 MiB   at peak, 10 processes

His job passes the C-C gate with 4.5x headroom and would not have started at 647
MiB free. The processes were uncoordinated: HOUSE launched a rho sweep while three
agents were dispatched, and MERCURY's replicate timings drifted +269%/+356%/+304%
on identical work as a direct result. Nothing on this machine coordinates the card,
and no per-job gate can see aggregate occupancy.

WHY THIS MATTERS MORE IN PHASE 2 THAN IT DID HERE. Iterations 24-26 are three full
training runs at the 25.7M shape against a 12-hour session cap. An OOM ninety
minutes into a training run, caused by a sibling agent starting a sweep, presents
as a sizing bug in the trainer. It is not one, and the trainer is where everyone
will look. NEPTUNE's words: "the next OOM will look like a sizing bug and will not
be one."

WHAT THIS IS NOT. It is not a scheduler and it does not serialise anything -- there
is no lock, no queue, and no way for one process here to stop another. It answers
one question, in writing, before a job spends: does this fit in what is free RIGHT
NOW, and if not, what is holding the card. That converts a mid-run OOM into a
refusal at second zero with the occupant named.

It degrades to a stated UNKNOWN rather than a false GREEN when `nvidia-smi` is
absent or there is no CUDA device, because a gate that silently passes when it
cannot measure is the vacuous-control shape this campaign catalogues.
"""
from __future__ import annotations

import dataclasses
import shutil
import subprocess
import sys


@dataclasses.dataclass(frozen=True)
class Card:
    total_mib: int
    used_mib: int
    free_mib: int
    processes: int
    utilization_pct: int

    def __str__(self) -> str:
        return (f"{self.free_mib} MiB free of {self.total_mib} "
                f"({self.used_mib} used, {self.processes} processes, "
                f"{self.utilization_pct}% util)")


def read_card() -> Card | None:
    """Current occupancy, or None if it cannot be measured. Never guesses."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip().splitlines()[0]
        total, used, free, util = (int(x.strip()) for x in q.split(","))
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        n = len([p for p in procs.splitlines() if p.strip()])
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None
    return Card(total, used, free, n, util)


def preflight(job_mib: float, *, name: str, margin: float = 1.25) -> tuple[bool, str]:
    """(fits, one-line verdict) for a job needing `job_mib` peak RESERVED memory.

    `margin` defaults to 1.25 because reserved exceeds allocated: NEPTUNE measured
    1701.4 allocated against 1810.0 reserved on the large sgate arm, a 1.064x
    caching-allocator overhead, and fragmentation on a card already holding four
    other jobs is worse than on an empty one. Size against reserved, not allocated.

    A card that cannot be read returns fits=True with an UNKNOWN verdict -- this
    must never block work on a machine without a GPU -- but the string says UNKNOWN
    so a caller printing it cannot mistake it for a measurement.
    """
    card = read_card()
    need = job_mib * margin
    if card is None:
        return True, (f"{name}: VRAM UNKNOWN -- nvidia-smi unavailable or no CUDA device. "
                      f"Needed {need:.0f} MiB ({job_mib:.0f} x {margin}). Not a measurement.")
    if need <= card.free_mib:
        return True, (f"{name}: FITS -- needs {need:.0f} MiB, {card.free_mib} MiB free. {card}")
    return False, (
        f"{name}: DOES NOT FIT -- needs {need:.0f} MiB ({job_mib:.0f} x {margin} reserved "
        f"margin), only {card.free_mib} MiB free. {card}. "
        f"The card is not short; it is occupied. Wait for a sibling job or dispatch fewer "
        f"seats. An OOM from here would present as a trainer sizing bug and would not be one."
    )


def require(job_mib: float, *, name: str) -> None:
    """Refuse at second zero rather than OOM ninety minutes in."""
    fits, verdict = preflight(job_mib, name=name)
    print(verdict, flush=True)
    if not fits:
        raise MemoryError(verdict)


def demo() -> None:
    """Self-check: the gate refuses something that cannot fit and passes something that can."""
    card = read_card()
    print(f"card: {card if card else 'UNREADABLE (no nvidia-smi or no CUDA device)'}")

    # Must-fire: nothing fits a request larger than the whole card.
    huge = (card.total_mib * 10) if card else 1e9
    fits, why = preflight(huge, name="must-fire")
    if card is None:
        assert fits and "UNKNOWN" in why, "an unreadable card must say UNKNOWN, never GREEN"
        print("demo OK (no card): gate reports UNKNOWN and does not block")
        return
    assert not fits, f"the gate passed a request of {huge} MiB on a {card.total_mib} MiB card"
    assert "DOES NOT FIT" in why

    # Must-not-fire: a 1 MiB request on a card with free memory must pass, or the
    # gate blocks everything and is worse than no gate at all.
    fits, why = preflight(1, name="must-not-fire")
    assert fits, f"the gate refused a 1 MiB job with {card.free_mib} MiB free: {why}"

    # The measured shape this module was written for: NEPTUNE's large sgate arm.
    fits, why = preflight(1810.0, name="large-sgate (measured 1701.4 alloc / 1810.0 reserved)")
    print(f"  {why}")
    print("demo OK: refuses the impossible, passes the trivial, prices the real job")


if __name__ == "__main__":
    sys.exit(demo())
