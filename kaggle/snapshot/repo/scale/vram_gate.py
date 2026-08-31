"""Price a job against what is free NOW, not against the machine's total.

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

ONE RESOURCE OVER, round 10 iteration 8. This gate was run against
`scale/r10_capacity_sweep.py`, which spends no VRAM at all: its harness is CPU-only
by construction (`m3_capability.py:261` -- "no .cuda() anywhere in this file").
`require(4275, name='capacity-sweep')` returned FITS citing 7162 MiB free VRAM -- a
true statement about the wrong resource. What binds that job is HOST RSS, measured
901.2 MiB at n=2048, 1567.9 MiB at n=8192 and 4275.1 MiB at n=32768 against 6.2 GiB
free host RAM.

SIZE THIS FROM A MEASUREMENT, NOT A FIT -- the fit is superlinear and a linear one
reads low. Those three points were fitted as `RSS = 665.5 + 0.1102*n`, which
reproduced a held-out n=2048 to 10.1 MiB (1.1%) and was believed on that evidence.
The held-out point lies INSIDE the fitted range, so it tested interpolation while
the model was being used to price n=65536, outside it. Measured later at n=49152:
peak 6584 MiB against a predicted 6082, under by 502 MiB (8.2%). The per-example
slope is not constant -- 0.1102 MiB/example over 8192->32768 and 0.1409 over
32768->49152 -- so a linear extrapolation understates, and understating is the
direction that lets a job through. Callers passing `host_mib=` for a rung larger
than 49152 should treat any linear estimate as a floor, not a price. Three concurrent n=32768 seats need 12.8 GiB of host and the VRAM
clause would have passed all three. That is the same aggregate-occupancy failure
the paragraphs above were written for, one resource over, and it is why `host_mib=`
exists: same 1.25 margin, same UNKNOWN rule, priced separately because a job that
fits the card can still take the box down.

It degrades to a stated UNKNOWN rather than a false GREEN when `nvidia-smi` is
absent, there is no CUDA device, or `psutil` is not installed, because a gate that
silently passes when it cannot measure is the vacuous-control shape this campaign
catalogues.
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


@dataclasses.dataclass(frozen=True)
class Host:
    total_mib: int
    available_mib: int

    def __str__(self) -> str:
        return (f"{self.available_mib} MiB available of {self.total_mib} host RAM "
                f"({self.total_mib - self.available_mib} in use)")


def read_host() -> Host | None:
    """Free host RAM, or None if it cannot be measured. Never guesses.

    `available`, not `free`: reclaimable page cache is spendable and `free` is not
    the number a job is racing its siblings for.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        m = psutil.virtual_memory()
    except Exception:
        return None
    return Host(m.total // 2 ** 20, m.available // 2 ** 20)


def preflight(job_mib: float, *, name: str, margin: float = 1.25,
              host_mib: float | None = None) -> tuple[bool, str]:
    """(fits, one-line verdict) for a job needing `job_mib` peak RESERVED VRAM and,
    if `host_mib` is given, that much peak HOST RSS. Both clauses must pass.

    Pass `job_mib=0` for a job that spends no VRAM -- the CPU-only case that made
    this argument necessary -- and the VRAM clause is skipped rather than answered
    vacuously. Pricing NEITHER resource raises: a gate asked to measure nothing is
    the false GREEN this module exists to refuse.

    `margin` defaults to 1.25 because reserved exceeds allocated: NEPTUNE measured
    1701.4 allocated against 1810.0 reserved on the large sgate arm, a 1.064x
    caching-allocator overhead, and fragmentation on a card already holding four
    other jobs is worse than on an empty one. Size against reserved, not allocated.
    The same 1.25 covers the host clause for a different reason -- a peak RSS
    measured alone is not the peak you pay beside three sibling seats and a foreign
    job, which `r10_it8_pricing.py` measured as a 3.71x hit on step time.

    A resource that cannot be read returns fits=True with an UNKNOWN verdict -- this
    must never block work on a machine without a GPU -- but the string says UNKNOWN
    so a caller printing it cannot mistake it for a measurement.
    """
    if not job_mib and host_mib is None:
        raise ValueError(f"{name}: preflight priced nothing (job_mib=0 and no "
                         f"host_mib). A gate over no resource is a vacuous GREEN.")
    fits, clauses = True, []

    if job_mib:
        card = read_card()
        need = job_mib * margin
        if card is None:
            clauses.append(
                f"{name}: VRAM UNKNOWN -- nvidia-smi unavailable or no CUDA device. "
                f"Needed {need:.0f} MiB ({job_mib:.0f} x {margin}). Not a measurement.")
        elif need <= card.free_mib:
            clauses.append(
                f"{name}: FITS -- needs {need:.0f} MiB, {card.free_mib} MiB free. {card}")
        else:
            fits = False
            clauses.append(
                f"{name}: DOES NOT FIT -- needs {need:.0f} MiB ({job_mib:.0f} x {margin} "
                f"reserved margin), only {card.free_mib} MiB free. {card}. "
                f"The card is not short; it is occupied. Wait for a sibling job or dispatch "
                f"fewer seats. An OOM from here would present as a trainer sizing bug and "
                f"would not be one.")

    if host_mib is not None:
        host = read_host()
        need = host_mib * margin
        if host is None:
            clauses.append(
                f"{name}: HOST UNKNOWN -- psutil unavailable. Needed {need:.0f} MiB "
                f"({host_mib:.0f} x {margin}). Not a measurement.")
        elif need <= host.available_mib:
            clauses.append(
                f"{name}: HOST FITS -- needs {need:.0f} MiB, {host.available_mib} MiB "
                f"available. {host}")
        else:
            fits = False
            clauses.append(
                f"{name}: HOST DOES NOT FIT -- needs {need:.0f} MiB ({host_mib:.0f} x "
                f"{margin} contention margin), only {host.available_mib} MiB available. "
                f"{host}. A CPU-only job cannot be saved by a free card: this is the "
                f"resource it actually spends. Three n=32768 capacity seats want 12.8 GiB "
                f"of host between them and no per-job gate sees the other two.")

    return fits, " | ".join(clauses)


def require(job_mib: float, *, name: str, host_mib: float | None = None) -> None:
    """Refuse at second zero rather than OOM ninety minutes in."""
    fits, verdict = preflight(job_mib, name=name, host_mib=host_mib)
    print(verdict, flush=True)
    if not fits:
        raise MemoryError(verdict)


def demo() -> None:
    """Self-check: the gate refuses the impossible on EITHER resource, passes the trivial."""
    card, host = read_card(), read_host()
    print(f"card: {card if card else 'UNREADABLE (no nvidia-smi or no CUDA device)'}")
    print(f"host: {host if host else 'UNREADABLE (no psutil)'}")

    # Must-fire: a gate asked to price no resource must refuse to answer at all,
    # rather than return the vacuous GREEN this module exists to catalogue.
    try:
        preflight(0, name="prices-nothing")
    except ValueError:
        pass
    else:
        raise AssertionError("preflight returned a verdict while pricing nothing")

    # --- VRAM clause: unchanged for every existing caller ---
    huge = (card.total_mib * 10) if card else 1e9
    fits, why = preflight(huge, name="must-fire")
    if card is None:
        assert fits and "VRAM UNKNOWN" in why, "an unreadable card must say UNKNOWN, never GREEN"
        print("  no card: the VRAM clause reports UNKNOWN and does not block")
    else:
        assert not fits, f"the gate passed a request of {huge} MiB on a {card.total_mib} MiB card"
        assert "DOES NOT FIT" in why

        # Must-not-fire: a 1 MiB request on a card with free memory must pass, or the
        # gate blocks everything and is worse than no gate at all.
        fits, why = preflight(1, name="must-not-fire")
        assert fits, f"the gate refused a 1 MiB job with {card.free_mib} MiB free: {why}"

        # The measured shape this module was written for: NEPTUNE's large sgate arm.
        _, why = preflight(1810.0, name="large-sgate (measured 1701.4 alloc / 1810.0 reserved)")
        print(f"  {why}")

    # --- host clause, round 10 iteration 8 ---
    huge_host = (host.total_mib * 10) if host else 1e9
    fits, why = preflight(0, name="must-fire-host", host_mib=huge_host)
    if host is None:
        assert fits and "HOST UNKNOWN" in why, "an unreadable box must say UNKNOWN, never GREEN"
        print("  no psutil: the host clause reports UNKNOWN and does not block")
    else:
        assert not fits, (f"the gate passed {huge_host:.0f} MiB of host RSS on a "
                          f"{host.total_mib} MiB box")
        assert "HOST DOES NOT FIT" in why

        fits, why = preflight(0, name="must-not-fire-host", host_mib=1)
        assert fits, f"the gate refused a 1 MiB host job with {host.available_mib} MiB free: {why}"

        # THE REGRESSION THIS CLAUSE EXISTS FOR: a job that trivially fits the card
        # and cannot fit the box must be refused. MERCURY's require(4275,
        # name='capacity-sweep') returned FITS on 7162 MiB of free VRAM for a job
        # that spends none of it.
        fits, why = preflight(1, name="fits-the-card-not-the-box", host_mib=huge_host)
        assert not fits, f"a free card waved through an impossible host request: {why}"

        _, why = preflight(0, name="capacity-sweep n=32768 (measured 4275.1 MiB peak RSS)",
                           host_mib=4275.1)
        print(f"  {why}")

    # PLANTED UNREADABLE (MISTAKES.md V-16), added after the INSPECTOR showed the
    # UNKNOWN assertions above are UNREACHABLE on any box that has nvidia-smi and
    # psutil -- which is every box this has ever run on. They sat inside
    # `if card is None:` / `if host is None:`, so replacing both UNKNOWN paths
    # with silent GREENs left demo() exiting 0. A must-fire guarding the
    # cannot-measure case must not itself depend on the tools being absent.
    #
    # Both readers are forced to fail here, so the branch runs on a healthy box.
    _self = sys.modules[__name__]   # works as a script AND as scale.vram_gate
    real_card, real_host = _self.read_card, _self.read_host
    try:
        _self.read_card = lambda: None
        _self.read_host = lambda: None
        fits, why = preflight(1e9, name="planted-unreadable-card")
        assert fits and "VRAM UNKNOWN" in why, (
            f"an unreadable card did not report UNKNOWN: {why!r}. A gate that "
            "silently passes when it cannot measure is the defect this module "
            "exists to refuse.")
        fits, why = preflight(0, name="planted-unreadable-host", host_mib=1e9)
        assert fits and "HOST UNKNOWN" in why, (
            f"an unreadable box did not report UNKNOWN: {why!r}")
        assert "FITS" not in why.replace("HOST UNKNOWN", ""), (
            f"an unreadable reading was dressed as a pass: {why!r}")
    finally:
        _self.read_card, _self.read_host = real_card, real_host

    print("demo OK: refuses the impossible on card and box, passes the trivial, "
          "prices the real jobs, prices nothing only by refusing to, and reports "
          "UNKNOWN on a planted-unreadable reader")


if __name__ == "__main__":
    sys.exit(demo())
