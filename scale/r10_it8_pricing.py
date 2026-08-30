"""The R10 P1a capacity sweep, priced BEFORE it ran. Producer for the DAG table.

METHOD, the one binding since Phase 0 (`PHASE2_CONTRACT_V_MAIN_4.md:192`): a count
of 600-step run-equivalents read off the source that will execute them, times a
seconds-per-RE rate re-measured on the machine at the moment of spend. The flat
per-point model this replaced was 15.95x out on the rho axis.

c_step IS MEASURED, NOT MODELLED. Eight steps of `scale/r10_capacity_sweep.py`'s
own loop on the softmax arm at s=64 d=24, after two warmup steps, this machine,
2026-08-30, one process holding the box. Both thread pins recorded because the
sweep runs at 8 and every published e3 number in `results/m3_capability.txt` was
taken at 2:

    n        2 threads     8 threads    16 threads
    2048     0.15403       0.11137       0.10103
    8192     0.87456       0.55395       0.53407
    32768    3.94342       2.54988       2.61226

Per-example cost RISES with n (75.2 -> 106.8 -> 120.3 us at 2 threads) rather than
staying flat: the arm is full-batch, the operator is [n,s,s], and the box leaves
cache. A per-example unit fitted at n=2048 and multiplied out to n=32768 would be
low by 1.60x. That is the same shape of error the rho-axis estimate made.

PEAK HOST RSS IS ALSO MEASURED, and it is the binding constraint, not VRAM. The
harness is CPU-only by construction (`m3_capability.py:261` -- "no .cuda()
anywhere in this file"), so `scale/vram_gate.require` prices a resource this job
does not spend. It was run anyway, as instructed, and it refused a 6000 MiB guess
and passed the 4704 MiB measurement -- but the number that decides this sweep is
6.2 GiB of FREE HOST RAM against a measured 4275 MiB peak at n=32768, which is
why the n=32768 column cannot run beside the other two.

    n        peak RSS MiB
    2048       901.2
    8192      1567.9
    32768     4275.1
"""
from __future__ import annotations

T_STARS = (2, 8, 32)
N_TRAIN = (2048, 8192, 32768)
STEP_RUNGS = (150, 600, 2400, 9600)

#: seconds per training step, MEASURED (see module docstring), 8 threads.
C_STEP = {2048: 0.11137, 8192: 0.55395, 32768: 2.54988}
C_STEP_2T = {2048: 0.15403, 8192: 0.87456, 32768: 3.94342}
PEAK_RSS_MIB = {2048: 901.2, 8192: 1567.9, 32768: 4275.1}

#: Measured this round on this machine. Quote both, per the contract.
S_PER_RE_UNCONTENDED = 18.01
S_PER_RE_FOUR_SEATS = 48.89

#: RE-MEASURED AT SPEND, which is the half of the costing law that is easy to
#: skip and was the expensive one here. The estimate above was taken with one
#: process holding the box. The sweep actually ran as THREE concurrent seats
#: beside a foreign 2-hour job already on the machine (PID 33660, 20,247 CPU-s
#: at launch), i.e. 32 OpenMP threads on 28 logical cores. Re-probed 36 minutes
#: into the spend, same shape, same 8-thread pin, n=2048:
#:
#:     uncontended  0.11137 s/step
#:     contended    0.41327 s/step      ->  3.71x
#:
#: That is the same magnitude as the +269%/+356%/+304% replicate drift MERCURY
#: recorded in iteration 3 on identical work, and it has the same cause: nothing
#: on this machine coordinates the box, so a rate measured alone is not the rate
#: you pay. Planning against the uncontended rate under-priced the n=8192 leg by
#: 3.7x and it was cut mid-flight rather than allowed to run 69 minutes.
CONTENTION_3_SEATS = 3.71

#: What actually ran. Per t*: the step cap for each n. Rungs above the cap are
#: DROPPED, named, and priced below -- a silent truncation reads as "covered
#: everything" and this round has recorded that failure mode fourteen times.
#:
#: 8192's cap was cut 2400 -> 600 MID-FLIGHT, after the re-measured contended
#: rate above turned that leg from a 22-minute buy into a 69-minute one. The
#: n=2048 ladder had by then returned the finding that reprices the whole grid
#: (see OVERFIT below), and it argued for spending the remainder on the n axis
#: rather than the steps axis. The cut is recorded here rather than absorbed.
RAN_CAP = {2048: 9600, 8192: 600, 32768: 150}

#: THE FINDING THAT REPRICED THE GRID. At n=2048 the steps axis is not a
#: capability axis, it is an overfitting axis. Measured, seed 0, eval NRMSE
#: against train NRMSE at t*=2 / 8 / 32:
#:
#:     steps    train             eval              gap
#:      150     0.770/0.866/0.879 0.971/1.129/1.158 +0.20/+0.26/+0.28
#:      600     0.408/0.560/0.645 1.334/1.497/1.347 +0.93/+0.94/+0.70
#:     2400     0.191/0.302/0.350 1.611/1.920/1.994 +1.42/+1.62/+1.64
#:     9600     0.054/0.125/0.167 1.687/2.254/2.238 +1.63/+2.13/+2.07
#:
#: Train NRMSE falls to 0.054 while eval RISES past 2.2. The arm memorises 2048
#: examples and every additional step makes the held-out reading worse, at every
#: t*. So the 9600-step corner the stake statement names is not the cell that
#: decides anything at this n -- the DATA axis is, and the cheapest rung (150)
#: is the best rung. That inverted the remaining spend onto n and is why 8192
#: and 32768 were bought at 600 and 150 steps instead of 8192 being bought deep.
OVERFIT_AT_2048 = True

#: PROVENANCE WART, recorded rather than smoothed. The (t*=2, 150, 2048, seed 0)
#: cell was measured twice: 0.971432 at 8 threads in stage A and 0.969087 at 6
#: threads in the seed wave. Same seed, same data, same code -- CPU matmul
#: reduction order varies with thread count, which is exactly why
#: `m3_capability.py:66` pins threads at all. The delta is 0.0023, far below any
#: verdict boundary here, but a table that silently kept one of the two would be
#: hiding a reproducibility fact rather than reporting it.
THREAD_REPRO_DELTA = 0.971432 - 0.969087


def re_of(seconds: float, rate: float = S_PER_RE_UNCONTENDED) -> float:
    return seconds / rate


def cost_independent() -> float:
    """36 separate trainings: every (n, t*, steps) cell retrains from scratch."""
    per_col = sum(STEP_RUNGS)                      # 12,750 steps
    return len(T_STARS) * sum(per_col * C_STEP[n] for n in N_TRAIN)


def cost_nested() -> float:
    """9 trainings: the steps axis is read off ONE trajectory per (n, t*).

    A 9600-step run passes through 150, 600 and 2400. `t*` is NOT sliceable the
    same way -- see `r10_capacity_sweep.__doc__` -- so the t* axis stays 3 real
    trainings per n and only the steps axis collapses.
    """
    return len(T_STARS) * sum(max(STEP_RUNGS) * C_STEP[n] for n in N_TRAIN)


def cost_ran() -> float:
    return len(T_STARS) * sum(RAN_CAP[n] * C_STEP[n] for n in N_TRAIN)


def dropped_cells():
    """(t*, n, steps, marginal seconds) for every cell the budget could not buy."""
    out = []
    for t in T_STARS:
        for n in N_TRAIN:
            over = [r for r in STEP_RUNGS if r > RAN_CAP[n]]
            if not over:
                continue
            # Nested: the whole column past the cap costs one extra run to the
            # TOP rung, not one per rung. Attributed to the deepest cell.
            marginal = (max(over) - RAN_CAP[n]) * C_STEP[n]
            out.append((t, n, tuple(over), marginal))
    return out


def report() -> None:
    print("=== R10 P1a CAPACITY SWEEP -- PRICED DAG (softmax only) ===")
    print("rate: %.2f s/RE uncontended, %.2f s/RE at four seats (both measured "
          "this round)" % (S_PER_RE_UNCONTENDED, S_PER_RE_FOUR_SEATS))
    print()
    print("c_step MEASURED, 8 threads, s=64 d=24, 8 steps after 2 warmup:")
    for n in N_TRAIN:
        print("  n=%-6d %8.5f s/step   (2 threads: %.5f)  peak RSS %7.1f MiB"
              % (n, C_STEP[n], C_STEP_2T[n], PEAK_RSS_MIB[n]))
    print("  per-example cost is NOT flat: %.1f -> %.1f -> %.1f us. A unit fitted"
          % tuple(C_STEP[n] / n * 1e6 for n in N_TRAIN))
    print("  at n=2048 and multiplied out to n=32768 is low by %.2fx."
          % ((C_STEP[32768] / 32768) / (C_STEP[2048] / 2048)))
    print()

    ind, nest, ran = cost_independent(), cost_nested(), cost_ran()
    print("36 CELLS OR 12 RUNS? -- 36 cells, 9 TRAININGS.")
    print("  t* is a CORPUS, not an eval slice: M3_TASKS registers e3_t{2,8,32}")
    print("  as partial(make_equilibrium_batch, t_star=t), which zeroes the")
    print("  sub-diagonal at head=s-1-t*, giving a different label law N(0,t*+1)")
    print("  and a different bar 2/sqrt(t*). So the t* axis costs 3x, always.")
    print("  steps IS nested: one 9600-step run is read at all four rungs.")
    print("  So the grid is 3 t* x 3 n = 9 trainings read at 4 rungs = 36 cells,")
    print("  NOT 36 trainings and NOT 12.")
    print()
    print("  36 independent trainings : %10.0f s = %6.2f h = %8.1f RE"
          % (ind, ind / 3600, re_of(ind)))
    print("   9 nested trainings      : %10.0f s = %6.2f h = %8.1f RE"
          % (nest, nest / 3600, re_of(nest)))
    print("  nesting saves            : %10.0f s = %6.2f h = %8.1f RE  (%.1f%%)"
          % (ind - nest, (ind - nest) / 3600, re_of(ind - nest),
             100 * (ind - nest) / ind))
    print()
    print("UNAFFORDABLE, with the arithmetic. The corner cell alone:")
    corner = max(STEP_RUNGS) * C_STEP[32768]
    print("  (9600 steps, n=32768), ONE t*  : %8.0f s = %.2f h = %.1f RE"
          % (corner, corner / 3600, re_of(corner)))
    print("  the same cell at all three t*  : %8.0f s = %.2f h = %.1f RE = %.0f%% "
          "of the whole nested grid"
          % (3 * corner, 3 * corner / 3600, re_of(3 * corner),
             100 * 3 * corner / nest))
    print()
    print("WHAT RAN, and what was dropped:")
    for n in N_TRAIN:
        keep = [r for r in STEP_RUNGS if r <= RAN_CAP[n]]
        drop = [r for r in STEP_RUNGS if r > RAN_CAP[n]]
        print("  n=%-6d ran steps %-22s dropped %s"
              % (n, str(keep), str(drop) if drop else "none"))
    print("  cells covered: %d of %d.  spend: %.0f s = %.1f RE (%.1f%% of the grid)"
          % (len(T_STARS) * sum(len([r for r in STEP_RUNGS if r <= RAN_CAP[n]])
                                for n in N_TRAIN),
             len(T_STARS) * len(N_TRAIN) * len(STEP_RUNGS),
             ran, re_of(ran), 100 * ran / nest))
    dc = dropped_cells()
    tot_drop = sum(m for _, _, _, m in dc)
    ncells = sum(len(s) for _, _, s, _ in dc)
    print("  cells dropped: %d, marginal cost %.0f s = %.2f h = %.1f RE (%.1f%% "
          "of the grid)" % (ncells, tot_drop, tot_drop / 3600, re_of(tot_drop),
                            100 * tot_drop / nest))
    for t, n, steps, m in dc:
        print("    t*=%-3d n=%-6d steps=%-18s marginal %8.0f s = %6.1f RE"
              % (t, n, str(steps), m, re_of(m)))


def demo() -> None:
    """One runnable check: the accounting has to close and the ordering has to hold."""
    ind, nest, ran = cost_independent(), cost_nested(), cost_ran()
    assert ran < nest < ind, (ran, nest, ind)

    # Nesting saves exactly the rungs below the top, at every (n, t*).
    saved = len(T_STARS) * sum(sum(STEP_RUNGS[:-1]) * C_STEP[n] for n in N_TRAIN)
    assert abs((ind - nest) - saved) < 1e-6, (ind - nest, saved)

    # Ran + dropped-marginal must reconstruct the full nested grid exactly.
    tot_drop = sum(m for _, _, _, m in dropped_cells())
    assert abs(ran + tot_drop - nest) < 1e-6, (ran, tot_drop, nest, ran + tot_drop)

    # Every dropped column is named; nothing is silently truncated.
    named = {(t, n) for t, n, _, _ in dropped_cells()}
    expected = {(t, n) for t in T_STARS for n in N_TRAIN
                if any(r > RAN_CAP[n] for r in STEP_RUNGS)}
    assert named == expected, (named ^ expected)

    # The measured rate must be the one the contract quotes, not a re-derivation.
    assert (S_PER_RE_UNCONTENDED, S_PER_RE_FOUR_SEATS) == (18.01, 48.89)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    report()
    demo()
