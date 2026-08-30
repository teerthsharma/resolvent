"""Read the capacity-sweep journals and print the verdict table. Reads, never runs.

ONE BLOCK PER t*, rows = steps, cols = n_train, per the iteration's ask. Every
cell carries its SEED COUNT, because a single-seed number and an eight-seed
number are not the same claim and a table that hides which is which is the
failure mode this round has already recorded fourteen times.

C-D, THE LEARNABILITY PRECONDITION. `NO READING` is printed for any cell at or
above NRMSE 1.0 -- never "loss", never "win". 1.0 is the predict-the-mean
predictor by construction, so a cell that does not clear it has not established
that anything is learnable there, and a between-arm comparison in that cell
compares two things that both failed to start.

`--` is a cell that was never run. It is NOT a failure and must not be read as
one; `results/r10_it8_priced_dag.txt` names each one and prices it.

THE CEILING COLUMN is what stops a "learns" from being read as "solves". A 1-hop
model's best possible reading of a t*-hop label is `sqrt((t*-1)/t*)`, exact and
closed-form from `negation_scope.equilibrium_hop_reading`. At t*=32 that is
0.984251, so the entire learnable band at that rung is 1.6% wide.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
STEP_RUNGS = (150, 600, 2400, 9600)
N_TRAIN = (2048, 8192, 32768)


def load(tag: str = "it8"):
    cells, ceil, dropped = {}, {}, set()
    for p in sorted((ROOT / "results").glob("r10_%s_capacity_softmax_t*.jsonl" % tag)):
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("t") == "ceiling":
                ceil[e["t_star"]] = e["nrmse_ceiling"]
            elif e.get("t") == "cell":
                cells.setdefault((e["t_star"], e["steps"], e["n_train"]),
                                 {})[e["seed"]] = e["eval_nrmse"]
            elif e.get("t") == "dropped":
                for s in e["steps"]:
                    dropped.add((e["t_star"], s, e["n_train"]))
    return cells, ceil, dropped


def bootstrap_mean_ci(vals, *, b: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap over SEEDS, B=10^4, as the contract's strong form asks.

    Resamples the seed-level NRMSEs, not the eval examples -- the question is
    whether the CELL clears 1.0 across initialisations, and the per-example
    bootstrap already in the journal answers a different question (whether THIS
    trained model's reading is stable), which is why both are kept.
    """
    import random
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(b):
        means.append(sum(rng.choices(vals, k=len(vals))) / len(vals))
    means.sort()
    lo = means[int((alpha / 2) * b)]
    hi = means[min(b - 1, int((1 - alpha / 2) * b))]
    return lo, hi


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "it8"
    cells, ceil, dropped = load(tag)
    if not cells:
        print("no cells yet")
        return 1
    t_stars = sorted({t for t, _, _ in cells} | set(ceil))

    print("=== R10 P1a -- SOFTMAX CAPACITY, e3 chain family, s=64 d=24 "
          "n_eval=4096, 8 threads ===")
    print("bar: NRMSE < 1.0 (predict-the-mean). NO READING = did not clear it.")
    print("--  = cell not run (priced and named in results/r10_it8_priced_dag.txt)")
    print()

    any_learn = False
    crossing = []
    for t in t_stars:
        c = ceil.get(t)
        print("--- t* = %d   (softmax hop budget 1; best possible NRMSE for a "
              "1-hop model = %.6f) ---" % (t, c) if c else "--- t* = %d ---" % t)
        print("%-8s %s" % ("steps", "".join("%26s" % ("n_train=%d" % n)
                                            for n in N_TRAIN)))
        for st in STEP_RUNGS:
            row = "%-8d" % st
            for n in N_TRAIN:
                per = cells.get((t, st, n))
                if not per:
                    # Every un-run cell here was dropped for budget and is priced
                    # by name in results/r10_it8_priced_dag.txt. `!` marks one the
                    # journal explicitly recorded as dropped, so a cell missing
                    # for any OTHER reason shows up as a bare `--` and is visible.
                    row += "%26s" % ("-- (dropped)" if (t, st, n) in dropped else "--")
                    continue
                vals = [per[s] for s in sorted(per)]
                m = statistics.fmean(vals)
                learns = m < 1.0
                any_learn = any_learn or learns
                if learns:
                    crossing.append((t, st, n, m, len(vals)))
                tagtxt = "LEARNS" if learns else "NO READING"
                row += "%26s" % ("%.4f %s n=%d" % (m, tagtxt, len(vals)))
            print(row)
        print()

    print("=== SEED CIs (bootstrap over seeds, B=10^4) ===")
    multi = [(k, v) for k, v in sorted(cells.items()) if len(v) >= 2]
    if not multi:
        print("  none yet -- every cell above is SINGLE SEED (n=1). Sign floor "
              "for a 1-seed claim is 2*2^-1 = 1.0: no sign is established.")
    for (t, st, n), per in multi:
        vals = [per[s] for s in sorted(per)]
        ci = bootstrap_mean_ci(vals)
        floor = 2 * 2.0 ** (-len(vals))
        print("  t*=%-3d steps=%-5d n=%-6d  seeds=%d  mean=%.6f  sd=%.6f  "
              "CI95=[%.6f,%.6f]  excludes 1.0: %s  sign floor 2*2^-%d=%.4f"
              % (t, st, n, len(vals), statistics.fmean(vals),
                 statistics.stdev(vals) if len(vals) > 1 else 0.0,
                 ci[0], ci[1], "YES" if ci[1] < 1.0 else "NO", len(vals), floor))

    print()
    print("=== VERDICT ===")
    print("  region LEARNABLE iff at least one cell reads NRMSE < 1.0: %s"
          % ("LEARNABLE" if any_learn else "UNTESTABLE (nothing cleared the bar)"))
    for t in t_stars:
        got = [x for x in crossing if x[0] == t]
        if got:
            best = min(got, key=lambda x: x[3])
            print("  t*=%-3d LEARNABLE -- best %.6f at steps=%d n=%d (%d seed(s)); "
                  "1-hop ceiling %.6f" % (t, best[3], best[1], best[2], best[4],
                                          ceil.get(t, float("nan"))))
        else:
            print("  t*=%-3d NO READING in every cell run -- the region is "
                  "UNTESTED here, not refuted." % t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
