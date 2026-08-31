"""Adjudicate V13_PREDICTION_HOP2.md against the journal, mechanically.

The prediction was filed while `results/r10_v13pilot_capacity_pivot_unsigned_t8.jsonl`
held zero cell rows (sha256 of the prediction recorded at filing time:
cdd2ae78820c13a06fa0b7bfafec2f15...). It states four claims about the N=8 cell at
t*=8, n=32768. This script reads them off the journal rather than leaving the
verdict to whoever writes the report.

P1  mean eval_nrmse in [0.960, 0.990]
P2  the N=8 seed CI does not clear floor_1 = sqrt((t*-1)/t*), so cap_verdict
    returns "consistent with 1 hop" and proven_hops = 0
P3  the reading sits at least 0.09 above its own floor_2 = sqrt((t*-2)/t*)
P4  h_hat = t*(1 - NRMSE^2) stays below 1.0

REFUSES below N=8, for the same reason `verdict` and `cap_verdict` do: the
prediction is about an eight-seed cell and a verdict on fewer seeds answers a
question nobody registered.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scale.it11_verdict import REQUIRED_SEEDS, cap_verdict, hop_floor, seed_ci

JOURNAL = ROOT / "results" / "r10_v13pilot_capacity_pivot_unsigned_t8.jsonl"
T_STAR, N_TRAIN, STEPS, THREADS = 8, 32768, 150, 12
SOFTMAX_REF_MEAN, SOFTMAX_REF_SD = 0.975371, 0.002360   # N=8, threads=12, same cell


def cells():
    v = {}
    for line in JOURNAL.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (e.get("t") == "cell" and e.get("t_star") == T_STAR
                and e.get("n_train") == N_TRAIN and e.get("steps") == STEPS
                and e.get("threads") == THREADS):
            v[e["seed"]] = e["eval_nrmse"]
    return v


def main() -> int:
    v = cells()
    n = len(v)
    print("cell: t*=%d n=%d steps=%d threads=%d" % (T_STAR, N_TRAIN, STEPS, THREADS))
    print("seeds on record: %d/%d  %s" % (n, REQUIRED_SEEDS,
                                          ["%.6f" % v[s] for s in sorted(v)]))
    if n < REQUIRED_SEEDS:
        print("\nINSUFFICIENT -- the prediction is about an N=%d cell and this has %d."
              % (REQUIRED_SEEDS, n))
        print("No verdict is reported, because a verdict over %d seeds answers a "
              "question nobody registered." % n)
        return 2

    xs = [v[s] for s in sorted(v)]
    mean, sd = statistics.fmean(xs), statistics.stdev(xs)
    lo, hi = seed_ci(xs)
    f1, f2 = hop_floor(T_STAR, 1), hop_floor(T_STAR, 2)
    h_hat = T_STAR * (1.0 - mean ** 2)
    cap = cap_verdict(T_STAR, xs)

    print("\nmean %.6f  sd %.6f  CI95 [%.6f, %.6f]" % (mean, sd, lo, hi))
    print("softmax at the same cell: %.6f (sd %.6f)  ->  contrast %+.6f"
          % (SOFTMAX_REF_MEAN, SOFTMAX_REF_SD, mean - SOFTMAX_REF_MEAN))
    print("floor_1 %.6f   floor_2 %.6f   h_hat %.3f" % (f1, f2, h_hat))
    print("cap_verdict: %s" % cap)

    checks = [
        ("P1  mean in [0.960, 0.990]", 0.960 <= mean <= 0.990, "%.6f" % mean),
        ("P2  CI does not clear floor_1", cap.proven_hops == 0,
         "proven_hops=%d, CI_hi %.6f vs floor_1 %.6f" % (cap.proven_hops, hi, f1)),
        ("P3  at least 0.09 above floor_2", (mean - f2) >= 0.09, "%.6f" % (mean - f2)),
        ("P4  h_hat below 1.0", h_hat < 1.0, "%.3f" % h_hat),
    ]
    print()
    held = 0
    for name, ok, detail in checks:
        print("  %-34s %-6s  %s" % (name, "HOLDS" if ok else "FAILS", detail))
        held += bool(ok)
    print("\n%d/4 held." % held)
    if held == 4:
        print("The filed prediction stands: the hop-2 term does not lift this arm off")
        print("softmax, and no increase in n from this arm produces a floor crossing.")
    else:
        print("The filed prediction is wrong as written. Record which clause failed and")
        print("what it implies; do not restate the prediction to match the outcome.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
