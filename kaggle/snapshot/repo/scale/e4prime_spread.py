"""Sampling spread of the number the E4' strike is conditioned on.

`tests/cameron/test_e4_rips_gate.py::test_the_degree_decoder_passes_at_criticality_so_e4_is_struck`
strikes E4 iff `fit_eval(local_features(pairs, [adj], 0), y) < PASS_BAR` on
`CriticalLarge_S2Rips_1024`. The shipped reading is 0.471045, a margin of 0.028955.

`draw_balanced_marginal` seeds its pair sampler `random.Random(0x33960000 ^ case.seed)`
and its train/test split `np.random.RandomState(0)`. Neither is a parameter, so the
sampling spread of the deciding number was never measured when the strike was made.

WHAT IS VARIED HERE AND WHAT IS HELD. Only the DRAW seed moves. The graph, the
partition, the feature matrix construction, the split permutation and the bar are all
held at their shipped values. The draw seed is reachable without editing tracked
source because `draw_balanced_marginal` reads `case.seed` as a plain field: replacing
it on the frozen dataclass re-seeds the sampler and leaves `edges` and `partition`
identical objects. `_assert_graph_untouched` proves that rather than asserting it.

THE SEED SET IS DECLARED BY RULE, NOT PICKED. Block A, the 20 that the it.12 verdict
is read on, is the 20 consecutive `case.seed` values after the shipped 0x33960006, i.e.
0x33960007..0x3396001A, giving `random.Random` seeds 7..26. The shipped draw is seed 6.
No seed was inspected before the set was fixed.

BLOCK B EXISTS BECAUSE BLOCK A TURNED OUT NOT TO BE INDEPENDENT OF THE CLAIM IT CHECKS.
Block A reproduces the min, max and mean of MARS_REPORT_IT4 to six decimals, so that
rule evidently reselected his seeds rather than sampling elsewhere. Block B is a
disjoint 20 (`case.seed` 0x33960100..0x33960113, `random.Random` seeds 256..275) and is
the part of this file that is genuinely new evidence. It does not enter the it.12
verdict, which is defined on 20 seeds.

    python -m scale.e4prime_spread
"""
from __future__ import annotations

import dataclasses
import json
import math
import statistics
import sys

from ceq.rips import corpus
from scale.rips_gate import (PASS_BAR, adjacency, draw_balanced_marginal, fit_eval,
                             local_features)

CASE_NAME = "CriticalLarge_S2Rips_1024"
SHIPPED_CASE_SEED = 0x33960006
SEED_XOR = 0x33960000
SWEEP_A = [SHIPPED_CASE_SEED + k for k in range(1, 21)]
SWEEP_B = [(SHIPPED_CASE_SEED & ~0xFF) + 0x100 + k for k in range(20)]


def score_for(case, adj) -> float:
    """The deciding line of the strike test, verbatim, on whatever draw `case` seeds."""
    drawn, counts = draw_balanced_marginal(case)
    if drawn is None:
        raise RuntimeError(f"draw returned None: {counts}")
    pairs, y = drawn
    return fit_eval(local_features(pairs, [adj], 0), y)


def _assert_graph_untouched(shipped, swapped) -> None:
    if swapped.edges is not shipped.edges or swapped.partition is not shipped.partition:
        raise RuntimeError("reseeding perturbed the graph; the sweep would be invalid")


def main() -> int:
    cases = {c.name: c for c in corpus()}
    case = cases[CASE_NAME]
    assert case.seed == SHIPPED_CASE_SEED, case.seed
    adj = adjacency(case.node_count, case.edges)

    rows = [{"role": "shipped", "case_seed": hex(case.seed),
             "rng_seed": SEED_XOR ^ case.seed, "score": score_for(case, adj),
             "status": "ok"}]

    for role, seeds in (("sweep", SWEEP_A), ("sweep_b", SWEEP_B)):
        for cs in seeds:
            swapped = dataclasses.replace(case, seed=cs)
            _assert_graph_untouched(case, swapped)
            row = {"role": role, "case_seed": hex(cs), "rng_seed": SEED_XOR ^ cs}
            try:
                row["score"] = score_for(swapped, adj)
                row["status"] = "ok"
            except Exception as exc:                  # counted, never read as a pass
                row["score"] = None
                row["status"] = f"{type(exc).__name__}: {exc}"
            row["strike_holds"] = (row["score"] is not None and row["score"] < PASS_BAR)
            rows.append(row)

    # An instrument that cannot fail proves nothing: the SAME fit_eval on the SAME
    # corpus must be seen to return >= PASS_BAR somewhere, or a sweep that never
    # crosses the bar is unreadable.
    stable = cases["StableSparse_S2Rips_64"]
    rows.append({"role": "instrument_check", "case_seed": hex(stable.seed),
                 "rng_seed": SEED_XOR ^ stable.seed,
                 "score": score_for(stable, adjacency(64, stable.edges)),
                 "status": "ok", "note": "StableSparse_S2Rips_64, same decoder"})

    shipped = rows[0]["score"]
    for role, tag in (("sweep", "summary"), ("sweep_b", "summary_b")):
        sweep = [r for r in rows if r["role"] == role]
        ok = [r["score"] for r in sweep if r["status"] == "ok"]
        errors = [r for r in sweep if r["status"] != "ok"]
        k = sum(1 for r in sweep if r.get("strike_holds"))
        mu = statistics.fmean(ok)
        sd = statistics.stdev(ok)                     # ddof=1, the unbiased estimator
        rows.append({"role": tag, "case": CASE_NAME, "pass_bar": PASS_BAR,
                     "shipped": shipped, "n_sweep": len(sweep), "n_ok": len(ok),
                     "n_error": len(errors), "mu_hat": mu, "sigma_hat": sd,
                     "sigma_hat_pop": statistics.pstdev(ok),
                     "min": min(ok), "max": max(ok), "z": (shipped - mu) / sd,
                     "z_pop": (shipped - mu) / statistics.pstdev(ok),
                     "k_strike_holds": k, "reversals": len(ok) - k,
                     "verdict": "STANDS" if k >= 18 else "WITHDRAWN"})

    with open("results/r10_it12_e4prime_spread.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    for r in rows:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
