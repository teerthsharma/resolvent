"""IT.10 MERCURY. Three RED assertions and one GREEN control.

RED-1  every scored cell in this round is read on ONE eval draw (n_eff = 1).
RED-2  `softmax` has no cell on the fresh seeds 8..15 anywhere in `results/`.
RED-3  the it.8 aggregate's registered verdict disagrees with its own cells.
GREEN  the aggregate row is reconstructible from the cells with the runner's
       own estimator, bitwise -- without which RED-3 reads a journal that
       cannot be recomputed and means nothing.
"""
import json
import math
import pathlib
import statistics

import pytest
from scipy import stats as _st

ROOT = pathlib.Path(__file__).resolve().parents[2]
IT8 = ROOT / "results" / "v20_r15_it8_armpl_b.jsonl"
NEW = ROOT / "results" / "v20_r15_it10_mercury_rescore.jsonl"
FRESH = [8, 9, 10, 11, 12, 13, 14, 15]


def rows(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def ci(v, floor1):
    n = len(v)
    m, sd = statistics.fmean(v), statistics.stdev(v)
    half = _st.t.ppf(0.975, df=n - 1) * sd / math.sqrt(n)
    return dict(n=n, mean=m, sd=sd, ci_lo=m - half, ci_hi=m + half,
                crosses=bool(m + half < floor1))


def test_every_arm_pl_cell_is_scored_on_more_than_one_eval_draw():
    """RED-1. n_eff = 1 is not a rate; a rate needs >= 2 eval draws per cell."""
    cells = [r for r in rows(NEW) + rows(IT8)
             if r.get("t") in ("cell", "rescore") and r.get("kind") == "arm_pl"]
    by_cell = {}
    for c in cells:
        by_cell.setdefault(c["seed"], set()).add(c.get("eval_seed", 12345))
    single = sorted(s for s, e in by_cell.items() if len(e) < 2)
    assert not single, (
        f"arm_pl seeds scored on ONE eval draw only (n_eff=1, no binomial rate "
        f"licensed): {single}; eval seeds seen per cell: "
        f"{ {s: sorted(e) for s, e in sorted(by_cell.items())} }")


def test_the_fresh_seeds_have_a_softmax_control_on_the_same_seeds():
    """RED-2. 7-of-8 against 0-of-8 is unpaired until softmax runs on 8..15."""
    seen = set()
    for p in sorted((ROOT / "results").glob("*.jsonl")):
        for r in rows(p):
            if r.get("t") in ("cell", "rescore") and r.get("kind") == "softmax":
                seen.add(r["seed"])
    missing = [s for s in FRESH if s not in seen]
    assert not missing, (
        f"softmax cells exist for seeds {sorted(seen)}; the fresh seeds "
        f"{missing} have no softmax control anywhere in results/")


def test_it8_aggregate_verdict_agrees_with_its_own_cells():
    """RED-3. 8 of 9 cells below floor_1; the registered verdict says false."""
    rs = rows(IT8)
    agg = [r for r in rs if r.get("t") == "agg"][0]
    cells = [r for r in rs if r.get("t") == "cell"]
    crossed = [c for c in cells if c["eval_nrmse"] < agg["floor_1"]]
    assert (len(crossed) > len(cells) / 2) == bool(agg["crosses"]), (
        f"{len(crossed)}/{len(cells)} cells sit below floor_1="
        f"{agg['floor_1']!r} but the agg row registers crosses="
        f"{agg['crosses']!r} (sd={agg['sd']!r}, ci_hi={agg['ci_hi']!r})")


def test_control_the_it8_aggregate_is_reconstructible_bitwise():
    """GREEN control. Without it RED-3 is a claim about arithmetic I cannot do."""
    rs = rows(IT8)
    agg = [r for r in rs if r.get("t") == "agg"][0]
    v = [c["eval_nrmse"] for c in rs if c.get("t") == "cell"]
    got = ci(v, agg["floor_1"])
    for k in ("n", "mean", "sd", "ci_lo", "ci_hi", "crosses"):
        assert got[k] == agg[k], f"{k}: recomputed {got[k]!r} != journalled {agg[k]!r}"
