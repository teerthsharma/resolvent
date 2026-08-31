"""LADDER E's "the curve rises" verdict, as a named statistic instead of an eye.

    python -m scale.page_trend

WHAT THIS REPLACES. `scale/e_ladder.py:217 verdict()` decides row A against row B
by `mono = all(b >= a for a, b in zip(deltas, deltas[1:]))` -- four seed-means
compared pairwise, which is a monotonicity CHECK on four numbers and not a test.
It has no null, so it has no error rate: four independent noisy numbers land in
non-decreasing order by chance one time in twenty-four, and nothing in the row-A
branch knows that. This module supplies the null.

THE TWO STATISTICS, AND WHY BOTH.

  PAGE'S TREND TEST answers "is the ordering real". It is the ordered-alternatives
  rank test: rank the `k` conditions WITHIN each of the `n` blocks, smallest to
  largest, sum the ranks per condition into `R_j`, and take `L = sum_j j * R_j`.
  Under the null that the conditions are exchangeable within a block, every one
  of the `k!` rank assignments is equally likely and independent across blocks,
  so the exact null distribution of `L` is the `n`-fold convolution of the `k!`
  point distribution of `sum_j j * r_j`. At `k = 4`, `n = 5` that is 24 values
  convolved five times -- 7962624 rank assignments enumerated exactly, in closed
  form, with no table lookup and no normal approximation. `L` is scale-free: it
  cannot report an effect size and it does not try to.

  ISOTONIC REGRESSION answers "how big is it at the top rung". Pool-adjacent-
  violators fits the least-squares non-decreasing sequence to the rung means.
  Where the raw means are already non-decreasing the fit returns them unchanged,
  which is the honest behaviour: isotonic regression is a constrained fit, not a
  smoother, and it must not manufacture a rise that the data do not carry.

  Neither alone is a verdict. Page's `L` is significant for a rise of any size,
  and a positive top rung is not evidence of a trend. The pre-registered branch
  requires both.

THE PRE-REGISTRATION, IN CODE AND EXHAUSTIVE BEFORE THE DATA.

    RISES  := page_p < ALPHA  AND  isotonic top-rung value > 0 with its
              bootstrap CI excluding zero
    FLAT   := neither clause fires
    SPLIT  := exactly one clause fires

`SPLIT` exists because "rises = both, flat = neither" leaves a hole, and this
repo has already paid for one: row H was added to
`E_LADDER_PREREGISTERED_READING.md` because rows A-G all conditioned on settled
winning somewhere and a strictly-losing ladder fell through the entire table. A
branch table with a hole in it reports whichever adjacent row the reader prefers.

IMMUNITY TO RUNG-PICKING. Page's `L` quantifies over every rung by construction
-- there is no subset of rungs it can be computed on -- and the isotonic clause
reads the LAST rung, fixed by the ladder's own definition rather than chosen
after the numbers were seen. `RUNGS` here is imported from `scale.e_ladder` so
the two cannot drift apart.

NO THRESHOLD IS REFITTED TO THE DATA IT JUDGES. `ALPHA = 0.05` is the
conventional level, `N_BOOT` and `BOOT_SEED` are `scale/m3_synthetic_settled.py`'s
own bootstrap settings reused verbatim, and the exact null is a combinatorial
object that does not see the data at all. This is the discipline
`scale/e_ladder.py:23-27` freezes `RESOLUTION_13 = 0.027260` for.

THE READING, ON THE COMPLETE CPU LADDER. `results/e_ladder_reading.txt`, five
seeds, four rungs. The pre-registered branch returns `RISES`: `L = 139`, exact
`p = 0.016724`, permutation `p = 0.016255` over 200000 draws (`se 0.000283`), and
an isotonic top rung of `+0.016035` with CI `[+0.002826, +0.033167]`.

TWO THINGS THAT VERDICT RESTS ON, BOTH MEASURED AND BOTH PRINTED BESIDE IT.

  THE SIZE CLAUSE IS CARRIED BY THE CONSTRAINT, NOT BY THE DATA. The same
  bootstrap without the monotone constraint gives the top rung as `+0.016035`
  with CI `[-0.004711, +0.033167]`, which COVERS ZERO -- and whose lower bound
  reproduces `results/e_ladder_reading.txt`'s shipped `ci_lo` of `-0.004711`
  exactly, so the two bootstraps are the same bootstrap. PAVA pooled the top
  rung in `13.07%` of resamples and lifted the lower bound by `+0.007537`. The
  pooling only ever raises a low top, never lowers a high one, so that lift is
  a property of the estimator under a flat truth. The size clause fires on the
  strength of it.

  THE TREND CLAUSE SURVIVES 2 OF 5 SINGLE-SEED DELETIONS. Dropping seed 1 leaves
  `p = 0.003864` and dropping seed 2 leaves `p = 0.021741`, but dropping seed 0
  or seed 4 leaves `p = 0.050411` and dropping seed 3 leaves `p = 0.072401`.
  Five seeds is what the ladder was run at and this is what five seeds buys.

THE SCOPE CONSTRAINT THAT OUTRANKS ALL OF THE ABOVE. Row G of
`E_LADDER_PREREGISTERED_READING.md` credits a rung NOTHING in either direction
when either cell sits at or above predict-the-mean, and it fires on THREE of the
four rungs this statistic is computed on: `e3_t2` (settled 1.013958),
`e3_t8` (1.096009 / twin 1.091725) and `e3_t32` (1.103711 / 1.119745). Only
`e3_t1` has both cells below 1.0. **A trend in the difference between two arms
that both lose to predict-the-mean is an ordering, not a capability.** `RISES`
here is a statement about the contrast as a number. It is not a claim that
settling bought anything at depth, and `report()` prints the cell means beside
the verdict so that it cannot be read as one.

None of these three observations changes which row fires. All three are printed
by `report()` so that no reader takes `RISES` without them.
"""
from __future__ import annotations

import argparse
import fractions
import itertools
import json
import math
import pathlib
import sys

import numpy as np
from sklearn.isotonic import isotonic_regression

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scale.e_ladder import RUNGS, read                              # noqa: E402

__all__ = ["ALPHA", "N_BOOT", "BOOT_SEED", "page_l", "page_null",
           "page_null_exact", "achievable_p_values", "critical_value",
           "bootstrap_p_floor",
           "page_p_value", "page_p_value_monte_carlo", "isotonic_curve",
           "isotonic_top_ci", "trend", "verdict_of", "report"]

#: The pre-registered level. Conventional, and frozen before any `L` was formed.
ALPHA = 0.05

#: Bootstrap settings, taken verbatim from `scale/m3_synthetic_settled.py:172-182`
#: so the isotonic clause's interval is drawn the same way every other interval
#: in this repo is drawn. Not tuned here.
N_BOOT = 10000
BOOT_SEED = 0


def page_l(blocks: np.ndarray) -> tuple[float, np.ndarray, bool]:
    """`(L, rank_sums, had_ties)` for an `n x k` block-by-condition table.

    Ranks run within a block, 1 for the smallest. Ties take midranks, which
    keeps `L` well defined but voids the exactness of `page_null` -- the flag
    travels out so the caller can refuse rather than quote an exact p-value on
    a tied table.
    """
    b = np.asarray(blocks, dtype=float)
    if b.ndim != 2:
        raise ValueError("blocks must be n_blocks x n_conditions")
    n, k = b.shape
    ranks = np.empty_like(b)
    had_ties = False
    for i in range(n):
        row = b[i]
        order = np.argsort(row, kind="mergesort")
        plain = np.empty(k)
        plain[order] = np.arange(1, k + 1, dtype=float)
        # midranks for any tied group
        srt = row[order]
        start = 0
        for stop in range(1, k + 1):
            if stop == k or srt[stop] != srt[start]:
                if stop - start > 1:
                    had_ties = True
                    plain[order[start:stop]] = (start + stop + 1) / 2.0
                start = stop
        ranks[i] = plain
    rank_sums = ranks.sum(axis=0)
    weights = np.arange(1, k + 1, dtype=float)
    return float(weights @ rank_sums), rank_sums, had_ties


def page_null(k: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """`(support, probability)` for `L` under exchangeability, EXACTLY.

    One block contributes `sum_j j * r_j` over the `k!` equally likely rank
    assignments `r`; blocks are independent, so the distribution of `L` is the
    `n`-fold convolution of that. At `k = 4, n = 5` this enumerates the full
    `24^5 = 7962624` assignment space without materialising it.
    """
    weights = np.arange(1, k + 1, dtype=float)
    per_block = [int(weights @ np.array(r, dtype=float))
                 for r in itertools.permutations(range(1, k + 1))]
    lo, hi = min(per_block), max(per_block)
    single = np.zeros(hi - lo + 1)
    for v in per_block:
        single[v - lo] += 1.0
    single /= single.sum()
    dist = single.copy()
    for _ in range(n - 1):
        dist = np.convolve(dist, single)
    support = np.arange(lo * n, hi * n + 1, dtype=float)
    if len(support) != len(dist):                                   # pragma: no cover
        raise AssertionError("convolution support mismatch")
    return support, dist


def page_null_exact(k: int, n: int) -> tuple[list[int], list[int], int]:
    """`(support, integer counts, total)` for `L`. THE SECOND PATH.

    `page_null` convolves float probabilities with `np.convolve`. This multiplies
    integer polynomials with Python big integers and never sees a float, so
    rounding cannot be common to both. The two are checked against each other in
    `tests/jupiter/test_page_trend.py`, and both are checked against literal
    enumeration of every one of the `(k!)^n` rank assignments at sizes where that
    is tractable -- a third route that shares no arithmetic with either.
    """
    weights = range(1, k + 1)
    per_block = [sum(j * r for j, r in zip(weights, perm))
                 for perm in itertools.permutations(range(1, k + 1))]
    lo, hi = min(per_block), max(per_block)
    block = [0] * (hi - lo + 1)
    for v in per_block:
        block[v - lo] += 1
    counts = [1]
    for _ in range(n):
        out = [0] * (len(counts) + len(block) - 1)
        for i, a in enumerate(counts):
            if a:
                for j, b in enumerate(block):
                    out[i + j] += a * b
        counts = out
    return list(range(lo * n, hi * n + 1)), counts, sum(counts)


def achievable_p_values(k: int, n: int) -> list[tuple[int, fractions.Fraction]]:
    """Every upper-tail p-value the design can produce, as exact rationals.

    `[(L, P(L' >= L))]` descending in `L`. A discrete null admits only these; any
    other number quoted as a p-value on this design is an interpolation. At
    `k = 4`, `n = 5` there are 51 of them, the smallest non-zero being
    `1 / 24^5 = 1.2558674e-07`.
    """
    support, counts, total = page_null_exact(k, n)
    out, cum = [], 0
    for idx in range(len(counts) - 1, -1, -1):
        cum += counts[idx]
        out.append((support[idx], fractions.Fraction(cum, total)))
    return out


def critical_value(k: int, n: int, alpha: float = ALPHA) -> dict:
    """The attainable level of the trend clause, and its true size.

    A discrete test cannot in general have size exactly `alpha`. The critical
    value is the smallest `L` whose exact upper-tail p is at or below `alpha`,
    and `effective_size` is that p -- the test's ACTUAL false-positive rate,
    always at or below the nominal one. At `k = 4`, `n = 5`, `alpha = 0.05`:
    `L_crit = 137`, `effective_size = 0.037002877`, and the next coarser rung
    `L = 136` reads `0.052384114`, above `alpha`. Twelve of the 51 achievable
    p-values sit at or below `0.05`, so the level is reachable with room.
    """
    lattice = achievable_p_values(k, n)
    below = [(t, p) for t, p in lattice if float(p) <= alpha]
    if not below:
        return {"k": k, "n": n, "alpha": alpha, "reachable": False,
                "L_crit": None, "effective_size": None, "n_below_alpha": 0,
                "finest_p": float(lattice[0][1]),
                "n_achievable": len({p for _t, p in lattice})}
    t_crit, p_crit = below[-1]                     # smallest L still under alpha
    return {"k": k, "n": n, "alpha": alpha, "reachable": True,
            "L_crit": t_crit, "effective_size": float(p_crit),
            "n_below_alpha": len(below),
            "finest_p": float(lattice[0][1]),
            "n_achievable": len({p for _t, p in lattice})}


def bootstrap_p_floor(n_blocks: int) -> float:
    """The finest two-sided p a sign-pattern statistic on `n` blocks can produce.

    `2 / 2^n`. It is the granularity floor of the SIZE clause, not of Page's `L`.
    A paired percentile bootstrap CI over `n` seeds is driven by which seeds agree
    in sign -- measured on the shipped `contrast()` over 1000 samples, a 5-0
    unanimity excludes zero 385/385 times, a 4-1 split 20-44% and a 3-2 split
    0-3.7% -- so "the CI excludes zero" at `n = 5` is very nearly "all five seeds
    agreed", and the finest two-sided p it can express is `2/32 = 0.0625`,
    ABOVE 0.05. Page's `L` is not subject to this: it ranks `k` conditions within
    each block rather than reading one sign, so its outcome space is `(k!)^n`
    rather than `2^n` -- `24^5 = 7962624` against `2^5 = 32`, a factor of
    `12^5 = 248832`, or `log2(24) = 4.585` bits per block against 1.
    """
    return 2.0 / 2.0 ** n_blocks


def page_p_value(observed: float, k: int, n: int) -> float:
    """`P(L >= observed)` under the exact null. Upper tail: the ordered
    alternative is `theta_1 <= ... <= theta_k`, which drives `L` up."""
    support, dist = page_null(k, n)
    return float(dist[support >= observed - 1e-9].sum())


def page_p_value_monte_carlo(observed: float, k: int, n: int, *,
                             draws: int = 200000, seed: int = 0) -> dict:
    """THE SECOND PATH. Same p-value by permutation, so a bug in the
    convolution's support arithmetic cannot hide behind a bug in the
    enumeration. Returns the estimate and its binomial standard error."""
    rng = np.random.default_rng(seed)
    weights = np.arange(1, k + 1, dtype=float)
    hits = 0
    chunk = 20000
    remaining = draws
    while remaining > 0:
        take = min(chunk, remaining)
        remaining -= take
        # independent uniform rank assignments per block, `take` replicates
        keys = rng.random((take, n, k))
        ranks = np.argsort(np.argsort(keys, axis=2), axis=2) + 1
        stat = (ranks.sum(axis=1) * weights).sum(axis=1)
        hits += int((stat >= observed - 1e-9).sum())
    p = hits / draws
    return {"p": p, "se": math.sqrt(max(p * (1 - p), 0.0) / draws),
            "draws": draws, "seed": seed}


def isotonic_curve(values, weights=None) -> np.ndarray:
    """The least-squares non-decreasing fit. PAVA, via sklearn 1.9.0."""
    y = np.asarray(values, dtype=float)
    return np.asarray(isotonic_regression(
        y, sample_weight=None if weights is None else np.asarray(weights,
                                                                 dtype=float),
        increasing=True), dtype=float)


def isotonic_top_ci(blocks: np.ndarray, *, n_boot: int = N_BOOT,
                    seed: int = BOOT_SEED) -> dict:
    """The isotonic fit of the condition means, and a percentile CI on its TOP
    rung from a bootstrap over BLOCKS.

    Blocks are seeds, so the resample is over seeds and stays paired across
    rungs -- the same pairing `scale/m3_synthetic_settled.py:contrast` uses.
    With five seeds the resample has only `5^5 = 3125` distinct atoms, so the
    interval is coarse by construction; the count is returned rather than
    hidden.

    THE CONSTRAINT IS NOT FREE AND ITS COST IS RETURNED, NOT ARGUED. When a
    resample puts the top rung below its neighbour, PAVA pools the two and the
    fitted top comes back at their average -- ABOVE the raw top. The pooling is
    one-sided, so the isotonic top's bootstrap lower bound is biased UPWARD
    exactly when the underlying curve is flat, which is the hypothesis the
    clause is supposed to be able to reject. `raw_ci_lo` / `raw_ci_hi` are the
    same bootstrap without the constraint and `pooled_fraction` is the share of
    resamples in which the pooling fired; a reader comparing the two intervals
    sees the size of the lift instead of taking the constrained one on trust.
    """
    b = np.asarray(blocks, dtype=float)
    n, _k = b.shape
    fit = isotonic_curve(b.mean(axis=0))
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n, size=(n_boot, n))
    means = np.array([b[idx].mean(axis=0) for idx in draw])
    tops = np.array([isotonic_curve(m)[-1] for m in means])
    raw = means[:, -1]
    lo, hi = np.percentile(tops, [2.5, 97.5])
    raw_lo, raw_hi = np.percentile(raw, [2.5, 97.5])
    return {"fit": fit.tolist(), "top": float(fit[-1]),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "raw_top": float(b.mean(axis=0)[-1]),
            "raw_ci_lo": float(raw_lo), "raw_ci_hi": float(raw_hi),
            "constraint_lift": float(lo - raw_lo),
            "pooled_fraction": float((tops > raw + 1e-12).mean()),
            "raw_excludes_zero": bool(raw_lo > 0.0 or raw_hi < 0.0),
            "n_boot": n_boot, "seed": seed, "distinct_atoms": n ** n,
            "excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "positive_excluding_zero": bool(fit[-1] > 0.0 and lo > 0.0)}


def leave_one_block_out(blocks, *, n_boot: int = N_BOOT,
                        seed: int = BOOT_SEED) -> list[dict]:
    """The verdict refit with each block dropped in turn.

    Five seeds is the resolution the ladder was run at, and a verdict that one
    seed can flip is a verdict about that seed. This is a stability diagnostic,
    NOT a second test and NOT a branch: it changes nothing about which row
    fires, it only prints how much of the row rests on one draw.
    """
    b = np.asarray(blocks, dtype=float)
    n, k = b.shape
    out = []
    for drop in range(n):
        sub = np.delete(b, drop, axis=0)
        L, _sums, ties = page_l(sub)
        p = page_p_value(L, k, n - 1)
        iso = isotonic_top_ci(sub, n_boot=n_boot, seed=seed)
        out.append({"dropped": drop, "L": L, "p": p, "had_ties": ties,
                    "top": iso["top"], "ci_lo": iso["ci_lo"],
                    "ci_hi": iso["ci_hi"],
                    "rises": bool(p < ALPHA and iso["positive_excluding_zero"])})
    return out


def verdict_of(page: dict, iso: dict) -> tuple[str, str]:
    """The pre-registered branch. Exhaustive over the two clauses' four states.

    Structured to match `scale/m3_synthetic_settled.py:172 verdict_of` and
    `scale/e_ladder.py:217 verdict()`: a tag and a sentence, no numbers
    reformatted, the caller prints them beside the evidence.
    """
    trend_fires = page["p"] < ALPHA
    size_fires = iso["positive_excluding_zero"]
    if trend_fires and size_fires:
        return ("RISES", "Page's L is significant at alpha={:.2f} (L={:.0f}, "
                         "exact p={:.6f}) AND the isotonic top rung is {:+.6f} "
                         "with CI [{:+.6f}, {:+.6f}] excluding zero."
                .format(ALPHA, page["L"], page["p"], iso["top"],
                        iso["ci_lo"], iso["ci_hi"]))
    if not trend_fires and not size_fires:
        return ("FLAT", "neither clause fires: Page's L={:.0f} has exact "
                        "p={:.6f} >= {:.2f}, and the isotonic top rung is "
                        "{:+.6f} with CI [{:+.6f}, {:+.6f}], which does not "
                        "exclude zero on the positive side."
                .format(page["L"], page["p"], ALPHA, iso["top"],
                        iso["ci_lo"], iso["ci_hi"]))
    if trend_fires:
        return ("SPLIT", "the ORDERING is significant (L={:.0f}, exact "
                         "p={:.6f}) but its SIZE at the top rung is not: "
                         "{:+.6f}, CI [{:+.6f}, {:+.6f}]. An ordering with no "
                         "measurable magnitude licenses nothing about the "
                         "mechanism."
                .format(page["L"], page["p"], iso["top"], iso["ci_lo"],
                        iso["ci_hi"]))
    return ("SPLIT", "the top rung is positive with a CI excluding zero "
                     "({:+.6f}, [{:+.6f}, {:+.6f}]) but the ordering across the "
                     "rungs is not significant (L={:.0f}, exact p={:.6f}). That "
                     "is a single-rung effect, which row B of "
                     "E_LADDER_PREREGISTERED_READING.md already names."
            .format(iso["top"], iso["ci_lo"], iso["ci_hi"], page["L"],
                    page["p"]))


def trend(blocks, *, labels=None, mc_draws: int = 200000) -> dict:
    """Both statistics, the exact and Monte Carlo p-values, and the verdict."""
    b = np.asarray(blocks, dtype=float)
    n, k = b.shape
    L, rank_sums, had_ties = page_l(b)
    exact = page_p_value(L, k, n)
    mc = page_p_value_monte_carlo(L, k, n, draws=mc_draws, seed=BOOT_SEED)
    support, dist = page_null(k, n)
    page = {"critical": critical_value(k, n, ALPHA),
            "L": L, "p": exact, "rank_sums": rank_sums.tolist(),
            "had_ties": had_ties, "k": k, "n": n,
            "null_mean": float(support @ dist),
            "null_min": float(support[0]), "null_max": float(support[-1]),
            "assignments": math.factorial(k) ** n,
            "monte_carlo": mc,
            "paths_agree": abs(exact - mc["p"]) < 4.0 * max(mc["se"], 1e-12)}
    iso = isotonic_top_ci(b)
    row, sentence = verdict_of(page, iso)
    stability = leave_one_block_out(b)
    return {"page": page, "isotonic": iso, "row": row, "sentence": sentence,
            "stability": stability,
            "stability_rises": sum(1 for s in stability if s["rises"]),
            "labels": list(labels) if labels else list(range(k)),
            "means": b.mean(axis=0).tolist(), "blocks": b.tolist()}


def from_ladder(journal=None, **cfg) -> dict:
    """The statistic on the real ladder. Reads `scale.e_ladder.read`, which
    reads `results/m3_quintuple_v2.jsonl`; nothing is trained here and no
    number is recomputed from weights."""
    cur = read(journal, **cfg)
    rows = {r["task"]: r for r in cur["ladder"]}
    missing = [t for t in RUNGS if rows.get(t, {}).get("state") != "RUN"]
    if missing:
        raise ValueError(f"the ladder is partial at {missing}; Page's test "
                         "quantifies over every rung and cannot be taken on a "
                         "subset without becoming the rung-picking it replaces")
    blocks = np.array([[rows[t]["per_seed_delta"][i] for t in RUNGS]
                       for i in range(len(cur["seeds"]))])
    out = trend(blocks, labels=RUNGS)
    out["seeds"] = cur["seeds"]
    out["t_star"] = [rows[t]["t_star"] for t in RUNGS]
    # The ladder's OWN pre-registration outranks this statistic. Row G of
    # E_LADDER_PREREGISTERED_READING.md credits a rung NOTHING in either
    # direction when either cell sits at or above predict-the-mean, and a trend
    # in the difference between two arms that both lose to the mean is an
    # ordering, not a capability. This travels with the verdict rather than
    # being left for a reader to notice.
    out["n_plus_top"] = rows[RUNGS[-1]]["seeds_favouring_settled"]
    out["uncredited"] = [t for t in RUNGS if not rows[t].get("credited")]
    out["all_credited"] = bool(cur["all_credited"])
    out["cell_means"] = {t: {"settled": rows[t]["settled_mean"],
                             "twin": rows[t]["twin_mean"]} for t in RUNGS}
    return out


def report(journal=None, **cfg) -> str:
    cur = from_ladder(journal, **cfg)
    p, iso = cur["page"], cur["isotonic"]
    w = [].append
    lines: list[str] = []
    w = lines.append
    w("=" * 78)
    w("LADDER E TREND -- Page's L and isotonic regression on settled - twin")
    w("=" * 78)
    w("    convention: delta = mean(NRMSE_twin - NRMSE_settled). "
      "POSITIVE MEANS SETTLED WINS.")
    w(f"    blocks = seeds {cur['seeds']}   conditions = t* {cur['t_star']}")
    w("    pre-registration is in this file's docstring and was frozen before "
      "the statistic ran.")
    w("")
    w("    seed  " + "".join(f"{t:>14}" for t in cur["labels"]))
    for i, sd in enumerate(cur["seeds"]):
        w(f"    {sd:<6}" + "".join(f"{v:>+14.6f}" for v in cur["blocks"][i]))
    w("    mean  " + "".join(f"{v:>+14.6f}" for v in cur["means"]))
    w("    iso   " + "".join(f"{v:>+14.6f}" for v in iso["fit"]))
    w("")
    w("PAGE'S TREND TEST (ordered alternative theta_1 <= ... <= theta_k)")
    w(f"    rank sums R_j        {['%.1f' % v for v in p['rank_sums']]}")
    w(f"    L = sum_j j * R_j    {p['L']:.0f}")
    w(f"    exact null           support [{p['null_min']:.0f}, "
      f"{p['null_max']:.0f}], mean {p['null_mean']:.1f}, "
      f"{p['assignments']} rank assignments")
    w(f"    exact p (L >= obs)   {p['p']:.6f}")
    cv = p["critical"]
    w(f"    GRANULARITY          {cv['n_achievable']} achievable p-values on the "
      f"whole support; finest {cv['finest_p']:.3e}")
    w(f"    alpha={ALPHA:.2f} reachable  {cv['reachable']}  -- critical L = "
      f"{cv['L_crit']}, TRUE size {cv['effective_size']:.9f}, "
      f"{cv['n_below_alpha']} achievable p-values at or below alpha")
    w(f"    permutation p        {p['monte_carlo']['p']:.6f}  "
      f"(se {p['monte_carlo']['se']:.6f}, {p['monte_carlo']['draws']} draws, "
      f"seed {p['monte_carlo']['seed']})")
    w(f"    two paths agree      {p['paths_agree']}")
    w(f"    ties present         {p['had_ties']}"
      f"{'' if not p['had_ties'] else '   -- the exact p-value is VOID'}")
    w("")
    w("ISOTONIC REGRESSION (PAVA, non-decreasing, on the rung means)")
    w(f"    top rung             {iso['top']:+.6f}")
    w(f"    bootstrap CI         [{iso['ci_lo']:+.6f}, {iso['ci_hi']:+.6f}]  "
      f"({iso['n_boot']} draws over {len(cur['seeds'])} seeds, seed "
      f"{iso['seed']}, {iso['distinct_atoms']} distinct resamples)")
    w(f"    excludes zero        {iso['excludes_zero']}")
    w(f"    seeds favouring settled at the top rung: {cur['n_plus_top']} of "
      f"{len(cur['seeds'])}")
    w("")
    w("    GRANULARITY OF THIS CLAUSE, WHICH IS NOT PAGE'S")
    w(f"    A percentile CI over {len(cur['seeds'])} blocks is driven by sign "
      f"agreement, so the finest")
    w(f"    two-sided p it can express is 2/2^{len(cur['seeds'])} = "
      f"{bootstrap_p_floor(len(cur['seeds'])):.4f} -- ABOVE alpha={ALPHA:.2f}.")
    w("    This clause therefore CANNOT be a 0.05-level statement at this seed")
    w("    count, whatever its interval reads. Page's L is not subject to it:")
    w(f"    its outcome space is {math.factorial(p['k'])}^{p['n']} = "
      f"{p['assignments']}, not 2^{p['n']} = {2 ** p['n']}.")
    w("")
    w("    WHAT THE CONSTRAINT IS WORTH -- the same bootstrap WITHOUT isotonic")
    w(f"    unconstrained top    {iso['raw_top']:+.6f}  "
      f"CI [{iso['raw_ci_lo']:+.6f}, {iso['raw_ci_hi']:+.6f}]  "
      f"excludes zero {iso['raw_excludes_zero']}")
    w(f"    PAVA pooled the top in {iso['pooled_fraction'] * 100:.2f}% of "
      f"resamples and lifted the lower bound by {iso['constraint_lift']:+.6f}.")
    w("    The pooling is ONE-SIDED, so this lift is a property of the "
      "estimator, not of")
    w("    the data. Read the size clause knowing that.")
    w("")
    w("LEAVE-ONE-SEED-OUT (diagnostic only; changes no branch)")
    w("    dropped       L         p        iso top        CI          RISES")
    for s in cur["stability"]:
        w(f"    seed {s['dropped']}      {s['L']:>4.0f}  {s['p']:>8.6f}  "
          f"{s['top']:>+9.6f}  [{s['ci_lo']:+.6f},{s['ci_hi']:+.6f}]  "
          f"{s['rises']}")
    w(f"    the verdict survives {cur['stability_rises']} of "
      f"{len(cur['stability'])} single-seed deletions.")
    w("")
    w("SCOPE -- LADDER E's OWN ROW G OUTRANKS THIS STATISTIC")
    w("    rung      settled       twin   credited")
    for t in cur["labels"]:
        m = cur["cell_means"][t]
        w(f"    {t:<8}{m['settled']:>9.6f}  {m['twin']:>9.6f}   "
          f"{t not in cur['uncredited']}")
    if cur["uncredited"]:
        w(f"    {len(cur['uncredited'])} of {len(cur['labels'])} rungs have a "
          f"cell AT OR ABOVE predict-the-mean: {', '.join(cur['uncredited'])}.")
        w("    Row G credits those rungs NOTHING in either direction. A trend in")
        w("    the difference between two arms that both lose to the mean is an")
        w("    ORDERING, not a capability. The verdict below is a statement about")
        w("    the contrast as a number and NOT a claim that settling bought")
        w("    anything at depth.")
    w("")
    w("=== PRE-REGISTERED VERDICT ===")
    w(f"    {cur['row']}: {cur['sentence']}")
    return "\n".join(lines)


def demo() -> None:
    """Self-checks. Every one fails if the corresponding logic is deleted."""
    # the exact null is a probability distribution on the right support
    support, dist = page_null(4, 5)
    assert abs(dist.sum() - 1.0) < 1e-12
    assert support[0] == 100 and support[-1] == 150
    assert abs(float(support @ dist) - 125.0) < 1e-9
    # a perfectly increasing table maximises L and is significant
    perfect = np.array([[1.0, 2.0, 3.0, 4.0]] * 5)
    L, _sums, ties = page_l(perfect)
    assert L == 150 and not ties
    assert page_p_value(L, 4, 5) < 1e-6
    # a perfectly DECREASING table minimises L and is as far from significant
    # as the statistic can get -- the one-sidedness is not decorative
    L_down, _s, _t = page_l(perfect[:, ::-1])
    assert L_down == 100
    assert abs(page_p_value(L_down, 4, 5) - 1.0) < 1e-9
    # ties are detected rather than silently midranked into an exact p-value
    _L, _s, tied = page_l(np.array([[1.0, 1.0, 2.0, 3.0]] * 5))
    assert tied
    # isotonic is a constrained fit, not a smoother: already-monotone data is
    # returned unchanged, and a violation is pooled to its weighted mean
    assert np.allclose(isotonic_curve([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])
    assert np.allclose(isotonic_curve([3.0, 1.0]), [2.0, 2.0])
    # the branch table is exhaustive: all four clause states return a row
    seen = set()
    for tp in (0.001, 0.5):
        for pos in (True, False):
            row, _text = verdict_of(
                {"p": tp, "L": 139.0}, {"positive_excluding_zero": pos,
                                        "top": 0.1, "ci_lo": 0.01,
                                        "ci_hi": 0.2})
            seen.add(row)
    assert seen == {"RISES", "FLAT", "SPLIT"}
    # the two p-value paths agree on a table that is not degenerate
    rng = np.random.default_rng(7)
    noisy = rng.normal(size=(5, 4)) + np.arange(4) * 0.4
    L_n, _s, _t = page_l(noisy)
    exact = page_p_value(L_n, 4, 5)
    mc = page_p_value_monte_carlo(L_n, 4, 5, draws=200000, seed=0)
    assert abs(exact - mc["p"]) < 4.0 * max(mc["se"], 1e-9), (exact, mc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        demo()
        print("page_trend self-checks pass")
    elif args.json:
        print(json.dumps(from_ladder(args.journal), indent=2, sort_keys=True))
    else:
        print(report(args.journal))
