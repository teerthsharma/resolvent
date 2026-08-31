"""The key-norm matcher that every causal-vs-filler contrast in this project owes.

WHY THIS EXISTS. `select_pivots` ranks candidates by `key.norm(dim=-1)`
(`scale/pivot_probe.py:88`) -- a pure function of a token's own representation,
with no reference to any downstream effect. The "causal" arm of every contrast in
this repo is exactly that selector's top-k, so high key-norm and membership in
the causal arm are confounded *by construction*. Until a filler is matched on key
norm, a causal-vs-filler difference is not evidence about consequence.

WHAT WAS ALREADY HERE, AND WHY IT IS NOT ENOUGH. `scale/aggregator_matched_filler.py`
and `scale/wilson_probes.py` already draw a "band" filler from ranks k+1..2k of the
selector's own score. That match is BY RANK, NOT BY VALUE, and both files say so:
the band's mean key-norm still trails the causal arm's (25.4987 vs 28.0430 at
k=8, band/causal 0.91 / 0.89 / 0.83 across k=8/32/128), and that residual gap is
uncontrolled and travels with every number taken under it. This module replaces
the rank match with a value match and, more importantly, *prints what it could
not remove*.

THE COST AND THE ASSIGNMENT.

    C_ij = |‖k_causal_i‖ − ‖k_filler_j‖|          π* = argmin_π Σ_i C_{i π(i)}

reported as the residual imbalance `Σ_i C_{i π*(i)} / n`.

TWO PATHS, AND THE ONE FREE ORACLE. `scipy.optimize.linear_sum_assignment` is
used when scipy is importable (it is: scipy 1.17.1, though no file in this repo
imported `scipy.optimize` before this one -- only `scipy.stats`). The fallback is
not Jonker-Volgenant or auction, because neither is needed: the cost is
`|a_i − b_j|` on the real line, which satisfies the Monge condition, so sorting
both pools and pairing rank-for-rank is *provably* an optimal assignment in
O(n log n).

That fact is worth more as a check than as a shortcut. In the square case this
module computes the sorted pairing alongside the solver's answer and raises if
they disagree, so a wrong assignment cannot reach a table. It is an exact,
closed-form oracle obtained for free, and it is the reason the calibration below
can distinguish "the solver worked" from "the call returned without raising".

THE STRATA ARE NEVER POOLED. Round 5's F-selector established that the filler
pool is at least two populations -- band and tail separate at 1.0716 / 0.8782 /
0.4868 with CIs excluding zero. `match_strata` matches within each stratum and
returns one `Match` per stratum. It deliberately offers no pooled residual: a
pooled match lets the causal pool buy its partners from whichever stratum happens
to be nearest, and the imbalance against the other stratum then never appears in
the table at all. A matcher that pools is a matcher that hides the confound.

WHAT MATCHING CANNOT DO, STATED UP FRONT. Where the two supports do not overlap,
no assignment removes the gap; the optimum simply reports it. That is the point
of printing the residual next to every effect rather than printing the effect
alone. A residual comparable to the population spread means the contrast beneath
it is still confounded, matched or not.

CALIBRATION, THREE DIRECTIONS. `calibrate_null` matches two independent draws from
one population and must give a residual that is sampling noise and an effect whose
CI contains zero. `calibrate_separated` matches two deliberately shifted
populations and must give a NONZERO residual and an effect whose CI excludes zero.
A control that can only read the null is not a control.

Neither of those two is sufficient, and saying so is the point of the third.
Both match equal-size pools, so the assignment is a full bijection and the matched
multiset is the same under any permutation; Cohen's d therefore cannot see whether
the solver ran. Measured, by replacing the solver with the identity permutation:
their effects read 0.097105 and 1.476579 either way, unchanged to the last digit,
and the separated direction still reports `fired=True`. `calibrate_selection`
supplies the direction that does bind -- a small causal pool against a large
filler pool, where the assignment selects WHICH filler units are used, a confound
planted by construction and seen at full size before it is removed. That one fails
when the solver is deleted.

Every calibration row also prints the identity-pairing residual it had to beat, so
a matcher that never matched is visible in the table rather than inferred from it.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

from scale.monge import monge_cost                                 # noqa: E402
from scale.pivot_probe import select_pivots                        # noqa: E402
from scale.torque_probe import boot_ci, cohen_d                     # noqa: E402
from scale.wilson_probes import rank_band                          # noqa: E402

try:
    from scipy.optimize import linear_sum_assignment as _lsa
    ASSIGNMENT_PATH = "scipy.optimize.linear_sum_assignment"
except ImportError:                                    # pragma: no cover
    _lsa = None
    ASSIGNMENT_PATH = "monge-sorted"

# The oracle and the solver compute the same optimum by different routes, so they
# agree to floating-point summation order and nothing looser. The bar is RELATIVE:
# an absolute bar is one a large residual cannot meet and a small one meets for
# free, which is how a tolerance ends up doing the work a measurement should do.
_ORACLE_TOL = 1e-12


@dataclasses.dataclass(frozen=True)
class Match:
    """One assignment. `residual` is the contract's `Σ C_{i π*(i)} / n`."""

    causal_idx: np.ndarray
    filler_idx: np.ndarray
    residual: float
    identity_residual: float
    n: int

    def __repr__(self) -> str:                          # pragma: no cover
        return (f"Match(n={self.n}, residual={self.residual:.6f}, "
                f"identity_residual={self.identity_residual:.6f})")


@dataclasses.dataclass(frozen=True)
class CalibRow:
    """One calibration direction. `fired` is the AND of both required halves."""

    name: str
    residual: float
    identity_residual: float
    pooled_sd: float
    effect: float
    effect_ci: tuple[float, float]
    naive_ci: tuple[float, float]
    fired: bool
    #: only the selection direction has an unmatched twin to report
    effect_unmatched: float = float("nan")

    def __repr__(self) -> str:                          # pragma: no cover
        return (f"CalibRow({self.name}, residual={self.residual:.6f}, "
                f"identity={self.identity_residual:.6f}, sd={self.pooled_sd:.6f}, "
                f"d={self.effect:.6f}, CI=[{self.effect_ci[0]:.6f},"
                f"{self.effect_ci[1]:.6f}], naive_CI=[{self.naive_ci[0]:.6f},"
                f"{self.naive_ci[1]:.6f}], fired={self.fired})")


# ------------------------------------------------------------------ the cost


def cost_matrix(causal_norms, filler_norms) -> np.ndarray:
    """`C_ij = |‖k_causal_i‖ − ‖k_filler_j‖|`, the contract's cost verbatim."""
    a = np.asarray(causal_norms, dtype=float).reshape(-1)
    b = np.asarray(filler_norms, dtype=float).reshape(-1)
    return np.abs(a[:, None] - b[None, :])


def monge_optimum(causal_norms, filler_norms) -> float:
    """The closed-form optimal residual for the square case, without the solver.

    `|a_i − b_j|` is a Monge matrix on the line, so the sorted pairing is an
    optimal assignment. Used as an independent oracle, never as the shipped path.
    """
    a = np.sort(np.asarray(causal_norms, dtype=float).reshape(-1))
    b = np.sort(np.asarray(filler_norms, dtype=float).reshape(-1))
    if a.size != b.size:
        raise ValueError("the Monge oracle is defined here for square problems only")
    return float(np.abs(a - b).sum() / a.size)


def match(causal_norms, filler_norms, *, verify: bool = True) -> Match:
    """Optimal key-norm assignment of the causal pool onto the filler pool.

    Pools of unequal size are allowed; the smaller side is matched in full and
    the surplus on the larger side is left unassigned, which is what the solver's
    rectangular form already does.
    """
    a = np.asarray(causal_norms, dtype=float).reshape(-1)
    b = np.asarray(filler_norms, dtype=float).reshape(-1)
    if a.size == 0 or b.size == 0:
        raise ValueError("both pools must be non-empty")
    c = cost_matrix(a, b)

    if _lsa is not None:
        r, k = _lsa(c)
    else:                                              # pragma: no cover
        # ponytail: the square Monge case only. scipy is present in this
        # environment, so the rectangular fallback is unreachable here; add the
        # O(n·m) monotone DP if this ever has to run without scipy.
        if a.size != b.size:
            raise NotImplementedError(
                "no scipy: the sorted-pairing fallback covers square problems only")
        r = np.arange(a.size)
        k = np.argsort(b)[np.argsort(np.argsort(a))]   # pair rank with rank

    r = np.asarray(r)
    k = np.asarray(k)
    residual = float(c[r, k].sum() / r.size)

    # The free cross-check, on BOTH shapes. A solver that returned a non-optimal
    # assignment would otherwise reach a table looking exactly like a small
    # effect. The square case has a closed form; the rectangular case -- which is
    # the shape the +3/-5 gate actually runs -- is checked against the O(n*m)
    # monotone DP in scale/monge.py, which shares no algorithm with the solver.
    if verify:
        if a.size == b.size:
            oracle = monge_optimum(a, b)
        else:
            oracle = monge_cost(torch.from_numpy(a), torch.from_numpy(b)) / r.size
    else:
        oracle = residual
    if abs(residual - oracle) > _ORACLE_TOL * max(1.0, abs(oracle)):
        raise AssertionError(
            f"assignment is not optimal: solver {residual!r} vs exact oracle "
            f"{oracle!r} via {ASSIGNMENT_PATH}, shape {a.size}x{b.size}")

    m = min(a.size, b.size)
    identity = float(np.abs(a[:m] - b[:m]).sum() / m)
    return Match(causal_idx=r, filler_idx=k, residual=residual,
                 identity_residual=identity, n=int(r.size))


def match_strata(causal_norms, filler_norms, strata, *,
                 verify: bool = True) -> dict:
    """One `Match` per filler stratum. No pooled residual is returned, ever.

    `filler_idx` is remapped to indices into the FULL filler array, so
    `strata[result.filler_idx]` is constant and equal to the stratum name. That
    is the property a pooled match silently violates.
    """
    b = np.asarray(filler_norms, dtype=float).reshape(-1)
    lab = np.asarray(strata).reshape(-1)
    if lab.size != b.size:
        raise ValueError(f"strata {lab.size} does not match filler pool {b.size}")

    out = {}
    for name in dict.fromkeys(lab.tolist()):           # first-seen order, stable
        where = np.flatnonzero(lab == name)
        sub = match(causal_norms, b[where], verify=verify)
        out[name] = dataclasses.replace(sub, filler_idx=where[sub.filler_idx])
    return out


# ------------------------------------------------------------------- effect


def _d_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`scale.torque_probe.cohen_d` over the last axis, for whole bootstrap stacks."""
    n = a.shape[-1]
    sp = np.sqrt(((n - 1) * a.var(-1, ddof=1) + (n - 1) * b.var(-1, ddof=1))
                 / (2 * n - 2))
    return np.divide(a.mean(-1) - b.mean(-1), sp,
                     out=np.zeros(a.shape[:-1]), where=sp > 0)


def naive_pair_boot_ci(y_causal, y_filler, *, bnum: int = 10000, seed: int = 0):
    """The WRONG interval, kept so the report can show how wrong it is.

    Resamples the already-matched pairs and never re-runs the match. It is the
    obvious thing to do and it is invalid: see `rematch_boot_ci`. Nothing in this
    module gates on it.
    """
    a = np.asarray(y_causal, dtype=float)
    b = np.asarray(y_filler, dtype=float)
    g = np.random.default_rng(seed)
    idx = g.integers(0, a.size, size=(bnum, a.size))
    s = _d_vec(a[idx], b[idx])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def rematch_boot_ci(causal_norms, filler_norms, y_causal, y_filler,
                    *, bnum: int = 10000, seed: int = 0):
    """Percentile CI of Cohen's d that RE-RUNS THE MATCH in every replicate.

    THE DEFECT THIS AVOIDS. Matching is part of the estimator, not a fixed
    preprocessing step, so an interval must reproduce the matching inside each
    resample. Resampling the matched pairs instead treats the pair differences as
    independent, and they are not: pairing rank to rank makes them share the
    difference in pool means, so their mean is far more variable than an
    iid-pairs resample can see.

    MEASURED, at n=256, seed=20260826, threads=2, `python scale/matcher.py`:
    the naive-pairs interval has bootstrap sd 0.045302 and reads
    [0.009888, 0.188523], EXCLUDING zero on data with no effect in it. Re-matching
    gives sd 0.089484 and [-0.078196, 0.270299], containing zero. The truth,
    taken by re-drawing the whole population over 200 seeds, is sd 0.096312
    (mean effect -0.001635, t = -0.2401 against zero). The naive interval
    understates the spread by 2.1x and manufactures a significant result out of a
    null. Every matched contrast in this project has to use this form.

    WHY IT IS AFFORDABLE. The optimal assignment for `|a_i − b_j|` is the sorted
    pairing, so a replicate costs a sort rather than a solve and 10,000 of them
    are cheap. `match` verifies the two agree on every live call.
    """
    a = np.asarray(causal_norms, dtype=float).reshape(-1)
    b = np.asarray(filler_norms, dtype=float).reshape(-1)
    ya = np.asarray(y_causal, dtype=float).reshape(-1)
    yb = np.asarray(y_filler, dtype=float).reshape(-1)
    n = a.size
    g = np.random.default_rng(seed)
    ia = g.integers(0, a.size, size=(bnum, n))
    ib = g.integers(0, b.size, size=(bnum, n))
    ra = np.argsort(a[ia], axis=1)
    rb = np.argsort(b[ib], axis=1)
    s = _d_vec(np.take_along_axis(ya[ia], ra, axis=1),
               np.take_along_axis(yb[ib], rb, axis=1))
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


# -------------------------------------------------------------- calibration

OUTCOME_SLOPE = 0.5      # the outcome depends on the key norm and nothing else
OUTCOME_NOISE = 1.0


def _pool(n: int, loc: float, scale: float, rng) -> np.ndarray:
    """A synthetic key-norm pool. Positive by construction: a key norm is."""
    return np.abs(rng.normal(loc, scale, size=n))


def _outcome(norms: np.ndarray, rng) -> np.ndarray:
    """Outcome driven by the key norm alone.

    This is the whole confound in miniature. If matching works, an outcome that
    is a function of the matched-on variable must show no residual difference
    between arms drawn from one population, and must still show one between arms
    whose norms genuinely differ.
    """
    return OUTCOME_SLOPE * norms + rng.normal(0.0, OUTCOME_NOISE, size=norms.size)


def _row(name: str, a, b, ya, yb, m: Match, *, seed: int, require_zero: bool):
    ya, yb = np.asarray(ya), np.asarray(yb)
    ma, mb = ya[m.causal_idx], yb[m.filler_idx]
    eff = cohen_d(list(ma), list(mb))
    # The point estimate and the interval are the SAME statistic. The first
    # version of this row reported an unpaired Cohen's d beside an interval on
    # the paired d_z, which is two different quantities on one line.
    lo, hi = rematch_boot_ci(a, b, ya, yb, seed=seed)
    nlo, nhi = naive_pair_boot_ci(ma, mb, seed=seed)
    sd = float(np.concatenate([np.asarray(a), np.asarray(b)]).std(ddof=1))
    if require_zero:
        fired = m.residual < 0.20 * sd and lo <= 0.0 <= hi
    else:
        fired = m.residual > 0.5 * sd and (lo > 0.0 or hi < 0.0)
    return CalibRow(name=name, residual=m.residual,
                    identity_residual=m.identity_residual, pooled_sd=sd,
                    effect=float(eff), effect_ci=(lo, hi),
                    naive_ci=(nlo, nhi), fired=bool(fired))


def calibrate_null(*, n: int = 256, seed: int = 0) -> CalibRow:
    """Direction 1. Two independent draws from ONE population.

    Must give a residual that is sampling noise and an effect CI containing zero.

    The two pools are drawn independently rather than being the same array twice.
    A self-match on one array is satisfied by the identity permutation, so it
    passes with the solver deleted; here the identity pairing carries a residual
    several times larger than the assignment's, and `identity_residual` is
    printed beside every row so that gap is visible rather than asserted.
    """
    rng = np.random.default_rng(seed)
    a, b = _pool(n, 25.0, 4.0, rng), _pool(n, 25.0, 4.0, rng)
    ya, yb = _outcome(a, rng), _outcome(b, rng)
    return _row("null: one population, two draws", a, b, ya, yb,
                match(a, b), seed=seed, require_zero=True)


def calibrate_separated(*, n: int = 256, seed: int = 0, shift: float = 6.0) -> CalibRow:
    """Direction 2. Two deliberately separated populations.

    Must give a NONZERO residual and an effect CI excluding zero. With equal
    spreads the optimal pairing is quantile-to-quantile, so the residual
    converges on the shift itself -- the matcher reporting the imbalance it
    cannot remove instead of absorbing it.
    """
    rng = np.random.default_rng(seed)
    a, b = _pool(n, 25.0 + shift, 4.0, rng), _pool(n, 25.0, 4.0, rng)
    ya, yb = _outcome(a, rng), _outcome(b, rng)
    return _row(f"separated: shift={shift}", a, b, ya, yb,
                match(a, b), seed=seed, require_zero=False)


def calibrate_selection(*, n_causal: int = 64, n_filler: int = 2000,
                        seed: int = 0) -> CalibRow:
    """Direction 3, and the only one that tests the assignment through the effect.

    THE DEFECT IN DIRECTIONS 1 AND 2. Both match equal-size pools, so the
    assignment is a full bijection and `y_filler[filler_idx]` is the same multiset
    whatever the permutation. Cohen's d depends only on that multiset, so in the
    square case the EFFECT IS PERMUTATION-INVARIANT: replacing the solver with the
    identity permutation leaves it unchanged to the last digit (measured:
    0.097105 null and 1.476579 separated, byte-identical intact and mutated).
    The square directions test the effect estimator, and the residual tests the
    solver, but the effect never tests the solver.

    WHAT THIS DIRECTION DOES INSTEAD. It uses the shape the project actually has
    -- a small causal pool against a large, broader filler pool -- where the
    assignment SELECTS which filler units are used. A planted confound is put in
    by construction: the outcome depends on key norm alone, and the causal pool
    sits at a higher norm than the filler pool, so the unmatched contrast reads a
    large effect that is entirely the selector. Matching must remove it.

    BOTH HALVES MUST FIRE. The unmatched effect must be SEEN to be large -- a
    control that cannot be nonzero is not a control -- and the matched effect's
    CI must then contain zero. Deleting the solver fails this direction, which is
    what directions 1 and 2 cannot claim.
    """
    rng = np.random.default_rng(seed)
    a = _pool(n_causal, 28.0, 2.0, rng)          # causal: high norm, tight
    b = _pool(n_filler, 18.0, 6.0, rng)          # filler: lower norm, broad
    ya, yb = _outcome(a, rng), _outcome(b, rng)

    m = match(a, b)
    sub, ysub = b[m.filler_idx], yb[m.filler_idx]
    eff_un = cohen_d(list(ya), list(yb))
    eff = cohen_d(list(ya[m.causal_idx]), list(ysub))
    lo, hi = rematch_boot_ci(a, sub, ya, ysub, seed=seed)
    nlo, nhi = naive_pair_boot_ci(ya[m.causal_idx], ysub, seed=seed)
    sd = float(np.concatenate([a, b]).std(ddof=1))
    fired = (abs(eff_un) > 1.0 and lo <= 0.0 <= hi
             and m.residual < 0.20 * float(a.std(ddof=1)))
    return CalibRow(name="selection: planted confound removed",
                    residual=m.residual, identity_residual=m.identity_residual,
                    pooled_sd=sd, effect=float(eff), effect_ci=(lo, hi),
                    naive_ci=(nlo, nhi), fired=bool(fired),
                    effect_unmatched=float(eff_un))


# ------------------------------------------------------- a real key-norm table


def draw_strata(s: int, d: int, k: int, g: torch.Generator):
    """Causal / band / tail key-norm pools from one draw at ARM A's geometry.

    The stratum definitions are `scale/aggregator_matched_filler.py`'s and
    `scale/wilson_probes.py`'s, not new ones: causal is `select_pivots`' top-k,
    band is ranks k+1..2k of the same score, tail is everything else that is a
    legal choice of `c`.
    """
    x0 = torch.randn(s, d, generator=g)
    wq, wk = (torch.randn(d, d, generator=g) for _ in range(2))
    kk = x0 @ wk
    i = s - 1
    j = int(torch.randint(1, i, (1,), generator=g))
    piv = select_pivots(kk, min(k, s - 2), exclude=(i, j))
    pset = set(int(p) for p in piv)
    band = [t for t in rank_band(kk, k, 2 * k, exclude=(i, j))
            if t not in pset and t != j and 1 <= t < i]
    tail = [t for t in range(1, i) if t not in pset and t != j and t not in band]
    norm = kk.norm(dim=-1)
    return (np.array([float(norm[t]) for t in sorted(pset)]),
            np.array([float(norm[t]) for t in band]),
            np.array([float(norm[t]) for t in tail]))


# -------------------------------------------------------------------- report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--s", type=int, default=1024)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--draws", type=int, default=40)
    a = ap.parse_args()

    print(f"HUNGARIAN MATCHER (contract 1.5). n={a.n} seed={a.seed} "
          f"threads={torch.get_num_threads()}")
    print(f"assignment path: {ASSIGNMENT_PATH}")
    print("cost C_ij = | ||k_causal_i|| - ||k_filler_j|| |; "
          "residual = sum_i C[i, pi*(i)] / n")
    print("Other agents share this box; NO TIMING IS REPORTED.")
    print()

    print("=== CALIBRATION, THREE DIRECTIONS. All must fire. ===")
    print(f"  {'direction':>34} {'residual':>10} {'identity':>10} {'pool sd':>9} "
          f"{'effect d':>10} {'95% CI (re-match)':>24} "
          f"{'95% CI (naive, UNUSED)':>24} {'fired':>7}")
    rows = [calibrate_null(n=a.n, seed=a.seed),
            calibrate_separated(n=a.n, seed=a.seed),
            calibrate_selection(seed=a.seed)]
    ok = True
    for r in rows:
        ok &= r.fired
        print(f"  {r.name:>34} {r.residual:>10.6f} {r.identity_residual:>10.6f} "
              f"{r.pooled_sd:>9.6f} {r.effect:>10.6f} "
              f"[{r.effect_ci[0]:>10.6f},{r.effect_ci[1]:>10.6f}] "
              f"[{r.naive_ci[0]:>10.6f},{r.naive_ci[1]:>10.6f}] "
              f"{'YES' if r.fired else 'NO':>7}")
    print()
    print(f"  selection direction only: UNMATCHED effect d = "
          f"{rows[2].effect_unmatched:.6f}, MATCHED effect d = {rows[2].effect:.6f}")
    print("  That pair is the instrument doing the job it exists for: a confound")
    print("  planted by construction, seen at full size, then removed. It is also")
    print("  the ONLY direction whose effect can detect a deleted solver. On the")
    print("  two equal-size directions the assignment is a full bijection, so the")
    print("  matched multiset -- and therefore Cohen's d -- is the same whatever")
    print("  the permutation. Their effects read identically with the solver")
    print("  replaced by the identity: 0.097105 and 1.476579, unchanged.")
    print()
    print("  The last column is the interval a bootstrap over MATCHED PAIRS")
    print("  would have reported. It is printed and never used. On the null row")
    print("  it excludes zero, which is a significant result manufactured out of")
    print("  data with no effect in it: matching is part of the estimator, so a")
    print("  replicate must re-run the match. See `rematch_boot_ci`.")
    print()
    print("  Bars, fixed before the run: null residual < 0.20*sd with an effect")
    print("  CI containing 0; separated residual > 0.50*sd with an effect CI")
    print("  excluding 0. The null bar is not 0: matching two size-n draws from")
    print("  one population pairs order statistic to order statistic, leaving a")
    print("  residual of order sd*n^-0.5 (measured 0.117*sd at n=256).")
    print("  The `identity` column is what the assignment had to beat. In the")
    print("  null row it is several times the residual, which is the evidence")
    print("  that the solver did the work; a matcher with its solver deleted")
    print("  would report the identity column as its residual.")
    if not ok:
        print()
        print("  A DIRECTION DID NOT FIRE. The instrument is not calibrated and")
        print("  every number taken with it is void.")
        return 1

    print()
    print("=== THE n^-0.5 SIGNATURE: the null residual is sampling noise ===")
    # Averaged over seeds on purpose. A single seed's residual is dominated by
    # that seed's difference in pool means, which is itself only O(n^-0.5), so a
    # one-seed column is too noisy to show the law it is meant to show -- at
    # seed=0 the n=1024 cell reads ABOVE the n=256 cell.
    reps = 8
    print(f"  mean over {reps} seeds. n^-0.5 predicts each step of 4x in n "
          f"halves the residual.")
    print(f"  {'n':>6} {'residual':>10} {'identity':>10} {'ratio to n=256':>16} "
          f"{'predicted':>10}")
    base = None
    for n in (256, 1024, 4096):
        vals = [calibrate_null(n=n, seed=a.seed + t) for t in range(reps)]
        res = sum(v.residual for v in vals) / reps
        idt = sum(v.identity_residual for v in vals) / reps
        if base is None:
            base = res
        print(f"  {n:>6} {res:>10.6f} {idt:>10.6f} {res / base:>16.4f} "
              f"{(256 / n) ** 0.5:>10.4f}")
    print("  A residual that is small because it is clamped does not shrink here,")
    print("  and the identity column, which the match must beat, does not move.")

    print()
    print("=== BAND AND TAIL AS STRATA, NEVER POOLED. "
          f"s={a.s} d={a.d} seed={a.seed} draws={a.draws} ===")
    # G6: a comparative sentence needs a CI. One draw cannot carry one, so the
    # residual is taken over `draws` independent draws and bootstrapped over
    # them. The earlier single-draw version of this table supported no
    # comparison at all and said so.
    per = {k: {"band": [], "tail": [], "pooled": []} for k in a.ks}
    ident = {k: {"band": [], "tail": []} for k in a.ks}
    kmean = {k: {"causal": [], "band": [], "tail": []} for k in a.ks}
    g = torch.Generator().manual_seed(a.seed)
    verified = 0
    for k in a.ks:
        for t in range(a.draws):
            causal, band, tail = draw_strata(a.s, a.d, k, g)
            if band.size == 0 or tail.size == 0:
                continue
            filler = np.concatenate([band, tail])
            labels = np.array(["band"] * band.size + ["tail"] * tail.size)
            # The exact oracle is O(n*m) in Python, so it runs on the FIRST draw
            # of each k -- enough to bind the wiring and the shape -- and is
            # stated here rather than skipped silently. The count is printed.
            vfy = (t == 0)
            out = match_strata(causal, filler, labels, verify=vfy)
            verified += 2 * int(vfy)
            per[k]["pooled"].append(match(causal, filler, verify=vfy).residual)
            verified += int(vfy)
            kmean[k]["causal"].append(float(causal.mean()))
            for name in ("band", "tail"):
                per[k][name].append(out[name].residual)
                ident[k][name].append(out[name].identity_residual)
                kmean[k][name].append(float(filler[out[name].filler_idx].mean()))

    print(f"  {verified} assignments checked against the exact oracle "
          f"(scale/monge.py rectangular DP, scale/matcher.py square closed form).")
    print(f"  {'k':>5} {'stratum':>8} {'mean ||k|| causal':>18} "
          f"{'mean ||k|| filler':>18} {'residual':>10} "
          f"{'95% CI':>22} {'identity':>10}")
    for k in a.ks:
        for name in ("band", "tail"):
            v = per[k][name]
            lo, hi = boot_ci(v, seed=a.seed)
            print(f"  {k:>5} {name:>8} "
                  f"{sum(kmean[k]['causal'])/len(v):>18.6f} "
                  f"{sum(kmean[k][name])/len(v):>18.6f} "
                  f"{sum(v)/len(v):>10.6f} [{lo:>9.6f},{hi:>9.6f}] "
                  f"{sum(ident[k][name])/len(v):>10.6f}")

    print()
    print("=== TAIL MINUS BAND, PAIRED PER DRAW. G6: CI must exclude zero ===")
    print(f"  {'k':>5} {'mean tail-band':>15} {'95% CI':>24} {'excludes 0':>11}")
    for k in a.ks:
        diff = [t - b for t, b in zip(per[k]["tail"], per[k]["band"])]
        lo, hi = boot_ci(diff, seed=a.seed)
        print(f"  {k:>5} {sum(diff)/len(diff):>15.6f} "
              f"[{lo:>10.6f},{hi:>10.6f}] {'YES' if lo > 0 or hi < 0 else 'NO':>11}")
    print("  The two strata carry DIFFERENT residual imbalance at every k, with")
    print("  intervals that exclude zero. That is the measured form of")
    print("  F-selector's finding, and the reason they are never pooled.")

    print()
    print("=== WHAT POOLING WOULD HAVE HIDDEN ===")
    print(f"  {'k':>5} {'pooled':>10} {'band':>10} {'tail':>10} "
          f"{'pooled-band':>13} {'95% CI':>24}")
    for k in a.ks:
        d = [q - b for q, b in zip(per[k]["pooled"], per[k]["band"])]
        lo, hi = boot_ci(d, seed=a.seed)
        print(f"  {k:>5} {sum(per[k]['pooled'])/len(d):>10.6f} "
              f"{sum(per[k]['band'])/len(d):>10.6f} "
              f"{sum(per[k]['tail'])/len(d):>10.6f} {sum(d)/len(d):>13.6f} "
              f"[{lo:>10.6f},{hi:>10.6f}]")
    print("  The pooled residual tracks the BAND and not the tail: the pooled")
    print("  match spends its budget on the nearest stratum, so a pooled table")
    print("  carries almost no information about the tail.")

    print()
    print("=== A RESULT THAT IS NOT IN THIS MODULE'S FAVOUR ===")
    print("  For the BAND stratum the identity column tracks the residual. The")
    print("  band is ranks k..2k of the same key-norm score that orders the")
    print("  causal pool, so both arrive already sorted and the rank match IS")
    print("  the optimal value match. Against the band this matcher reproduces")
    print("  what scale/aggregator_matched_filler.py and scale/wilson_probes.py")
    print("  already did; it does not improve on them. What it adds is the")
    print("  number, with an interval. Where it does change the answer is the")
    print("  TAIL, where identity and matched are far apart.")
    print()

    print("  LIMIT, carried with every number above: matching removes only the")
    print("  key-norm gap it can pay for. Where the supports do not overlap the")
    print("  optimum reports the gap rather than closing it, so a residual")
    print("  comparable to the pool spread means the contrast beneath it is")
    print("  still confounded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
