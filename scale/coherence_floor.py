"""How far apart k random unit vectors in R^d actually sit, against three closed
forms that claim to say -- and a fourth that says something different from what
the author wrote down.

WHAT THIS IS. Four numbers are compared for k unit vectors in R^d: (a) the Welch
bound, the provably-optimal floor on the worst-case pairwise coherence achievable
by ANY code of k unit vectors, exactly 0 whenever k <= d because d orthonormal
directions cover k <= d vectors with no forced overlap; (b) the author's stated
closed form for the expected MAXIMUM pairwise coherence of k independent uniform
unit vectors, mu_jl(d, k) = sqrt(2 ln(k) / d), built on a union bound over k
events; (c) the same union bound done correctly -- the event being bounded is
"some PAIR exceeds t", and there are C(k, 2) pairs, not k, so the corrected form
is mu_pairs(d, k) = sqrt(2 ln(C(k, 2)) / d); (d) the expected MEAN absolute
pairwise coherence, which has a closed form exactly (a ratio of Gamma functions,
from the density of a single coordinate of a random unit vector being proportional
to (1 - t^2)^((d - 3) / 2) on [-1, 1]) and a large-d limit, sqrt(2 / (pi d)).

WHAT WAS MEASURED, on this machine, numpy 1.26.4, at d=256, k=16, seed=0,
20000 Monte Carlo trials of k fresh normalised-Gaussian unit vectors each:

    mu_welch(256, 16)          = 0.000000   (k <= d, exact zero, not a claim under test)
    mu_jl(256, 16)             = 0.147176   (the author's stated figure, ~0.147)
    mu_pairs(256, 16)          = 0.193397   (C(16, 2) = 120 pairs, not 16)
    mean_coherence_exact(256)  = 0.049917   (exact Gamma-function ratio)
    mean_coherence_limit(256)  = 0.049868   (large-d limit, agrees to 5e-5)

    Monte Carlo max_{i<j} |<u_i,u_j>|,  mean over 20000 trials = 0.174795
        95% CI [0.174460, 0.175131]
    Monte Carlo mean_{i<j} |<u_i,u_j>|, mean over 20000 trials = 0.049917
        95% CI [0.049869, 0.049964]

THE MEAN-COHERENCE FORMS ARE CONFIRMED. The Monte Carlo mean CI covers
mean_coherence_exact(256) with room on both sides, and the exact and large-d-limit
closed forms agree with each other to 5e-5 at d=256. Nothing about the mean claim
is in question.

THE MAX-COHERENCE CLOSED FORMS ARE BOTH WRONG, IN OPPOSITE DIRECTIONS, AND NEITHER
CI COVERS ITS TARGET. The measured Monte Carlo max sits at 0.174795, which is
outside BOTH 95% CIs' reach of BOTH candidate closed forms: it clears mu_jl's
0.147176 by 18.9% and falls short of mu_pairs's 0.193397 by 9.6%. The author's
figure (mu_jl = 0.147) undercounts because its union bound runs over the wrong
event count -- k = 16 events instead of the C(16, 2) = 120 actual pairs that can
each be the maximiser -- and correcting the count (mu_pairs) overshoots, because a
union-bound threshold is a high-probability upper bound on the tail, not an
estimate of the order statistic's mean, and at k=16 pairs the two are not close
enough to coincide. Do not read mu_pairs as "the corrected answer": it is closer
than mu_jl (9.6% vs 18.9%) and still not what a draw returns.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

__all__ = [
    "welch_bound", "jl_coherence", "pair_union_coherence",
    "mean_coherence_exact", "mean_coherence_limit", "monte_carlo", "compare",
    "report", "demo",
]

# Author's stated figure for mu_jl(256, 16), quoted verbatim for the honesty check.
AUTHOR_MU_JL = 0.147184


def welch_bound(d: int, k: int) -> float:
    """Optimal floor on the worst-case pairwise coherence of k unit vectors in R^d.
    Exactly 0 whenever k <= d (d orthonormal directions suffice)."""
    if k < 2:
        return 0.0
    return math.sqrt(max(0.0, (k - d) / (d * (k - 1))))


def jl_coherence(d: int, k: int) -> float:
    """The author's stated closed form: a union bound over k events. Left in place,
    uncorrected, so the honesty check in this module has something to check against."""
    if k < 1:
        return 0.0
    return math.sqrt(2.0 * math.log(k) / d) if k > 1 else 0.0


def pair_union_coherence(d: int, k: int) -> float:
    """The union bound done over the actual event count, C(k, 2) pairs, not k."""
    pairs = k * (k - 1) / 2.0
    if pairs < 1:
        return 0.0
    return math.sqrt(2.0 * math.log(pairs) / d)


def mean_coherence_exact(d: int) -> float:
    """E|<u,v>| for two independent uniform unit vectors in R^d, exact for any
    d >= 2: Gamma(d/2) / (sqrt(pi) Gamma((d+1)/2)), evaluated in log-space because
    the two Gamma values individually overflow long before their ratio does."""
    return math.exp(gammaln(d / 2.0) - gammaln((d + 1) / 2.0)) / math.sqrt(math.pi)


def mean_coherence_limit(d: int) -> float:
    """Large-d limit of mean_coherence_exact."""
    return math.sqrt(2.0 / (math.pi * d))


def monte_carlo(d: int, k: int, trials: int = 20000, seed: int = 0) -> dict:
    """Draw k normalised-Gaussian unit vectors in R^d, `trials` independent times,
    and report the sample mean (with a normal 95% CI on that mean, over trials) of
    both max_{i<j} |<u_i,u_j>| and mean_{i<j} |<u_i,u_j>|."""
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(k, k=1)
    max_vals = np.empty(trials)
    mean_vals = np.empty(trials)
    batch = 2000
    filled = 0
    while filled < trials:
        n = min(batch, trials - filled)
        g = rng.standard_normal((n, k, d))
        u = g / np.linalg.norm(g, axis=2, keepdims=True)
        gram = np.einsum("nik,njk->nij", u, u)
        pair_abs = np.abs(gram[:, iu[0], iu[1]])
        max_vals[filled:filled + n] = pair_abs.max(axis=1)
        mean_vals[filled:filled + n] = pair_abs.mean(axis=1)
        filled += n

    def ci(vals: np.ndarray) -> tuple[float, float, float]:
        m = float(vals.mean())
        halfwidth = 1.96 * float(vals.std(ddof=1)) / math.sqrt(trials)
        return m, m - halfwidth, m + halfwidth

    max_mean, max_lo, max_hi = ci(max_vals)
    mean_mean, mean_lo, mean_hi = ci(mean_vals)
    return {
        "max_mean": max_mean, "max_ci_lo": max_lo, "max_ci_hi": max_hi,
        "mean_mean": mean_mean, "mean_ci_lo": mean_lo, "mean_ci_hi": mean_hi,
        "trials": trials, "seed": seed,
    }


def compare(d: int, k: int, trials: int = 20000, seed: int = 0) -> dict:
    """Every closed form and every Monte Carlo number side by side, plus a boolean
    per closed form for whether the relevant Monte Carlo 95% CI covers it (max-type
    forms checked against the max CI, mean-type forms against the mean CI)."""
    mc = monte_carlo(d, k, trials=trials, seed=seed)
    welch = welch_bound(d, k)
    jl = jl_coherence(d, k)
    pairs = pair_union_coherence(d, k)
    mean_exact = mean_coherence_exact(d)
    mean_limit = mean_coherence_limit(d)

    def covers(lo: float, hi: float, x: float) -> bool:
        return lo <= x <= hi

    return {
        "d": d, "k": k,
        "mu_welch": welch, "mu_jl": jl, "mu_pairs": pairs,
        "mu_mean_exact": mean_exact, "mu_mean_limit": mean_limit,
        **mc,
        "mu_welch_covered_by_mc_max_ci": covers(mc["max_ci_lo"], mc["max_ci_hi"], welch),
        "mu_jl_covered_by_mc_max_ci": covers(mc["max_ci_lo"], mc["max_ci_hi"], jl),
        "mu_pairs_covered_by_mc_max_ci": covers(mc["max_ci_lo"], mc["max_ci_hi"], pairs),
        "mu_mean_exact_covered_by_mc_mean_ci": covers(mc["mean_ci_lo"], mc["mean_ci_hi"], mean_exact),
        "mu_mean_limit_covered_by_mc_mean_ci": covers(mc["mean_ci_lo"], mc["mean_ci_hi"], mean_limit),
    }


def report(d: int = 256, k: int = 16, trials: int = 20000, seed: int = 0) -> str:
    c = compare(d, k, trials=trials, seed=seed)
    lines: list[str] = []
    w = lines.append
    w("=" * 78)
    w(f"COHERENCE FLOOR  d={d}  k={k}  trials={trials}  seed={seed}")
    w("=" * 78)
    w(f"  mu_welch(d,k)              = {c['mu_welch']:.6f}   (exact 0, k<=d)")
    w(f"  mu_jl(d,k)   [author form] = {c['mu_jl']:.6f}   "
      f"covered by MC max CI: {c['mu_jl_covered_by_mc_max_ci']}")
    w(f"  mu_pairs(d,k) [C(k,2) fix] = {c['mu_pairs']:.6f}   "
      f"covered by MC max CI: {c['mu_pairs_covered_by_mc_max_ci']}")
    w(f"  mean_coherence_exact(d)    = {c['mu_mean_exact']:.6f}   "
      f"covered by MC mean CI: {c['mu_mean_exact_covered_by_mc_mean_ci']}")
    w(f"  mean_coherence_limit(d)    = {c['mu_mean_limit']:.6f}   "
      f"covered by MC mean CI: {c['mu_mean_limit_covered_by_mc_mean_ci']}")
    w("")
    w(f"  MC max_i<j |<u_i,u_j>|  mean={c['max_mean']:.6f}  "
      f"95% CI [{c['max_ci_lo']:.6f}, {c['max_ci_hi']:.6f}]")
    w(f"  MC mean_i<j|<u_i,u_j>|  mean={c['mean_mean']:.6f}  "
      f"95% CI [{c['mean_ci_lo']:.6f}, {c['mean_ci_hi']:.6f}]")
    w("")
    w("  READING: the mean-coherence forms are confirmed (CI covers the exact form,")
    w("  exact and limit agree to <1e-3). Neither max-coherence closed form is")
    w("  confirmed: the author's mu_jl misses because its union bound runs over k")
    w("  events instead of C(k,2) pairs; the corrected mu_pairs is closer but still")
    w("  outside the CI, because a union-bound tail threshold is not the same object")
    w("  as an order statistic's mean.")
    w("=" * 78)
    return "\n".join(lines)


def demo() -> None:
    # (a) Welch bound is exactly 0 whenever k <= d.
    assert welch_bound(256, 16) == 0.0
    assert welch_bound(10, 10) == 0.0
    assert welch_bound(10, 5) == 0.0

    # (a) positive and matches the closed form once k > d: d=4, k=9 -> sqrt(5/32).
    got = welch_bound(4, 9)
    want = math.sqrt(5.0 / 32.0)
    assert got > 0.0
    assert abs(got - want) < 1e-12, (got, want)

    # (c) exact and large-d-limit mean-coherence forms agree to better than 1e-3 at d=256.
    exact = mean_coherence_exact(256)
    limit = mean_coherence_limit(256)
    assert abs(exact - limit) < 1e-3, (exact, limit)

    # Monte Carlo mean-coherence CI covers the exact closed form at d=256, k=16.
    mc = monte_carlo(256, 16, trials=20000, seed=0)
    assert mc["mean_ci_lo"] <= exact <= mc["mean_ci_hi"], (mc, exact)

    # Honesty check, run rather than assumed: does the MC max CI cover the
    # author's stated 0.147184, or mu_pairs? Measured answer is printed, not
    # asserted, because asserting an outcome that was not verified first is the
    # exact failure mode this module exists to catch.
    covers_author = mc["max_ci_lo"] <= AUTHOR_MU_JL <= mc["max_ci_hi"]
    covers_pairs = mc["max_ci_lo"] <= pair_union_coherence(256, 16) <= mc["max_ci_hi"]
    print(f"demo(): MC max CI covers author's mu_jl={AUTHOR_MU_JL}: {covers_author}")
    print(f"demo(): MC max CI covers mu_pairs: {covers_pairs}")
    print("demo(): all assertions passed")


if __name__ == "__main__":
    print(report())
    demo()
