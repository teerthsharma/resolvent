"""REMOVAL-ECHO v1, iteration 1: the three instruments and their must-fires.

THE QUESTION. Does a system's reaction to a dated removal break additivity -- "one
number per thing" -- and can that breaking be MEASURED rather than inferred?
Pre-registered at docs/prereg/REMOVAL_ECHO_v1.md before any data was touched.

WHY THIS REPLACES THE CURL PROGRAMME. Five domains failed in a row (betting,
power grids, road traffic, FX levels, Binance delistings), each for one of two
reasons: the system could not break the rule, or the removal hid the reaction.
The first reason has a name. A SUPERPOSITION system cannot break additivity --
a grid obeys Kirchhoff's voltage law, so the curl of any flow response is zero by
theorem, and ceqjepa/market_curl.py's own check reads 2.220e-16 there. An
EQUILIBRIUM system can: removing an edge from a Braess network lowers equilibrium
travel time. So the admission test for a domain is whether the rest
RE-EQUILIBRATES after the removal.

THE THREE INSTRUMENTS

  S0  MOBIUS INTERACTION -- what the one number hides.
        I(A,B) = f(AB) - f(A) - f(B) + f(empty)
      Additive iff I = 0 for every pair. Where every subset is observed (a
      designed double knockout), I is computed OUTRIGHT with no counterfactual.
      Shapley values are the "one number per thing" it is scored against.

  S1  SYNTHETIC CONTROL -- the counterfactual f(empty) when it is not observed.
      Weights w >= 0, sum w = 1, fit on the pre-removal window only; the effect
      is y - Y_donors w after the removal, ranked among placebos-in-space (a fake
      removal on each donor) and placebos-in-time (a fake date). An effect with no
      placebo rank is not a number.

  S2  THE ECHO -- the geometry of the reaction in time.
      h_j(tau) = corr(removal(t), y_j(t + tau)) over lags; the peak lag and its
      width. Binance's 1.73x rerouting was the zero-lag value of this curve.

MUST-FIRES, each required before any domain number is reported:
  - a planted ADDITIVE game reads I = 0 to 1e-12
  - a planted SYNERGY reads its planted value
  - the additive game at MATCHED NOISE yields no Benjamini-Hochberg discoveries
    beyond q -- additivity broken by measurement noise is the most likely false
    positive this protocol can produce, so it is tested at the noise the data has
  - synthetic-control placebo ranks on a no-removal unit are approximately uniform
  - a planted lag-k echo peaks at k; a no-response survivor stays in the null band

RUN: python -m ceqjepa.removal_echo
"""

import itertools
import math

import numpy as np
from scipy.optimize import minimize

__all__ = ["mobius_pair", "mobius_all_pairs", "shapley", "bh", "synthetic_control",
           "placebo_rank", "echo", "REFUSED"]

REFUSED = None


# ---------------------------------------------------------------- S0 ----------
def mobius_pair(f, a, b):
    """I(A,B) = f(AB) - f(A) - f(B) + f(empty). f maps a frozenset to a value."""
    return f(frozenset({a, b})) - f(frozenset({a})) - f(frozenset({b})) + f(frozenset())


def mobius_all_pairs(f, units):
    return {(a, b): mobius_pair(f, a, b) for a, b in itertools.combinations(units, 2)}


def shapley(f, units):
    """Exact Shapley values -- the additive 'one number per thing'."""
    n = len(units)
    phi = {u: 0.0 for u in units}
    for u in units:
        rest = [v for v in units if v != u]
        for k in range(n):
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for S in itertools.combinations(rest, k):
                S = frozenset(S)
                phi[u] += w * (f(S | {u}) - f(S))
    return phi


def bh(pvals, q=0.05):
    """Benjamini-Hochberg: boolean mask of discoveries at FDR q."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    passed = p[order] <= thresh
    k = int(np.max(np.nonzero(passed)[0]) + 1) if passed.any() else 0
    mask = np.zeros(m, dtype=bool)
    mask[order[:k]] = True
    return mask


# ---------------------------------------------------------------- S1 ----------
def synthetic_control(y_treated, Y_donors, t0):
    """Simplex weights fit on t < t0. Returns (weights, counterfactual path).

    Refuses when there are fewer than two donors or fewer than three pre-removal
    points -- a synthetic control fit to nothing is not a counterfactual.
    """
    y = np.asarray(y_treated, dtype=float)
    Y = np.asarray(Y_donors, dtype=float)
    if Y.ndim != 2 or Y.shape[1] < 2 or t0 < 3:
        return REFUSED
    J = Y.shape[1]
    pre_y, pre_Y = y[:t0], Y[:t0]
    res = minimize(lambda w: float(np.sum((pre_y - pre_Y @ w) ** 2)),
                   np.full(J, 1.0 / J), method="SLSQP",
                   bounds=[(0.0, 1.0)] * J,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"ftol": 1e-12, "maxiter": 500})
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum()
    return w, Y @ w


def _rmspe_ratio(y, cf, t0):
    pre = np.sqrt(np.mean((y[:t0] - cf[:t0]) ** 2))
    post = np.sqrt(np.mean((y[t0:] - cf[t0:]) ** 2))
    return post / max(pre, 1e-12)


def placebo_rank(Y_all, treated, t0):
    """Rank of the treated unit's post/pre RMSPE ratio among placebos-in-space.

    Each unit in turn is treated as if removed, fit from the others. Returns
    (rank, n) with rank 1 = largest effect. Under the null the rank is uniform on
    1..n, so the smallest achievable p is 1/n -- stated, never hidden.
    """
    Y_all = np.asarray(Y_all, dtype=float)
    n = Y_all.shape[1]
    ratios = []
    for u in range(n):
        donors = np.delete(Y_all, u, axis=1)
        out = synthetic_control(Y_all[:, u], donors, t0)
        if out is REFUSED:
            return REFUSED
        ratios.append(_rmspe_ratio(Y_all[:, u], out[1], t0))
    ratios = np.asarray(ratios)
    return int(1 + np.sum(ratios > ratios[treated])), n


# ---------------------------------------------------------------- S2 ----------
def echo(removal, y, max_lag):
    """h(tau) = corr(removal(t), y(t + tau)) for tau = 0..max_lag."""
    r = np.asarray(removal, dtype=float)
    y = np.asarray(y, dtype=float)
    h = []
    for tau in range(max_lag + 1):
        a, b = r[: len(r) - tau], y[tau:]
        if a.std() == 0 or b.std() == 0:
            h.append(0.0)
        else:
            h.append(float(np.corrcoef(a, b)[0, 1]))
    return np.asarray(h)


def _perm_pvals_from_noise(I_obs, sd_I):
    """Two-sided normal p-values for interaction estimates with known sd."""
    from scipy.stats import norm
    z = np.abs(np.asarray(I_obs)) / sd_I
    return 2.0 * norm.sf(z)


# ---------------------------------------------------------------- demo ---------
def demo():
    rng = np.random.default_rng(0)
    units = list(range(6))

    print("(a) S0 MUST-FIRE: an ADDITIVE game reads I = 0 to 1e-12 on every pair.")
    vals = rng.normal(size=len(units))
    f_add = lambda S: float(sum(vals[u] for u in S))
    I_add = mobius_all_pairs(f_add, units)
    worst = max(abs(v) for v in I_add.values())
    print("    max |I(A,B)| over %d pairs = %.3e" % (len(I_add), worst))
    assert worst < 1e-12, "an additive game reads a nonzero interaction"
    phi = shapley(f_add, units)
    print("    Shapley recovers the per-unit values: max |phi - v| = %.3e"
          % max(abs(phi[u] - vals[u]) for u in units))
    assert max(abs(phi[u] - vals[u]) for u in units) < 1e-12

    print("(b) S0 MUST-FIRE: a planted SYNERGY reads its planted value.")
    planted = {(1, 3): 0.7, (2, 5): -0.4}
    f_syn = lambda S: f_add(S) + sum(v for (a, b), v in planted.items() if a in S and b in S)
    I_syn = mobius_all_pairs(f_syn, units)
    for (a, b), v in planted.items():
        print("    pair (%d,%d): planted %+.2f  measured %+.12f" % (a, b, v, I_syn[(a, b)]))
        assert abs(I_syn[(a, b)] - v) < 1e-12
    others = max(abs(I_syn[p]) for p in I_syn if p not in planted)
    print("    every unplanted pair: max |I| = %.3e" % others)
    assert others < 1e-12

    print("(c) S0 MUST-FIRE AT MATCHED NOISE. Additivity broken by measurement noise is")
    print("    the most likely false positive, so the additive game is scored at the")
    print("    noise level the data will have. BH at q = 0.05 must hold its FDR.")
    sd_f = 0.3
    n_rep = 40
    big = list(range(20))
    vals_b = rng.normal(size=len(big))
    pairs = list(itertools.combinations(big, 2))
    discoveries, planted_hits = [], []
    for _ in range(n_rep):
        noisy = lambda S: float(sum(vals_b[u] for u in S)) + rng.normal() * sd_f
        I_obs = np.array([mobius_pair(noisy, a, b) for a, b in pairs])
        sd_I = 2.0 * sd_f                   # four independent draws of sd sd_f
        discoveries.append(bh(_perm_pvals_from_noise(I_obs, sd_I)).sum())
    fdr_frac = float(np.mean(discoveries)) / len(pairs)
    print("    %d pairs, %d replicates, noise sd %.2f: mean BH discoveries = %.3f"
          % (len(pairs), n_rep, sd_f, float(np.mean(discoveries))))
    print("    fraction of pairs called non-additive on a purely ADDITIVE game = %.4f"
          % fdr_frac)
    assert fdr_frac < 0.05, "BH calls an additive game non-additive at matched noise"

    # THE DETECTION FLOOR, measured rather than assumed. The first version of this
    # must-fire planted synergies at 2.5 sd_I, found them with power 0.16, and failed
    # its own assert. That is not an instrument bug: with m = 190 tests, BH's
    # threshold for five discoveries is 0.05*5/190 = 0.0013, and a 2.5-sd effect has
    # a two-sided p of about 0.012. So BH at this family size cannot see a 2.5-sd
    # interaction, and the effect size was NOT raised to turn the assert green.
    # Instead the power curve is swept and the minimum detectable effect reported.
    #
    # THIS BEARS DIRECTLY ON D1, and it is recorded here as a DIAGNOSTIC ADDED AFTER
    # the pre-registration (docs/prereg/REMOVAL_ECHO_v1.md at e238db6), not as a
    # change to it: the D1 counter predicts "fewer than 5% survive BH". A low
    # survival fraction is only evidence of additivity if the family's minimum
    # detectable effect is below the effects the biology produces. D1's fraction is
    # therefore reported beside BH's floor at D1's own m and noise, or it is not
    # interpretable.
    print("    THE DETECTION FLOOR. Planted synergies swept over effect size; BH at")
    print("    m = %d tests. Power at each size, 5 planted pairs per replicate:" % len(pairs))
    planted_idx = (0, 17, 60, 120, 180)
    sd_I = 2.0 * sd_f
    powers = {}
    for size in (2.5, 3.5, 4.5, 5.5, 6.5):
        syn = {pairs[i]: size * sd_I for i in planted_idx}
        hits = []
        for _ in range(n_rep):
            noisy = lambda S: (float(sum(vals_b[u] for u in S))
                               + sum(v for (a, b), v in syn.items() if a in S and b in S)
                               + rng.normal() * sd_f)
            I_obs = np.array([mobius_pair(noisy, a, b) for a, b in pairs])
            mask = bh(_perm_pvals_from_noise(I_obs, sd_I))
            hits.append(sum(mask[pairs.index(p)] for p in syn))
        powers[size] = float(np.mean(hits)) / len(syn)
        print("      effect %.1f sd_I -> power %.3f" % (size, powers[size]))
    mde = next((s for s in sorted(powers) if powers[s] >= 0.8), None)
    print("    minimum detectable effect at 80%% power, m = %d: %s"
          % (len(pairs), ("%.1f sd_I" % mde) if mde else "ABOVE 6.5 sd_I"))
    assert powers[6.5] > 0.8, "BH cannot find even a 6.5-sd synergy: the check is vacuous"
    assert powers[2.5] < powers[6.5], "power does not rise with effect size"
    print("    FIRED both ways: an additive game stays below q (FDR %.4f), and a large"
          % fdr_frac)
    print("    planted synergy is found. Anything smaller than the floor is INVISIBLE")
    print("    to this family, so a null there is a statement about power, not biology.")

    print("(d) S1 MUST-FIRE: placebo ranks on a NO-REMOVAL unit are UNIFORM.")
    # STRUCK BY THE INSPECTOR, AND REPAIRED. The first version asserted only that
    # the MEAN rank sat within 1.0 of (J+1)/2. A scratch copy fed half rank-1 and
    # half rank-J -- rank-1 frequency 0.500, as non-uniform as a rank distribution
    # gets -- and it PASSED, because that distribution's mean is exactly (J+1)/2.
    # A check named "uniform" that cannot fail on non-uniformity is vacuous; it was
    # the sixth such check shipped in this repository. It now tests the WHOLE
    # histogram by chi-square, and must be SEEN to reject a planted non-uniform
    # histogram with the uniform one's mean before it is allowed to accept the real
    # draws. The uniformity FACT was never in doubt -- a 400-panel chi-square gave
    # p = 0.8698 and was audited clean -- only this module's own check was.
    from scipy.stats import chisquare

    def rank_uniform_p(rk, J):
        counts = np.bincount(np.asarray(rk, dtype=int), minlength=J + 1)[1:]
        return float(chisquare(counts).pvalue)

    T, t0, J = 40, 25, 8
    planted_bad = [1] * 30 + [J] * 30
    p_bad = rank_uniform_p(planted_bad, J)
    print("    PLANTED NEGATIVE: half rank-1, half rank-%d -- mean %.2f, identical to a"
          % (J, float(np.mean(planted_bad))))
    print("    uniform's -- chi-square p = %.3e" % p_bad)
    assert p_bad < 0.01, "the uniformity test ACCEPTS a maximally non-uniform histogram"
    print("    FIRED: the mean-only check passed this; the histogram test rejects it.")
    ranks = []
    for rep in range(200):
        F = rng.normal(size=(T, 2)).cumsum(axis=0)
        L = rng.normal(size=(2, J))
        Y = F @ L + rng.normal(size=(T, J)) * 0.5
        ranks.append(placebo_rank(Y, treated=0, t0=t0)[0])
    ranks = np.asarray(ranks)
    p_real = rank_uniform_p(ranks, J)
    print("    200 panels, J=%d, no removal anywhere: mean rank %.2f (uniform %.2f),"
          % (J, ranks.mean(), (J + 1) / 2))
    print("    rank-1 frequency %.3f (uniform %.3f), chi-square p = %.4f"
          % ((ranks == 1).mean(), 1.0 / J, p_real))
    assert p_real > 0.01, "placebo ranks on a no-removal unit are NOT uniform"
    print("    smallest achievable p with J=%d units is 1/J = %.3f -- stated, not hidden."
          % (J, 1.0 / J))

    print("    PLANTED removal effect: the treated unit must rank near the top.")
    F = rng.normal(size=(T, 2)).cumsum(axis=0)
    Y = F @ rng.normal(size=(2, J)) + rng.normal(size=(T, J)) * 0.5
    Y[t0:, 0] += 6.0
    r, n = placebo_rank(Y, treated=0, t0=t0)
    print("    treated unit with a planted +6.0 step: rank %d of %d" % (r, n))
    assert r <= 2, "a planted removal effect does not rank among the largest placebos"

    print("(e) S2 MUST-FIRE: a planted lag-k echo peaks at k; no response stays null.")
    T2, t_rm, k = 300, 150, 7
    removal = (np.arange(T2) >= t_rm).astype(float)
    y_echo = np.zeros(T2)
    y_echo[t_rm + k:] = 1.0
    y_echo += rng.normal(size=T2) * 0.05
    h = echo(np.diff(removal, prepend=0), np.diff(y_echo, prepend=0), 20)
    print("    planted lag %d: peak |h| at lag %d (|h| = %.3f)"
          % (k, int(np.argmax(np.abs(h))), float(np.abs(h).max())))
    assert int(np.argmax(np.abs(h))) == k, "the echo peak is not at the planted lag"
    null = np.abs(echo(np.diff(removal, prepend=0), rng.normal(size=T2), 20)).max()
    print("    no-response survivor: max |h| = %.3f (must sit in the null band)" % null)
    assert null < float(np.abs(h).max()) / 2, "a no-response series echoes as strongly"
    print("ALL SELF-CHECKS PASSED")


if __name__ == "__main__":
    demo()
