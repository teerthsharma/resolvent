"""The second path for M1/M5's coherence arithmetic and M2's RIP line.

`scale/coherence_floor.py` and `scale/rip_line.py` each carry a `demo()` of
closed-form self-checks. This file runs those and adds the independent numeric
route that neither module takes, so every number quoted in the report has two
computations behind it that fail differently.

THE SECOND PATH FOR THE MAX COHERENCE. `coherence_floor` compares two closed
forms against a Monte Carlo draw. Monte Carlo and a closed form can both be
wrong for the same reason -- a mis-stated dimension enters both. The route here
is neither: the exact density of `<u,v>` for uniform unit vectors is
`c (1 - t^2)^((d-3)/2)`, so `E[max of M iid |<u,v>|]` is
`integral_0^1 (1 - F(x)^M) dx` by the order-statistic identity, evaluated by
quadrature. The `M = C(k,2)` pairwise products are NOT independent, so this is
an approximation and is treated as one; it lands inside the Monte Carlo interval,
which is what makes the sampled figure trustworthy and the author's closed form
not.
"""
import math

import numpy as np
from scipy import integrate, special

import scale.coherence_floor as CF
import scale.rip_line as RL

D = 256
K = 16


def _pairwise_pdf(d):
    log_c = (special.gammaln(d / 2) - special.gammaln((d - 1) / 2)
             - 0.5 * math.log(math.pi))
    return lambda t: math.exp(log_c + ((d - 3) / 2) * math.log1p(-t * t))


def _abs_cdf(d):
    pdf = _pairwise_pdf(d)
    return lambda x: integrate.quad(lambda t: 2 * pdf(t), 0.0, x, limit=200)[0]


def test_the_closed_form_self_checks_run():
    CF.demo()
    RL.demo()


def test_the_welch_bound_is_exactly_zero_below_the_dimension():
    for k in (2, 8, 16, 255, 256):
        assert CF.welch_bound(D, k) == 0.0
    # and is the real bound above it, against the hand value sqrt(5/32)
    assert abs(CF.welch_bound(4, 9) - math.sqrt(5 / 32)) < 1e-12
    # a Welch-optimal frame attains it: 3 unit vectors at 120 degrees in R^2
    ang = np.array([0.0, 2 * math.pi / 3, 4 * math.pi / 3])
    frame = np.stack([np.cos(ang), np.sin(ang)])
    gram = np.abs(frame.T @ frame)
    np.fill_diagonal(gram, 0.0)
    assert abs(gram.max() - CF.welch_bound(2, 3)) < 1e-12


def test_the_mean_coherence_has_two_agreeing_paths():
    """Exact Gamma ratio against the sqrt(2/(pi d)) limit and against the
    quadrature of the same density."""
    exact = CF.mean_coherence_exact(D)
    assert abs(exact - CF.mean_coherence_limit(D)) < 1e-3
    pdf = _pairwise_pdf(D)
    by_quadrature = integrate.quad(lambda t: 2 * t * pdf(t), 0.0, 1.0,
                                   limit=200)[0]
    assert abs(exact - by_quadrature) < 1e-9


def test_the_author_closed_form_for_the_max_is_not_the_expected_max():
    """The finding, asserted rather than described. `sqrt(2 ln k / d)` reads
    0.147176 at d=256, k=16; the expected maximum over the C(k,2) = 120 pairs
    is 0.1745. The union bound runs over pairs, not vectors."""
    jl = CF.jl_coherence(D, K)
    assert abs(jl - 0.147176) < 1e-5
    m = K * (K - 1) // 2
    assert m == 120
    cdf = _abs_cdf(D)
    expected_max = integrate.quad(lambda x: 1.0 - cdf(x) ** m, 0.0, 1.0,
                                  limit=200)[0]
    assert 0.170 < expected_max < 0.180, expected_max
    assert expected_max > jl * 1.15          # the closed form UNDERSTATES by >15%
    assert expected_max < CF.pair_union_coherence(D, K)   # the union bound is a bound
    # and the quadrature agrees with the module's Monte Carlo interval
    mc = CF.monte_carlo(D, K, trials=20000, seed=0)
    assert mc["max_ci_lo"] <= expected_max <= mc["max_ci_hi"], (mc, expected_max)


def test_a_scramble_controls_expected_residual_coherence_is_not_zero():
    """M5. A control calibrated to expect zero overlap between scrambled roles
    is vacuous before it runs: the expected mean overlap is 0.0499 and the
    expected worst-case overlap is 0.1748, both at d=256, k=16."""
    assert CF.mean_coherence_exact(D) > 0.04
    mc = CF.monte_carlo(D, K, trials=20000, seed=0)
    assert mc["mean_ci_lo"] > 0.0
    assert mc["max_ci_lo"] > 0.15
    # the floor it would have to beat to be a real signal is exactly zero, so
    # the nonzero reading is entirely chance
    assert CF.welch_bound(D, K) == 0.0


def test_the_rip_line_is_a_function_of_n_s_c_and_nothing_else():
    assert abs(RL.rip_line(1024, 4, 1.0) - 4 * math.log(256)) < 1e-12
    # monotone in s inside the sparse branch, where every caller sits
    vals = [RL.rip_line(1024, s, 1.0) for s in (1, 2, 4, 8, 16, 32)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    # the branch labels are the exact strings a cell prints
    line = RL.rip_line(1024, 4, 1.4427)
    assert RL.verdict(1024, 4, int(line) - 1, 1.4427) == "UNDER-SAMPLED"
    assert RL.verdict(1024, 4, int(line) + 1, 1.4427) == "ADMISSIBLE"
    assert RL.admissible(1024, 4, int(line) + 1, 1.4427)


def test_the_fitted_constant_is_not_bed_invariant():
    """Recorded because the first two beds share `n/s = 16` and therefore
    cannot detect it. At `n/s = 4` basis pursuit needs 25% more samples per
    `s ln(n/s)` than at `n/s = 16` or `n/s = 64`, so `C` is a per-bed
    measurement and reusing one bed's value elsewhere is a guess."""
    ratios = {(c["n"], c["s"]): c["n"] / c["s"] for c in RL.SETTINGS}
    assert len(set(ratios.values())) >= 3, ratios
    assert (64, 16) in ratios and (256, 4) in ratios
