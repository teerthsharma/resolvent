"""X35-prime (c) Kramers-Kronig causality residual and (d) the onset
localization floor -- the must-fires, written before either module exists.

The KK probe is rebuilt from scratch: CEQ_V15_2_DELTA.md struck the author's
own because it read `0.000` on a planted anticipating kernel, which is
MISTAKES.md V-16 (the instrument that cannot measure, reporting a pass). The
must-fire is therefore two-sided -- a planted acausal kernel MUST read nonzero
AND a causal one MUST read ~0 -- because a probe that fires on everything is
as useless as one that fires on nothing.

The (d) tests do NOT assert a CRB. They assert that the classical CRB does not
exist for this bed's parameter, by measuring the two regularizations that
would create one and showing they disagree in the limit.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ceq.x35p import kk, crb
from ceq.beds import bed_k


SQRT2 = math.sqrt(2.0)


# ==========================================================================
# (c) KK -- the null half. A probe that fires on everything measures nothing.
# ==========================================================================


@pytest.mark.parametrize("d", [0, 1, 2, 5, 13])
def test_causal_delay_kernel_reads_zero(d):
    out = kk.kk_residual(np.array([d]), np.array([1.0]))
    assert out["residual"] < 1e-13, out


def test_random_causal_kernels_read_zero():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(64):
        taps = rng.standard_normal(24)
        worst = max(worst, kk.kk_residual(np.arange(24), taps)["residual"])
    assert worst < 1e-12, worst


@pytest.mark.parametrize("kind,params", [("delay", dict(d=4)),
                                         ("powerlaw", dict(H=0.8))])
def test_bed_k_kernels_read_zero(kind, params):
    K = bed_k.kernel_matrix(kind, 64, **params)
    lags, taps = kk.kernel_from_matrix(K, 48)
    assert kk.kk_residual(lags, taps)["residual"] < 1e-12


# ==========================================================================
# (c) KK -- THE MUST-FIRE. A planted anticipating kernel must read nonzero.
# ==========================================================================


def test_planted_anticipating_tap_fires_at_its_predicted_value():
    """MUST-FIRE. Expected value computed before the run (MISTAKES.md V-10):
    a single anticausal tap of amplitude `a` reads exactly `a * sqrt(2)`."""
    lags = np.array([-3, 0, 1, 2])
    taps = np.array([0.7, 1.0, -0.5, 0.25])
    out = kk.kk_residual(lags, taps)
    assert out["residual"] == pytest.approx(0.7 * SQRT2, abs=1e-12)
    assert out["residual"] > 0.9


def test_probe_is_graded_in_the_anticipating_amplitude():
    """Not a flag: linear in the violation, so a kernel that is slightly
    acausal gets a number rather than a verdict."""
    reads = [kk.kk_residual(np.array([-2, 0, 1]), np.array([a, 1.0, 0.3]))["residual"]
             for a in (0.0, 1e-6, 1e-3, 0.1, 1.0)]
    assert reads[0] < 1e-13
    assert reads == sorted(reads)
    for a, r in zip((1e-6, 1e-3, 0.1, 1.0), reads[1:]):
        assert r == pytest.approx(a * SQRT2, rel=1e-9)


def test_closed_form_matches_the_frequency_domain_route():
    """Two independent routes to the same number (MISTAKES.md V-3): the
    frequency-domain Hilbert reconstruction, and `sqrt(2) * ||anticausal||`
    computed in the time domain. Agreement is a check, not an identity."""
    rng = np.random.default_rng(7)
    for _ in range(32):
        lags = np.arange(-6, 10)
        taps = rng.standard_normal(lags.size)
        out = kk.kk_residual(lags, taps)
        anti = np.linalg.norm(taps[lags < 0])
        assert out["residual"] == pytest.approx(SQRT2 * anti, rel=1e-10, abs=1e-14)


def test_spectrum_only_route_agrees_with_the_tap_route():
    """The probe's whole point is that it runs on H(omega) alone."""
    lags, taps = np.arange(-4, 8), np.random.default_rng(3).standard_normal(12)
    a = kk.kk_residual(lags, taps)
    b = kk.kk_residual_from_spectrum(a["spectrum"])
    assert b["residual"] == pytest.approx(a["residual"], rel=1e-12)


def test_grid_too_small_raises_rather_than_aliasing():
    with pytest.raises(ValueError, match="guard band"):
        kk.kk_residual(np.array([-8, 0, 8]), np.ones(3), n_freq=16)


# ==========================================================================
# (c) DIAGNOSIS -- why the original read 0.000. Each variant is a plausible
# naive implementation; the test records which ones cannot fire.
# ==========================================================================


ANTICIPATING = (np.array([-5, -1, 0, 2, 6]), np.array([1.0, -0.8, 1.0, 0.4, -0.2]))


def test_bare_array_variant_reads_zero_on_the_anticipating_kernel():
    """The origin-forgetting bug: taps handed over without a lag axis, so
    index 0 is taken as lag 0. Every finitely-supported sequence is causal
    once you forget where its time origin is."""
    assert kk.broken_bare_array(ANTICIPATING[1])["residual"] < 1e-13


def test_magnitude_variant_reads_zero_by_construction():
    """Residual taken on |H| rather than on Re/Im: the minimum-phase
    reconstruction has the input's magnitude by construction, so the
    comparison has no rejection region (MISTAKES.md V-10)."""
    assert kk.broken_magnitude(*ANTICIPATING)["residual"] < 1e-12


def test_unpadded_variant_can_read_zero_on_an_aliased_kernel():
    """No zero-padding: a negative lag wraps into the causal half of the
    circle and is indistinguishable from a long positive delay."""
    lags, taps = np.array([-7, -6, 0, 1]), np.array([1.0, 1.0, 1.0, 0.5])
    assert kk.broken_no_padding(lags, taps)["residual"] < 1e-13
    assert kk.kk_residual(lags, taps)["residual"] > 1.0


def test_lag_negation_moves_the_rebuilt_reading_and_not_the_bare_one():
    """Discriminating experiment (a). A correct probe reads a different number
    when the lag axis is negated; a probe that never had the lag axis cannot."""
    lags, taps = ANTICIPATING
    assert kk.kk_residual(lags, taps)["residual"] == pytest.approx(1.811077, abs=1e-6)
    assert kk.kk_residual(-lags, taps)["residual"] == pytest.approx(0.632456, abs=1e-6)
    assert kk.broken_bare_array(taps)["residual"] < 1e-13


def test_zero_phase_kernel_has_no_imaginary_part_to_normalize_by():
    """Discriminating experiment (b), and the brief's real-spectrum candidate.
    A symmetric kernel is maximally anticausal for its energy AND has an
    exactly real spectrum, so every relative normalization is undefined while
    the absolute reading is exact. Report the absolute."""
    out = kk.kk_residual(np.array([-1, 0, 1]), np.array([0.5, 1.0, 0.5]))
    assert np.abs(out["spectrum"].imag).max() == 0.0
    assert out["residual"] == pytest.approx(SQRT2 * 0.5, abs=1e-12)
    assert math.isnan(out["relative"])
    assert kk.broken_magnitude(np.array([-1, 0, 1]),
                               np.array([0.5, 1.0, 0.5]))["residual"] < 1e-12


def test_bin_normalized_variant_shrinks_but_is_not_the_mechanism():
    """A normalization that divides by the bin count rather than its root:
    the same kernel reads smaller the finer you sample the spectrum, exactly
    as `1/sqrt(M)`. It is ELIMINATED as the cause of an exact `0.000` -- 64x
    more bins buys 8x, so reaching 0.000 would take a grid nobody runs."""
    coarse = kk.broken_bin_normalized(*ANTICIPATING, n_freq=1024)["residual"]
    fine = kk.broken_bin_normalized(*ANTICIPATING, n_freq=1 << 16)["residual"]
    assert fine == pytest.approx(coarse / 8.0, rel=1e-9)
    assert fine > 1e-4


# ==========================================================================
# (d) The bound. NO CRB NUMBER IS ASSERTED -- these tests establish that the
# classical CRB does not exist for the bed's onset parameter.
# ==========================================================================


def test_residual_is_the_variance_change_point_the_bound_is_derived_for():
    """Guards MISTAKES.md M-18: a bound derived against a quantity whose
    distribution in the corpus was never checked. Measured on the production
    path, `ceq.x35.residual.run_residual`."""
    got = crb.measure_residual_variance_profile(onset=32, sigma=1.0, magnitude=1.5,
                                                n=64, runs=600, seed0=900)
    assert got["pre_var"] == pytest.approx(1.0, rel=0.08)
    assert got["post_var"] == pytest.approx(1.0 + 1.5 ** 2, rel=0.08)


def test_classical_crb_does_not_exist_for_the_abrupt_discrete_onset():
    """Fisher information is identically 0: the variance profile is a step in
    tau, so d v_i / d tau vanishes almost everywhere. CRB = +inf, a bound no
    estimator can cross -- vacuous (MISTAKES.md V-10)."""
    out = crb.fisher_information_ramp(tau=32.0, width=0.0, sigma=1.0,
                                      magnitude=1.0, n=64)
    assert out["fisher"] == 0.0
    assert out["crb"] == math.inf


def test_the_likelihood_is_flat_between_integer_onsets():
    """The same fact from the data side: moving a continuous onset within one
    sample changes no likelihood, so there is no score to take a variance of."""
    r = np.random.default_rng(11).standard_normal(64)
    a = crb.abrupt_loglik(r, 32.0, sigma=1.0, magnitude=1.0)
    b = crb.abrupt_loglik(r, 32.9, sigma=1.0, magnitude=1.0)
    c = crb.abrupt_loglik(r, 33.0, sigma=1.0, magnitude=1.0)
    assert a == b
    assert a != c


def test_smoothing_makes_a_crb_exist_but_the_ramp_width_sets_it():
    """The other regularization. CRB(w) -> 0 as w -> 0 while the abrupt
    reading is +inf: the two natural limits disagree, so no limit defines the
    bound, and any distance-to-CRB would be a statement about w."""
    ws = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    crbs = [crb.fisher_information_ramp(tau=32.0, width=w, sigma=1.0,
                                        magnitude=1.0, n=64)["crb"] for w in ws]
    assert crbs == sorted(crbs)
    assert crbs[-1] / crbs[0] > 20.0


def test_the_smoothed_crb_also_depends_on_the_sub_sample_phase_of_tau():
    """And the ramp width is not the only free knob. At w << 1 the bound
    depends on where tau falls BETWEEN samples, which is not a property of the
    estimation problem at all."""
    kw = dict(width=0.05, sigma=1.0, magnitude=1.0, n=64)
    on_grid = crb.fisher_information_ramp(tau=32.0, **kw)["crb"]
    off_grid = crb.fisher_information_ramp(tau=32.5, **kw)["crb"]
    assert off_grid / on_grid > 1e5


# ==========================================================================
# (d) The replacement bound, and ITS must-fire: the noise sweep.
# ==========================================================================


def test_zzb_reaches_the_prior_variance_when_the_data_are_uninformative():
    """The bound's own calibration against a case with a known answer: at
    magnitude 0 no test beats a coin, P_min = 1/2, and the bound must equal
    Var(Uniform{0..n-1}) = (n^2-1)/12 exactly."""
    n = 64
    got = crb.zzb(n=n, sigma=1.0, magnitude=0.0)
    assert got["bound"] == pytest.approx((n * n - 1) / 12.0, rel=1e-12)
    assert got["p_min"][1] == pytest.approx(0.5, rel=1e-12)


def test_pmin_is_non_increasing_in_the_separation():
    p = crb.zzb(n=64, sigma=1.0, magnitude=1.0)["p_min"][1:]
    assert np.all(np.diff(p) <= 1e-12)


def test_zzb_increases_with_noise():
    bounds = [crb.zzb(n=64, sigma=s, magnitude=1.0)["bound"]
              for s in (0.25, 0.5, 1.0, 2.0, 4.0)]
    assert bounds == sorted(bounds)


def test_noise_sweep_never_falls_below_the_bound():
    """MUST-FIRE for (d). The Bayes-optimal estimator (posterior mean under
    the same uniform prior the bound assumes) is run against the curve at
    every noise level. A sub-bound point is an instrument failure, not a
    result, and this test is what makes it one."""
    sweep = crb.noise_sweep(n=64, magnitude=1.0,
                            sigmas=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
                            runs=1000, seed0=4000)
    for row in sweep:
        assert row["mse_bayes"] >= row["bound"] - 3.0 * row["se_bayes"], row
    assert sweep[0]["bound"] < sweep[-1]["bound"]
    assert sweep[0]["mse_bayes"] < sweep[-1]["mse_bayes"]


def test_the_bound_is_still_below_the_mmse_where_it_is_nearly_tight():
    """The `3 se` tolerance above admits Monte-Carlo sub-bound points, and one
    was seen (sigma=16, 4000 runs, 0.53 se below). This is the analytic check
    that resolves it without Monte Carlo at all: as the data go uninformative
    the posterior mean goes to the prior mean and the MMSE goes to the prior
    variance, so the bound must sit strictly below `(n^2-1)/12` -- and does,
    by a margin that shrinks but never changes sign."""
    prior_var = (64 * 64 - 1) / 12.0
    gaps = [prior_var - crb.zzb(n=64, sigma=s, magnitude=1.0)["bound"]
            for s in (8.0, 16.0, 64.0, 256.0)]
    assert all(g > 0.0 for g in gaps)
    assert gaps == sorted(gaps, reverse=True)
