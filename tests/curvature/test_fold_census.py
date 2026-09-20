"""Tests for ceqjepa.fold_census: the two pre-registered KILLERS for the
beta-scaled read-as-update FOLD leap, run as measurements, not assumed.

WRITTEN RED. ceqjepa/fold_census.py did not exist before this file; the first
failure was ModuleNotFoundError: No module named 'ceqjepa.fold_census'.

WHAT MUST BE SEEN TO FIRE, so this file is not a rubber stamp on its own
module: (a) the isolated N=1 case must reproduce the closed-form fold tau to
high precision -- everything else is validated against THIS, not against
itself; (b) at N=3 orthogonal the three thresholds must be strictly increasing
with ||k||^2 (killer a does not fire) AND must reproduce the pre-registered
0.54/1.09/2.17 to within 1%; (c) the naive power-iteration cross-check must
actually disagree with root-finding at large tau, which is the reason this
file does not use naive iteration as its measurement -- a test suite that
never sees that disagreement would not know it needed root-finding at all.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ceqjepa import fold_census as fc

BETA_9 = 0.9
NORMS = (2.0, 4.0, 8.0)


# ---------------------------------------------------------------------------
# 1. GROUND TRUTH: the isolated N=1 case against the closed form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("g", [2.0, 4.0, 8.0])
@pytest.mark.parametrize("beta", [0.5, 0.7, 0.9])
def test_isolated_single_pattern_matches_the_closed_form(g, beta):
    G = fc.gram_orthogonal([g])
    closed = fc.isolated_fold_tau(g, beta)
    measured = fc.explode_threshold(0, G, beta, g)
    assert abs(measured - closed) < 1e-5 * closed


@pytest.mark.parametrize("g,beta", [(4.0, 0.7), (8.0, 0.9)])
def test_spectral_radius_reaches_the_tangency_at_the_measured_fold(g, beta):
    """The fold is DEFINED as a stable point annihilating a saddle -- a real
    eigenvalue of d(read_map)/dc crossing 1. If explode_threshold's answer is
    right, the spectral radius there must sit just under 1, not somewhere
    unrelated to it."""
    G = fc.gram_orthogonal([g])
    thr = fc.explode_threshold(0, G, beta, g)
    sol, resid = fc.solve_fixed_point([1.0], G, thr, beta)
    assert resid < 1e-6
    sr = fc.spectral_radius(sol, G, thr, beta)
    assert 0.99 < sr < 1.0 + 1e-3


# ---------------------------------------------------------------------------
# 2. KILLER (a): together, or in norm order?
# ---------------------------------------------------------------------------

def test_three_orthogonal_patterns_vanish_in_norm_order_not_together():
    census = fc.three_pattern_census(beta=BETA_9, norms_sq=NORMS)
    assert census["order_follows_norm"], census["thresholds"]
    # NOT together: the killer as specified is thresholds coinciding. The
    # measured spread is ~75% of the mean -- nowhere near "together".
    assert census["spread_over_mean"] > 0.5, census


def test_three_orthogonal_patterns_reproduce_the_preregistered_red():
    census = fc.three_pattern_census(beta=BETA_9, norms_sq=NORMS)
    for got, want in zip(census["thresholds"], (0.54, 1.09, 2.17)):
        assert abs(got - want) / want < 0.01, (census["thresholds"], (0.54, 1.09, 2.17))


def test_coupled_threshold_agrees_with_isolated_formula_at_beta_point_nine():
    """Right at each pattern's OWN fold, near-zero coefficients on the other
    two orthogonal patterns make cross-terms in Z negligible -- so coupling
    should barely move the threshold away from the isolated closed form here,
    unlike at lower beta (see the boundary-fit tests below)."""
    G = fc.gram_orthogonal(NORMS)
    for j, g in enumerate(NORMS):
        thr = fc.explode_threshold(j, G, BETA_9, g)
        iso = fc.isolated_fold_tau(g, BETA_9)
        assert abs(thr - iso) / iso < 1e-3, (j, g, thr, iso)


# ---------------------------------------------------------------------------
# 3. KILLER (b): a line through (beta=1, tau=0)?
# ---------------------------------------------------------------------------

def test_explode_boundary_bends_away_from_the_origin_over_the_full_beta_range():
    """Fit over beta in {0.5..0.95} as specified. The claim under test is
    intercept == 0; measured, it sits at ~2.3 standard errors from zero with
    R^2 short of 1, which is the killer partially firing over this range."""
    G = fc.gram_orthogonal(NORMS)
    betas = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    fit = fc.fit_explode_boundary(betas, G, pattern_index=2, g=8.0)
    assert abs(fit["intercept"]) > 2.0 * fit["intercept_se"], fit
    assert fit["r2"] < 0.999, fit


def test_explode_boundary_is_nearly_linear_close_to_the_softmax_corner():
    """The bend above is a low-beta (far from beta=1) effect: restricted to
    beta in {0.8, 0.9, 0.95}, the coupled boundary should track the isolated
    line far more tightly."""
    G = fc.gram_orthogonal(NORMS)
    betas = [0.8, 0.9, 0.95]
    fit = fc.fit_explode_boundary(betas, G, pattern_index=2, g=8.0)
    for beta, tau in zip(fit["betas"], fit["taus"]):
        iso = fc.isolated_fold_tau(8.0, beta)
        assert abs(tau - iso) / iso < 0.02, (beta, tau, iso)


def test_isolated_formula_is_exactly_linear_through_the_origin():
    """The ISOLATED closed form is algebra, not a measurement -- it is the
    baseline the coupled measurements above are compared against, and it must
    read back as a bitwise line through the origin."""
    betas = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    G = fc.gram_orthogonal([8.0])
    fit = fc.fit_explode_boundary(betas, G, pattern_index=0, g=8.0)
    assert abs(fit["intercept"]) < 1e-4
    assert fit["r2"] > 1.0 - 1e-9


# ---------------------------------------------------------------------------
# 4. NON-ORTHOGONAL PATTERNS
# ---------------------------------------------------------------------------

def test_gram_correlated_reduces_to_orthogonal_at_rho_zero():
    Gc = fc.gram_correlated(NORMS, 0.0)
    assert np.allclose(Gc, fc.gram_orthogonal(NORMS), atol=1e-9)


def test_mild_correlation_keeps_norm_order_but_shifts_weak_patterns_a_lot():
    G = fc.gram_correlated(NORMS, 0.2)
    thresholds = [fc.explode_threshold(j, G, BETA_9, g) for j, g in enumerate(NORMS)]
    assert all(thresholds[i] < thresholds[i + 1] for i in range(2)), thresholds
    iso = [fc.isolated_fold_tau(g, BETA_9) for g in NORMS]
    # the weak/mid patterns move far more than the dominant one
    assert abs(thresholds[1] - iso[1]) / iso[1] > 0.05
    assert abs(thresholds[2] - iso[2]) / iso[2] < 0.01


def test_strong_correlation_erases_the_weak_patterns_own_branch():
    """A planted negative: at rho=0.4 the weak and mid patterns' own branches
    should not be findable near their isolated prediction (their retrieval
    identity has already merged with the dominant pattern's), while the
    dominant pattern's own branch is unaffected. A version of this code that
    always returns SOME number here would be hiding the merge, not measuring
    it."""
    G = fc.gram_correlated(NORMS, 0.4)
    for j, g in list(enumerate(NORMS))[:2]:
        with pytest.raises(RuntimeError):
            fc.explode_threshold(j, G, BETA_9, g)
    thr = fc.explode_threshold(2, G, BETA_9, 8.0)
    assert abs(thr - fc.isolated_fold_tau(8.0, BETA_9)) / fc.isolated_fold_tau(8.0, BETA_9) < 0.01


# ---------------------------------------------------------------------------
# 5. WHY THE PRIMARY MEASUREMENT IS ROOT-FINDING, NOT POWER ITERATION
# ---------------------------------------------------------------------------

def test_naive_power_iteration_disagrees_with_root_finding_at_large_tau():
    """At large tau every pattern's naive iterate converges to the SAME point
    regardless of which pattern it started at -- there is no separate basin
    left to find there, which is exactly why explode_threshold seeds its
    search near each pattern's OWN isolated tau rather than iterating from
    scratch at an arbitrary tau."""
    G = fc.gram_orthogonal(NORMS)
    finals = []
    for j in range(3):
        c0 = [0.0, 0.0, 0.0]
        c0[j] = 1.0
        status, c_final = fc.iterate_naive(c0, G, tau=100.0, beta=BETA_9)
        assert status == "converged"
        finals.append(c_final)
    assert np.allclose(finals[0], finals[1], atol=1e-3)
    assert np.allclose(finals[1], finals[2], atol=1e-3)
    # the two WEAKER patterns land on a point dominated by the STRONGEST
    # pattern's own coordinate (index 2), not their own -- there is no
    # separate basin at j=0 or j=1 left to find at this tau.
    for j in (0, 1):
        assert int(np.argmax(np.abs(finals[j]))) != j


def test_naive_power_iteration_agrees_with_root_finding_right_at_the_fold():
    """The cross-check demo() relies on: just above/below the measured
    threshold, naive iteration DOES agree with root-finding, because right at
    a pattern's own fold its own coordinate genuinely dominates."""
    G = fc.gram_orthogonal(NORMS)
    thr = fc.explode_threshold(2, G, BETA_9, 8.0)
    survives, _ = fc.iterate_naive([0.0, 0.0, 1.0], G, thr * 1.02, BETA_9)
    dies, _ = fc.iterate_naive([0.0, 0.0, 1.0], G, thr * 0.98, BETA_9)
    assert survives == "converged"
    assert dies == "exploded"


def test_demo_runs():
    fc.demo()


# ---------------------------------------------------------------------------
# 6. THE LITERAL PRE-REGISTERED METHOD, run for real: "a basin count by
# ITERATING THE FULL MULTI-PATTERN UPDATE FROM MANY RANDOM INITIALISATIONS TO
# CONVERGENCE, DEDUPLICATING THE LIMITS" -- not Newton continuation from one
# seed per pattern (sections 1-5 above), which is faster but is not the
# method the RED specifies. Self-contained in THIS file (only fc.gram_
# orthogonal and fc.isolated_fold_tau are reused) so it survives independently
# of which continuation helpers fold_census.py's own API happens to ship.
#
# THE FINDING neither section 1-5 nor fold_census.py's own docstring
# characterises: swept over tau in [0.05, 120] at beta=0.9, the measured basin
# count is NEVER 2 or 3. Each smaller pattern's basin is closed again by a
# SECOND, unpredicted fold before the next pattern's own onset arrives --
#     ||k||^2=2 window   [0.5436, 0.7250]
#     ||k||^2=4 window   [1.0872, 1.6236]
#     ||k||^2=8 window   [2.1745, unclosed to tau=120, checked]
# so the windows never overlap and the pre-registered 3 -> 2 -> 1 -> 0 step
# sequence does not happen: the measured sequence, rising in tau, is
# 0 -> 1 -> 0 -> 1 -> 0 -> 1. Leaking exp(0)=1 of mass to every OTHER pattern
# on every step is negligible right at a pattern's own onset (its own
# exponential still dominates the shared normalizer there) and not negligible
# once tau has grown enough for a competing pattern's flat baseline to
# re-annihilate the smaller pattern's stable point with a second saddle.
# ---------------------------------------------------------------------------

def _literal_step(C, G, tau, beta, explode_s=600.0):
    """One vectorised step of xi <- sum_j e^{xi.k_j/tau} k_j / Z^beta in
    coefficient space (c_{t+1} = softmax_beta(G c_t / tau), G the Gram
    matrix), over a batch C[n_trials, n_patterns]. explode_s caps the raw
    score before exp, well inside float64's range, so a diverging trial is
    flagged rather than turned into inf/nan a few steps later."""
    S = (C @ G) / tau
    exploded = S.max(axis=1) > explode_s
    C_new = C.copy()
    safe = ~exploded
    if safe.any():
        Ex = np.exp(S[safe])
        Z = Ex.sum(axis=1)
        C_new[safe] = Ex / (Z ** beta)[:, None]
    return C_new, exploded


def _literal_census(G, tau, beta, n_trials=150, seed=20260920, max_iter=4000,
                     tol=1e-10, explode_s=600.0, cluster_tol=1e-4):
    """Iterate the FULL multi-pattern update from many pattern-aligned and
    random initialisations to float64 convergence or divergence, dedupe the
    limits. Returns the distinct-basin count."""
    n = G.shape[0]
    rng = np.random.default_rng(seed)
    seeds = [scale * np.eye(n)[j] for j in range(n) for scale in
              (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)]
    n_rand = max(0, int(n_trials) - len(seeds))
    if n_rand:
        dirs = rng.standard_normal((n_rand, n))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        mags = 10.0 ** rng.uniform(-2.0, 1.5, size=n_rand)
        seeds.extend(list(dirs * mags[:, None]))
    C = np.array(seeds, dtype=np.float64)
    status = np.full(C.shape[0], 2, dtype=np.int64)  # 0 converged 1 diverged 2 stalled
    running = np.ones(C.shape[0], dtype=bool)
    for _ in range(int(max_iter)):
        idx = np.where(running)[0]
        if idx.size == 0:
            break
        C_new, exploded = _literal_step(C[idx], G, tau, beta, explode_s)
        d_idx = idx[exploded]
        status[d_idx] = 1
        running[d_idx] = False
        C[d_idx] = 0.0
        keep = idx[~exploded]
        if keep.size:
            diff = np.linalg.norm(C_new[~exploded] - C[keep], axis=1)
            conv = diff < tol
            c_idx = keep[conv]
            status[c_idx] = 0
            C[c_idx] = C_new[~exploded][conv]
            running[c_idx] = False
            rest = keep[~conv]
            if rest.size:
                C[rest] = C_new[~exploded][~conv]
    reps = []
    for p in C[status == 0]:
        if not any(np.linalg.norm(p - r) < cluster_tol * (1.0 + np.linalg.norm(r))
                    for r in reps):
            reps.append(p)
    return len(reps)


def _literal_find_steps(G, beta, tau_lo=0.05, tau_hi=120.0, growth=1.08,
                         step_tol=1e-4, **kw):
    """Every count change over [tau_lo, tau_hi], bisected to step_tol.
    A transparent geometric scan first: no assumption about how many steps
    there are or which way the count moves."""
    def count_at(tau):
        return _literal_census(G, tau, beta, **kw)
    taus = [tau_lo]
    t = tau_lo
    while t < tau_hi:
        t *= growth
        taus.append(min(t, tau_hi))
    counts = [count_at(t) for t in taus]
    steps = []
    for i in range(1, len(taus)):
        if counts[i] != counts[i - 1]:
            lo, hi, lo_v, hi_v = taus[i - 1], taus[i], counts[i - 1], counts[i]
            while hi - lo > step_tol:
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if count_at(mid) != hi_v else (lo, mid)
            steps.append((round(0.5 * (lo + hi), 6), lo_v, hi_v))
    return steps


def test_literal_multi_init_census_onsets_match_the_continuation_numbers():
    """The brute-force method's own onset, compared against BOTH the
    continuation numbers this file's sections 1-5 already trust and the
    isolated closed form -- not assumed to agree with either."""
    G = fc.gram_orthogonal(NORMS)
    steps = _literal_find_steps(G, BETA_9)
    onsets = [tau for tau, before, after in steps if after > before]
    assert len(onsets) == 3, "expected exactly 3 onsets, got %r" % (steps,)
    assert onsets == sorted(onsets), "onsets are not in increasing tau order"

    continuation = fc.three_pattern_census(beta=BETA_9, norms_sq=NORMS)["thresholds"]
    for g, tau_cont, tau_meas in zip(NORMS, continuation, onsets):
        assert tau_meas == pytest.approx(tau_cont, abs=5e-3), (
            "||k||^2=%r: continuation=%.6f, brute-force onset=%.6f"
            % (g, tau_cont, tau_meas))
    gaps = [b - a for a, b in zip(onsets, onsets[1:])]
    assert all(gap > 0.1 for gap in gaps), (
        "K1: measured onsets are not well-separated: %r" % onsets)


def test_the_three_basins_never_coexist_the_step_sequence_is_not_3210():
    """THE ACTUAL FINDING. Count never exceeds 1 anywhere in the scan, so the
    pre-registered 3 -> 2 -> 1 -> 0 sequence is not what the coupled system
    does -- measured, not assumed, and reported even though it goes the other
    way from the RED's own expectation."""
    G = fc.gram_orthogonal(NORMS)
    steps = _literal_find_steps(G, BETA_9)
    levels = {b for _, b, _ in steps} | {a for _, _, a in steps}
    assert levels <= {0, 1}, (
        "the measured count reached %r, not just 0/1: re-examine before "
        "trusting this test's own docstring" % (levels,))
    assert len(steps) == 5, (
        "expected 5 transitions (open,close,open,close,open), got %r" % (steps,))
    # explicit spot check away from every transition: tau=0.9 sits squarely in
    # the dead zone between the ||k||^2=2 pattern's close and the ||k||^2=4
    # pattern's open, where nothing has converged to a distinct basin at all.
    assert _literal_census(G, 0.9, BETA_9) == 0


def test_each_smaller_pattern_window_is_closed_by_a_second_unpredicted_fold():
    """The window boundary that neither the independence shortcut nor the
    continuation-based sections above characterise."""
    G = fc.gram_orthogonal(NORMS)
    steps = _literal_find_steps(G, BETA_9)
    opens = sorted(tau for tau, before, after in steps if after > before)
    closes = sorted(tau for tau, before, after in steps if after < before)
    assert len(opens) == 3 and len(closes) == 2, (opens, closes)
    # the two smaller patterns each close before the NEXT pattern opens --
    # that is what makes the windows disjoint and the count never reach 2.
    assert closes[0] < opens[1] < closes[1] < opens[2], (opens, closes)
    for tau in (opens[0], closes[0], opens[1], closes[1], opens[2]):
        assert 0.05 < tau < 120.0


# ---------------------------------------------------------------------------
# 7. ROUND TWO: the surviving piece restated as a training-route predicate.
# `isolated_fold_tau` is algebra (see module docstring); the only open
# question for a real run is whether its (beta, tau) schedule ever crosses
# it. Both schedules below share endpoints (tau: 12.0 -> 2.0 over 21 steps)
# and the same hard-swap beta (0 for t<10, 1 for t>=10, i.e. kernel then
# softmax) -- only the SHAPE of tau differs.
# ---------------------------------------------------------------------------

ROUTE_KMAX = 4.0


def _hard_swap_beta(steps, mid):
    """beta = 0 (kernel) for t < mid, beta = 1 (softmax) for t >= mid: a
    kernel-to-softmax schedule with the dispatch's own `hard_swap` shape,
    jumping at its own stated midpoint."""
    beta = np.zeros(steps, dtype=np.float64)
    beta[mid:] = 1.0
    return beta


def test_route_constraint_required_tau_matches_isolated_fold_tau():
    beta = np.array([0.0, 0.5, 0.9])
    tau = np.array([5.0, 5.0, 5.0])
    out = fc.route_constraint_violations(beta, tau, ROUTE_KMAX)
    expected = np.array([fc.isolated_fold_tau(ROUTE_KMAX, b) for b in beta])
    assert np.allclose(out["required"], expected)
    assert np.allclose(out["shortfall"], expected - tau)


def test_route_constraint_rejects_mismatched_schedule_lengths():
    with pytest.raises(ValueError):
        fc.route_constraint_violations([0.0, 0.5], [1.0], ROUTE_KMAX)


def test_straight_kernel_to_softmax_schedule_fails_the_route_constraint():
    """A tau schedule ramped down in a straight line over the whole run, blind
    to the hard swap in beta, is still deep in the kernel phase (beta=0,
    required = e*KMAX = 10.8731...) partway through -- by t=3 the straight
    ramp has already fallen below that floor, seven steps before the swap
    ever arrives and drops the floor to zero."""
    steps, mid = 21, 10
    beta = _hard_swap_beta(steps, mid)
    tau = np.linspace(12.0, 2.0, steps)  # straight line, ignores the swap
    out = fc.route_constraint_violations(beta, tau, ROUTE_KMAX)
    assert out["steps"] == list(range(3, 10)), out["steps"]
    assert out["worst_step"] == 9  # last step before the swap: biggest shortfall
    assert out["shortfall"][9] == pytest.approx(math.e * ROUTE_KMAX - tau[9])


def test_curved_kernel_to_softmax_schedule_passes_the_route_constraint():
    """Same endpoints (12.0 -> 2.0) as the straight schedule above, but held
    flat through the whole kernel phase and only eased down after the swap,
    once beta=1 has already dropped the floor to zero -- the shape tracks the
    constraint instead of a straight line ignoring it."""
    steps, mid = 21, 10
    beta = _hard_swap_beta(steps, mid)
    tau = np.empty(steps, dtype=np.float64)
    tau[:mid] = 12.0
    tau[mid:] = np.linspace(12.0, 2.0, steps - mid)
    out = fc.route_constraint_violations(beta, tau, ROUTE_KMAX)
    assert out["steps"] == [], out["steps"]
    assert out["worst_step"] is None


def test_literal_census_is_seed_independent_at_the_measured_steps():
    """Not a fluke of one seed: three different seeds land on the same steps
    to the bisection tolerance, because the pattern-aligned inits that find
    these narrow basins carry no randomness at all."""
    G = fc.gram_orthogonal(NORMS)
    ref = _literal_find_steps(G, BETA_9)
    for seed in (1, 42, 7):
        got = _literal_find_steps(G, BETA_9, seed=seed)
        assert len(got) == len(ref), (seed, got, ref)
        for (t0, b0, a0), (t1, b1, a1) in zip(ref, got):
            assert b0 == b1 and a0 == a1, (seed, ref, got)
            assert abs(t0 - t1) < 1e-3, (seed, t0, t1)


# ---------------------------------------------------------------------------
# 8. ROUND TWO, PASS 2: the operating point on the ARM THAT IS ACTUALLY WIRED.
# `smprime_operating_point` runs the REAL `ceq.hf.modeling_ceq.CEQAttention`
# at `operator="smprime"` (24 independent layer draws, see its own docstring
# for why that is exact and not an approximation at init) and answers whether
# the anneal route ever drives a_eff up to 1/e using the model's OWN measured
# key norms, not the census bed's planted 2/4/8. This is slow (real torch
# tensors, 24 layers at hidden=1280) relative to the pure-numpy tests above,
# so the report is computed ONCE per test module and shared.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def operating_point_report():
    return fc.smprime_operating_point()


def test_the_wired_arm_is_read_at_its_own_defaults(operating_point_report):
    """No override was passed: hidden=1280, heads=20 (d_head=64), 24 layers
    -- the ORDER's own numbers -- are what `CEQConfig(operator="smprime")`
    gives without anyone choosing them here."""
    r = operating_point_report
    assert r["d_head"] == 64, r["d_head"]
    assert len(r["max_k2_by_layer"]) == 24, r["max_k2_by_layer"]


def test_qk_is_the_only_value_ever_on_record_for_this_arm(operating_point_report):
    """`ceq/hf/configuration_ceq.py` names the fact directly: no checkpoint
    has ever trained `smprime` at this hidden size, so every layer's `self.qk`
    reads back the constructed corner, 1.0, and `smprime_layer_draws` already
    raises if that were not uniform across layers -- this just checks the
    value the ORDER calls "the trained value" really is that corner."""
    assert operating_point_report["qk"] == 1.0
    assert operating_point_report["tau"] == pytest.approx(64.0 ** 0.5 / 1.0)


def test_measured_max_k2_per_layer_is_in_the_tens_not_at_2_4_8(operating_point_report):
    """(3) of the ORDER. `d_head=64, hidden=1280, initializer_range=0.02`
    puts the fan-in variance at `hidden * initializer_range**2 ~= 0.512` per
    key component, so `||k||^2` (a sum of 64 such components) should read in
    the TENS -- nowhere near the census bed's planted 2, 4 or 8, and nowhere
    near scout 4's 0.0256 (which dropped the sum over 1280 input dims)."""
    per_layer = operating_point_report["max_k2_by_layer"]
    assert all(20.0 < v < 150.0 for v in per_layer), per_layer
    for planted in (0.0256, 2.0, 4.0, 8.0):
        assert all(abs(v - planted) > 10.0 for v in per_layer), (planted, per_layer)


def test_beta_one_control_reads_a_eff_exactly_zero(operating_point_report):
    """The beta=1 control the ORDER asks for: `a_eff = (1-beta)*...` is `0.0`
    at every value of `qk` and `max_k2` purely because its own leading factor
    is `1 - 1 = 0` -- an identity, and it must read back bitwise, not merely
    small."""
    assert operating_point_report["beta1_control_a_eff"] == 0.0


def test_the_anneal_route_crosses_1_over_e_within_the_first_tenth_of_the_way(operating_point_report):
    """(1)+(2) of the ORDER. The route crosses well before beta reaches 0 --
    in fact within the first ~10% of the anneal's own distance from beta=1,
    which is the finding that keeps consequence (8) alive rather than vacuous
    (see `smprime_operating_point`'s own docstring for the measured beta)."""
    crossing = operating_point_report["crossing"]
    assert crossing is not None, "expected a crossing before beta=0"
    assert 0.9 < crossing["beta"] < 1.0, crossing
    assert crossing["a_eff"] > 1.0 / fc.E, crossing
    assert operating_point_report["beta0_a_eff"] > 1.0 / fc.E


def test_measured_crossing_beta_agrees_with_the_closed_form(operating_point_report):
    """The swept, discretized crossing (read off the REAL `arm_beta("anneal",
    ...)` grid via `route_constraint_violations`) and the continuous algebraic
    solution should agree to the grid's own step spacing (~1/599), not merely
    to the same order of magnitude."""
    r = operating_point_report
    assert abs(r["crossing"]["beta"] - r["beta_closed_form"]) < 2.0 / r["steps"]


def test_route_violations_first_step_is_the_reported_crossing(operating_point_report):
    """`smprime_operating_point` reads its crossing off `route_constraint_
    violations` (already written for exactly this predicate) rather than
    re-deriving the inequality a second time -- checked here bitwise."""
    r = operating_point_report
    assert min(r["route_violations"]["steps"]) == r["crossing"]["t"]


def test_realized_loggain_crosses_at_a_lower_beta_than_the_idealized_bound(operating_point_report):
    """`a_eff` assumes the QUERY-ALIGNED case, q_i = k_j exactly -- the fold's
    own idealisation. A real, independently-drawn q rarely reads a dot
    product as large as ||k_j|| itself, so the REALIZED row logit
    (`max_row_max_w`) is smaller than `max_k2/sqrt(d_head)`, and its own
    crossing beta must therefore sit LOWER (further from beta=1, i.e. later
    in the anneal) than the idealized one -- not merely different."""
    r = operating_point_report
    realized_beta = r["realized_loggain"]["beta_closed_form_realized"]
    assert 0.5 < realized_beta < r["crossing"]["beta"], r


# ---------------------------------------------------------------------------
# 9. CONSEQUENCE (2): fold, not cusp. `fold_is_a_fold_not_a_cusp` is the
# analytic half (the single-pattern germ cannot be a cusp at any point on its
# own fold curve, by the algebra of F_cc alone); this section adds the
# EMPIRICAL half, reusing section 6's own `_literal_find_steps` rather than a
# second copy of it -- a cusp point would show up there as two boundary
# events (an open and a close) COINCIDING, which is exactly what section 6
# already measured as NOT happening (disjoint windows, strict gaps).
# ---------------------------------------------------------------------------

def test_the_single_pattern_fold_is_a_fold_not_a_cusp():
    cusp = fc.fold_is_a_fold_not_a_cusp()
    assert abs(cusp["F_c"]) < 1e-9, cusp
    assert abs(cusp["F_c"] - cusp["F_c_fd"]) < 1e-6, cusp
    assert abs(cusp["F_cc"] - cusp["F_cc_fd"]) < 1e-3, cusp
    assert abs(cusp["F_a"] - cusp["F_a_fd"]) < 1e-6, cusp
    assert cusp["F_cc"] > 0.3, cusp   # 1/e, strictly positive -- never the cusp's own F_cc=0
    assert cusp["is_generic_fold"] and not cusp["cusp_degeneracy_present"]
    assert cusp["cusp_rejected"]


def test_no_measured_boundary_event_coincides_so_no_cusp_candidate_exists():
    """A cusp point needs two fold branches to meet TANGENTIALLY -- here, a
    window's close and the next window's open landing at the same tau.
    Section 6 already measured five distinct, gapped events; re-read as the
    cusp question, the smallest gap between any two consecutive events is
    nowhere near zero, so consequence (2) has no candidate point to fit
    `8a^3+27b^2=0` against in the swept range -- REJECTED for want of a
    tangency, not merely unconfirmed."""
    G = fc.gram_orthogonal(NORMS)
    steps = _literal_find_steps(G, BETA_9)
    events = sorted(tau for tau, _before, _after in steps)
    gaps = [b - a for a, b in zip(events, events[1:])]
    assert len(gaps) == 4, (events, gaps)
    assert min(gaps) > 0.05, ("a gap this small would be a cusp candidate", events, gaps)
