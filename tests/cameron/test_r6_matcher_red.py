"""CAMERON round 6 it.0 -- the RED tests behind the Hungarian matcher (1.5).

Contract 1.5 asks for `C_ij = |‖k_causal_i‖ − ‖k_filler_j‖|` and
`π* = argmin_π Σ_i C_{i π(i)}`, with the residual imbalance `Σ C_{i π*(i)} / n`
and the band/tail stratification printed per table, and a must-fire calibration
in BOTH directions.

Each test below is written against `scale.matcher` as it does not yet exist, so
the first run is RED by import. What each one is FOR:

* `test_cost_matrix_is_abs_norm_difference` pins the cost to the contract's
  formula rather than to any monotone stand-in.

* `test_match_attains_the_monge_optimum` is the load-bearing one. The cost
  `|a_i − b_j|` on the line is a Monge matrix, so the optimal assignment is
  exactly the sorted pairing. That gives an **independent closed-form oracle**
  for the optimum, computed without the solver under test. An implementation
  that returned the identity permutation, or any greedy pairing, fails this;
  "it ran without raising" does not pass it.

* `test_match_beats_the_identity_pairing` is the anti-instrument-#15 test. A
  self-match on the *same array in the same order* is satisfied by the identity
  permutation, so a matcher with its solver deleted would pass a naive
  "residual ≈ 0" check. Here the two pools are independent draws from one
  population, where the identity pairing has a large residual and only a real
  assignment drives it down.

* The two calibration directions are separate tests and must BOTH fire. Round 5
  shipped four gates that measured something adjacent to what was
  pre-registered; a calibration that can only read the null is the fifth.

* `test_strata_are_never_pooled` encodes F-selector: the filler pool is at least
  two populations, and a matcher that is allowed to pay for a causal token with
  a token from the other stratum has hidden the confound it was built to expose.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

from scale import matcher                                          # noqa: E402

SEED = 20260826


def _norms(n: int, loc: float, scale: float, seed: int) -> np.ndarray:
    """A synthetic key-norm pool. Positive by construction: a key norm is."""
    g = np.random.default_rng(seed)
    return np.abs(g.normal(loc, scale, size=n))


def _sorted_pairing_residual(a: np.ndarray, b: np.ndarray) -> float:
    """The closed-form optimum for `|a_i − b_j|`, computed WITHOUT the solver.

    `C_ij = |a_i − b_j|` satisfies the Monge condition on the line, so sorting
    both pools and pairing rank-for-rank is an optimal assignment. This is the
    oracle the solver is checked against; it is deliberately not imported from
    the module under test.
    """
    return float(np.abs(np.sort(a) - np.sort(b)).sum() / len(a))


# --------------------------------------------------------------- the cost


def test_cost_matrix_is_abs_norm_difference():
    a = _norms(9, 25.0, 3.0, SEED)
    b = _norms(11, 22.0, 3.0, SEED + 1)
    c = matcher.cost_matrix(a, b)
    assert c.shape == (9, 11)
    for i in range(9):
        for j in range(11):
            assert c[i, j] == pytest.approx(abs(a[i] - b[j]), abs=0.0, rel=1e-12)


# ------------------------------------------------- the assignment is optimal


def test_match_attains_the_monge_optimum():
    """The residual must equal the closed-form optimum, not merely be finite."""
    for n, seed in ((8, SEED), (64, SEED + 7), (257, SEED + 13)):
        a = _norms(n, 28.0, 4.0, seed)
        b = _norms(n, 25.0, 4.0, seed + 1000)
        res = matcher.match(a, b)
        assert res.residual == pytest.approx(_sorted_pairing_residual(a, b), rel=1e-12)
        # a permutation, not a multiset with repeats
        assert sorted(res.filler_idx.tolist()) == list(range(n))
        assert sorted(res.causal_idx.tolist()) == list(range(n))


def test_match_beats_the_identity_pairing():
    """Anti-#15. Two independent draws from ONE population.

    The identity pairing is what a matcher with its solver deleted returns. It
    must be strictly and largely worse than the assignment, or the null-direction
    calibration below proves nothing about the solver.
    """
    n = 128
    a = _norms(n, 25.0, 4.0, SEED + 21)
    b = _norms(n, 25.0, 4.0, SEED + 22)
    identity = float(np.abs(a - b).sum() / n)
    res = matcher.match(a, b)
    assert res.residual < identity / 4.0, (res.residual, identity)


# --------------------------------------------- calibration, both directions


def test_calibration_null_direction_residual_and_effect_are_zero():
    """Same population both sides: residual ≈ 0 AND effect CI contains zero.

    The bar is derived before the run rather than read off it. Matching two
    independent size-n draws from one population pairs order statistic to order
    statistic, so the residual is a sampling artifact of size O(sd·n^-0.5), not
    zero. At n=256 that is ~0.1·sd, so 0.20·sd is the bar; 0.05·sd was written
    first here and is arithmetically unreachable by any correct matcher, which
    is a defect in the bar and not in the instrument.
    """
    row = matcher.calibrate_null(n=256, seed=SEED)
    assert row.residual < 0.20 * row.pooled_sd, row
    assert row.effect_ci[0] <= 0.0 <= row.effect_ci[1], row
    assert row.fired is True, row


def test_null_residual_is_sampling_noise_and_shrinks_as_n_grows():
    """The null residual must behave like O(n^-0.5), not like a constant.

    This is the test that separates "the residual is small because the pools
    genuinely match" from "the residual is small because something is clamped".
    A bug that returns a fixed small number passes the bar above and fails here.
    """
    small = matcher.calibrate_null(n=256, seed=SEED).residual
    large = matcher.calibrate_null(n=4096, seed=SEED).residual
    assert large < small / 2.0, (small, large)


def test_calibration_separated_direction_residual_and_effect_are_nonzero():
    """Deliberately separated populations: residual > 0 AND effect CI excludes zero.

    This is the direction round 5 kept omitting. A control that cannot be
    nonzero is not a control. With equal spreads the sorted pairing is
    quantile-to-quantile, so the residual converges on the shift itself: the
    matcher reports the imbalance it cannot remove instead of hiding it.
    """
    row = matcher.calibrate_separated(n=256, seed=SEED, shift=6.0)
    assert row.residual > 0.5 * row.pooled_sd, row
    assert row.effect_ci[0] > 0.0 or row.effect_ci[1] < 0.0, row
    assert row.fired is True, row


def test_calibration_selection_direction_removes_a_planted_confound():
    """Direction 3, and the only one whose EFFECT can detect a deleted solver.

    Directions 1 and 2 match equal-size pools, so the assignment is a full
    bijection and the matched multiset -- hence Cohen's d -- is identical under
    any permutation. Their effect halves are permutation-invariant by
    construction and cannot test the assignment at all. This direction uses the
    rectangular shape the project actually has, where the assignment selects
    WHICH filler units are used.
    """
    row = matcher.calibrate_selection(n_causal=64, n_filler=2000, seed=SEED)
    assert abs(row.effect_unmatched) > 1.0, row       # the confound is SEEN
    assert row.effect_ci[0] <= 0.0 <= row.effect_ci[1], row   # and removed
    assert abs(row.effect) < abs(row.effect_unmatched) / 3.0, row
    assert row.fired is True, row


def test_the_selection_direction_fails_when_the_solver_is_deleted(monkeypatch):
    """The mutation the other two directions survive.

    Replacing the assignment with the identity permutation must break this
    direction. Without this test the calibration would be satisfiable by a
    matcher that never matched.
    """
    monkeypatch.setattr(matcher, "_ORACLE_TOL", float("inf"))
    monkeypatch.setattr(matcher, "_lsa",
                        lambda c: (np.arange(min(c.shape)), np.arange(min(c.shape))))
    row = matcher.calibrate_selection(n_causal=64, n_filler=2000, seed=SEED)
    assert row.fired is False, row


def test_the_shipped_interval_re_runs_the_match_and_the_naive_one_does_not():
    """The naive paired bootstrap is invalid here and must not be what gates.

    Matching is part of the estimator, so a resample has to re-run it. Resampling
    the matched pairs treats the pair differences as independent when rank-to-rank
    pairing has made them share the difference in pool means. The consequence is
    not academic: on null data the naive interval excludes zero.
    """
    n = 256
    rng = np.random.default_rng(SEED)
    a, b = matcher._pool(n, 25.0, 4.0, rng), matcher._pool(n, 25.0, 4.0, rng)
    ya, yb = matcher._outcome(a, rng), matcher._outcome(b, rng)
    m = matcher.match(a, b)

    lo, hi = matcher.rematch_boot_ci(a, b, ya, yb, seed=SEED)
    nlo, nhi = matcher.naive_pair_boot_ci(ya[m.causal_idx], yb[m.filler_idx],
                                          seed=SEED)
    assert (hi - lo) > 1.5 * (nhi - nlo), ((lo, hi), (nlo, nhi))
    # and the row the module reports must carry the re-match interval
    row = matcher.calibrate_null(n=n, seed=SEED)
    assert row.effect_ci == pytest.approx((lo, hi), rel=1e-12)


def test_null_interval_covers_zero_at_about_its_nominal_rate():
    """A single seed cannot settle a null: a 95% interval misses 5% of the time.

    Over 24 seeds a correct interval should contain zero for the large majority.
    A systematically biased matcher, or an interval as narrow as the naive one,
    fails here even when it passes on a lucky seed.
    """
    hits = sum(1 for s in range(24)
               if (lambda r: r.effect_ci[0] <= 0.0 <= r.effect_ci[1])(
                   matcher.calibrate_null(n=256, seed=SEED + s)))
    assert hits >= 20, hits


# ------------------------------------------------------------- the strata


def test_strata_are_never_pooled():
    """No matched pair may cross a stratum, and no pooled residual is offered."""
    a = _norms(40, 28.0, 2.0, SEED + 31)
    band = _norms(40, 25.0, 2.0, SEED + 32)
    tail = _norms(40, 8.0, 2.0, SEED + 33)
    filler = np.concatenate([band, tail])
    labels = np.array(["band"] * 40 + ["tail"] * 40)

    out = matcher.match_strata(a, filler, labels)
    assert set(out) == {"band", "tail"}
    for name, res in out.items():
        assert np.all(labels[res.filler_idx] == name), name
    # the whole point: the two strata do NOT share a residual
    assert out["band"].residual != out["tail"].residual
    assert not hasattr(out, "residual")


def test_pooling_beats_every_stratum_by_identity_not_by_measurement():
    """Labelled as what it is: an identity, kept only as an implementation check.

    A pooled match minimises over a strictly larger feasible set than any
    stratified match, so `pooled <= every stratum residual` cannot fail for a
    correct implementation and cannot fail for an incorrect one either as long as
    it still optimises. Measured over 400 drawn instances with randomised pool
    sizes and separations: 400/400 strict, 0/400 equal, 0/400 violations.

    It is recorded here so nobody later mistakes it for evidence about strata.
    The claim that carries content is the magnitude, in the test below.
    """
    a = _norms(40, 28.0, 2.0, SEED + 31)
    band = _norms(40, 25.0, 2.0, SEED + 32)
    tail = _norms(40, 8.0, 2.0, SEED + 33)
    filler = np.concatenate([band, tail])
    labels = np.array(["band"] * 40 + ["tail"] * 40)
    pooled = matcher.match(a, filler).residual
    strat = matcher.match_strata(a, filler, labels)
    assert pooled <= min(m.residual for m in strat.values()) + 1e-12


def test_pooling_understates_the_tail_by_a_margin_that_could_have_been_small():
    """The measured claim: the gap is LARGE, not merely present.

    "Pooled is smaller" is forced. "Pooled is less than half the tail's residual
    in almost every drawn instance" is not -- a geometry where the band and tail
    sat close together would falsify it, and that is a geometry this project
    could plausibly have had. Instances are drawn rather than hand-built, because
    a hand-built minimal case is exactly where a control goes vacuous: the
    smallest instance is usually one where the right and wrong answers agree.
    """
    rng = np.random.default_rng(SEED + 99)
    hits = 0
    trials = 40
    for _ in range(trials):
        nc = int(rng.integers(4, 24))
        ca = matcher._pool(nc, 28.0, 2.0, rng)
        bd = matcher._pool(nc, 25.0, 2.0, rng)
        tl = matcher._pool(int(rng.integers(50, 200)), 10.0, 3.0, rng)
        filler = np.concatenate([bd, tl])
        labels = np.array(["band"] * nc + ["tail"] * tl.size)
        pooled = matcher.match(ca, filler, verify=False).residual
        strat = matcher.match_strata(ca, filler, labels, verify=False)
        if strat["tail"].residual > 2.0 * pooled:
            hits += 1
    assert hits >= int(0.9 * trials), (hits, trials)


# ---------------------------------------------------------- provenance


def test_module_reports_which_assignment_path_it_took():
    assert matcher.ASSIGNMENT_PATH in (
        "scipy.optimize.linear_sum_assignment",
        "monge-sorted",
    )
