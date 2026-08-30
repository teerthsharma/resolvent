"""The trend verdict's own error rates, measured on drawn tables.

`scale/page_trend.py` replaces an eyeball with a branch, and a branch with no
measured error rate is the same eyeball wearing a p-value. Every table below is
DRAWN, never hand-built, and the count of draws on which each clause fires is
asserted rather than described.

WHAT THE SWEEP FOUND, AND IT IS THE REASON THE VERDICT IS A CONJUNCTION.
Over 200 drawn tables per arm at `n_boot=300` (seed 11, this machine):

    flat truth      trend clause 0.065   size clause 0.325   RISES 0.040
    rising truth    trend clause 0.810   size clause 1.000   RISES 0.810

The SIZE clause alone fires on a third of flat tables -- six times its nominal
rate -- because PAVA's pooling is one-sided and lifts a low top rung. The TREND
clause alone is close to nominal. Their CONJUNCTION reads 0.040 against a
nominal 0.05, because the trend clause gates the miscalibrated one. That is the
whole argument for requiring both, and it is measured, not asserted.
"""
import numpy as np
import pytest

from scale.page_trend import (ALPHA, isotonic_curve, isotonic_top_ci,
                              leave_one_block_out, page_l, page_null,
                              page_p_value, page_p_value_monte_carlo,
                              verdict_of)

BOOT = 200          # smaller than the shipped N_BOOT; this file measures rates,
DRAWS = 60          # not intervals, and the rates converge far faster


def clauses(blocks, n_boot=BOOT):
    """(trend fires, size fires, unconstrained size fires) for one table."""
    L, _sums, _ties = page_l(blocks)
    p = page_p_value(L, blocks.shape[1], blocks.shape[0])
    iso = isotonic_top_ci(blocks, n_boot=n_boot, seed=0)
    return (p < ALPHA, iso["positive_excluding_zero"],
            iso["raw_excludes_zero"] and iso["raw_top"] > 0.0)


def sweep(shift, draws=DRAWS, seed=11):
    rng = np.random.default_rng(seed)
    trend = size = raw = both = 0
    for _ in range(draws):
        b = rng.normal(scale=0.03, size=(5, 4)) + np.arange(4) * shift
        t, s, r = clauses(b)
        trend += t
        size += s
        raw += r
        both += t and s
    return {k: v / draws for k, v in
            dict(trend=trend, size=size, raw=raw, rises=both).items()}


def test_the_exact_null_is_a_distribution_on_the_right_support():
    support, dist = page_null(4, 5)
    assert abs(dist.sum() - 1.0) < 1e-12
    assert (support[0], support[-1]) == (100.0, 150.0)
    assert abs(float(support @ dist) - 125.0) < 1e-9
    # symmetric about its mean: the ordered alternative is one-sided, and a
    # symmetric null is what makes the one-sidedness a choice rather than an
    # artefact of the statistic
    assert np.allclose(dist, dist[::-1])


def test_the_exact_and_permutation_paths_agree():
    """Two paths that fail differently: a convolution over the k! per-block
    values against a permutation draw. A support off-by-one breaks one only."""
    rng = np.random.default_rng(3)
    for _ in range(4):
        b = rng.normal(size=(5, 4)) + np.arange(4) * 0.3
        L, _s, ties = page_l(b)
        assert not ties
        exact = page_p_value(L, 4, 5)
        mc = page_p_value_monte_carlo(L, 4, 5, draws=100000, seed=1)
        assert abs(exact - mc["p"]) < 4.0 * max(mc["se"], 1e-9)


def test_page_l_is_maximal_on_a_perfect_rise_and_minimal_on_a_perfect_fall():
    up = np.array([[1.0, 2.0, 3.0, 4.0]] * 5)
    assert page_l(up)[0] == 150.0
    assert page_p_value(150.0, 4, 5) < 1e-6
    assert page_l(up[:, ::-1])[0] == 100.0
    assert abs(page_p_value(100.0, 4, 5) - 1.0) < 1e-9


def test_isotonic_is_a_constrained_fit_and_not_a_smoother():
    assert np.allclose(isotonic_curve([1.0, 2.0, 3.0, 4.0]), [1.0, 2.0, 3.0, 4.0])
    assert np.allclose(isotonic_curve([3.0, 1.0]), [2.0, 2.0])
    # the pooling is ONE-SIDED: a violated top comes back RAISED, never lowered
    fit = isotonic_curve([0.0, 0.0, 0.4, -0.4])
    assert fit[-1] > -0.4


def test_the_branch_table_is_exhaustive_over_its_two_clauses():
    seen = {}
    for p in (0.001, 0.5):
        for positive in (True, False):
            row, sentence = verdict_of(
                {"p": p, "L": 139.0},
                {"positive_excluding_zero": positive, "top": 0.1,
                 "ci_lo": 0.01, "ci_hi": 0.2})
            seen[(p < ALPHA, positive)] = row
            assert sentence
    assert len(seen) == 4
    assert seen[(True, True)] == "RISES"
    assert seen[(False, False)] == "FLAT"
    assert seen[(True, False)] == "SPLIT"
    assert seen[(False, True)] == "SPLIT"


@pytest.mark.slow
def test_the_verdict_discriminates_a_flat_truth_from_a_rising_one():
    """THE ANTI-VACUITY CHECK. Both arms drawn from the same generator, one
    with a planted trend and one without. A verdict that fires on both, or on
    neither, is measuring nothing."""
    flat = sweep(0.00)
    rising = sweep(0.02)
    assert 0.0 < flat["rises"] <= 0.20, flat
    assert rising["rises"] >= 0.50, rising
    assert rising["rises"] > flat["rises"] + 0.3


@pytest.mark.slow
def test_the_size_clause_alone_is_miscalibrated_and_the_conjunction_is_not():
    """The measured reason the branch requires BOTH clauses. Asserted, because
    it is the load-bearing design decision in `page_trend.py` and a later edit
    that dropped the trend clause would leave this failing."""
    flat = sweep(0.00)
    assert flat["size"] > 0.15, flat            # the isotonic clause over-fires
    assert flat["size"] > flat["rises"] + 0.15  # the trend clause gates it
    assert flat["trend"] <= 0.15                # the gate is near nominal


def test_leave_one_block_out_returns_one_refit_per_block():
    rng = np.random.default_rng(5)
    b = rng.normal(size=(5, 4))
    out = leave_one_block_out(b, n_boot=BOOT)
    assert [r["dropped"] for r in out] == [0, 1, 2, 3, 4]
    assert all(80 <= r["L"] <= 120 for r in out)    # k=4, n=4 support is [80,120]
    assert all(isinstance(r["rises"], bool) for r in out)


def test_the_real_ladder_reading_is_reproduced_exactly():
    """The shipped numbers, recomputed. If the journal or `e_ladder.read`
    changes underneath this, the report's quoted figures are stale and this
    fails rather than the report quietly drifting."""
    from scale.page_trend import from_ladder
    cur = from_ladder()
    assert cur["page"]["L"] == 139.0
    assert abs(cur["page"]["p"] - 0.016724) < 1e-6
    assert abs(cur["isotonic"]["top"] - 0.016035) < 1e-6
    assert abs(cur["isotonic"]["ci_lo"] - 0.002826) < 1e-6
    assert abs(cur["isotonic"]["raw_ci_lo"] - (-0.004711)) < 1e-6
    assert cur["row"] == "RISES"
    assert cur["stability_rises"] == 2
    # Row G's scope constraint must travel with the verdict. Three of the four
    # rungs carry a cell at or above predict-the-mean and are credited nothing;
    # a run that silently lost this field would let RISES be read as a
    # capability claim.
    assert cur["uncredited"] == ["e3_t2", "e3_t8", "e3_t32"]
    assert not cur["all_credited"]
    assert cur["cell_means"]["e3_t32"]["settled"] > 1.0
    assert cur["cell_means"]["e3_t32"]["twin"] > 1.0
    assert "credited" in __import__("scale.page_trend", fromlist=["report"]
                                    ).report()
