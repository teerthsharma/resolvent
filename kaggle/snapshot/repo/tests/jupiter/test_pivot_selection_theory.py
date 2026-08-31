"""The two derivations `results/r9_maths_survey.md` rests on, as runnable checks.

Both are theorems about this repo's own objects rather than citations, so both
are asserted rather than described. The headline constants are pinned so that a
later edit to `scale/pivot_selection_theory.py` cannot quietly move a number the
survey quotes.
"""
import numpy as np

import scale.pivot_selection_theory as T


def test_the_module_self_checks_run():
    T.demo()


def test_the_content_blind_rate_is_one_over_s_and_k_cancels():
    """FACT 1. Two paths: the closed form, and drawing. `k` cancelling is the
    whole point -- it is why K4 could not have read anything about selection."""
    for s in (16, 64, 256, 1024):
        assert abs(T.content_blind_rate(s, 8) - 1 / s) < 1e-15
        rates = {T.content_blind_rate(s, k) for k in (1, 2, 4, 8) if k <= s}
        assert len(rates) == 1, f"k did not cancel at s={s}: {rates}"
        sampled = T.sampled_content_blind_rate(s, 8, draws=100000, seed=0)
        assert abs(sampled - 1 / s) < 0.15 / s + 1e-9, (s, sampled)


def test_captured_mass_is_supermodular_on_a_nonnegative_operator():
    """The finding that kills the obvious approach: greedy has NO 1-1/e on
    `||A[:,P]A[P,:]||_F^2`, because its second difference is +2*M_ab >= 0."""
    rng = np.random.default_rng(0)
    positive = 0
    for _ in range(50):
        a = rng.random((5, 5))
        m = T.pairwise_M(a)
        for i, j in ((0, 1), (1, 3), (2, 4)):
            d = T.second_difference(lambda p: T.captured_mass(a, p), (), i, j)
            assert abs(d - 2 * m[i, j]) < 1e-8 * max(1.0, abs(d))
            assert d >= -1e-9
            if d > 1e-9:
                positive += 1
    assert positive > 100, "no strictly increasing return seen; check is vacuous"


def test_reconstruction_is_monotone_submodular_on_a_nonnegative_operator():
    """And therefore carries the 1-1/e. Monotonicity against the DERIVED lower
    bound on the marginal gain, not merely against zero."""
    rng = np.random.default_rng(1)
    for _ in range(30):
        a = rng.random((5, 5))
        m = T.pairwise_M(a)
        for i, j in ((0, 1), (2, 3)):
            d = T.second_difference(lambda p: T.reconstruction_gain(a, p),
                                    (4,), i, j)
            assert abs(d + 2 * m[i, j]) < 1e-8 * max(1.0, abs(d))
            assert d <= 1e-9
        for i in range(5):
            gain = T.reconstruction_gain(a, (i,)) - T.reconstruction_gain(a, ())
            ua = np.outer(a[:, i], a[i, :])
            assert gain >= (ua * ua).sum() - 1e-9
    assert T.reconstruction_gain(rng.random((5, 5)), ()) == 0.0


def test_the_signed_operator_voids_the_guarantee():
    """The PASS half is non-degenerate: nonnegative operators keep submodularity
    on every draw, signed ones lose it on nearly every draw. A control that read
    the same on both would be measuring the draw, not the sign."""
    rng = np.random.default_rng(2)
    kept = sum(1 for _ in range(200)
               if T.submodular_objectives(rng.random((6, 6)))
               ["reconstruction_submodular"])
    lost = sum(1 for _ in range(200)
               if not T.submodular_objectives(rng.normal(size=(6, 6)))
               ["reconstruction_submodular"])
    assert kept == 200, f"nonnegative lost it {200 - kept}/200"
    assert lost > 150, f"signed kept it {200 - lost}/200"


def test_the_greedy_bound_constant_is_pinned():
    assert abs(T.GREEDY_BOUND - 0.6321205588285577) < 1e-15
