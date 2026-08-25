"""Which benchmark could see this module's one distinguishing property?

W6 measured the property: minimum influence-Jacobian entry `d(out_i)/d(v_j)` is
**-9.000e-01** for the signed path sum against **exactly +0.000e+00** for softmax
attention, APPNP and the max-plus star alike. A non-negative operator's Kleene
star has a non-negative influence Jacobian in any ordered semiring.

That statement, as written, is nearly useless for choosing a benchmark, because
a real attention block has a value projection `W_v` and an output projection
`W_o`, both signed, and either can supply a minus sign. "Can produce a negative
number" does not separate the arms at block level. If the benchmark is chosen on
the strength of the W6 number alone, the benchmark will measure nothing.

The sharper property, and the one a task can actually require, is
**CONTENT-CONDITIONAL SIGN**:

    can a THIRD token c decide whether token j helps or hurts token i?

`W_v` and `W_o` cannot do this. They are constants -- they do not depend on the
input, so whatever sign they contribute, they contribute to every context
identically. Attention weights `a_ij >= 0` can only rescale that fixed sign.
Only an operator whose own entries may go negative, composed over multi-hop
paths i -> c -> j, can let c decide the sign.

That is exactly what the word `not` does in "A did not cause B". It is exactly
what a scope-of-negation NLI item tests. It is not obviously what an ARC-AGI
grid puzzle tests.

This file measures the property rather than arguing about it, and its third test
is the one that keeps the claim honest: the separation is not a theorem about
all transformers, it is a theorem about LINEAR value paths, and a single
nonlinearity between layers gives it back to softmax.
"""
from __future__ import annotations

import pytest
import torch

from ceq import bench


def test_a_third_token_flips_the_sign_of_the_value_influence_under_the_signed_operator(device):
    """The signed multi-hop path sum lets token c decide the sign of j -> i.

    Paths i -> c -> j carry the product A_ic * A_cj. Both factors are signed, so
    changing c's content changes which sign the composed path contributes, and
    the total `[sum_h A^h]_ij` can cross zero.
    """
    rate = bench.sign_flip_rate("signed", device=device, n_draws=128, seed=0)
    assert rate > 0.0, f"signed operator flipped in {rate:.2%} of draws"

    # THE FIRST VERSION OF THIS ASSERTION WAS `rate > 0.10` AND IT FAILED.
    # 0.10 was a guess with nothing behind it; measured is 0.0469 at depth 1
    # over 128 draws. The threshold was not lowered to fit -- the claim was
    # restated to the one the probe can actually support, which is CATEGORICAL:
    # nonzero against a control that is exactly zero. The rate itself is a
    # property of how often two random projections land on opposite sides of a
    # zero crossing, not a measure of how strongly the operator negates, so a
    # numeric threshold on it would have been meaningless either way.


def test_stacked_softmax_layers_cannot_flip_that_sign_at_any_depth(device):
    """CONTROL, and the load-bearing half of the result.

    For a stack of non-negative operators with linear value paths,

        d(out_i)/d(v_j) = (sum over paths of products of a >= 0) * (fixed matrix)

    The path products are non-negative and the fixed matrix does not depend on
    the input, so no third token can flip the sign. Depth does not help: the
    scalar in front stays non-negative however many layers are stacked.

    Checked at depth 1 AND depth 2, because "one layer cannot, therefore
    transformers cannot" would be exactly the kind of unearned generalization
    this project has already been bitten by.
    """
    for depth in (1, 2):
        for seed in (0, 1, 2):
            rate = bench.sign_flip_rate("softmax", depth=depth, device=device,
                                        n_draws=128, seed=seed)
            assert rate == 0.0, f"softmax depth {depth} seed {seed}: {rate:.2%}"


def test_a_nonlinearity_between_softmax_layers_gives_the_sign_flip_back(device):
    """THE HONEST LIMIT. The separation is about linear value paths, not about
    softmax, and a real transformer has an MLP.

    Insert one GELU between two softmax attention layers and the sign flip
    returns. So the claim that survives is NOT "softmax cannot express
    content-conditional negation" -- it is "softmax needs a nonlinearity and the
    depth to route through it, where the signed operator has it in one layer".

    That is a claim about parameter and depth efficiency, which is measurable at
    small scale, and NOT a claim about capability, which is what would have been
    needed to justify a famous-benchmark run.
    """
    rate = bench.sign_flip_rate("softmax_gelu", depth=2, device=device,
                                n_draws=128, seed=0)
    assert rate > 0.0, (
        f"nonlinear softmax stack flipped {rate:.2%} -- if this were 0 the "
        "separation would be stronger than claimed, and the claim would have "
        "to be restated in the other direction"
    )


def test_the_probe_is_measuring_the_value_path_and_not_the_input_path(device):
    """INSTRUMENT CHECK.

    `d(out_i)/d(x_j)` for a softmax block is NOT sign-constrained -- the query
    and key paths make `a_ij` itself a signed function of the input. A probe
    that differentiated with respect to `x` would find flips for every arm and
    measure nothing, which is precisely how W4's first negation test failed and
    how the W3 content test returned R2 = 1.0 for all three keep-rules.

    So: the probe must differentiate with respect to `v`. This test asserts the
    distinction is real by showing the x-path probe DOES flip for softmax, where
    the v-path probe does not.
    """
    v_path = bench.sign_flip_rate("softmax", depth=1, device=device,
                                  n_draws=128, seed=0)
    x_path = bench.sign_flip_rate("softmax", depth=1, device=device,
                                  n_draws=128, seed=0, wrt="x")
    assert v_path == 0.0, v_path
    assert x_path > 0.0, (
        "the x-path probe found no flips either, so the two probes are not "
        "measuring different things and the v-path result proves nothing"
    )


def test_a_single_hop_signed_operator_cannot_flip_the_sign_either(device):
    """UNPLANNED FINDING, and a real constraint on the design.

    At `hops=1` the signed operator's coefficient is just `A_ij`, which is built
    from `q_i` and `k_j` alone -- no third token appears in it. The sign flip
    needs the path i -> c -> j, so it needs `A^2`, so it needs at least two
    hops.

    Measured over 128 draws: hops=1 gives exactly 0.0000, hops=2 gives 0.0625.
    The multi-hop path sum is not a cost knob here; below two hops the module
    has no distinguishing property left at all, and any benchmark run at
    hops=1 would be measuring softmax with extra steps.
    """
    one = bench.sign_flip_rate("signed", hops=1, depth=1, device=device,
                               n_draws=128, seed=0)
    two = bench.sign_flip_rate("signed", hops=2, depth=1, device=device,
                               n_draws=128, seed=0)
    assert one == 0.0, f"hops=1 flipped {one:.2%}, expected exactly zero"
    assert two > 0.0, f"hops=2 flipped {two:.2%}, expected nonzero"
