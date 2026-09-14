"""What the beta switch actually MOVES, with the gate at its default.

RED-FIRST, and the RED is the point of the file. The naive reading of
`lean/CEQ/V16Domain.lean:433 three_corners_containment` -- one family whose beta
interior runs BETWEEN softmax attention and linear attention -- says the mixing
pattern must change along beta, because softmax and linear attention are
different operators (`:445 corners_are_distinct` proves they are distinct
points). The first assertion written here was that reading:

    def test_the_beta_interior_moves_the_mixing_pattern():
        soft = _row_normalised(1.0)
        lin  = _row_normalised(0.0)
        gap  = float((soft - lin).abs().max())
        assert gap > 1e-6

and it failed, verbatim, at commit b4c6620 on WIN-16QAL06O9GB, python 3.11.9,
torch 2.14.0+cpu, `python -m pytest tests/curvature/test_beta_axis_is_a_row_gain.py -q`,
exit 1:

    E  AssertionError: the beta axis left the row-normalised mixing pattern
    E  unmoved: worst |softmax_row - linear_row| = 1.110223e-16 over a
    E  [3, 12, 12] operator
    E  assert 1.1102230246251565e-16 > 1e-06

The corners ARE distinct -- `corners_are_distinct` is not wrong -- but they are
distinct only by a PER-ROW SCALAR. `W_ij(beta) = num_ij / Z_i^beta` divides the
whole row i by one number, and `num_ij` carries no beta at all, so every row of
every beta is the same probability vector up to a gain `Z_i^(1-beta)`. With the
gate at its default (`u = 1`, `theta = 0`, which is `g = 0`'s corner reached
through `blend`) the beta axis of the containment is a LEARNED PER-ROW GAIN ON
SOFTMAX and nothing else. That is what these tests now assert.
"""
import pytest
import torch

import ceq.arm_smprime as arm

SEED = 5501
S, DK, DV = 12, 8, 5
BETAS = (0.0, 0.37, 1.0)
TOL = 1e-13


def _draw(seed=SEED):
    g = torch.Generator().manual_seed(seed)
    kw = dict(generator=g, dtype=torch.float64)
    return (torch.randn(3, S, DK, **kw), torch.randn(3, S, DK, **kw),
            torch.randn(3, S, DV, **kw))


def _rows(beta):
    q, k, _ = _draw()
    return arm.operator(q, k, beta=beta).real


def _row_normalised(beta):
    w = _rows(beta)
    return w / w.sum(-1, keepdim=True)


@pytest.mark.parametrize("beta", BETAS)
def test_row_normalised_operator_is_beta_invariant(beta):
    """The replacement for the RED: normalise the row and beta vanishes."""
    ref = _row_normalised(1.0)
    got = _row_normalised(beta)
    worst = float((got - ref).abs().max())
    assert worst < TOL, "beta=%r moved the normalised row by %.6e" % (beta, worst)


@pytest.mark.parametrize("beta", BETAS)
def test_the_whole_beta_axis_is_the_gain_Z_to_the_one_minus_beta(beta):
    """And the scalar it IS: `W(beta) = softmax * Z^(1 - beta)`, exactly."""
    q, k, _ = _draw()
    _, mod = arm.numerator(q, k)
    z = mod.sum(-1, keepdim=True)
    predicted = _row_normalised(1.0) * z ** (1.0 - beta)
    worst = float((_rows(beta) - predicted).abs().max())
    assert worst < TOL, ("beta=%r is not the gain Z^(1-beta) on softmax: worst "
                         "%.6e" % (beta, worst))


def test_the_gain_is_the_only_thing_a_downstream_normaliser_would_see():
    """A per-row gain is what an RMS/LayerNorm after the read removes. The
    read-outs at two betas, each divided by its own row RMS, agree."""
    q, k, v = _draw()
    def unit(beta):
        o = arm.readout(q, k, v, beta=beta).real
        return o / o.pow(2).mean(-1, keepdim=True).sqrt()
    worst = float((unit(0.0) - unit(1.0)).abs().max())
    assert worst < 1e-12, "a row normaliser did not absorb the beta gain: %.6e" % worst


def test_the_gate_is_the_axis_that_does_move_the_pattern():
    """The kill needs a replacement route and this is it, measured: the GATE
    head, not beta, is where the family's extra mixing lives. A non-unit
    magnitude head moves the normalised row far past every tolerance above."""
    q, k, _ = _draw()
    g = torch.Generator().manual_seed(SEED + 1)
    u = torch.rand(3, S, generator=g, dtype=torch.float64)
    th = torch.zeros(3, S, dtype=torch.float64)
    w = arm.operator(q, k, u, th, beta=1.0, g=1.0).real
    gated = w / w.sum(-1, keepdim=True)
    moved = float((gated - _row_normalised(1.0)).abs().max())
    assert moved > 1e-2, ("the gate did not move the normalised row either, so "
                          "the family has no interior at all: %.6e" % moved)
