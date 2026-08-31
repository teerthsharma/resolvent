"""The e3-harmonic ladder: its oracle, its two gates, and the band it cannot reach.

Every assertion here is a property of `scale/e4_harmonic.py` that a wrong
implementation breaks. Three of them are the round's gates and one is the round's
kill: `test_lambda2_band_is_unreachable_on_a_single_bridge` fails if a future edit
ever quietly moves the substrate into the contract's `lambda_2` band, because doing
so would mean the bottleneck is gone and with it the reason the label is global.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ceq.rips import rips_edges, sample_sphere
from scale.rips_gate import FAIL_BAR, PASS_BAR, nrmse
import scale.e4_harmonic as EH


# --------------------------------------------------------------------------
# the accelerator must not change the corpus
# --------------------------------------------------------------------------

def test_numpy_edges_match_the_ported_reference():
    """`_np_edges` is an accelerator, so it must be bit-identical to the port.

    `ceq/rips.py` carries a provenance claim that its arithmetic is unchanged from
    the upstream file. An accelerator that disagreed anywhere would silently void
    that claim, so the two are compared on a case from the ported corpus itself.
    """
    points = sample_sphere(256, 0x33960002)
    degree = 0.75 * math.log(256.0)
    assert sorted(EH._np_edges(points, degree)) == sorted(rips_edges(points, degree))


# --------------------------------------------------------------------------
# the oracle is an absorbing-chain solve, and it is the fixed point
# --------------------------------------------------------------------------

def test_solve_is_the_fixed_point_of_the_iteration():
    """`B = N R` must equal the limit of `z <- Q z + R`, to solver precision.

    This is the whole reason the oracle qualifies: the label is defined by a fixed
    point, and the two routes to it -- inverting `I - Q` and iterating -- have to
    agree. If they did not, one of them would not be computing an equilibrium.
    """
    q, r, _transient = EH.absorbing_chain(*EH.case_graph(*EH.SMALL_CASE[1:]))
    z_star = EH.fixed_point(q, r)
    z = np.zeros_like(r)
    for _ in range(20_000):
        z = q @ z + r
    assert np.abs(z - z_star).max() < 1e-9


def test_the_label_is_not_a_closed_form_of_a_bounded_neighbourhood():
    """A `k`-hop reading must be WRONG at every `k` below the dial and exact at it.

    Exactness at `k = t*` is what makes the rung well-posed; being bounded away
    below it is what makes the rung require the iteration. Both are needed: a rung
    that is exact at `k = 1` is a static task wearing an equilibrium's name.
    """
    q, r, _transient = EH.absorbing_chain(*EH.case_graph(*EH.SMALL_CASE[1:]))
    for t in EH.LADDER_T:
        label = EH.hop_reading(q, r, t)
        assert nrmse(EH.hop_reading(q, r, t), label) == 0.0
        for k in range(t):
            assert nrmse(EH.hop_reading(q, r, k), label) > 1e-3, (t, k)


def test_the_truncation_ladder_tightens_monotonically():
    """Gate (a). The reading must improve at every rung, never worsen."""
    q, r, _transient = EH.absorbing_chain(*EH.case_graph(*EH.SMALL_CASE[1:]))
    label = EH.fixed_point(q, r)
    errs = [nrmse(EH.hop_reading(q, r, k), label) for k in (0, 1, 2, 4, 8, 16, 32)]
    assert all(b <= a + 1e-12 for a, b in zip(errs, errs[1:])), errs
    assert errs[0] >= 1.0


# --------------------------------------------------------------------------
# gate (b), BOTH halves, and the PASS half carries its own non-degeneracy check
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shipped():
    return EH.measure(*EH.SHIPPED_CASE[1:])


def test_decoder_fails_on_the_fixed_point(shipped):
    """Gate (b), FAIL half. The static local decoder must not read the label.

    `FAIL_BAR` is `scale/rips_gate.py`'s own bar, unchanged, and the decoder is
    given the strongest strictly-local features available -- degree, ball sizes out
    to radius 5, and whether either absorbing node is inside each of those balls.
    The last group is the feature that struck E4 at iteration 11, so it is included
    deliberately rather than omitted to make the gate easier.
    """
    assert shipped["decoder"]["fixed_point"] >= FAIL_BAR


def test_decoder_passes_at_rung_one_and_that_label_is_non_degenerate(shipped):
    """Gate (b), PASS half -- and the check the fourteenth vacuous control lacked.

    A decoder that fails everywhere fails for the wrong reason, so the same decoder,
    the same feature matrix and the same train/test split are shown reading rung 1,
    whose label is one step of the iteration and therefore genuinely local. The
    non-degeneracy check is the second assertion: rung 1's label must actually vary.
    A constant label would make this control unfailable, which is the defect struck
    at iteration 11 on `SupercriticalDense_S2Rips_256`.
    """
    assert shipped["decoder"]["t1"] <= PASS_BAR
    assert shipped["label_sd"]["t1"] > 0.0
    assert 0.0 < shipped["label_nonzero"]["t1"] < 1.0


def test_planted_degree_control_fires_on_the_same_instances(shipped):
    """The absence is worthless without this: the fit works on these exact rows."""
    assert shipped["decoder"]["planted_degree"] <= PASS_BAR


def test_the_leak_channel_is_isolated(shipped):
    """Ball sizes alone must be the mean predictor, so the leak is attributable.

    Reporting one decoder number leaves it unknown WHICH local feature read the
    label. Splitting the feature set says it exactly, and it is the difference
    between a gate and an anecdote.
    """
    assert shipped["decoder"]["ball_sizes_only"] >= FAIL_BAR


def test_the_decoder_loses_ground_along_the_ladder(shipped):
    """The dose the gate itself shows: further along the ladder, less legible.

    Rung `t*`'s label has support inside a ball of radius `t*`, so a fixed-radius
    decoder should lose ground as the rung grows and the fixed point should be the
    least legible of all. The ENDPOINTS are what this asserts, because they are what
    is load-bearing and they hold with margin.

    STRICT MONOTONICITY DOES NOT HOLD AND THE CLAIM WAS WRONG WHEN FIRST WRITTEN.
    The measured sequence on the shipped case is `0.000002, 0.393213, 0.756967,
    0.731839, 0.973819`, so `t8 -> t32` DIPS by `0.025127`. The fit is deterministic
    on a fixed split, so that dip is a property of the labels and not sampling
    spread. Asserting monotonicity with a tolerance wide enough to swallow `0.025127`
    would be a bar fitted after the number was seen, so the assertion is the
    constant-free statement instead and the dip is recorded here and in `report()`.
    """
    seq = [shipped["decoder"][k] for k in ("t1", "t2", "t8", "t32", "fixed_point")]
    assert shipped["decoder"]["fixed_point"] == max(seq), seq
    assert shipped["decoder"]["t1"] == min(seq), seq


# --------------------------------------------------------------------------
# the kill: the contract's own two clauses contradict each other here
# --------------------------------------------------------------------------

def test_lambda2_band_is_unreachable_on_a_single_bridge(shipped):
    """Cheeger forbids the contract's band on any one-edge-bridged graph.

    The spectral gap obeys `g <= 2*phi`. A single bridge between two components of
    volume `vol` has `phi = 1/vol`, so `g <= 2/vol` and `t_rel >= vol/2`. The
    contract asks for `t_rel` in `[10, 20]`, which would need `vol <= 40` -- forty
    edge-endpoints total, a graph far too small to carry a global label at all.

    This test is the finding, kept live. It fails if the substrate is ever moved
    into the band, because arriving there means the bottleneck was removed, and the
    bottleneck is the only reason the label is not local.
    """
    assert shipped["t_rel"] > 20.0
    assert shipped["t_rel"] >= shipped["cheeger_t_rel_floor"]
