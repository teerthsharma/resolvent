"""RED-first: does the predictor's VALUE AXIS give the corner rule a referent?

The rule is beta_d = 1 - alpha_d, where alpha_d is the exponent of coordinate d's
read in the token count n. On pi_jepa's encoder that exponent is 0 by
construction -- pj.Encoder is x_t -> s_t and aggregates nothing over the sequence
axis -- so the rule has no referent there and the mask in use reached those
coordinates by index equality alone.

The reroute under test: measure alpha on the PREDICTOR'S VALUE AXIS instead.
`read_per_coordinate(q, k, v, beta)` contracts `num @ v` over the key axis, and
that contraction is the only place in the module where a token count varies.

MUST-FIRES. A measurement is only a measurement if it can come out both ways.

  1. A planted EXTENSIVE value coordinate -- one whose summands carry a nonzero
     mean, so the unnormalised total grows with the count -- reads alpha near 1.
  2. A planted INTENSIVE value coordinate reads alpha near 0, SEPARABLY from
     (1). This one is not in the brief and is the test that decides the reroute:
     if every coordinate reads the same alpha regardless of what was planted in
     it, then alpha on this axis is a property of the CONTRACTION and not of the
     coordinate, and `beta = 1 - alpha` is an identity again -- the same defect
     as on the encoder, with the sign flipped.
  3. The exponent must not depend on the beta it is measured at. `output_alpha`
     takes a beta, and the rule sets beta from alpha; if alpha measured at beta=0
     and at beta=1 differ by about 1, the rule has no fixed point and cannot be
     evaluated without already knowing its own answer.
  4. Refused coordinates stay refused: the measurement returns a Refusal
     carrying a reason, never NaN and never a silent default.

RUN: python -m pytest tests/curvature/test_value_axis_alpha.py -v
"""

import math

import numpy as np
import pytest
import torch

import ceqjepa.pi_assign as pa
import ceqjepa.pi_jepa as pj

#: The count axis under test: the context length the key axis contracts over.
S_GRID = (16, 32, 64, 128)

#: A planted class is recovered only if it lands this close to its own corner.
ALPHA_TOL = 0.15

#: Two planted classes must be separated by more than this, or the axis is
#: reading the contraction rather than the coordinate.
SEPARATION_BAR = 0.50

#: Beyond this, alpha depends on the beta it was measured at and the rule is
#: circular.
FIXED_POINT_BAR = 0.25


def _fn():
    """The surface under test, or a named failure saying it is not there yet."""
    f = getattr(pj, "value_axis_alpha", None)
    if f is None:
        pytest.fail(
            "ceqjepa.pi_jepa exposes no `value_axis_alpha`. The reroute needs "
            "value_axis_alpha(beta_at=0.0, s_grid=(16,32,64,128), seed=..., "
            "plant=None) returning one entry per latent coordinate: a float "
            "exponent of that coordinate's read RMS in the context length, or a "
            "Refusal carrying its reason. `plant` accepts 'extensive', "
            "'intensive' or None and forces the named class into the value "
            "stream so the measurement can be seen to come out both ways.")
    return f


def test_a_planted_extensive_value_coordinate_reads_alpha_near_one():
    """MUST-FIRE 1. The unnormalised total over n terms grows like n."""
    out = _fn()(beta_at=0.0, s_grid=S_GRID, plant="extensive")
    vals = [a for a in out if not pj.is_refusal(a)]
    assert vals, "every coordinate refused under a planted extensive stream"
    worst = max(abs(a - 1.0) for a in vals)
    assert worst <= ALPHA_TOL, (
        "a planted EXTENSIVE value coordinate does not read alpha near 1 on the "
        "value axis: alphas %s, worst distance from 1.0 is %.4f against a %.2f "
        "tolerance" % (["%+.4f" % a for a in vals], worst, ALPHA_TOL))


def test_a_planted_intensive_value_coordinate_reads_alpha_near_zero():
    """MUST-FIRE 2. The test that decides whether this axis measures anything."""
    out = _fn()(beta_at=0.0, s_grid=S_GRID, plant="intensive")
    vals = [a for a in out if not pj.is_refusal(a)]
    assert vals, "every coordinate refused under a planted intensive stream"
    worst = max(abs(a - 0.0) for a in vals)
    assert worst <= ALPHA_TOL, (
        "a planted INTENSIVE value coordinate does not read alpha near 0: alphas "
        "%s, worst distance from 0.0 is %.4f against a %.2f tolerance. If this "
        "fires while the extensive plant passes, the value axis reports the "
        "contraction and not the coordinate"
        % (["%+.4f" % a for a in vals], worst, ALPHA_TOL))


def test_the_two_planted_classes_are_separated_on_this_axis():
    """MUST-FIRE 2, the discriminating form. One number decides the reroute."""
    f = _fn()
    ext = [a for a in f(beta_at=0.0, s_grid=S_GRID, plant="extensive")
           if not pj.is_refusal(a)]
    inten = [a for a in f(beta_at=0.0, s_grid=S_GRID, plant="intensive")
             if not pj.is_refusal(a)]
    assert ext and inten, "a planted class refused every coordinate"
    gap = float(np.mean(ext) - np.mean(inten))
    assert gap >= SEPARATION_BAR, (
        "the value axis does not separate a planted extensive coordinate from a "
        "planted intensive one: mean alpha %+.4f against %+.4f, gap %+.4f under "
        "the %.2f bar. beta = 1 - alpha on this axis is then an identity of the "
        "contraction, exactly as it is on the encoder, and the reroute is dead"
        % (float(np.mean(ext)), float(np.mean(inten)), gap, SEPARATION_BAR))


def test_the_exponent_does_not_depend_on_the_beta_it_is_measured_at():
    """MUST-FIRE 3. A rule that sets beta from alpha needs alpha to be fixed."""
    f = _fn()
    a0 = [a for a in f(beta_at=0.0, s_grid=S_GRID) if not pj.is_refusal(a)]
    a1 = [a for a in f(beta_at=1.0, s_grid=S_GRID) if not pj.is_refusal(a)]
    assert len(a0) == len(a1) and a0, "the two sweeps disagree on the axis length"
    drift = max(abs(x - y) for x, y in zip(a0, a1))
    assert drift <= FIXED_POINT_BAR, (
        "alpha depends on the beta it was measured at: worst per-coordinate "
        "drift between beta=0 and beta=1 is %.4f over the %.2f bar (beta=0 %s, "
        "beta=1 %s). The rule sets beta from alpha, so it has no fixed point and "
        "cannot be evaluated without already knowing its own answer"
        % (drift, FIXED_POINT_BAR, ["%+.3f" % x for x in a0],
           ["%+.3f" % y for y in a1]))


def test_a_refused_coordinate_comes_back_as_a_refusal_not_a_number():
    """MUST-FIRE 4. Refusals are values carrying a reason, never NaN."""
    out = _fn()(beta_at=0.0, s_grid=S_GRID)
    assert len(out) == pj.D_LATENT, (
        "the measurement returned %d entries for a %d-coordinate axis: the axis "
        "cannot silently shorten" % (len(out), pj.D_LATENT))
    for i, a in enumerate(out):
        if pj.is_refusal(a):
            assert str(a).strip(), "coordinate %d refused with an empty reason" % i
        else:
            assert math.isfinite(a), (
                "coordinate %d came back as %r: a non-finite exponent is a "
                "refusal wearing a float's clothing" % (i, a))


def test_the_measurement_is_not_reading_a_constant_bed():
    """Planted-negative control: the two plants must differ in the VALUE stream.

    Without this, both must-fires above could pass on a bed where the plant does
    nothing and every alpha is whatever the contraction happens to give.
    """
    f = _fn()
    raw = getattr(pj, "_planted_value_stream", None)
    if raw is None:
        pytest.fail(
            "ceqjepa.pi_jepa exposes no `_planted_value_stream(plant, s, seed)` "
            "returning the value tensor the sweep actually contracts, which is "
            "what makes the plant checkable rather than asserted")
    ext = raw("extensive", 64, pj.SEED)
    inten = raw("intensive", 64, pj.SEED)
    assert ext.shape == inten.shape, (ext.shape, inten.shape)
    mext = float(ext.mean().abs())
    minten = float(inten.mean().abs())
    assert mext > 10.0 * max(minten, 1e-9), (
        "the two plants do not differ where it matters: |mean| of the extensive "
        "value stream is %.6e against the intensive stream's %.6e. An extensive "
        "total needs a nonzero summand mean; if both are centred, both sums grow "
        "like sqrt(n) and the plant is decoration" % (mext, minten))
