"""MARS: adversarial controls against the value-axis retirement, not for it.

The finding under attack is this repository's own: that on the predictor's value
axis the measured exponent moves with the beta it is measured at, at slope
-1.0027, so beta = one minus alpha has an empty solution set and the corner rule
is inadmissible there.

Two ways that finding could be an artefact rather than a mechanism, each given
its own control:

  POSITIONAL ABLATION. If the count dependence were carried by the positional
  structure of the attention weights rather than by the contraction itself, the
  slope would be a fact about how this bed orders its tokens and not about the
  read. Control: destroy the ordering by permuting the context positions and
  re-measure. A slope that survives is not positional.

  NORM-FROZEN. The claimed mechanism is that Z = mod.sum(-1) carries the count
  exactly as the numerator does, so dividing by Z**beta subtracts beta from the
  exponent. Control: divide by a Z rescaled to remove its count dependence and
  re-measure. If the slope goes to zero, Z is the whole mechanism and the claim
  is mechanistic rather than curve-fitted. If the slope survives, the stated
  mechanism is wrong even if the number is right.

RUN: python -m pytest tests/curvature/test_value_axis_mars.py -v
"""

import numpy as np
import torch

import ceqjepa.pi_jepa as pj
from ceq import arm_smprime as arm

BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)

#: A slope this close to -1 is the shipped mechanism reproducing.
SLOPE_TOL = 0.10

#: With Z's count dependence removed, the slope must fall below this or the
#: stated mechanism is not the mechanism.
FROZEN_BAR = 0.25


def _alphas(beta_at, permute_positions=False, freeze_z=False,
            s_grid=pj.S_GRID_VALUE, seed=pj.SEED, batch=pj.VALUE_SWEEP_BATCH):
    """Per-coordinate exponent, with either control optionally applied.

    Mirrors pj.value_axis_alpha and calls arm.numerator, so the operator is the
    shipped one and only the two named interventions differ.
    """
    model = pj.build(seed)
    d = pj.D_LATENT
    norms = np.zeros((len(s_grid), d))
    for i, s in enumerate(s_grid):
        x, _ = pj.draw_bed(seed + i, batch, s, pj.HORIZON, pj.X_DIM)
        ctx = x[:, :s]
        if permute_positions:
            g = torch.Generator().manual_seed(int(seed) + 77 * i)
            perm = torch.randperm(s, generator=g)
            ctx = ctx[:, perm]
        with torch.no_grad():
            st = model.online(ctx)
            q, k, v = model.pred.wq(st), model.pred.wk(st), model.pred.wv(st)
            num, mod = arm.numerator(q, k)
            z = mod.sum(-1, keepdim=True)
            if freeze_z:
                # Remove Z's count dependence while keeping its shape: divide by
                # the batch-mean Z at this length, so z_used has no growth in s
                # but still varies across rows and positions.
                z = z / z.mean()
            o = num @ v.to(num.dtype)
            zb = z ** float(beta_at)
            out = torch.complex(o.real / zb, o.imag / zb)[:, -1, :]
        mag = out.abs().to(torch.float64).numpy()
        norms[i] = np.sqrt((mag ** 2).mean(axis=0))
    lg_s = np.log2(np.asarray(s_grid, dtype=np.float64))
    return np.array([np.polyfit(lg_s, np.log2(norms[:, c]), 1)[0]
                     for c in range(d)])


def _slope_over_beta(**kw):
    """d(alpha)/d(beta_at), the statistic the finding rests on."""
    a = [float(np.mean(_alphas(b, **kw))) for b in BETAS]
    return float(np.polyfit(np.asarray(BETAS), np.asarray(a), 1)[0]), a


def test_the_shipped_slope_reproduces_in_this_file():
    """The control's control: this file must reproduce the finding it attacks."""
    slope, a = _slope_over_beta()
    print("\n  shipped      slope %+.4f   alphas %s"
          % (slope, ["%+.4f" % x for x in a]))
    assert abs(slope + 1.0) <= SLOPE_TOL, (
        "this file does not reproduce the finding it exists to attack: slope "
        "%+.4f against -1 +- %.2f. Every verdict below is void until it does"
        % (slope, SLOPE_TOL))


def test_the_count_dependence_is_not_positional():
    """MARS 1. Destroy the token ordering; the slope must survive."""
    slope, a = _slope_over_beta(permute_positions=True)
    print("\n  positions permuted  slope %+.4f   alphas %s"
          % (slope, ["%+.4f" % x for x in a]))
    assert abs(slope + 1.0) <= SLOPE_TOL, (
        "permuting the context positions moved the slope to %+.4f, outside -1 "
        "+- %.2f. The count dependence would then be a fact about how this bed "
        "orders tokens, and the retirement of the corner rule on this axis does "
        "not follow" % (slope, SLOPE_TOL))


def test_freezing_the_normaliser_removes_the_slope():
    """MARS 2. The stated mechanism, made to carry the whole effect."""
    slope, a = _slope_over_beta(freeze_z=True)
    print("\n  Z count-frozen      slope %+.4f   alphas %s"
          % (slope, ["%+.4f" % x for x in a]))
    assert abs(slope) <= FROZEN_BAR, (
        "removing Z's count dependence left a slope of %+.4f, over the %.2f "
        "bar. The finding claims Z = mod.sum(-1) carries the count and that "
        "dividing by Z**beta is what subtracts beta from the exponent; if the "
        "slope survives without that growth, the number may be right but the "
        "mechanism stated for it is wrong" % (slope, FROZEN_BAR))


def test_the_two_controls_are_not_the_same_intervention():
    """Planted negative: a control that changes nothing proves nothing.

    Measured at beta = 1, not beta = 0: Z**0 is one, so freezing Z is inert at
    beta = 0 by identity and a difference there would be the bug, not the check.
    """
    base = _alphas(1.0)
    perm = _alphas(1.0, permute_positions=True)
    froz = _alphas(1.0, freeze_z=True)
    d_perm = float(np.abs(base - perm).max())
    d_froz = float(np.abs(base - froz).max())
    print("\n  max |alpha| change at beta=1: permute %.4f   freeze %.4f"
          % (d_perm, d_froz))
    assert d_froz > 1e-3, (
        "freezing Z left every exponent unchanged at beta = 1 (worst %.2e): the "
        "intervention is inert where it is supposed to bite, so the MARS verdict "
        "above is about nothing" % d_froz)
    inert = _alphas(0.0, freeze_z=True)
    assert float(np.abs(_alphas(0.0) - inert).max()) < 1e-9, (
        "freezing Z changed the exponent at beta = 0, where Z**0 is one by "
        "identity: the control is doing something other than what it says")
