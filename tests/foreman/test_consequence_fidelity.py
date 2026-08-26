"""RED-first bind for CONSEQUENCE FIDELITY (`LOOP_PROMPT.md` 1.7c).

The metric under test is: over drawn single-token `do()` interventions whose
oracle effect is known in closed form, the fraction of interventions on which
`sign(model dy_hat) == sign(oracle dy)`, with an EXACT binomial interval, plus
the regression slope of `dy_hat` on `dy` with a CI.

Every test here is written before `scale/foreman_consequence.py` exists, and
each one names the failure mode it is guarding:

* the closed-form effect could disagree with the oracle -- then the metric is
  scoring a different intervention than the one it applied;
* the probe could be structurally unable to read either extreme -- so the
  oracle itself is plugged in as the model and must read 1.0, and its negation
  must read 0.0, both SEEN;
* the probe could manufacture fidelity out of an untrained network -- so a
  0-step arm must read chance, with the interval covering 0.5;
* the interval could be an approximation wearing the word "exact" -- so the
  endpoints are checked against the binomial-tail inversion that DEFINES
  Clopper-Pearson, and against the Wilson interval they must NOT equal;
* the slope could be a constant -- so a planted gain must be recovered with an
  interval that excludes the un-planted value.
"""
from __future__ import annotations

import math

import pytest
import torch

from scale.negation_scope import CH_FLIP, CH_NOISE, CH_PAYLOAD, make_batch, oracle

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

S, D, DMODEL, N = 64, 24, 16, 256


def _batch(n=N, seed=7):
    return make_batch(n, S, D, d_model=DMODEL, seed=seed)


def test_the_closed_form_intervention_effect_equals_rerunning_the_oracle():
    """1.7c requires a CLOSED-FORM oracle effect. This is the only thing that
    makes that admissible: the closed form and the re-run oracle are computed on
    the SAME drawn perturbed tensors and must agree entrywise."""
    from scale.foreman_consequence import draw_do

    x, y, f, p = _batch()
    do = draw_do(x, f, p, seed=3)
    assert set(do.kind.tolist()) == {0, 1, 2}, "all three intervention kinds must be drawn"

    rerun = oracle(do.x_do, f, p) - y
    resid = (rerun - do.dy).abs()
    assert float(resid.max()) < 1e-6, f"closed form disagrees with the oracle: {float(resid.max()):.3e}"

    null = do.kind == 2
    assert int(null.sum()) > 0
    # The null intervention touches only the distractor channels, so BOTH the
    # closed form and the re-run oracle must be EXACTLY zero, not merely small.
    assert float(do.dy[null].abs().max()) == 0.0
    assert float(rerun[null].abs().max()) == 0.0
    # ... and it must have actually changed the tensor, or it is not a do() at all.
    assert float((do.x_do - x).abs().max()) > 0.0
    changed = (do.x_do[null] - x[null]).abs().amax(dim=-1)      # [n_null, s]
    assert float(changed.sum()) > 0.0
    assert float(do.x_do[null][..., CH_FLIP].sub(x[null][..., CH_FLIP]).abs().max()) == 0.0
    assert float(do.x_do[null][..., CH_PAYLOAD].sub(x[null][..., CH_PAYLOAD]).abs().max()) == 0.0
    assert CH_NOISE == 2


def test_the_probe_reads_one_for_the_oracle_and_zero_for_its_negation():
    """Both ends SEEN on drawn instances. A metric that cannot reach 1.0 and
    cannot reach 0.0 is not measuring the sign of anything."""
    from scale.foreman_consequence import consequence_fidelity

    x, y, f, p = _batch()
    perfect = consequence_fidelity(lambda xx: oracle(xx, f, p), x, f, p, seed=11)
    assert perfect["n_used"] > 100
    assert perfect["sign_match"] == 1.0
    assert perfect["ci_lo"] > 0.97

    inverted = consequence_fidelity(lambda xx: -oracle(xx, f, p), x, f, p, seed=11)
    assert inverted["sign_match"] == 0.0
    assert inverted["ci_hi"] < 0.03


def test_an_untrained_arm_reads_chance():
    """The must-fire null. An arm at 0 training steps has no consequence
    fidelity to have; if the probe reports one, the probe is the source."""
    from scale import m3_capability as m3
    from scale.foreman_consequence import consequence_fidelity

    x, y, f, p = _batch()
    torch.manual_seed(0)
    arm = m3.Arm("pivot_signed", S)
    arm.eval()

    def fwd(xx):
        with torch.no_grad():
            return arm(xx)

    r = consequence_fidelity(fwd, x, f, p, seed=11)
    assert r["ci_lo"] <= 0.5 <= r["ci_hi"], (
        f"untrained arm reads {r['sign_match']:.6f} with interval "
        f"[{r['ci_lo']:.6f}, {r['ci_hi']:.6f}] -- it does not cover chance")


def test_the_metric_is_computable_for_softmax():
    """1.7c's shippability clause: a column only the signed arm can be scored on
    is a plea, not a capability table."""
    from scale import m3_capability as m3
    from scale.foreman_consequence import consequence_fidelity

    x, y, f, p = _batch()
    for kind in m3.ARMS:
        torch.manual_seed(0)
        arm = m3.Arm(kind, S)
        arm.eval()

        def fwd(xx, _a=arm):
            with torch.no_grad():
                return _a(xx)

        r = consequence_fidelity(fwd, x, f, p, seed=11)
        assert r["n_used"] > 100, kind
        for key in ("sign_match", "ci_lo", "ci_hi", "slope", "slope_lo", "slope_hi"):
            assert math.isfinite(r[key]), (kind, key, r[key])


def test_the_interval_is_the_exact_binomial_inversion_and_not_wilson():
    """The flip-rate caveat that forced this clause: 0.004000 is 1 event in 250.
    An exact interval is DEFINED by inverting the binomial tails; this checks
    that identity holds at the endpoints, and that the result is not the Wilson
    interval carrying the word 'exact'."""
    from scipy.stats import binom

    from scale.foreman_consequence import clopper_pearson

    k, n, alpha = 1, 250, 0.05
    lo, hi = clopper_pearson(k, n, alpha=alpha)
    assert 0.0 < lo < hi < 1.0
    assert binom.sf(k - 1, n, lo) == pytest.approx(alpha / 2, abs=1e-9)
    assert binom.cdf(k, n, hi) == pytest.approx(alpha / 2, abs=1e-9)

    z = 1.959963984540054
    ph, m = k / n, n + z * z
    centre = (k + z * z / 2) / m
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) * (n / m)
    assert abs(hi - (centre + half)) > 1e-4, "the interval coincides with Wilson"

    assert clopper_pearson(0, n, alpha=alpha)[0] == 0.0
    assert clopper_pearson(n, n, alpha=alpha)[1] == 1.0


def test_consequence_fidelity_and_nrmse_dissociate_in_both_directions():
    """The check that this is a NEW measurement plane and not NRMSE renamed.

    Two planted models, drawn on the same batch, dissociate the two columns:

      the ORACLE PLUS A CONSTANT -- every intervention effect is reproduced
      exactly, so consequence fidelity is 1.0 and the calibration slope is 1.0,
      while NRMSE is driven arbitrarily far above the 1.0 bar by the offset;

      the CONSTANT PREDICTOR -- NRMSE is exactly 1.0 when the constant is the
      mean, the definition of the bar, while `dy_hat` is identically 0 on every
      draw and consequence fidelity is 0.0.

    So neither column determines the other, in either direction. That is why
    the metric can still order arms in the region where NRMSE has saturated at
    'everything is worse than a constant'.
    """
    from scale.foreman_consequence import consequence_fidelity
    from scale.negation_scope import nrmse

    x, y, f, p = _batch()

    off = consequence_fidelity(lambda xx: oracle(xx, f, p) + 10.0, x, f, p, seed=11)
    assert off["sign_match"] == 1.0
    assert off["slope"] == pytest.approx(1.0, abs=1e-4)
    assert nrmse(oracle(x, f, p) + 10.0, y) > 5.0

    const = y.mean().expand_as(y)
    flat = consequence_fidelity(lambda xx: y.mean().expand(xx.shape[0]), x, f, p, seed=11)
    assert nrmse(const, y) == pytest.approx(1.0, abs=1e-5)
    assert flat["sign_match"] == 0.0
    assert flat["n_dyhat_exact_zero"] == flat["n_used"]


def test_the_slope_recovers_a_planted_gain():
    """(b) of 1.7c. The slope must be a measurement: a planted gain of 3.0 must
    come back as 3.0 with an interval that EXCLUDES the un-planted 1.0."""
    from scale.foreman_consequence import consequence_fidelity

    x, y, f, p = _batch()
    r = consequence_fidelity(lambda xx: 3.0 * oracle(xx, f, p), x, f, p, seed=11)
    assert r["slope"] == pytest.approx(3.0, abs=1e-4)
    assert r["slope_lo"] <= 3.0 <= r["slope_hi"]
    assert not (r["slope_lo"] <= 1.0 <= r["slope_hi"]), "the slope cannot exclude anything"
