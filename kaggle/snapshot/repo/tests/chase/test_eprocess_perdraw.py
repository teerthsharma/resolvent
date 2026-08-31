"""X18 must-fire: the per-draw Ville process, calibrated in BOTH directions.

The claim the file makes is that swapping the unit from the training seed to the
evaluation draw removes a ceiling that was a property of the schedule. Two
things have to be true for that to be worth anything, and neither is checkable
by reading the arithmetic:

  * the process must CROSS on a planted effect (it is not merely conservative);
  * the process must NOT cross on the null more often than Ville allows.

`LOOP_PROMPT.md` §5: every must-fire's PASS half carries its own non-degeneracy
check. The fourteenth vacuous control struck in this project was a PASS case
whose label was constant, and a planted stream of constant `+B` would cross this
process trivially while proving nothing. `test_the_planted_stream_is_not
_degenerate` is that check and it runs on the SAME stream the PASS half uses.
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scale import eprocess as EP                                   # noqa: E402
from scale import eprocess_perdraw as PD                           # noqa: E402

HORIZON = 2048          #: n_eval of the ladder run -- the real per-cell budget
N_REP = 400
PLANTED = 0.20          #: a per-draw effect, in clipped-NRMSE units
NOISE = 0.60


def _stream(rng, mean, n):
    """Draws clipped to the a-priori bound, exactly as `eprocess._draw` does."""
    return np.clip(rng.normal(mean, NOISE, n), -EP.B, EP.B)


def _crosses(d) -> bool:
    p = EP.Pair()
    for v in d:
        p.update(float(v))
    return p.decision is not None


# ------------------------------------------------------------ the ceiling ---
def test_the_ceiling_arithmetic_is_the_claim_it_prints():
    """The old unit could not cross and the new one can, by arithmetic alone."""
    assert EP.max_attainable(5) == 3.80169140625
    assert EP.max_attainable(5) < EP.THRESHOLD          # 5 seeds: forbidden
    assert EP.MIN_T_MIXTURE == 13
    # the per-draw ceiling overflows a double, which is why it is in log space
    with pytest.raises(OverflowError):
        EP.max_attainable(HORIZON)
    assert PD.log10_max_attainable(HORIZON) > PD.LOG10_THRESHOLD
    # and the log form agrees with the direct form wherever the direct form
    # can be evaluated at all -- otherwise the log form is unbound arithmetic
    for t in (1, 5, 13, 100):
        assert math.isclose(PD.log10_max_attainable(t),
                            math.log10(EP.max_attainable(t)), rel_tol=1e-12)


# --------------------------------------------------------- the PASS half ---
def test_a_planted_per_draw_effect_crosses():
    rng = np.random.default_rng(0)
    hits = sum(_crosses(_stream(rng, PLANTED, HORIZON)) for _ in range(20))
    assert hits == 20, f"planted effect crossed only {hits}/20 times"


def test_the_planted_stream_is_not_degenerate():
    """THE NON-DEGENERACY CHECK ON THE PASS HALF. A constant `+B` stream would
    cross and would prove nothing; this asserts the stream that DOES cross is a
    real two-sided draw with the planted mean, not a rail."""
    rng = np.random.default_rng(0)
    d = _stream(rng, PLANTED, HORIZON)
    assert (d > 0).any() and (d < 0).any(), "the planted stream is one-sided"
    assert 0.05 < float((d < 0).mean()) < 0.95, "the planted stream is a rail"
    assert abs(float(d.mean()) - PLANTED) < 0.05
    assert float(d.std()) > 0.5 * NOISE
    assert float(np.abs(d).max()) <= EP.B


# --------------------------------------------------------- the FAIL half ---
def test_the_null_does_not_cross_more_often_than_ville_allows():
    """`P(sup_t E_t >= tau) <= 1/tau` per direction; the reported event is the
    union of two directions, so the family bound is `ALPHA_FAMILY = 0.05`."""
    rng = np.random.default_rng(1)
    hits = sum(_crosses(_stream(rng, 0.0, HORIZON)) for _ in range(N_REP))
    rate = hits / N_REP
    assert rate <= EP.ALPHA_FAMILY + 0.02, (
        f"null crossed {hits}/{N_REP} = {rate}, above the family bound "
        f"{EP.ALPHA_FAMILY}")


def test_the_bound_is_enforced_not_clamped():
    """A difference outside `B` voids Ville's inequality, so it must RAISE."""
    p = EP.Pair()
    p.update(EP.B)                       # exactly at the bound is legal
    with pytest.raises(ValueError):
        p.update(EP.B + 1e-9)


# ------------------------------------------------- the bind on real weights ---
def test_per_draw_errors_reproduce_the_journalled_nrmse():
    """The UNCLIPPED per-draw errors' root-mean-square must be the journalled
    `eval_nrmse`, bitwise. This is what makes the per-draw stream a
    decomposition of the reported metric rather than a second metric wearing its
    units.

    SKIPS when no ladder weights are on disk yet; it is a bind on the real
    artifact, and a fabricated stand-in would defeat its purpose.
    """
    import torch
    from scale.m3_quintuple import WEIGHTS_DIR, load_unit
    from scale import negation_scope as NS

    have = sorted(WEIGHTS_DIR.glob("*_taske3_*.pt")) if WEIGHTS_DIR.exists() \
        else []
    if not have:
        pytest.skip("no e3 ladder weights on disk yet")
    model, rec = load_unit(have[0])
    bfn = NS.M3_TASKS[rec["task"]][0]
    xe, ye, _f, _p = bfn(rec["n_eval"], rec["s"], rec["d"],
                         d_model=rec["d_model"], seed=rec["seed"] + 12345)
    with torch.no_grad():
        pred = model(xe) * rec["sigma"] + rec["mu"]
    # THE BITWISE BIND: the saved weights reproduce the journalled number.
    assert NS.nrmse(pred, ye) == rec["eval_nrmse"]

    # THE PER-DRAW DECOMPOSITION, AND IT IS *NOT* BITWISE, FOR A NAMED REASON.
    # `negation_scope.nrmse` is `((pred-y)**2).mean().sqrt() / sd` -- it divides
    # AFTER the square root. The per-draw stream divides BEFORE squaring, so in
    # float32 the two orders round differently. Measured here: 0.9783142805099487
    # against 0.9783142763084641, a relative gap of 4.3e-09. That is the
    # arithmetic order, not a second metric, and it is asserted at a tolerance
    # with the reason attached rather than papered over with `pytest.approx`.
    e = (pred - ye).abs() / float(ye.std(unbiased=False))
    rms = float((e ** 2).mean().sqrt())
    assert math.isclose(rms, rec["eval_nrmse"], rel_tol=1e-6), (
        f"per-draw RMS {rms!r} is not the journalled {rec['eval_nrmse']!r} "
        f"even at float32 tolerance")
    # the control: the tolerance must be tight enough to reject a real change
    assert not math.isclose(rms * 1.001, rec["eval_nrmse"], rel_tol=1e-6)
