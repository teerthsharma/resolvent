"""tests/cameron/test_prediction_margin.py -- binds ceqjepa/sharpness.py's own
self-check as pytest contracts instead of print statements someone has to read.

THE ROW THIS SERVES. The prediction row has never had an honest number because
every exact win in this repo used the TRUE operator (an oracle, not a model).
sharpness.py's identity split -- ce = H(pi) + KL(pi||qbar) + J(q) - I_q -- is
the instrument that tells an oracle-shaped win from a model-shaped one: I_q is
the read's label covariance (it looked at the input), J(q) is its sharpness
(it is merely confident). The three things this file pins down:

  1. the identity is roundoff, not a diagnostic, on more than one (n,K) shape;
  2. sharpness.py's PREREG bar/kill (margin vs se_margin) fire on the sign
     they claim, on beds the module can already reach;
  3. THE TRAP: argmax accuracy is a strictly-monotone-per-row invariant, so it
     is frozen across a temperature sweep while the margin crosses zero.
     Accuracy cannot see that crossing -- this is asserted, not narrated.
"""

from __future__ import annotations

from ceqjepa import sharpness


# --------------------------------------------------------------------------- #
# 1. the identity: exact on any finite (q, y), independent of what generated it


def test_identity_residual_is_roundoff_on_four_shapes():
    import torch

    settings = [(1000, 3, 1), (50000, 7, 2), (37, 2, 3), (5000, 20, 4)]
    residuals = []
    for n, K, seed in settings:
        g = torch.Generator().manual_seed(seed)
        q = torch.softmax(1.5 * torch.randn(n, K, generator=g, dtype=torch.float64), 1)
        y = torch.randint(K, (n,), generator=g)
        d = sharpness.decompose(q, y, K)
        residuals.append(d["residual"])
        assert abs(d["residual"]) < 1e-10, f"n={n} K={K}: residual {d['residual']:.3e} is not roundoff"
    # roundoff, not a fluke of one shape: every one of the four is well under
    # the module's own 1e-8 self-check tolerance.
    assert len(residuals) == 4
    assert max(abs(r) for r in residuals) < 1e-10


# --------------------------------------------------------------------------- #
# 2. the PREREG bar/kill, on the two beds sharpness.py's self-check already
# draws: a calibrated read (must pass the bar) and the same read sharpened to
# T=0.3 (must be killed).


def test_bar_passes_on_the_calibrated_read():
    q, y, _ = sharpness._grouped_draw(60000, 7)
    d = sharpness.decompose(q, y, 4)
    se = sharpness.bootstrap_se(q, y, 4, n_boot=200, seed=1)
    assert sharpness.passes_bar(d, se), (
        f"calibrated read should clear the bar: margin={d['margin']:+.4f}, "
        f"se_margin={se['se_margin']:.4f}, sigma={sharpness.margin_sigma(d, se):.1f}"
    )
    assert not sharpness.kill_fired(d, se)


def test_kill_fires_on_the_overconfident_read():
    q, y, _ = sharpness._grouped_draw(60000, 7)
    q_hot = sharpness._temper(q, 0.3)
    d = sharpness.decompose(q_hot, y, 4)
    se = sharpness.bootstrap_se(q_hot, y, 4, n_boot=200, seed=1)
    assert sharpness.kill_fired(d, se), (
        f"T=0.3 read should be killed: margin={d['margin']:+.4f}, "
        f"se_margin={se['se_margin']:.4f}, sigma={sharpness.margin_sigma(d, se):.1f}"
    )
    assert not sharpness.passes_bar(d, se)


# --------------------------------------------------------------------------- #
# 3. THE TRAP. Bound as an assertion: accuracy is frozen across the sweep
# while the bar/kill verdict flips, so accuracy is proven -- not claimed --
# incapable of adjudicating this margin.


def test_accuracy_cannot_adjudicate_the_margin_crossing():
    q, y, _ = sharpness._grouped_draw(60000, 7)
    baseline_argmax = q.argmax(1)
    baseline_acc = sharpness._acc(q, y)

    verdicts = {}
    for T in [1.5, 1.25, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3]:
        qt = sharpness._temper(q, T)
        # the read's ranking is untouched: temperature is a strictly monotone
        # per-row rescaling, so argmax -- and therefore accuracy -- cannot move.
        assert qt.argmax(1).equal(baseline_argmax), f"T={T} moved the argmax"
        assert sharpness._acc(qt, y) == baseline_acc, f"T={T} moved accuracy"

        dt = sharpness.decompose(qt, y, 4)
        st = sharpness.bootstrap_se(qt, y, 4, n_boot=100, seed=2)
        if sharpness.passes_bar(dt, st):
            verdicts[T] = "bar"
        elif sharpness.kill_fired(dt, st):
            verdicts[T] = "kill"
        else:
            verdicts[T] = "gray"

    # every T shares one accuracy, yet the margin verdict is not constant --
    # a metric that cannot tell these T apart is not adjudicating this bet.
    assert "bar" in verdicts.values(), verdicts
    assert "kill" in verdicts.values(), verdicts
    assert verdicts[0.4] == "kill" and verdicts[0.5] == "bar", (
        f"expected the crossing between T=0.5 and T=0.4, got {verdicts}"
    )
