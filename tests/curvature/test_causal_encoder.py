"""P1: the encoder alone, against a bar a position-wise map cannot reach.

WHAT THIS FILE BINDS. ceqjepa/pi_jepa.py:387 documents `x_{<=t} -> s_t, per
position` and implements `x_t -> s_t`: an nn.Sequential over the last axis with
no reduction over the sequence axis. Every test here is about the piece that
replaces it, ceqjepa/causal_encoder.py, and about the contrast with the shipped
one -- nothing here trains a predictor, a read, or a probe.

THE BAR. A strict-prefix EMA bank y_t = scale_d * sum_{j<t} lam_d^(t-1-j) *
(x_j . u_d), on an i.i.d. Gaussian bed. Because x_t is independent of x_{<t},
y_t is independent of x_t, so E[y_t | x_t] = E[y_t] = 0 and the MMSE of ANY
measurable function of x_t alone equals Var(y_t). Pooled over positions that is
exactly R^2 <= 0. A position-wise encoder cannot clear a positive R^2 on this
target at any width, any depth, any number of steps -- the ceiling is
information-theoretic, not an optimisation accident.

RETIREMENT RECEIPT. test_shipped_encoder_is_position_wise_today asserts the
DEFECT, not a contract. It is GREEN today and it is supposed to go RED the day
pi_jepa.py's Encoder starts aggregating. Delete it in that same commit.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ceqjepa.pi_jepa import Encoder, S_LEN, X_DIM, D_LATENT, SEED
from ceqjepa.causal_encoder import (
    CausalEncoder,
    make_bed,
    ema_target,
    permute_context,
    train_arm,
    r_squared,
)


def _params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# The planted negative, pinned as it stands today.
# ---------------------------------------------------------------------------

def test_shipped_encoder_is_position_wise_today():
    """RETIREMENT RECEIPT, not a contract. Delete when pi_jepa.py is fixed.

    VARIES: the order of context positions 0..S-2.
    PINS: the multiset of context observations, x_{S-1}, the weights, the seed,
    the dtype, the shape.
    """
    torch.manual_seed(SEED)
    enc = Encoder().double()
    x = make_bed(8, seed=SEED)
    xp = permute_context(x, seed=SEED + 1)

    assert torch.equal(xp[:, -1], x[:, -1]), "the pin is broken: x_{S-1} moved"
    assert not torch.equal(xp[:, :-1], x[:, :-1]), "the vary is a no-op"

    with torch.no_grad():
        a, b = enc(x)[:, -1], enc(xp)[:, -1]
    assert torch.equal(a, b), (
        "the shipped Encoder moved under a context permutation; if this is a "
        "real fix in pi_jepa.py, delete this test in that commit"
    )
    assert (a - b).abs().max().item() == 0.0


def test_shipped_encoder_ignores_its_whole_prefix():
    """The stronger form: zeroing every context position changes nothing.

    VARIES: x_{0..S-2}, set to zero.  PINS: x_{S-1}, weights, seed, dtype.
    """
    torch.manual_seed(SEED)
    enc = Encoder().double()
    x = make_bed(8, seed=SEED)
    xz = x.clone()
    xz[:, :-1] = 0.0
    with torch.no_grad():
        assert torch.equal(enc(x)[:, -1], enc(xz)[:, -1])


# ---------------------------------------------------------------------------
# The replacement: the three properties the docstring always claimed.
# ---------------------------------------------------------------------------

def test_causal_encoder_moves_under_context_permutation():
    """The single cleanest contrast with the shipped arm.

    VARIES: the order of context positions 0..S-2.
    PINS: the multiset of context observations, x_{S-1}, the weights, the seed,
    the dtype, the shape.
    """
    torch.manual_seed(SEED)
    enc = CausalEncoder().double()
    x = make_bed(8, seed=SEED)
    xp = permute_context(x, seed=SEED + 1)

    assert torch.equal(xp[:, -1], x[:, -1])
    assert not torch.equal(xp[:, :-1], x[:, :-1])

    with torch.no_grad():
        a, b = enc(x)[:, -1], enc(xp)[:, -1]
    assert not torch.equal(a, b), "the causal encoder is order-blind"
    assert (a - b).abs().max().item() > 1e-6


def test_causal_encoder_has_no_future_leak():
    """s_t may see x_{<=t} and never x_{>t}. Asserted bitwise, per position.

    VARIES: x_k for one k, replaced by fresh noise.
    PINS: x_j for every j != k, the weights, the seed, the dtype.
    """
    torch.manual_seed(SEED)
    enc = CausalEncoder().double()
    x = make_bed(4, seed=SEED)
    with torch.no_grad():
        s = enc(x)

    g = torch.Generator().manual_seed(SEED + 7)
    for k in range(1, S_LEN):
        xk = x.clone()
        xk[:, k] = torch.randn(x.shape[0], X_DIM, generator=g, dtype=x.dtype)
        with torch.no_grad():
            sk = enc(xk)
        assert torch.equal(s[:, :k], sk[:, :k]), f"future leak from position {k}"
        assert not torch.equal(s[:, k], sk[:, k]), f"position {k} ignores its own x"


def test_causal_encoder_is_not_larger_than_the_arm_it_replaces():
    """A win with MORE parameters is a tie by this project's rule."""
    torch.manual_seed(SEED)
    mine = _params(CausalEncoder().double())
    shipped = _params(Encoder().double())
    assert mine <= shipped, f"causal {mine} > shipped {shipped}"


def test_target_moves_under_the_same_permutation():
    """The bar and the permutation control must agree, or the control is empty.

    VARIES: the order of context positions 0..S-2.
    PINS: the multiset of context observations, x_{S-1}, the target constants.
    """
    x = make_bed(8, seed=SEED)
    xp = permute_context(x, seed=SEED + 1)
    y, yp = ema_target(x), ema_target(xp)
    assert not torch.equal(y[:, -1], yp[:, -1]), (
        "the target is order-invariant, so the permutation control measures "
        "nothing about the bar"
    )


def test_target_is_orthogonal_to_the_position_wise_arms_input():
    """The one-line reason a position-wise map cannot clear the bar, measured.

    y_t is built from x_{<t} only and the bed is i.i.d., so the empirical
    correlation between every (x_t coordinate, y_t coordinate) pair is a
    sampling fluctuation around zero, not signal a position-wise map could use.
    """
    x = make_bed(512, seed=SEED)
    y = ema_target(x)
    xf = x[:, 1:].reshape(-1, X_DIM)
    yf = y[:, 1:].reshape(-1, D_LATENT)
    xf = (xf - xf.mean(0)) / xf.std(0)
    yf = (yf - yf.mean(0)) / yf.std(0)
    corr = (xf.T @ yf) / xf.shape[0]
    assert corr.abs().max().item() < 0.05, corr.abs().max().item()


# ---------------------------------------------------------------------------
# The bar itself.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["shipped", "causal"])
def test_the_bar_separates_the_two_arms(arm):
    """Thresholds declared from the argument, not fitted to the outcome.

    The position-wise ceiling is R^2 <= 0 in population, so 0.01 is slack for
    sampling; 0.20 is the judgment call for "clears it", fixed before the run.

    VARIES: the encoder class.
    PINS: the bed seed, the eval bed seed, the target constants, the optimiser,
    the learning rate, the step count, the loss, the positions scored, the dtype.
    """
    torch.manual_seed(SEED)
    model = (Encoder() if arm == "shipped" else CausalEncoder()).double()
    out = train_arm(model, steps=400, seed=SEED)
    if arm == "shipped":
        assert out["r2"] <= 0.01, out
    else:
        assert out["r2"] >= 0.20, out


def test_r_squared_zero_for_the_constant_predictor():
    """The metric's own control: predicting the eval mean scores exactly 0."""
    y = ema_target(make_bed(64, seed=SEED))[:, 1:]
    assert abs(r_squared(y.mean(dim=(0, 1)).expand_as(y), y)) < 1e-12


# ---------------------------------------------------------------------------
# The ablation that isolates aggregation from "it had an extra layer".
# ---------------------------------------------------------------------------

def test_self_span_is_position_wise_by_construction():
    """span='self' restricts each query to its own position and nothing else.

    VARIES: the set of positions a query may attend to, {j <= t} against {t}.
    PINS: every parameter and its initialisation seed, the sinusoid, both layer
    widths, the input, the dtype.

    The prefix arm is asserted in the SAME test against the SAME input, so the
    assertion carries its own positive control: an invariance test that nothing
    can fail is not a test.
    """
    x = make_bed(8, seed=SEED)
    xp = permute_context(x, seed=SEED + 1)

    torch.manual_seed(SEED)
    null = CausalEncoder(span="self").double()
    torch.manual_seed(SEED)
    full = CausalEncoder(span="prefix").double()

    assert _params(null) == _params(full), (_params(null), _params(full))
    for a, b in zip(null.parameters(), full.parameters()):
        assert torch.equal(a, b), "the two spans did not start from one init"

    with torch.no_grad():
        assert torch.equal(null(x)[:, -1], null(xp)[:, -1]), \
            "span='self' saw the context it was forbidden"
        assert not torch.equal(full(x)[:, -1], full(xp)[:, -1]), \
            "positive control failed: span='prefix' is order-blind too"


def test_self_span_and_wider_position_wise_both_stay_at_the_ceiling():
    """Capacity is not the reason the position-wise arm fails.

    VARIES: the encoder class and width.
    PINS: the bed seeds, the target constants, Adam, the learning rate, the step
    count, the batch size and order, the loss, the scored positions, the dtype.
    """
    for label, ctor in (("wide", lambda: Encoder(hidden=64)),
                        ("self_span", lambda: CausalEncoder(span="self"))):
        torch.manual_seed(SEED)
        out = train_arm(ctor().double(), steps=400, seed=SEED)
        assert out["r2"] <= 0.01, (label, out["r2"], out["params"])
