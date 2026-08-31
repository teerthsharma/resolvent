"""R2's falsifier, with a learned weighting rather than a sampled Jacobian.

REQUIREMENTS.md R2:

    "Influence must reflect what a token causes downstream ... Small, dissimilar,
     high-consequence tokens -- `not`, a bound, a flag -- must outrank large,
     similar, inert ones."
    "If the learned weighting collapses back to similarity, it is a
     reparameterization -- delete it."

`not` is the requirement's own example and it is the one that decides the
question, because `not` does not merely reweight, it INVERTS. The target here
is a ground-truth influence matrix S = (I - M)^-1 with M signed, so some
positions genuinely suppress others. Everything else is held equal: no encoder,
no readout, no nonlinearity outside the propagation operator. The only thing
that differs between arms is the operator that propagates influence.

A free linear readout would make every arm equivalent to every other, since the
target is linear -- so there is no readout. What is being measured is the
constraint on the propagation operator and nothing else.

Parameter counts, so the result cannot be blamed on capacity:
    APPNP        n^2 + 1              = 101
    max-plus     n_act * (n^2 + n)    = 330
    signed       n^2                  = 100
The max-plus arm has 3.3x the parameters of the arm it must beat.
"""

import torch
import pytest

from _device import DEVICES
from _ceq import bellman

N = 10
N_ACT = 3
GAMMA = 0.9
STEPS = 1500
VI_UNROLL = 40
BATCH = 256


def _target(device, seed=0):
    """S = (I - M)^-1 with M signed: some positions suppress others."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    M = torch.randn((N, N), generator=g, dtype=torch.float64)
    M = 0.7 * M / torch.linalg.matrix_norm(M, 2)
    M = M.to(device)
    eye = torch.eye(N, dtype=M.dtype, device=device)
    S = torch.linalg.inv(eye - M)
    return M, S


def _batch(device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((BATCH, N), generator=g, dtype=torch.float64).to(device)


def _rel_mse(pred, y):
    return float(((pred - y) ** 2).mean() / (y ** 2).mean())


def _train(params, forward, S, device, seed=0):
    opt = torch.optim.Adam(params, lr=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-4)
    for k in range(STEPS):
        x = _batch(device, 1000 + k)
        y = x @ S.T
        loss = ((forward(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        x = _batch(device, 999999 + seed)
        return _rel_mse(forward(x), x @ S.T)


def _arm_appnp(device):
    logits = torch.zeros((N, N), dtype=torch.float64, device=device, requires_grad=True)
    a_raw = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
    eye = torch.eye(N, dtype=torch.float64, device=device)

    def forward(x):
        A_hat = torch.softmax(logits, dim=1)                # row-stochastic, >= 0
        alpha = torch.sigmoid(a_raw) * 0.9 + 0.05
        return alpha * torch.linalg.solve(eye - (1 - alpha) * A_hat, x.T).T

    return [logits, a_raw], forward


def _arm_maxplus(device):
    g = torch.Generator(device="cpu").manual_seed(3)
    A = (0.1 * torch.randn((N_ACT, N, N), generator=g, dtype=torch.float64)
         ).to(device).requires_grad_(True)
    b = (0.1 * torch.randn((N_ACT, N), generator=g, dtype=torch.float64)
         ).to(device).requires_grad_(True)

    def forward(x):
        R = x[:, None, :] + b[None]                          # r_a(x) = x + b_a
        z = torch.zeros_like(x)
        for _ in range(VI_UNROLL):
            z = bellman(z, R, A, GAMMA)
        return z

    return [A, b], forward


def _arm_signed(device):
    g = torch.Generator(device="cpu").manual_seed(4)
    M = (0.1 * torch.randn((N, N), generator=g, dtype=torch.float64)
         ).to(device).requires_grad_(True)
    eye = torch.eye(N, dtype=torch.float64, device=device)

    def forward(x):
        s = torch.linalg.matrix_norm(M, 2)
        Mc = M * torch.clamp(0.95 / s, max=1.0)              # contraction, signed
        return torch.linalg.solve(eye - Mc, x.T).T

    return [M], forward


@pytest.fixture(scope="module", params=DEVICES)
def fits(request):
    device = request.param
    M, S = _target(device)
    out = {"device": device, "min_S": float(S.min())}
    for name, arm in (("appnp", _arm_appnp), ("maxplus", _arm_maxplus),
                      ("signed", _arm_signed)):
        params, forward = arm(device)
        out[name] = _train(params, forward, S, device)
    print("\n  [%s] target min influence entry %.4f (a `not` exists in the ground truth)"
          % (device, out["min_S"]))
    print("  held-out relative MSE:  APPNP %.4f | max-plus %.4f | signed %.6f"
          % (out["appnp"], out["maxplus"], out["signed"]))
    return out


# ==========================================================================
# CLAIM AS WRITTEN -- RED
# ==========================================================================

def test_maxplus_beats_appnp_on_a_signed_consequence_target(fits):
    """B1's kill condition, verbatim: "arm 3 matches arm 2 post-intervention.
    Then the max bought nothing over Leontief." Here the separation is not even
    post-intervention -- it is in-distribution, on the training target."""
    assert fits["maxplus"] < 0.5 * fits["appnp"], (
        f"max-plus {fits['maxplus']:.4f} against APPNP {fits['appnp']:.4f} on a "
        f"target containing a genuine suppression (min influence "
        f"{fits['min_S']:.4f}), with 3.3x the parameters: the max bought nothing"
    )


def test_maxplus_reaches_the_signed_target_at_all(fits):
    """Being no worse than the control is not a defence if both are far from
    the answer. The signed arm sets the scale of what is reachable."""
    assert fits["maxplus"] < 10.0 * fits["signed"], (
        f"max-plus {fits['maxplus']:.4f} against a signed contraction's "
        f"{fits['signed']:.6f} on the same target at comparable parameter count: "
        f"the non-negative Kleene star cannot represent suppression, and the "
        f"semiring it is taken in makes no difference"
    )


# ==========================================================================
# GREEN
# ==========================================================================

def test_a_signed_contraction_fits_the_same_target(fits):
    """The escape, measured on the task rather than on a Jacobian."""
    assert fits["signed"] < 0.01
    assert fits["min_S"] < -0.05


def test_the_two_non_negative_arms_do_not_land_on_the_same_floor(fits):
    """Written first as "they tie, because they fail for the same reason", and
    refuted by its own measurement at ratio 55.0. Recorded as measured.

    The shared obstruction -- non-negative influence -- is established by the
    Jacobian sampling in test_r2_maxplus_reduction.py, not by this number. What
    this number adds is that max-plus lands ABOVE the predict-zero baseline of
    1.0, so its 55x is not a representation floor. It is a representation floor
    plus something else, and this test does not separate expressivity from
    optimization: value iteration's gradient is zero across the interior of a
    linearity cell, and the fixed point sits at 1/(1-gamma) = 10x the input
    scale, so both a dead-gradient and a conditioning explanation are live.
    That separation is OPEN. It does not change the R2 verdict, which rests on
    the exact reduction and the exact-zero Jacobian, not on this fit."""
    ratio = fits["maxplus"] / fits["appnp"]
    print("  max-plus / APPNP error ratio: %.3f   (predict-zero baseline = 1.000)" % ratio)
    assert ratio > 10.0
    assert fits["maxplus"] > 1.0        # worse than emitting zero
    assert fits["appnp"] < 1.0          # the control is not at that floor


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--tb=short",
                                  "-p", "no:cacheprovider"]))
