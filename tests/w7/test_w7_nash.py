"""W7 -- Nash-equilibrium attention over a set system.

WHY. W4 measured the tier ladder in the predicted direction (attention 5.8198,
APPNP 4.2107, signed 2.6151 OOD NRMSE, 4/5 seeds) and refuted the claim that
matters: all three sit ABOVE 1.0, so every arm is worse than predicting the mean
on a held-out composition. The signed arm degrades more gracefully. It does not
generalize.

Diagnosis: a token's stance is currently a learned regression, and a regression
has no reason to compose. The held-out cell asks for the composition of two sign
flips seen only separately.

THE FIX UNDER TEST. Make the stance a quantal-response (logit Nash) equilibrium
instead of a learned map:

    s      = sigmoid((M s + b) / tau)      support probability per token
    stance = 2 s - 1  in [-1, 1]           signed
    A_ij   = rho * P_ij * stance_j         |A_ij| <= rho * P_ij

A best response is ORDINAL. It turns on the sign of a payoff difference, not its
magnitude. Composing two sign flips is an ordinal operation, which is what the
held-out cell requires and what a magnitude regression cannot supply.

WHAT IS PRESERVED, and it is not luck. |A_ij| <= rho * P_ij keeps row L1 bounded
by rho, so A stays strictly causal with bounded rows, `CEQ.Nilpotent.pow_card_eq_zero`
still gives A^n = 0, and `occupancy_is_exact_inverse` still gives an exact
terminating resolvent. Nilpotency is sign-blind and the bound survives the
stance multiplication by construction.

WHAT IS NOT NEEDED. Nash/QRE exists by Brouwer. No contraction certificate, no
Perron vector, no power iteration -- so the measured reducible-A failure mode
(power iteration returning w with exact zeros, weighted sup norm undefined)
stays out of reach.

R1, ON FOREMAN'S OWN TERMS. He deleted R1 because an affine update needs no
iteration (residual 1.05e-15) and max-plus is only PIECEWISE non-affine (probe
ratio 5.5, blind below the cell radius). He stated what would survive: an update
non-affine EVERYWHERE, naming `tanh(Mz+b)` with signed M at probe ratios
5.3e5-4.7e9. `sigmoid((Ms+b)/tau)` is that shape, and the probe is run below.

THE DIFFERENTIAL GEOMETRY IS NOT DECORATION. The multiplicative replicator update
is natural-gradient ascent on the simplex under the Fisher-Rao metric; that is
why the update is multiplicative rather than additive, and it is why the iterate
stays in [0,1] without projection.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ceq import arms, corpus, nash

N_TRAIN, N_TEST, STEPS, SEED = 384, 128, 400, 0
S, D = 24, 16
RHO = 0.9


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    return torch.device(request.param)


@pytest.fixture(scope="module")
def data():
    return corpus.build(n_train=N_TRAIN, n_test=N_TEST, seed=SEED)


def game(device, seed=0, n=S, dtype=torch.float64):
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = torch.randn(n, n, generator=g, dtype=dtype)
    m = (m + m.T) / 2 / n ** 0.5          # symmetric => potential game
    b = torch.randn(n, generator=g, dtype=dtype)
    return m.to(device), b.to(device)


# ------------------------------------------------- the equilibrium is genuine

def test_qre_reaches_a_real_fixed_point(device):
    """The returned s must satisfy its own defining equation, not merely stop
    changing because the iteration cap ran out."""
    m, b = game(device)
    s = nash.qre_stance(m, b, tau=1.0, iters=500)
    resid = float((s - torch.sigmoid((m @ s + b) / 1.0)).abs().max())
    assert resid < 1e-10, f"best-response residual {resid:.3e}"


def test_stance_stays_in_range_without_projection(device):
    """Fisher-Rao / replicator geometry: the multiplicative form keeps the
    iterate on the simplex by construction. If this needs clamping, the update
    is not the natural gradient it claims to be."""
    m, b = game(device)
    s = nash.qre_stance(m, b, tau=1.0, iters=500)
    assert float(s.min()) >= 0.0 and float(s.max()) <= 1.0, (float(s.min()), float(s.max()))


def test_the_equilibrium_actually_needs_iteration(device):
    """R1, revived on Foreman's terms. One pass must NOT already be the answer.

    He deleted R1 because every gamma-contraction moves gamma/(1-gamma) of the
    way on pass 1, so 'the output changes across early iterations' is satisfied
    by np.linalg.solve. The bar here is the distance of pass 1 from the settled
    point, which for an affine update is a fixed fraction and for a genuine
    equilibrium is not.
    """
    m, b = game(device)
    star = nash.qre_stance(m, b, tau=0.25, iters=2000)
    one = nash.qre_stance(m, b, tau=0.25, iters=1)
    rel = float((one - star).norm() / star.norm())
    assert rel > 0.05, f"pass 1 is already the settled state (rel {rel:.3e}); R1 stays dead"


def test_update_is_non_affine_everywhere_not_piecewise(device):
    """Foreman's collapse probe, with the radius sweep he showed was necessary.

    Max-plus scored affine-fit residual ratios of 5.5 at small radius -- below
    the cell radius the probe certifies a piecewise-affine operator as affine.
    A smooth sigmoid has no cells, so the ratio must stay high at EVERY radius.
    """
    m, b = game(device)
    star = nash.qre_stance(m, b, tau=0.25, iters=2000)
    ratios = [nash.affine_fit_ratio(m, b, star, tau=0.25, radius=r)
              for r in (1e-6, 1e-4, 1e-2, 1e-1)]
    assert min(ratios) > 10.0, f"probe collapsed at some radius: {ratios}"


# --------------------------------------------------- the operator is preserved

def test_operator_is_signed_and_bounded(device):
    """Tier 3 kept, and the L1 bound the Lean depends on kept with it."""
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(1, 1, S, D, generator=g, dtype=torch.float64).to(device)
    a = nash.nash_operator(x, x, x, rho=RHO, tau=0.5, iters=200)
    assert float(a.min()) < 0.0, "operator is not signed; tier 3 lost"
    assert float(a.abs().sum(-1).max()) <= RHO + 1e-9, float(a.abs().sum(-1).max())
    assert float(a.triu(0).abs().max()) == 0.0, "not strictly lower triangular"


def test_nilpotency_survives_the_stance_multiplication(device):
    """`CEQ.Nilpotent.pow_card_eq_zero` has one hypothesis and no sign
    condition. Multiplying columns by a signed stance cannot break it."""
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(1, 1, S, D, generator=g, dtype=torch.float64).to(device)
    a = nash.nash_operator(x, x, x, rho=RHO, tau=0.5, iters=200)
    ev = torch.linalg.eigvals(a.reshape(-1, S, S).to(torch.complex128))
    assert float(ev.abs().max()) < 1e-12
    assert float(torch.linalg.matrix_power(a, S).abs().max()) == 0.0


# --------------------------------------------------------- W7's kill condition

#: Issue #2: both kill conditions fired, and stay recorded rather than red.
#: strict=True, so a change that ever earns either bar XPASSes, fails the run,
#: and forces docs/FAILS.md section 3 to be rewritten instead of going stale.
RETIRED = pytest.mark.xfail(strict=True, reason=(
    "issue #2: nash OOD NRMSE 5.2888 +- 0.6041 vs signed 2.7333 +- 1.0235 over "
    "5 seeds, 0/5 below 1.0, 0/5 beating signed, 4.6142 with both known "
    "defects repaired; python scripts/nash_repairs.py, docs/FAILS.md section 3"))


@RETIRED
def test_nash_arm_generalizes_under_intervention(device, data):
    """RED first. THE KILL CONDITION, absolute and non-negotiable.

    OOD NRMSE on the held-out composition must go BELOW 1.0 -- better than
    predicting the mean. Relative improvement over APPNP is not enough; that was
    already measured at 0.6770 while every arm sat above 1.0.
    """
    got = arms.run_all(data, device=device, steps=STEPS, seed=SEED, kinds=("nash",))
    assert got["nash"]["ood_nrmse"] < 1.0, (
        f"nash OOD NRMSE {got['nash']['ood_nrmse']:.4f} >= 1.0: still worse than "
        f"predicting the mean. The equilibrium stance did not buy composition. {got}")


@RETIRED
def test_nash_arm_beats_the_signed_regression_it_replaces(device, data):
    """Secondary. If the equilibrium stance is not better than the learned
    stance it replaced, the game-theoretic machinery is unpaid complexity."""
    got = arms.run_all(data, device=device, steps=STEPS, seed=SEED,
                       kinds=("signed", "nash"))
    assert got["nash"]["ood_nrmse"] < got["signed"]["ood_nrmse"], got


def test_parameter_count_stays_matched(device, data):
    """A win bought with capacity is not a win."""
    c = {k: arms.build(k, data["vocab"], device=device).n_params()
         for k in ("attention", "appnp", "signed", "nash")}
    assert max(c.values()) <= 1.10 * min(c.values()), c
