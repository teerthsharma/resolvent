"""THEORY.md sec 0 claims the resolvent stage is 'exists, mislabelled' -- an asset
already built in sigmoid. THEORY.md sec 1 calls the identification with the successor
representation 'the central identification'.

These tests ask what the stage IS, by computing it two ways and comparing arrays.

Reference forms, transcribed from the primary sources:
  APPNP / PPNP, Gasteiger et al. ICLR 2019, Eq. (3):
      Z_PPNP = softmax( alpha * (I - (1-alpha) A_hat)^-1 H ),  H = f_theta(X)
  Dayan 1993, Eq. (3.1):
      x_i = I + Q + Q^2 + ... = (I - Q)^-1,   Q = the non-absorbing block
      (Q is SUB-stochastic; no gamma appears in Dayan's equations at all)
  Leontief inverse, with Corollary 1:
      X = (I - C)^-1 B exists non-negative whenever C >= 0 has all row sums < 1
"""

import numpy as np
import pytest
import torch

from _device import DEVICES, t, n
from _lib import RNG, random_row_stochastic, resolvent, resolvent_solve


def theory_stage(P, gamma, b, device="cpu"):
    """THEORY.md sec 6: 'resolvent solve (I - gamma P)^-1 b, gamma explicit, P stochastic'"""
    return resolvent_solve(gamma * P, b, device=device)


def ppnp_stage(A_hat, alpha, H, device="cpu"):
    """APPNP Eq. (3), pre-softmax."""
    Ai = t(A_hat, device)
    Hi = t(H, device)
    eye = torch.eye(Ai.shape[0], dtype=torch.float64, device=device)
    z = alpha * torch.linalg.solve(eye - (1 - alpha) * Ai, Hi)
    return n(z)


# ==========================================================================
# CLAIM AS WRITTEN -- RED
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_claim_the_resolvent_stage_differs_from_the_appnp_propagation_equation(device):
    """If decision 2 is a new architectural component, it must compute something
    PPNP does not. Set gamma = 1-alpha and b = alpha*H and compare arrays."""
    n = 64
    P = random_row_stochastic(n)
    H = RNG.standard_normal((n, 8))
    worst = 0.0
    for alpha in (0.05, 0.1, 0.2, 0.5):
        gamma = 1 - alpha
        z_theory = theory_stage(P, gamma, alpha * H, device=device)
        z_ppnp = ppnp_stage(P, alpha, H, device=device)
        worst = max(worst, float(np.max(np.abs(z_theory - z_ppnp))))
    print("\n  max |THEORY sec 6 stage - APPNP Eq.(3)| over alpha grid: %.3e" % worst)
    assert worst > 1e-10, (
        f"the two are the same array to {worst:.2e}: THEORY.md sec 6's resolvent "
        f"stage IS the PPNP propagation of Gasteiger et al. ICLR 2019, with "
        f"gamma = 1 - alpha"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_claim_constraining_P_row_stochastic_yields_dayans_successor_representation(device):
    """Dayan Eq. (3.1) inverts a SUB-stochastic Q with per-state row sums.
    THEORY.md sec 1 constraint 1 forces every row sum to the SAME gamma.
    If sec 1 reproduces the SR, every Dayan-legal Q must be expressible as gamma*P."""
    Q = np.array([[0.10, 0.20],      # row sum 0.30
                  [0.60, 0.30]])     # row sum 0.90  -- legal absorbing-chain block
    assert (Q >= 0).all() and (Q.sum(axis=1) < 1).all()
    rs = Q.sum(axis=1)
    assert np.allclose(rs, rs[0]), (
        f"Dayan-legal Q has unequal row sums {rs}; THEORY.md sec 1 constraint 1 "
        f"('A = gamma P with gamma a free scalar and P normalized') cannot express it"
    )


# ==========================================================================
# WHAT THE OBJECT ACTUALLY IS -- GREEN
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_the_resolvent_is_the_leontief_inverse_under_the_row_sum_condition(device):
    """Leontief Corollary 1: C >= 0 with all row sums < 1 implies (I-C)^-1 exists
    and is non-negative. gamma*P with P row-stochastic satisfies exactly that."""
    for _ in range(20):
        P = random_row_stochastic(24, concentration=RNG.choice([0.05, 1.0]))
        C = 0.9 * P
        assert (C >= 0).all() and (C.sum(axis=1) < 1 - 1e-12).all()
        M = resolvent(C, device=device)
        assert (M >= -1e-12).all(), "Leontief inverse must be non-negative"
        neumann = sum(np.linalg.matrix_power(C, k) for k in range(400))
        assert np.allclose(M, neumann, atol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_dayans_substochastic_family_strictly_contains_the_gamma_P_family(device):
    """Every gamma*P is a legal Dayan Q, but not conversely. sec 1's constraint 1
    is a restriction OF the successor representation, not a route to it."""
    P = random_row_stochastic(8)
    gamma = 0.9
    assert ((gamma * P).sum(axis=1) < 1 + 1e-12).all()      # gamma*P is Dayan-legal
    Q = np.abs(RNG.standard_normal((8, 8)))
    Q = Q / (Q.sum(axis=1, keepdims=True) * RNG.uniform(1.05, 4.0, size=(8, 1)))
    rs = Q.sum(axis=1)
    assert (rs < 1).all() and rs.std() > 1e-3            # Dayan-legal, not any gamma*P


@pytest.mark.parametrize("device", DEVICES)
def test_appnp_teleport_optimum_is_far_from_the_gamma_the_lorenz_fit_wants(device):
    """APPNP Fig. 5: 'The optimum typically lies within alpha in [0.05, 0.2]'
    -> gamma in [0.8, 0.95]. The Lorenz fit in test_lorenz_occupancy drives
    gamma to 0.999, i.e. alpha = 0.001, an order of magnitude outside that band."""
    gamma_lorenz = 0.999
    alpha_lorenz = 1 - gamma_lorenz
    assert alpha_lorenz < 0.05, f"alpha={alpha_lorenz} lies inside APPNP's band"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--tb=short", "-p", "no:cacheprovider"]))
