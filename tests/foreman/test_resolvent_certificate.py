"""THEORY.md sec 1: 'With P row-stochastic, rho(gamma P) = gamma < 1 holds *by
construction*, not by measurement. The Banach certificate stops being something
sigmoid hopes for and becomes a structural guarantee.'

THEORY.md sec 1 also defines the certificate, in the same section, three lines earlier:

    (I - A)z = b       where A is the state->state block, rho = sigma_max(A)

sigmoid's own README (fetched 2026-08-24) says the same thing in words:
'Banach fixed-point theorem provides error bounds when spectral norm stays below unity.'

So the certificate is the SPECTRAL NORM sigma_max, not the spectral radius.
These tests assert THEORY.md's claim under THEORY.md's own definition.
"""

import numpy as np
import pytest

from _device import DEVICES
from _lib import RNG, random_row_stochastic, sigma_max, spectral_radius, neumann_iterate

GAMMA = 0.995   # sigmoid's documented rho_max, per THEORY.md sec 1


# ==========================================================================
# CLAIM AS WRITTEN  -- these are the RED tests
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_claim_row_stochastic_makes_the_banach_certificate_structural(device):
    """THEORY.md sec 1: the certificate holds 'by construction'.
    Certificate := sigma_max(A) < 1, per THEORY.md's own definition of rho."""
    worst = 0.0
    witness = None
    for n in (2, 4, 8, 16, 64, 256):
        for _ in range(50):
            P = random_row_stochastic(n, concentration=0.05)   # near-deterministic kernels
            s = sigma_max(GAMMA * P, device=device)
            if s > worst:
                worst, witness = s, (n, P)
    assert worst < 1.0, (
        f"certificate FAILS by construction: max sigma_max(gamma*P) = {worst:.4f} "
        f"at n={witness[0]}, gamma={GAMMA}"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_claim_certificate_holds_for_the_smallest_possible_counterexample(device):
    """Two states, both transitioning to state 1. Perfectly legal transition kernel."""
    P = np.array([[0.0, 1.0],
                  [0.0, 1.0]])
    assert np.allclose(P.sum(axis=1), 1.0) and (P >= 0).all()
    A = GAMMA * P
    assert sigma_max(A, device=device) < 1.0, (
        f"sigma_max = {sigma_max(A, device=device):.6f} >= 1 while spectral radius = "
        f"{spectral_radius(A, device=device):.6f}; THEORY.md sec 1 claims the certificate is structural"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_claim_stochastic_family_is_not_a_subset_of_the_rejected_rho_clip_ball(device):
    """THEORY.md sec 1 presents 'constrain P stochastic' as a DIFFERENT move from
    'clip rho to 0.995', which sigmoid measured and rejected. If every gamma*P
    already lies inside the rho-clip ball, it is the same move, tightened."""
    escapes = 0
    for n in (4, 16, 64):
        for _ in range(200):
            P = random_row_stochastic(n, concentration=RNG.choice([0.05, 1.0, 5.0]))
            if spectral_radius(GAMMA * P, device=device) > GAMMA + 1e-9:
                escapes += 1
    assert escapes > 0, (
        "0 of 600 sampled gamma*P escaped the rho<=0.995 ball that sigmoid rejected: "
        "the stochastic family is a strict subset of the rejected family"
    )


# ==========================================================================
# CORRECTED STATEMENTS -- these should be GREEN, and they name the real fix
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_correct_statement_is_that_the_infinity_norm_not_the_spectral_norm_is_gamma(device):
    """Row-stochasticity gives ||gamma P||_inf = gamma exactly. That IS a Banach
    certificate -- in the sup norm. It is not the norm sigmoid measures."""
    for n in (2, 8, 64):
        for _ in range(20):
            P = random_row_stochastic(n, concentration=0.05)
            assert abs(np.abs(GAMMA * P).sum(axis=1).max() - GAMMA) < 1e-12


@pytest.mark.parametrize("device", DEVICES)
def test_spectral_radius_is_exactly_gamma_but_does_not_bound_transient_growth(device):
    """rho(gamma P) = gamma is true. It gives asymptotic convergence, not a
    per-iteration error bound. Show a kernel where the Neumann iterate's error
    GROWS before it shrinks, so the sigma_max-based error bound is invalid."""
    P = np.array([[0.0, 1.0],
                  [0.0, 1.0]])
    A = GAMMA * P
    assert abs(spectral_radius(A, device=device) - GAMMA) < 1e-9
    assert sigma_max(A, device=device) > 1.0, "expected spectral norm above unity for this kernel"

    # start the solver off along A's top right-singular direction
    b = np.array([1.0, 0.0])
    zstar = np.linalg.solve(np.eye(2) - A, b)
    v1 = np.linalg.svd(A)[2][0]
    z0 = zstar - v1
    errs = [np.linalg.norm(neumann_iterate(A, b, k, z0=z0, device=device) - zstar) for k in range(3)]
    assert errs[1] > errs[0], (
        f"expected the equilibrium solver's error to GROW on step 1 despite "
        f"rho={spectral_radius(A, device=device):.3f}<1; errs={['%.4f' % e for e in errs]}"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_unconstrained_A_can_sit_in_the_rho_ball_and_still_be_outside_the_stochastic_family(device):
    """The converse containment: rho-clipping admits operators stochasticity forbids.
    A sign-inverting coupling with rho = gamma is the simplest one."""
    A = np.array([[0.0, -GAMMA],
                  [GAMMA, 0.0]])
    assert abs(spectral_radius(A, device=device) - GAMMA) < 1e-12
    assert (A < 0).any(), "witness must have a negative entry"
    # not representable as g*P for any g>0 with P>=0
    assert not np.all(A >= 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short", "-p", "no:cacheprovider"]))
