"""THEORY.md lists these as two separate items, in two separate sections:

  sec 1 (a benefit):  'With P row-stochastic, rho(gamma P) = gamma < 1 holds by
                       construction ... becomes a structural guarantee.'
  sec 8 risk 5 (a cost): 'Stochastic P may destroy expressiveness. Row-stochastic
                       non-negative P is a much smaller hypothesis class...'

These tests ask whether they are two facts or one fact stated twice with opposite
signs. The mechanism under test: row-stochasticity pins the TOP eigenvector of P to
the constant vector 1 (P1 = 1). The resolvent maps eigenvalue lambda -> 1/(1-gamma*lambda),
so the constant mode always receives the largest gain, 1/(1-gamma).
"""

import numpy as np
import pytest
import torch

from _device import DEVICES, t, n
from _lib import RNG, random_row_stochastic, resolvent


def modes(P, gamma, device="cpu"):
    """(gain on the Perron/constant mode, gains on all other modes)."""
    lam = torch.linalg.eigvals(t(P, device))
    order = torch.argsort(torch.abs(lam), descending=True)
    lam = lam[order]
    gains = 1.0 / torch.abs(1.0 - gamma * lam)
    return float(gains[0]), n(gains[1:]), n(lam)


# ==========================================================================
# CLAIM AS WRITTEN -- RED
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_claim_the_resolvent_can_amplify_structure_relative_to_the_mean(device):
    """If the certificate and the expressiveness loss were independent, some
    stochastic kernel would let a non-constant mode out-gain the constant mode."""
    best = 0.0
    for n in (4, 16, 64):
        for conc in (0.02, 0.2, 1.0, 5.0):
            for _ in range(40):
                P = random_row_stochastic(n, concentration=conc)
                for gamma in (0.3, 0.7, 0.9, 0.99):
                    g0, gk, _ = modes(P, gamma, device=device)
                    best = max(best, float(np.max(gk) / g0))
    print("\n  best non-constant/constant gain ratio over 1920 kernels: %.6f" % best)
    assert best > 1.0 + 1e-9, (
        f"max relative gain on any non-constant mode is {best:.6f} <= 1: the "
        f"resolvent of a stochastic kernel NEVER amplifies structure relative to "
        f"the mean, for any gamma. The certificate and the expressiveness loss "
        f"are the same constraint."
    )


@pytest.mark.parametrize("device", DEVICES)
def test_claim_some_gamma_gives_both_a_multi_hop_horizon_and_contrast_retention(device):
    """The successor representation is only interesting if it aggregates over many
    steps (horizon H = gamma/(1-gamma) hops). Ask for H >= 10 AND >= 50% retention
    of the second mode relative to the constant mode."""
    hits = []
    for n in (16, 64):
        for conc in (0.02, 0.2, 1.0):
            for _ in range(30):
                P = random_row_stochastic(n, concentration=conc)
                for gamma in (0.9, 0.95, 0.99, 0.999):
                    g0, gk, _ = modes(P, gamma, device=device)
                    H = gamma / (1.0 - gamma)
                    c = float(np.max(gk) / g0)
                    if H >= 10.0 and c >= 0.5:
                        hits.append((n, conc, gamma, H, c))
    print("  kernels achieving horizon>=10 AND retention>=50%%: %d" % len(hits))
    assert hits, (
        "no stochastic kernel achieves a 10-hop horizon while retaining half the "
        "contrast: horizon and contrast are one dial, not two"
    )


# ==========================================================================
# THE LAW, AND THE CONTRAST WITH AN UNCONSTRAINED A -- GREEN
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_retention_and_horizon_are_exactly_reciprocally_coupled(device):
    """c = (1-gamma)/(1-gamma*lambda2) and H = gamma/(1-gamma) satisfy
    c * (1 + H*(1-lambda2)) = 1 exactly. One dial, closed form."""
    for _ in range(30):
        n = 32
        P = random_row_stochastic(n, concentration=RNG.choice([0.05, 1.0, 5.0]))
        lam = np.linalg.eigvals(P)
        lam2 = float(np.real(lam[np.argsort(-np.abs(lam))][1]))
        for gamma in (0.5, 0.9, 0.99):
            c = (1 - gamma) / (1 - gamma * lam2)
            H = gamma / (1 - gamma)
            assert abs(c * (1 + H * (1 - lam2)) - 1.0) < 1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_an_unconstrained_operator_with_the_same_spectral_radius_amplifies_structure(device):
    """rho is NOT what costs the expressiveness. Same rho = 0.995, unconstrained:
    the second mode out-gains the first by 180x. The cost comes from P1 = 1."""
    A = np.diag([0.1, 0.995])
    R = resolvent(A, device=device)
    gains = np.abs(np.diag(R))
    print("  unconstrained diag(0.1, 0.995): mode gains %.3f and %.1f -> ratio %.1fx"
          % (gains[0], gains[1], gains[1] / gains[0]))
    assert float(np.max(np.abs(np.linalg.eigvals(A)))) == pytest.approx(0.995)
    assert gains[1] / gains[0] > 100


@pytest.mark.parametrize("device", DEVICES)
def test_normalized_resolvent_of_a_stochastic_kernel_is_itself_a_stochastic_kernel(device):
    """(1-gamma)(I - gamma P)^-1 is row-stochastic: it is a weighted average, i.e.
    exactly the personalized-PageRank / graph-diffusion operator, and can only smooth."""
    for _ in range(20):
        P = random_row_stochastic(48, concentration=RNG.choice([0.05, 1.0]))
        for gamma in (0.5, 0.9, 0.99):
            R = (1 - gamma) * resolvent(gamma * P, device=device)
            assert (R > -1e-12).all()
            assert np.allclose(R.sum(axis=1), 1.0, atol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_as_gamma_goes_to_one_the_resolvent_collapses_to_rank_one(device):
    """The limit of the 'structural guarantee' regime is total information loss."""
    P = random_row_stochastic(64, concentration=1.0)
    ranks = []
    for gamma in (0.5, 0.9, 0.99, 0.9999):
        R = (1 - gamma) * resolvent(gamma * P, device=device)
        s = np.linalg.svd(R, compute_uv=False)
        ranks.append((gamma, float(s[1] / s[0])))
    print("  sigma_2/sigma_1 of the normalized resolvent: "
          + ", ".join("g=%.4f:%.2e" % r for r in ranks))
    assert ranks[-1][1] < ranks[0][1] / 100


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--tb=short", "-p", "no:cacheprovider"]))
