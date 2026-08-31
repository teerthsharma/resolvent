"""R2 -- "weigh by consequence, not similarity".

REQUIREMENTS.md R2 names one control the module must beat:

    "APPNP / personalized PageRank, Z = alpha(I - (1-alpha)Ahat)^-1 H. A linear
     resolvent stage reproduces it to 2.22e-16, so beating vanilla attention
     proves nothing."

and BACKLOG.md B1 names the live hypothesis said to escape that control:

    "max-plus star -- z* = max_a(r_a(x) + gamma A_a z*) by value iteration"
    "APPNP *is* the Kleene star in the (+,x) semiring, so moving to (max,+)
     escapes the reduction because the max-plus star has no linear closed form."

"No linear closed form" is true and is not the property that matters. The
Bellman operator is monotone and additively homogeneous, so its fixed point is
exactly the linear resolvent of its own greedy policy -- and that policy matrix
is 0/1 row-stochastic, i.e. precisely the gamma*P of THEORY.md sec 1 and the
Ahat of APPNP. These tests compute both sides and subtract.

The general statement both candidates are special cases of: the Kleene star of
a NON-NEGATIVE matrix, in any ordered semiring, has a non-negative influence
Jacobian. Consequence has a sign; `not` suppresses. Neither candidate can
represent suppression, for any parameter setting. That is sampled, not asserted.
"""

import torch
import pytest

from _device import DEVICES, HAS_CUDA
from _ceq import (appnp, discounted_resolvent, greedy_policy, policy_affine,
                  maxplus_star, random_instance, random_row_stochastic)


def _star_and_its_policy(R, A, gamma):
    z, info = maxplus_star(R, A, gamma)
    a, j = greedy_policy(z, R, A, gamma)
    E, c = policy_affine(a, j, R, A, gamma)
    return z, info, E, c


# ==========================================================================
# CLAIM AS WRITTEN -- RED
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_maxplus_star_differs_from_the_appnp_of_its_own_greedy_policy(device):
    """If (max,+) escapes the (+,x) reduction, the settled max-plus state must
    not be reproducible by a linear resolvent built from the SAME operator."""
    worst = 0.0
    witness = None
    for seed in range(24):
        R, A, gamma = random_instance(n=24, n_act=4, gamma=0.9, seed=seed, device=device)
        z, info, E, c = _star_and_its_policy(R, A, gamma)
        z_lin = discounted_resolvent(E, c, gamma)
        gap = float((z - z_lin).abs().max())
        if gap > worst:
            worst, witness = gap, seed
    print("\n  max |maxplus_star - (I - gamma E)^-1 c| over 24 instances: %.3e "
          "(seed %s)" % (worst, witness))
    assert worst > 1e-8, (
        f"the max-plus star reproduces the linear resolvent of its own greedy "
        f"policy to {worst:.2e}: E is 0/1 row-stochastic, so this IS APPNP with "
        f"alpha = 1 - gamma and Ahat = E. The max selected an adjacency; it did "
        f"not leave the (+,x) semiring."
    )


@pytest.mark.parametrize("device", DEVICES)
def test_maxplus_star_differs_from_appnp_in_appnps_own_parameterization(device):
    """Same statement written in APPNP Eq. (3) variables, so no algebraic
    sleight of hand is available: alpha = 1 - gamma, Ahat = E, H = c / alpha."""
    worst = 0.0
    for seed in range(12):
        R, A, gamma = random_instance(n=32, n_act=3, gamma=0.85, seed=100 + seed,
                                      device=device)
        z, _, E, c = _star_and_its_policy(R, A, gamma)
        alpha = 1.0 - gamma
        z_appnp = appnp(E, (c / alpha)[:, None], alpha)[:, 0]
        worst = max(worst, float((z - z_appnp).abs().max()))
    print("  max |maxplus_star - APPNP Eq.(3)| over 12 instances: %.3e" % worst)
    assert worst > 1e-8, f"identical to {worst:.2e} under APPNP's parameterization"


@pytest.mark.parametrize("device", DEVICES)
def test_maxplus_star_gradient_differs_from_the_appnp_resolvent(device):
    """The value could differ pointwise and still train identically. The
    training signal is dz*/dR. Measure it by finite differences on the full
    value iteration and compare against (I - gamma E)^-1, the APPNP resolvent.

    A max that is locally constant contributes nothing to first order."""
    R, A, gamma = random_instance(n=10, n_act=3, gamma=0.9, seed=7, device=device)
    z, _, E, c = _star_and_its_policy(R, A, gamma)
    a_star, _ = greedy_policy(z, R, A, gamma)
    n = R.shape[-1]
    eye = torch.eye(n, dtype=R.dtype, device=device)
    J_lin = torch.linalg.inv(eye - gamma * E)          # APPNP's resolvent

    eps = 1e-6
    J_fd = torch.zeros((n, n), dtype=R.dtype, device=device)
    for i in range(n):
        Rp, Rm = R.clone(), R.clone()
        Rp[a_star[i], i] += eps
        Rm[a_star[i], i] -= eps
        zp, _ = maxplus_star(Rp, A, gamma)
        zm, _ = maxplus_star(Rm, A, gamma)
        J_fd[:, i] = (zp - zm) / (2 * eps)

    gap = float((J_fd - J_lin).abs().max())
    print("  max |d(maxplus_star)/dR  -  (I - gamma E)^-1| = %.3e (scale %.3f)"
          % (gap, float(J_lin.abs().max())))
    assert gap > 1e-4, (
        f"the max-plus star's Jacobian equals the APPNP resolvent to {gap:.2e}: "
        f"the operator is piecewise affine and the argmax is locally constant, "
        f"so to first order -- which is all gradient descent sees -- it IS APPNP"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_maxplus_star_can_give_one_token_negative_influence_on_another(device):
    """R2's own example is `not`: a token whose consequence is to SUPPRESS.
    dz*_k/dR_i < 0 for some pair is the minimum requirement. The influence
    matrix is (I - gamma E)^-1 = sum_k gamma^k E^k with E >= 0."""
    worst = 0.0
    for seed in range(40):
        R, A, gamma = random_instance(n=16, n_act=4, gamma=0.9, seed=200 + seed,
                                      device=device)
        z, _, E, c = _star_and_its_policy(R, A, gamma)
        n = R.shape[-1]
        eye = torch.eye(n, dtype=R.dtype, device=device)
        J = torch.linalg.inv(eye - gamma * E)
        worst = min(worst, float(J.min()))
    print("  most negative influence entry over 40 max-plus instances: %.3e" % worst)
    assert worst < -1e-9, (
        f"min influence entry is {worst:.3e} >= 0 across every instance: the "
        f"max-plus star cannot make any token suppress any other, because its "
        f"star is the Kleene star of a NON-NEGATIVE matrix. `not` is unreachable."
    )


@pytest.mark.parametrize("device", DEVICES)
def test_appnp_can_give_one_token_negative_influence_on_another(device):
    """The control fails the same way, for the same reason. Both are stars of
    non-negative matrices, so this is one obstruction, not two."""
    worst = 0.0
    for seed in range(40):
        A_hat = random_row_stochastic(24, seed=300 + seed, device=device)
        n = A_hat.shape[-1]
        eye = torch.eye(n, dtype=A_hat.dtype, device=device)
        for alpha in (0.05, 0.1, 0.2, 0.5):
            J = alpha * torch.linalg.inv(eye - (1 - alpha) * A_hat)
            worst = min(worst, float(J.min()))
    print("  most negative influence entry over 160 APPNP kernels: %.3e" % worst)
    assert worst < -1e-9, (
        f"min influence entry is {worst:.3e} >= 0: APPNP cannot represent "
        f"suppression either"
    )


# ==========================================================================
# WHAT IS ACTUALLY TRUE -- GREEN
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_the_reduction_survives_a_sweep_over_gamma_action_count_and_size(device):
    """The collapse is not an artefact of one setting."""
    worst = 0.0
    rows = []
    for gamma in (0.5, 0.8, 0.95, 0.99):
        for n_act in (2, 8):
            for n in (8, 48):
                R, A, g = random_instance(n=n, n_act=n_act, gamma=gamma,
                                          seed=n * n_act, device=device)
                z, _, E, c = _star_and_its_policy(R, A, g)
                gap = float((z - discounted_resolvent(E, c, g)).abs().max())
                worst = max(worst, gap)
                rows.append((gamma, n_act, n, gap))
    print("  worst gap over 16 (gamma, n_act, n) cells: %.3e" % worst)
    for r in rows[:4]:
        print("    gamma=%.2f n_act=%d n=%d  gap=%.2e" % r)
    assert worst < 1e-7, f"reduction failed somewhere: worst {worst:.3e}"


@pytest.mark.parametrize("device", DEVICES)
def test_the_selected_policy_matrix_is_exactly_a_row_stochastic_kernel(device):
    """The object the max hands to the resolvent is 0/1 and row-stochastic, so
    it lies inside THEORY.md sec 1's family gamma*P, and therefore inside R3's
    measured convex-hull ceiling as well."""
    R, A, gamma = random_instance(n=32, n_act=5, gamma=0.9, seed=11, device=device)
    z, _, E, c = _star_and_its_policy(R, A, gamma)
    ones = torch.ones(32, dtype=E.dtype, device=device)
    assert torch.all((E == 0) | (E == 1))
    assert torch.allclose(E.sum(dim=1), ones)
    Rn = (1 - gamma) * torch.linalg.inv(
        torch.eye(32, dtype=E.dtype, device=device) - gamma * E)
    assert torch.all(Rn >= -1e-12)
    assert torch.allclose(Rn.sum(dim=1), ones, atol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_a_signed_resolvent_does_achieve_negative_influence(device):
    """The escape is not a different semiring, it is a SIGN. Drop non-negativity
    and keep the contraction, and suppression becomes representable at once.
    This is what an R2 mechanism has to have."""
    n = 16
    g = torch.Generator(device="cpu").manual_seed(5)
    M = torch.randn((n, n), generator=g, dtype=torch.float64)
    M = (0.9 * M / torch.linalg.matrix_norm(M, 2)).to(device)
    eye = torch.eye(n, dtype=M.dtype, device=device)
    J = torch.linalg.inv(eye - M)
    rho = float(torch.linalg.eigvals(M).abs().max())
    print("  signed operator rho=%.4f  min influence entry %.4f  max %.4f"
          % (rho, float(J.min()), float(J.max())))
    assert rho < 1.0
    assert float(J.min()) < -1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_maxplus_star_is_a_strictly_larger_function_class_than_one_appnp(device):
    """The honest counterweight: no SINGLE linear resolvent reproduces the map
    r -> z*(r), because the selected policy changes with r. The reduction is
    pointwise, not uniform. What it costs is the gradient, tested above."""
    R, A, gamma = random_instance(n=12, n_act=4, gamma=0.9, seed=13, device=device)
    seen = set()
    for k in range(20):
        pert = torch.randn(R.shape, generator=torch.Generator().manual_seed(k),
                           dtype=torch.float64).to(device)
        Rk = R + 3.0 * pert
        z, _ = maxplus_star(Rk, A, gamma)
        a, j = greedy_policy(z, Rk, A, gamma)
        seen.add((tuple(a.tolist()), tuple(j.tolist())))
    print("  distinct greedy policies over 20 reward draws: %d" % len(seen))
    assert len(seen) > 1, "the argmax never switched; the map is globally affine"


# ==========================================================================
# WHAT "SEPARABLE FROM SIMILARITY" ACTUALLY MEANS -- GREEN
#
# R2 sets APPNP as "the control that must be beaten", which reads as though
# APPNP were the similarity baseline. It is not. A similarity weighting is a
# function of the PAIR alone, so perturbing a third token cannot change the
# RATIO of two weights in a row -- only the shared normalizer moves. That
# ratio invariance is a decidable test, and it splits the field in three:
#
#   tier 1  pairwise similarity        1-hop softmax attention      ratio-invariant
#   tier 2  non-negative multi-hop     APPNP == max-plus star       not invariant,
#                                                                   influence >= 0
#   tier 3  signed multi-hop           signed contraction           not invariant,
#                                                                   influence signed
#
# APPNP already leaves tier 1. So does the max-plus star -- by landing in the
# same tier, which is the whole finding. Every test above measures a tier-2
# object against another tier-2 object.
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_one_hop_similarity_attention_is_pairwise_decomposable(device):
    """Perturb a third token's key. The ratio of any two OTHER weights in a row
    is unchanged, exactly, because softmax ratios cancel the normalizer."""
    n, d = 12, 8
    g = torch.Generator(device="cpu").manual_seed(31)
    q = torch.randn((n, d), generator=g, dtype=torch.float64).to(device)
    k = torch.randn((n, d), generator=g, dtype=torch.float64).to(device)

    def W(kk):
        return torch.softmax(q @ kk.T / d ** 0.5, dim=1)

    k2 = k.clone()
    k2[7] += 3.0 * torch.randn((d,), generator=g, dtype=torch.float64).to(device)
    A0, A1 = W(k), W(k2)
    i, j, j2 = 0, 1, 2                      # none of them is token 7
    r0 = float(A0[i, j] / A0[i, j2])
    r1 = float(A1[i, j] / A1[i, j2])
    print("\n  softmax attention weight ratio w[0,1]/w[0,2] before %.9f after %.9f"
          % (r0, r1))
    assert abs(r0 - r1) < 1e-12


@pytest.mark.parametrize("device", DEVICES)
def test_appnp_influence_is_not_pairwise_decomposable(device):
    """The same perturbation moves APPNP's influence ratio, because influence
    is a path sum through the whole graph. APPNP is already not a similarity
    weighting -- which is why beating it, not beating attention, is the bar."""
    n, alpha = 12, 0.1
    eye = torch.eye(n, dtype=torch.float64, device=device)
    A_hat = random_row_stochastic(n, seed=32, device=device)

    def S(M):
        return alpha * torch.linalg.inv(eye - (1 - alpha) * M)

    M2 = A_hat.clone()
    g = torch.Generator(device="cpu").manual_seed(33)
    row = torch.rand((n,), generator=g, dtype=torch.float64).to(device)
    M2[7] = row / row.sum()                 # still row-stochastic
    S0, S1 = S(A_hat), S(M2)
    i, j, j2 = 0, 1, 2
    r0 = float(S0[i, j] / S0[i, j2])
    r1 = float(S1[i, j] / S1[i, j2])
    print("  APPNP influence ratio S[0,1]/S[0,2] before %.6f after %.6f (%.1f%% move)"
          % (r0, r1, 100 * abs(r1 - r0) / r0))
    assert abs(r1 - r0) / r0 > 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_the_maxplus_star_lands_in_the_same_tier_as_appnp(device):
    """Its influence ratio moves under a third-party perturbation too, and its
    influence entries are still non-negative. Same tier, both coordinates."""
    R, A, gamma = random_instance(n=12, n_act=4, gamma=0.9, seed=34, device=device)
    eye = torch.eye(12, dtype=R.dtype, device=device)

    def infl(Rx):
        z, _ = maxplus_star(Rx, A, gamma)
        a, j = greedy_policy(z, Rx, A, gamma)
        E, _ = policy_affine(a, j, Rx, A, gamma)
        return torch.linalg.inv(eye - gamma * E)

    R2 = R.clone()
    R2[:, 7] += 5.0
    S0, S1 = infl(R), infl(R2)
    moved = float((S1 - S0).abs().max())
    print("  max-plus influence moved %.4f under a third-token perturbation; "
          "min entry %.3e" % (moved, min(float(S0.min()), float(S1.min()))))
    assert moved > 1e-6                     # not pairwise: leaves tier 1
    assert float(S0.min()) >= 0.0 and float(S1.min()) >= 0.0   # stays in tier 2


@pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device available")
def test_maxplus_star_cpu_cuda_parity():
    """The CPU path is the parity oracle. float64 value iteration must agree."""
    R, A, gamma = random_instance(n=32, n_act=4, gamma=0.9, seed=21, device="cpu")
    z_cpu, i_cpu = maxplus_star(R, A, gamma)
    z_gpu, i_gpu = maxplus_star(R.cuda(), A.cuda(), gamma)
    gap = float((z_cpu - z_gpu.cpu()).abs().max())
    print("  cpu/cuda max-plus star parity: %.3e  (iters %d vs %d)"
          % (gap, i_cpu["iters"], i_gpu["iters"]))
    assert gap < 1e-12


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--tb=short",
                                  "-p", "no:cacheprovider"]))
