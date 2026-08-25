"""THEORY.md sec 8 risk 1: 'Nested fixed points. (I - gamma P)^-1 is itself a Bellman
fixed point, and the DEQ solves a second fixed point on top of it. No proof yet that
the composition is a contraction. It may be redundant, or it may not converge.'

THEORY.md sec 1 makes the state->state map AFFINE in z:
    W.phi = T_0 z + sum_k a_k T_k z + B a + c
For a fixed action a this is z -> A(a) z + b(a). So the question is not open.
These tests pin down the dichotomy the theory does not state:

  affine    -> the fixed point is np.linalg.solve; decision 1 contributes nothing
  nonlinear -> the fixed point is real, but it is NOT (I - gamma P)^-1 b,
               so decision 2's successor-representation reading is void

You can have decision 1 or decision 2. Not both.
"""

import time

import numpy as np
import pytest

from _device import DEVICES
from _lib import RNG, random_row_stochastic, resolvent_solve, neumann_iterate

N = 128
GAMMA = 0.95


@pytest.fixture(scope="module", params=DEVICES)
def sys_(request):
    device = request.param
    P = random_row_stochastic(N)
    A = GAMMA * P
    b = RNG.standard_normal(N)
    return A, b, device


# ==========================================================================
# CLAIM AS WRITTEN -- RED
# ==========================================================================

def test_claim_the_outer_equilibrium_solver_computes_something_the_solve_does_not(sys_):
    """If the DEQ is not redundant, its equilibrium must differ from the closed form."""
    A, b, device = sys_
    z_direct = resolvent_solve(A, b, device=device)
    z_deq = neumann_iterate(A, b, iters=4000, device=device)
    gap = float(np.max(np.abs(z_deq - z_direct)))
    print("\n  |z_deq - z_closed_form|_inf = %.3e" % gap)
    assert gap > 1e-8, (
        f"the equilibrium solver reproduces np.linalg.solve to {gap:.2e}: "
        f"decision 1 adds no fixed point that decision 2 has not already closed"
    )


def test_claim_nesting_a_deq_around_the_resolvent_creates_a_second_fixed_point(sys_):
    """THEORY.md calls this 'a fixed point on top of a Bellman fixed point'.
    Compose them: g(z) = C (I - gamma P)^-1 (W z + b). If g is affine, the nest
    collapses to ONE linear solve and there is no second fixed point."""
    A, b, device = sys_
    W = 0.3 * RNG.standard_normal((N, N)) / np.sqrt(N)
    C = 0.3 * RNG.standard_normal((N, N)) / np.sqrt(N)
    M = np.linalg.inv(np.eye(N) - A)

    def g(z):
        return C @ (M @ (W @ z + b))

    z = np.zeros(N)
    for _ in range(3000):
        z = g(z)
    z_nested = z
    # the whole nest as a single affine solve
    z_single = np.linalg.solve(np.eye(N) - C @ M @ W, C @ M @ b)
    gap = float(np.max(np.abs(z_nested - z_single)))
    print("  nested-iteration vs single collapsed solve: %.3e" % gap)
    assert gap > 1e-8, (
        f"the 'nested fixed point' collapses to one linear solve (gap {gap:.2e}); "
        f"it is not a composition of two fixed points"
    )


def test_claim_a_nonlinear_deq_still_yields_the_successor_representation(sys_):
    """The only way decision 1 does work is a nonlinearity in the loop. Put one in
    and ask whether the equilibrium is still the occupancy vector (I - gamma P)^-1 b."""
    A, b, device = sys_
    z_sr = resolvent_solve(A, b, device=device)
    z = np.zeros(N)
    for _ in range(5000):
        z = np.tanh(A @ z + b)
    rel = float(np.linalg.norm(z - z_sr) / np.linalg.norm(z_sr))
    print("  relative gap between tanh-DEQ equilibrium and (I-gP)^-1 b: %.3f" % rel)
    assert rel < 0.05, (
        f"a nonlinear equilibrium is {rel:.1%} away from the resolvent: "
        f"it is not discounted expected future occupancy, so decision 2's "
        f"successor-representation reading does not apply to it"
    )


# ==========================================================================
# COST OF THE REDUNDANCY -- GREEN
# ==========================================================================

def test_iterative_equilibrium_is_strictly_slower_than_the_direct_solve(sys_):
    A, b, device = sys_
    z_direct = resolvent_solve(A, b, device=device)

    t0 = time.perf_counter()
    for _ in range(20):
        resolvent_solve(A, b, device=device)
    t_direct = (time.perf_counter() - t0) / 20

    # iterations the fixed-point solver needs to match float64 accuracy
    z, iters = np.zeros(N), 0
    while np.max(np.abs(z - z_direct)) > 1e-12 and iters < 100000:
        z = A @ z + b
        iters += 1
    t0 = time.perf_counter()
    neumann_iterate(A, b, iters, device=device)
    t_iter = time.perf_counter() - t0

    print("  direct solve %.3e s | %d fixed-point iterations %.3e s | %.1fx"
          % (t_direct, iters, t_iter, t_iter / t_direct))
    assert iters > 100
    assert t_iter > t_direct


@pytest.mark.parametrize("device", DEVICES)
def test_row_stochastic_P_makes_the_resolvent_available_in_closed_form_for_every_action(device):
    """The bilinear lift is affine in z for EVERY action a, so every action-conditioned
    equilibrium is a closed-form solve, not a search."""
    T0 = 0.5 * random_row_stochastic(32)
    T1 = 0.5 * random_row_stochastic(32)
    for a in (-1.0, 0.0, 0.37, 1.0):
        A = GAMMA * (T0 + a * T1) / max(abs(1 + a), 1e-9)
        b = RNG.standard_normal(32)
        if np.max(np.abs(np.linalg.eigvals(A))) >= 1:
            continue
        z1 = resolvent_solve(A, b, device=device)
        z2 = neumann_iterate(A, b, 5000, device=device)
        assert np.allclose(z1, z2, atol=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--tb=short", "-p", "no:cacheprovider"]))
