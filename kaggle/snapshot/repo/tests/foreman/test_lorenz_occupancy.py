"""THEORY.md sec 1, the sentence the author flags as load-bearing:

  'This also disposes of the rho_max problem the sigmoid README already documents
   and rejects: clipping rho to 0.995 degraded Lorenz one-step NRMSE from 0.067 to
   0.317, because clipping misreports chaotic dynamics. Constraining P to be
   stochastic does not misreport anything.'

Operationalized: fit the SAME one-step Lorenz operator three ways on the same data
 (a) unconstrained ridge          -- what sigmoid measured
 (b) spectral-norm clip to 0.995  -- the move the README rejected
 (c) A = gamma*P, P row-stochastic non-negative -- what THEORY.md sec 1 proposes
and compare one-step NRMSE.

The lift is [1,x,y,z,x^2,xy,xz,y^2,yz,z^2] and the integrator is forward Euler, so
the true one-step map is EXACTLY linear in the lift. The unconstrained fit is
therefore not handicapped, and every gap below is caused purely by the constraint.
"""

import numpy as np
import pytest

from _device import DEVICES
from _lib import (lorenz_lifted_pairs, fit_ridge, clip_sigma, best_stochastic,
                  nrmse, sigma_max, spectral_radius)

RHO_MAX = 0.995


@pytest.fixture(scope="module", params=DEVICES)
def bench(request):
    device = request.param
    Zt, Zt1 = lorenz_lifted_pairs(n_steps=8000, dt=0.005)
    A_free = fit_ridge(Zt, Zt1, device=device)
    A_clip = clip_sigma(A_free, RHO_MAX, device=device)
    e_stoch, g_star, A_stoch = best_stochastic(Zt, Zt1, device=device)
    out = dict(
        free=nrmse(Zt @ A_free.T, Zt1),
        clip=nrmse(Zt @ A_clip.T, Zt1),
        stoch=e_stoch,
        gamma=g_star,
        sigma_free=sigma_max(A_free, device=device),
        rho_free=spectral_radius(A_free, device=device),
        device=device,
    )
    print("\n  NRMSE unconstrained      = %.4f  (sigma_max=%.3f, rho=%.3f)"
          % (out["free"], out["sigma_free"], out["rho_free"]))
    print("  NRMSE sigma-clip 0.995   = %.4f" % out["clip"])
    print("  NRMSE row-stochastic     = %.4f  (best gamma=%.3f)" % (out["stoch"], out["gamma"]))
    return out


# ==========================================================================
# CLAIM AS WRITTEN -- RED
# ==========================================================================

def test_claim_stochastic_constraint_costs_no_more_than_the_rejected_clip(bench):
    """'Constraining P to be stochastic does not misreport anything' -- while
    clipping does. Then the stochastic fit must not be worse than the clipped fit."""
    assert bench["stoch"] <= bench["clip"] + 1e-6, (
        f"row-stochastic NRMSE {bench['stoch']:.4f} is WORSE than the clip "
        f"{bench['clip']:.4f} that sigmoid already rejected "
        f"(unconstrained baseline {bench['free']:.4f})"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_claim_stochastic_cost_does_not_grow_with_step_size(device):
    """The identity IS row-stochastic, so at tiny dt the constraint is nearly free --
    and at tiny dt the operator is nearly doing nothing. Sweep dt to separate
    'the constraint is cheap' from 'the operator is trivial'."""
    Z, _ = lorenz_lifted_pairs(n_steps=8000, dt=0.005)
    rows = []
    for stride in (1, 4, 16, 64):
        Zt, Zt1 = Z[:-stride], Z[stride:]
        A_free = fit_ridge(Zt, Zt1, device=device)
        e_free = nrmse(Zt @ A_free.T, Zt1)
        e_st, g, _ = best_stochastic(Zt, Zt1, device=device)
        rows.append((stride, e_free, e_st, e_st / max(e_free, 1e-12)))
        print("  stride=%2d  free=%.5f  stochastic=%.5f  ratio=%.1fx  (gamma=%.3f)"
              % (stride, e_free, e_st, rows[-1][3], g))
    assert rows[-1][2] < 2.0 * rows[0][2], (
        f"stochastic-constraint cost grows as the operator moves away from identity: "
        f"{[('stride %d' % r[0], '%.4f' % r[2]) for r in rows]}"
    )


def test_claim_the_resolvent_solve_is_well_posed_on_the_system_sigmoid_benchmarks(bench):
    """THEORY.md sec 6 puts '(I - gamma P)^-1 b' in the forward path and cites
    sigmoid sec 5's working (I - A)z = b. On Lorenz the unconstrained fit has
    rho -> 1, so I - A is near-singular and the solve sigmoid reports is ill-posed."""
    device = bench["device"]
    Zt, Zt1 = lorenz_lifted_pairs(n_steps=4000, dt=0.005)
    A_free = fit_ridge(Zt, Zt1, device=device)
    cond = np.linalg.cond(np.eye(A_free.shape[0]) - A_free)
    print("  cond(I - A_free) = %.3e   rho(A_free) = %.6f"
          % (cond, spectral_radius(A_free, device=device)))
    assert cond < 1e6, f"cond(I - A) = {cond:.3e}: the resolvent solve is ill-posed"


def test_claim_occupancy_operator_preserves_the_attractor(bench):
    """'it changes what is being modelled from the dynamics to the occupancy of the
    dynamics'. Roll both operators forward and ask whether the attractor survives."""
    device = bench["device"]
    Zt, Zt1 = lorenz_lifted_pairs(n_steps=4000, dt=0.005)
    A_free = fit_ridge(Zt, Zt1, device=device)
    _, g, A_st = best_stochastic(Zt, Zt1, device=device)
    z0 = Zt[0].copy()
    spread = {}
    for name, A in (("free", A_free), ("stochastic", A_st)):
        z, traj = z0.copy(), []
        for _ in range(400):
            z = A @ z
            traj.append(z[1:4])
        spread[name] = float(np.mean(np.std(np.array(traj), axis=0)))
    ref = float(np.mean(np.std(Zt[:, 1:4], axis=0)))
    print("  attractor spread: data=%.4f  free=%.4f  stochastic=%.4f (gamma=%.3f)"
          % (ref, spread["free"], spread["stochastic"], g))
    assert spread["stochastic"] > 0.5 * ref, (
        f"occupancy rollout spread {spread['stochastic']:.4g} vs data {ref:.4g}: "
        f"the attractor collapsed"
    )


# ==========================================================================
# WHY -- the mechanism, GREEN
# ==========================================================================

def test_unconstrained_lorenz_operator_needs_spectral_radius_near_or_above_one(bench):
    """The reason any rho<gamma family fails: the true one-step Lorenz operator is
    expanding. Both the clip AND the stochastic family exclude it, for the same reason."""
    assert bench["rho_free"] > RHO_MAX, (
        f"unconstrained fit rho={bench['rho_free']:.4f}; expected > {RHO_MAX}"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_non_negative_operator_cannot_represent_lorenz_sign_inverting_coupling(device):
    """P >= 0 makes the map monotone: z <= z' implies Pz <= Pz'. Lorenz's x-equation
    is sigma*(y-x), which is anti-monotone in x. No non-negative kernel expresses it."""
    target = np.array([[1 - 0.05, 0.05], [0.05, 1 - 0.05]])
    target[0, 0] = 0.5
    target[0, 1] = -0.5          # the sigma*(y-x) sign structure
    best = np.inf
    for gamma in np.linspace(0.05, 0.999, 60):
        # best non-negative row-sum-gamma approximation of a row with a negative entry
        P = np.clip(target, 0, None)
        rs = P.sum(axis=1, keepdims=True)
        P = np.where(rs > 0, P / np.maximum(rs, 1e-12) * gamma, gamma / 2)
        best = min(best, np.linalg.norm(P - target))
    assert best > 0.3, f"expected an irreducible gap; got {best:.4f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--tb=short", "-p", "no:cacheprovider"]))
