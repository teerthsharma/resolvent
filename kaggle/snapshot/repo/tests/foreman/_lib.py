"""Shared bench for the FOREMAN differential on THEORY.md.

Nothing here encodes a conclusion. It encodes the objects THEORY.md talks about:
  - A row-stochastic transition kernel P and the discount gamma  (THEORY.md sec 1)
  - The resolvent (I - gamma P)^-1 b                             (THEORY.md sec 1, sec 6)
  - The unconstrained ridge-fit coupling operator A              (THEORY.md sec 1)
  - The rho-clip that sigmoid's README already measured+rejected (THEORY.md sec 1)
  - Lorenz, the system that clip degraded 0.067 -> 0.317         (THEORY.md sec 1)

The linear algebra below runs through torch (float64) so every test can be
parametrized over device -- see _device.py. Inputs/outputs stay numpy at the
boundary so callers that never pass `device` see no change (device="cpu").
"""

import numpy as np
import torch

from _device import t, n

RNG = np.random.default_rng(20260824)


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------

def random_row_stochastic(n, rng=RNG, concentration=1.0):
    """P >= 0, P @ 1 = 1. The 'transition kernel' of THEORY.md sec 1 constraint 2."""
    P = rng.dirichlet(np.full(n, concentration), size=n)
    return P


def spectral_radius(A, device="cpu"):
    ev = torch.linalg.eigvals(t(A, device))
    return float(torch.max(torch.abs(ev)))


def sigma_max(A, device="cpu"):
    """THEORY.md sec 1 literally defines sigmoid's certificate as `rho = sigma_max(A)`."""
    s = torch.linalg.svdvals(t(A, device))
    return float(s[0])


def resolvent(A, device="cpu"):
    """(I - A)^-1. Same object whether A is called gamma*P or a ridge fit."""
    Ai = t(A, device)
    eye = torch.eye(Ai.shape[0], dtype=torch.float64, device=device)
    return n(torch.linalg.inv(eye - Ai))


def resolvent_solve(A, b, device="cpu"):
    Ai = t(A, device)
    bi = t(b, device)
    eye = torch.eye(Ai.shape[0], dtype=torch.float64, device=device)
    return n(torch.linalg.solve(eye - Ai, bi))


def neumann_iterate(A, b, iters, z0=None, device="cpu"):
    """The 'equilibrium solver' of decision 1: z <- A z + b."""
    Ai = t(A, device)
    bi = t(b, device)
    z = torch.zeros_like(bi) if z0 is None else t(z0, device).clone()
    for _ in range(iters):
        z = Ai @ z + bi
    return n(z)


# --------------------------------------------------------------------------
# Lorenz, lifted so that the one-step map is (almost exactly) linear
# --------------------------------------------------------------------------

def lorenz_traj(n_steps=20000, dt=0.005, burn=2000, s=10.0, r=28.0, b=8.0 / 3.0):
    x = np.array([1.0, 1.0, 1.0])
    out = np.empty((n_steps + burn, 3))
    for i in range(n_steps + burn):
        dx = np.array([s * (x[1] - x[0]),
                       x[0] * (r - x[2]) - x[1],
                       x[0] * x[1] - b * x[2]])
        x = x + dt * dx
        out[i] = x
    return out[burn:]


def quad_lift(X):
    """[1, x, y, z, x^2, xy, xz, y^2, yz, z^2]. Lorenz+Euler is EXACTLY linear here,
    so the unconstrained baseline is not handicapped by the lift."""
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return np.column_stack([np.ones_like(x), x, y, z,
                            x * x, x * y, x * z, y * y, y * z, z * z])


def lorenz_lifted_pairs(**kw):
    traj = lorenz_traj(**kw)
    sc = traj.std(axis=0)
    traj = traj / sc            # standardize so NRMSE is comparable across fits
    Z = quad_lift(traj)
    return Z[:-1], Z[1:]        # (inputs, targets), both n x 10


def nrmse(pred, true, cols=slice(1, 4)):
    """NRMSE on the (x,y,z) block of the lift, matching sigmoid's reported metric."""
    p, t = pred[:, cols], true[:, cols]
    return float(np.sqrt(np.mean((p - t) ** 2)) / np.std(t))


# --------------------------------------------------------------------------
# the three fits THEORY.md sec 1 puts in competition
# --------------------------------------------------------------------------

def fit_ridge(Zt, Zt1, lam=1e-8, device="cpu"):
    """Unconstrained A. THEORY.md sec 1: 'A is an unconstrained ridge-fit matrix'."""
    Zi = t(Zt, device)
    Z1 = t(Zt1, device)
    eye = torch.eye(Zi.shape[1], dtype=torch.float64, device=device)
    W = torch.linalg.solve(Zi.T @ Zi + lam * eye, Zi.T @ Z1)
    return n(W.T)


def clip_sigma(A, rho_max=0.995, device="cpu"):
    """The move sigmoid's README documents and REJECTS: clip the certificate."""
    Ai = t(A, device)
    U, S, Vh = torch.linalg.svd(Ai)
    Sc = torch.clamp(S, max=rho_max)
    return n(U @ torch.diag(Sc) @ Vh)


def project_rows_to_simplex(V, total=1.0, device="cpu"):
    """Euclidean projection of each row onto {p >= 0, sum p = total} (Duchi et al. 2008)."""
    Vi = t(V, device)
    nrow, d = Vi.shape
    U, _ = torch.sort(Vi, dim=1, descending=True)
    css = torch.cumsum(U, dim=1) - total
    idx = torch.arange(1, d + 1, dtype=torch.float64, device=device)
    cond = (U - css / idx) > 0
    rho = d - 1 - torch.argmax(torch.flip(cond, dims=[1]).to(torch.int64), dim=1)
    ar = torch.arange(nrow, device=device)
    theta = css[ar, rho] / (rho.to(torch.float64) + 1.0)
    return n(torch.clamp(Vi - theta[:, None], min=0.0))


def fit_stochastic(Zt, Zt1, gamma, iters=4000, lr=None, device="cpu"):
    """Best A = gamma*P with P row-stochastic non-negative, by projected gradient.
    THEORY.md sec 1 constraints 1+2 applied to the same data as fit_ridge."""
    Zi = t(Zt, device)
    Z1 = t(Zt1, device)
    nrow = Zi.shape[1]
    G = Zi.T @ Zi
    C = Zi.T @ Z1
    if lr is None:
        lr = 1.0 / (float(torch.linalg.svdvals(G)[0]) + 1e-12)
    A = t(project_rows_to_simplex(np.full((nrow, nrow), 1.0 / nrow), total=gamma, device=device), device)
    for _ in range(iters):
        grad = (A @ G - C.T)          # d/dA ||Zt1 - Zt A^T||_F^2 / 2
        A = t(project_rows_to_simplex(n(A - lr * grad), total=gamma, device=device), device)
    return n(A)


def best_stochastic(Zt, Zt1, gammas=(0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999), device="cpu"):
    best = None
    for g in gammas:
        A = fit_stochastic(Zt, Zt1, g, device=device)
        e = nrmse(Zt @ A.T, Zt1)
        if best is None or e < best[0]:
            best = (e, g, A)
    return best   # (nrmse, gamma, A)
