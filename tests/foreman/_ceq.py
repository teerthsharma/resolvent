"""Standalone kernel for the FOREMAN R1/R2 differential.

Imports nothing from sigmoid / caustic / topoml / any other teerthsharma repo.
torch only, float64, runs on CPU and CUDA.

Two operators, stated once, both as they appear in REQUIREMENTS.md.

  R2 control -- linear resolvent / APPNP (Gasteiger et al., ICLR 2019, Eq. 3,
  pre-softmax):

      Z = alpha (I - (1 - alpha) Ahat)^-1 H

  R2 hypothesis -- discounted max-plus star (Bellman optimality):

      z*_i = max_a [ R[a,i] + gamma * max_j (A[a,i,j] + z*_j) ]

  The max-plus operator is a gamma-contraction in the sup norm: max is
  nonexpansive, + is nonexpansive, and gamma multiplies z. So the iteration
  converges from any start, and the fixed point is unique.

Nothing here asserts which of the two is better. The tests do that.
"""

import torch


# --------------------------------------------------------------------------
# R2 control: the linear resolvent
# --------------------------------------------------------------------------

def appnp(A_hat, H, alpha):
    """alpha (I - (1-alpha) A_hat)^-1 H.  A_hat need not be square-batched."""
    n = A_hat.shape[-1]
    eye = torch.eye(n, dtype=A_hat.dtype, device=A_hat.device)
    return alpha * torch.linalg.solve(eye - (1.0 - alpha) * A_hat, H)


def discounted_resolvent(M, c, gamma):
    """(I - gamma M)^-1 c.  Same object as appnp with alpha = 1 - gamma,
    H = c / alpha; kept separate so the tests can show the identification."""
    n = M.shape[-1]
    eye = torch.eye(n, dtype=M.dtype, device=M.device)
    return torch.linalg.solve(eye - gamma * M, c)


# --------------------------------------------------------------------------
# R2 hypothesis: the discounted max-plus star
# --------------------------------------------------------------------------

def bellman(z, R, A, gamma):
    """One application of T.  R: (n_act, n).  A: (n_act, n, n).  z: (..., n)."""
    inner = (A + z[..., None, None, :]).amax(dim=-1)      # (..., n_act, n)
    return (R + gamma * inner).amax(dim=-2)               # (..., n)


def greedy_policy(z, R, A, gamma):
    """The (action, successor) pair each row selects at z.

    Returns (a_star, j_star), both integer tensors of shape (n,)."""
    inner_vals, inner_arg = (A + z[None, None, :]).max(dim=-1)   # (n_act, n)
    scored = R + gamma * inner_vals                              # (n_act, n)
    a_star = scored.argmax(dim=0)                                # (n,)
    n = z.shape[-1]
    j_star = inner_arg[a_star, torch.arange(n, device=z.device)]
    return a_star, j_star


def policy_affine(a_star, j_star, R, A, gamma):
    """The frozen-policy update z -> gamma * E z + c.

    E is 0/1 with E[i, j*_i] = 1, i.e. a DETERMINISTIC row-stochastic matrix.
    So gamma*E is exactly THEORY.md sec 1's gamma*P, and the fixed point of
    this affine map is exactly APPNP with alpha = 1 - gamma and Ahat = E."""
    n = R.shape[-1]
    idx = torch.arange(n, device=R.device)
    E = torch.zeros((n, n), dtype=R.dtype, device=R.device)
    E[idx, j_star] = 1.0
    c = R[a_star, idx] + gamma * A[a_star, idx, j_star]
    return E, c


def maxplus_star(R, A, gamma, tol=1e-14, max_iter=20000, z0=None, log=False):
    """Value iteration on the discounted max-plus operator.

    Returns (z, info). info carries the per-step sup-norm residual and the
    greedy policy visited at each step -- the two things the DEQ that blew its
    budget at step 47 was not logging."""
    n = R.shape[-1]
    z = torch.zeros(n, dtype=R.dtype, device=R.device) if z0 is None else z0.clone()
    residuals, policies, first_pass = [], [], None
    for k in range(max_iter):
        z_next = bellman(z, R, A, gamma)
        r = float((z_next - z).abs().max())
        residuals.append(r)
        if k == 0:
            first_pass = z_next.clone()
        if log:
            a, j = greedy_policy(z_next, R, A, gamma)
            policies.append((tuple(a.tolist()), tuple(j.tolist())))
        z = z_next
        if r < tol:
            break
    info = dict(
        residuals=residuals,
        iters=len(residuals),
        first_pass=first_pass,
        policies=policies,
        n_distinct_policies=len(dict.fromkeys(policies)) if policies else None,
    )
    return z, info


# --------------------------------------------------------------------------
# collapse probe -- REQUIREMENTS.md R1 says "best affine fit to the learned
# update at the settled state". It does not say at what radius. That omission
# is measured in test_r1_settling.py.
# --------------------------------------------------------------------------

def affine_fit_residual(f, z0, radius, n_samples=None, seed=0):
    """Relative residual of the best affine fit to f on a cube of half-width
    `radius` around z0.  0 means f is exactly affine there."""
    n = z0.shape[-1]
    n_samples = n_samples or (8 * n + 16)
    g = torch.Generator(device="cpu").manual_seed(seed)
    U = (torch.rand((n_samples, n), generator=g, dtype=z0.dtype) * 2 - 1) * radius
    Z = z0[None, :] + U.to(z0.device)
    Y = torch.stack([f(Z[i]) for i in range(n_samples)])
    X = torch.cat([Z, torch.ones((n_samples, 1), dtype=Z.dtype, device=Z.device)], dim=1)
    W = torch.linalg.lstsq(X, Y).solution
    resid = Y - X @ W
    denom = (Y - Y.mean(dim=0, keepdim=True)).norm()
    return float(resid.norm() / max(float(denom), 1e-300))


# --------------------------------------------------------------------------
# random problem instances
# --------------------------------------------------------------------------

def random_instance(n=24, n_act=4, gamma=0.9, seed=0, device="cpu", scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn((n_act, n, n), generator=g, dtype=torch.float64) * scale
    R = torch.randn((n_act, n), generator=g, dtype=torch.float64) * scale
    return R.to(device), A.to(device), gamma


def random_row_stochastic(n, seed=0, device="cpu", conc=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = torch.distributions.Gamma(conc, 1.0).sample((n, n))
    W = torch.rand((n, n), generator=g, dtype=torch.float64).pow(1.0 / conc)
    return (W / W.sum(dim=1, keepdim=True)).to(device)


def policy_iteration(R, A, gamma, max_iter=200):
    """Howard policy iteration: extract greedy policy, solve exactly, repeat.

    The direct method for the same fixed point value iteration approaches.
    Returns (z, n_policy_steps)."""
    n = R.shape[-1]
    z = torch.zeros(n, dtype=R.dtype, device=R.device)
    prev = None
    for k in range(max_iter):
        a, j = greedy_policy(z, R, A, gamma)
        key = (tuple(a.tolist()), tuple(j.tolist()))
        if key == prev:
            return z, k
        prev = key
        E, c = policy_affine(a, j, R, A, gamma)
        z = discounted_resolvent(E, c, gamma)
    return z, max_iter


def affine_update(gamma, P, b):
    """z -> gamma P z + b. The update REQUIREMENTS.md R1 already declares dead
    (closed form, residual 1.05e-15). Kept as the calibration control."""
    def f(z):
        return gamma * (P @ z) + b
    return f
