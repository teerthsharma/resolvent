"""Minimal faithful transcription of THEORY.md, so that tests have something to be RED against.

This file is NOT a proposal. It is the smallest honest encoding of what THEORY.md
specifies, section by section, so failure modes can be bound to runnable evidence.

  §1  stochastic P, gamma explicit, resolvent (I - gamma P)^-1 b
  §3  CSR block schedule + scheduled attention (kernels#22 semantics)
  §5  caustic Theorem 1 floor err >= n - m, select_by_floor
  §8.1 nested fixed point: DEQ on top of the resolvent

Every function here is deliberately dumb and dense. Correctness of the reference is
the point; speed is not. Where THEORY.md leaves a choice open, both options are
implemented and the test says which one it is attacking.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- §1 stochastic P

def rows_softmax(logits: torch.Tensor) -> torch.Tensor:
    """P row-stochastic by construction. The differentiable option."""
    return torch.softmax(logits, dim=-1)


def project_rows_to_simplex(A: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of each row onto the probability simplex.

    Wang & Carreira-Perpinan 2013, the standard O(d log d) sort algorithm.
    The projection option: keeps the ridge-fit A and projects it back every step.
    """
    d = A.shape[-1]
    u, _ = torch.sort(A, dim=-1, descending=True)
    css = torch.cumsum(u, dim=-1)
    k = torch.arange(1, d + 1, device=A.device, dtype=A.dtype)
    cond = u - (css - 1.0) / k > 0
    rho = cond.to(A.dtype).cumsum(dim=-1).argmax(dim=-1)  # last True index
    theta = (css.gather(-1, rho.unsqueeze(-1)) - 1.0) / (rho + 1).unsqueeze(-1).to(A.dtype)
    return torch.clamp(A - theta, min=0.0)


def resolvent(P: torch.Tensor, gamma: float, b: torch.Tensor) -> torch.Tensor:
    """z = (I - gamma P)^-1 b. THEORY.md §1, sigmoid §5's solve with the scale factored out."""
    d = P.shape[-1]
    I = torch.eye(d, device=P.device, dtype=P.dtype)
    return torch.linalg.solve(I - gamma * P, b)


# ------------------------------------------------- §3 CSR block schedule + attention

def build_schedule(n_q_blocks, n_kv_blocks, local_window=1, n_sink=1, salience=None,
                   top_k=0, device="cpu"):
    """Causal CSR block schedule: sink blocks + local window + top-k salient blocks.

    Returns (indptr, indices) as int32, the format kernels#22 is described as consuming.
    salience: optional (n_q_blocks, n_kv_blocks) score used for the top_k far-field picks.
    """
    rows = []
    indptr = [0]
    for i in range(n_q_blocks):
        sel = set()
        for s in range(min(n_sink, n_kv_blocks)):
            if s <= i:
                sel.add(s)
        for w in range(max(0, i - local_window + 1), i + 1):
            sel.add(w)
        if top_k > 0 and salience is not None:
            cand = [j for j in range(i + 1) if j not in sel]
            cand.sort(key=lambda j: -float(salience[i, j]))
            for j in cand[:top_k]:
                sel.add(j)
        s = sorted(sel)
        rows.extend(s)
        indptr.append(len(rows))
    return (torch.tensor(indptr, dtype=torch.int32, device=device),
            torch.tensor(rows, dtype=torch.int32, device=device))


def schedule_to_mask(indptr, indices, n_q_blocks, n_kv_blocks, BLOCK, S, causal=True,
                     device="cpu"):
    """Expand a CSR block schedule to a boolean (S, S) allow-mask."""
    m = torch.zeros(S, S, dtype=torch.bool, device=device)
    for i in range(n_q_blocks):
        q0, q1 = i * BLOCK, min((i + 1) * BLOCK, S)
        for p in range(int(indptr[i]), int(indptr[i + 1])):
            j = int(indices[p])
            k0, k1 = j * BLOCK, min((j + 1) * BLOCK, S)
            m[q0:q1, k0:k1] = True
    if causal:
        m &= torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
    return m


def masked_attention(q, k, v, mask, nan_guard=False):
    """Reference scheduled attention. (B,H,S,D) x bool (S,S) -> (B,H,S,D).

    nan_guard=False is the literal kernels#22-style behaviour: softmax over an
    all -inf row. THEORY.md does not specify a guard.
    """
    scale = q.shape[-1] ** -0.5
    scores = (q @ k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    if nan_guard:
        p = torch.nan_to_num(p, nan=0.0)
    return p @ v


class ScheduledAttention(torch.autograd.Function):
    """Explicit fwd/bwd so the schedule used in each can be controlled independently.

    THEORY.md §3: "A schedule can be rebuilt per input." §4: stratum gates the rebuild.
    Neither says the backward pass must reuse the forward's schedule.
    """

    @staticmethod
    def forward(ctx, q, k, v, mask_fwd, mask_bwd):
        with torch.enable_grad():
            qd, kd, vd = q.detach().requires_grad_(), k.detach().requires_grad_(), v.detach().requires_grad_()
            out = masked_attention(qd, kd, vd, mask_fwd)
        ctx.save_for_backward(q, k, v)
        ctx.mask_bwd = mask_bwd
        return out.detach()

    @staticmethod
    def backward(ctx, go):
        q, k, v = ctx.saved_tensors
        with torch.enable_grad():
            qd, kd, vd = q.detach().requires_grad_(), k.detach().requires_grad_(), v.detach().requires_grad_()
            out = masked_attention(qd, kd, vd, ctx.mask_bwd)
            gq, gk, gv = torch.autograd.grad(out, (qd, kd, vd), go)
        return gq, gk, gv, None, None


# ---------------------------------------------------------- §5 caustic Theorem 1

def orbit_floor(predicted_values):
    """caustic Theorem 1: err >= n - m. n entities, m distinct produced values.

    Consults no answer key. Integer valued.
    """
    n = len(predicted_values)
    m = len(set(predicted_values))
    return n - m


def verify_injective(ground_values):
    """caustic's guard: the ground relation must map distinct entities to distinct values."""
    return len(set(ground_values)) == len(ground_values)


def select_by_floor(candidates):
    """branchcut select_by_floor: candidates compete, scored by the certified floor.

    candidates: list of (name, produced_values). Lower floor wins. Ties -> first entered,
    so the do-nothing candidate must be entered first to make declining first-class.
    """
    scored = [(orbit_floor(vals), name) for name, vals in candidates]
    best = min(range(len(scored)), key=lambda i: scored[i][0])
    return candidates[best][0], scored


# -------------------------------------------------- §8.1 nested fixed point / DEQ

def deq_forward(f, z0, max_iter=50, tol=1e-6):
    """Plain fixed-point iteration. Returns (z, n_iter, converged, residual)."""
    z = z0
    res = float("inf")
    for i in range(max_iter):
        zn = f(z)
        res = float((zn - z).norm())
        z = zn
        if res < tol:
            return z, i + 1, True, res
        if not torch.isfinite(z).all():
            return z, i + 1, False, res
    return z, max_iter, False, res


def jacobian_spectral_radius(f, z):
    """rho(df/dz) at z, dense. Small dims only."""
    J = torch.autograd.functional.jacobian(f, z, vectorize=True)
    d = z.numel()
    J = J.reshape(d, d)
    return float(torch.linalg.eigvals(J).abs().max()), J


def implicit_grad_matrix(J):
    """(I - J)^-1, the implicit-differentiation factor. Returns (M, cond)."""
    d = J.shape[0]
    A = torch.eye(d, dtype=J.dtype, device=J.device) - J
    M = torch.linalg.inv(A)
    return M, float(torch.linalg.cond(A))
