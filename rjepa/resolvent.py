"""Resolvents on the candidate axis: permutation-equivariant, pole-free by construction or refused.

Two operators act on a SET of K candidates (no order, no causal mask):

    signed      A = rho * offdiag(q k^T) / rowL1,   out = (I - A)^-1 v
                ||A||_inf <= rho < 1, so the spectral radius is < 1 and no pole is reachable.
    stochastic  P = softmax(q k^T / sqrt d),        out = (1 - g) P (I - g P)^-1 v
                P is row-stochastic, rho(gP) = g, the pole sits at g = 1.

A sequence (causal) resolvent is nilpotent off the diagonal and needs no bound; on a set the
diagonal-free matrix is not nilpotent, so rho < 1 (g < 1) is load-bearing and anything else is refused.
Low-precision inputs are solved in float32 (there is no fp16/bf16 LU) and cast back. Masked candidates
are removed on both axes and output as 0.
"""
import numpy as np
import torch


def build_A(q, k, rho=0.9, mask=None, eps=1e-3):
    """Signed, diagonal-free, row-L1-normalised coupling. eps floors the row L1: without it
    |dA/dq| ~ 1/|q| grows without bound as the logit scale goes to zero."""
    if not rho < 1.0:
        raise ValueError(f"rho={rho!r}: off the candidate axis A is not nilpotent; rho >= 1 admits a pole")
    ct = torch.promote_types(q.dtype, torch.float32)
    q, k = q.to(ct), k.to(ct)
    K = q.shape[-2]
    w = q @ k.transpose(-1, -2)
    keep = ~torch.eye(K, dtype=torch.bool, device=q.device)
    if mask is not None:
        keep = keep & mask[..., :, None] & mask[..., None, :]
    w = w.masked_fill(~keep, 0.0)
    return rho * w / (w.abs().sum(-1, keepdim=True) + eps).clamp_min(torch.finfo(ct).tiny)


def _masked(v, A, mask):
    v = v.to(A.dtype)
    return v if mask is None else v * mask[..., None].to(A.dtype)


def set_resolvent(q, k, v, rho=0.9, mask=None):
    """(I - A)^-1 v by dense LU. q, k: (..., K, d); v: (..., K, e); mask: (..., K) bool, True = real."""
    A = build_A(q, k, rho, mask)
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    return torch.linalg.solve(eye - A, _masked(v, A, mask)).to(v.dtype)


def neumann_resolvent(q, k, v, rho=0.9, hops=4, mask=None):
    """v + Av + ... + A^hops v; max error <= rho^(hops+1) / (1 - rho) * max|v|."""
    A = build_A(q, k, rho, mask)
    term = z = _masked(v, A, mask)
    for _ in range(hops):
        term = A @ term
        z = z + term
    return z.to(v.dtype)


def stochastic_resolvent(q, k, v, g=0.9, mask=None):
    """O = (1 - g) P (I - g P)^-1 V, P = softmax(q k^T / sqrt d) over the candidate set.

    Dense LU, not a triangular solve: on a non-causal P a triangular solve silently reads only the
    lower triangle. All-masked rows return 0 (softmax of all -inf is NaN and is zeroed)."""
    if not g < 1.0:
        raise ValueError(f"g={g!r}: P is row-stochastic, rho(gP) = g, pole at g = 1")
    ct = torch.promote_types(q.dtype, torch.float32)
    q, k, vv = q.to(ct), k.to(ct), v.to(ct)
    w = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if mask is not None:
        w = w.masked_fill(~mask[..., None, :], float("-inf"))
        vv = vv * mask[..., None].to(ct)
    P = torch.nan_to_num(torch.softmax(w, -1), nan=0.0)
    if mask is not None:
        P = P * mask[..., :, None].to(ct)
    eye = torch.eye(P.shape[-1], dtype=ct, device=P.device)
    return ((1 - g) * P @ torch.linalg.solve(eye - g * P, vv)).to(v.dtype)


def resolvent_apply(W, h, gamma):
    """(I - gamma W)^-1 h for a batch of (K, K) couplings W; the set head's multi-hop message."""
    eye = torch.eye(W.shape[-1], dtype=W.dtype, device=W.device)
    return torch.linalg.solve(eye - gamma * W, h)


def linear_equivariant_resolvent(s, alpha, beta, gamma):
    """(I - gamma (alpha I + beta J))^-1 s for scores s, J the all-ones matrix; None when the spectral
    radius of gamma W is >= 1. The only linear permutation-equivariant W; it never changes the ranks of s."""
    n = len(s)
    W = alpha * np.eye(n) + beta * np.ones((n, n))
    if max(abs(np.linalg.eigvals(gamma * W))) >= 1 - 1e-6:
        return None
    return np.linalg.solve(np.eye(n) - gamma * W, s)
