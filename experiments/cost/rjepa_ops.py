"""The resolvent on the candidate axis: signed, permutation-equivariant, pole-free by norm bound.

    A = rho * offdiag(q k^T) / rowL1        (repo ceq_operator, with tril(-1) replaced by "i != j")
    out = (I - A)^-1 v                      exact solve; ||A||_inf <= rho < 1 => rho(A) < 1, no pole

Dropping causality drops nilpotency (the repo's pole-freedom argument), so rho < 1 is now load-bearing
and is refused otherwise. Low-precision inputs are solved in float32 (no fp16/bf16 LU exists) and cast
back. Masked candidates are removed from A on both axes and output as 0.
"""
import torch


def build_A(q, k, rho=0.9, mask=None, eps=1e-3):
    """eps floors the row L1: without it |dA/dq| ~ 1/|q| (measured 1e8x at input scale 1e-8)."""
    if not rho < 1.0:
        raise ValueError(f"rho={rho!r}: off the candidate axis A is not nilpotent; rho >= 1 admits a pole")
    ct = torch.promote_types(q.dtype, torch.float32)
    q, k = q.to(ct), k.to(ct)
    K = q.shape[-2]
    w = q @ k.transpose(-1, -2)  # row-L1 normalization makes any logit scale irrelevant in the forward
    keep = ~torch.eye(K, dtype=torch.bool, device=q.device)
    if mask is not None:
        keep = keep & mask[..., :, None] & mask[..., None, :]
    w = w.masked_fill(~keep, 0.0)
    return rho * w / (w.abs().sum(-1, keepdim=True) + eps).clamp_min(torch.finfo(ct).tiny)


def _io(v, A, mask):
    v = v.to(A.dtype)
    return v if mask is None else v * mask[..., None].to(A.dtype)


def set_resolvent(q, k, v, rho=0.9, mask=None):
    A = build_A(q, k, rho, mask)
    vv = _io(v, A, mask)
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    return torch.linalg.solve(eye - A, vv).to(v.dtype)


def neumann_resolvent(q, k, v, rho=0.9, hops=4, mask=None):
    """Fast path: v + Av + ... + A^hops v; |error| <= rho^(hops+1)/(1-rho) * max|v|."""
    A = build_A(q, k, rho, mask)
    term = z = _io(v, A, mask)
    for _ in range(hops):
        term = A @ term
        z = z + term
    return z.to(v.dtype)


# ---------------------------------------------------------------- Wilson's form (ceqjepa/operator.py:219)
def stochastic_resolvent(q, k, v, g=0.9, mask=None):
    """O = (1-g) P (I - gP)^-1 V with P = softmax(q k^T / sqrt d) over the candidate SET.

    ceqjepa's state_solve uses solve_triangular: on a non-causal P it silently reads only the lower
    triangle (measured 64 % relative error). Here: dense LU, g < 1 refused (rho(P) = 1 exactly, so the
    pole sits at g = 1), masked keys removed before softmax, all-masked rows -> 0, solve in >= float32.
    """
    if not g < 1.0:
        raise ValueError(f"g={g!r}: P is row-stochastic, rho(gP) = g, pole at g = 1")
    ct = torch.promote_types(q.dtype, torch.float32)
    q, k, vv = q.to(ct), k.to(ct), v.to(ct)
    w = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if mask is not None:
        w = w.masked_fill(~mask[..., None, :], float("-inf"))
        vv = vv * mask[..., None].to(ct)
    P = torch.nan_to_num(torch.softmax(w, -1), nan=0.0)  # all-masked row: softmax of all -inf is NaN
    if mask is not None:
        P = P * mask[..., :, None].to(ct)
    eye = torch.eye(P.shape[-1], dtype=ct, device=P.device)
    return ((1 - g) * P @ torch.linalg.solve(eye - g * P, vv)).to(v.dtype)
