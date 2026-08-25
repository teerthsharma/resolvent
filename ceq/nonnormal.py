"""A strictly causal, nilpotent, maximally non-normal coupling operator.

WHY THIS SHAPE

The perturbation ratio -- how much more a causal token moves the settled state
than a filler token does -- reads the COLUMNS of the resolvent (I - A)^-1. Three
operator families were measured on it before this one:

    standard attention (one softmax hop)   1565.111
    row-stochastic gamma P                 1618.510
    Perron  A_ij = s_j P_ij                 396.868

The Perron arm freed the row sums, which is a degree of freedom on the operator's
HORIZON, not on its columns, and it lost by 3.94x. The surviving lead came from
elsewhere: `monodromy` measures Henon a=1.4 b=0.3 by Benettin QR at lambda_1 =
+0.42084 while det = 0.3 contracts. Amplify-while-contract exists, and it needs
NON-NORMALITY -- transient growth -- not a relaxed constraint.

A strictly lower-triangular matrix is the canonical non-normal family. Making the
operator strictly causal (token i reads only tokens j < i, never itself) buys
three things, and two of them are free:

  1. NILPOTENT. A^N = 0, so rho(A) = 0 exactly. Contraction needs no Perron
     certificate, no power iteration, and the measured reducible-A failure mode
     -- power iteration returning w with exact zeros at min(w) = 6.6e-14, leaving
     the weighted sup norm undefined -- cannot arise, because no w is consulted.

  2. EXACT RESOLVENT. (I - A)^-1 = I + A + ... + A^(N-1) terminates. It is not a
     truncated Neumann series whose error has to be bounded; the tail is
     identically zero. `CEQ.Occupancy.occupancy_telescope` proves
     (1 - A) * sum_{k<N} A^k = 1 - A^N; here A^N = 0, which is the hypothesis of
     `occupancy_eq_inverse_of_nilpotent`, so the sum IS the inverse.

  3. PATH STRUCTURE. A dense operator lets every token reach every other in one
     hop, so path length carries no information. Under causality, (A^k)_ij is the
     total weight of length-k paths from j to i, and the resolvent sums them. The
     influence of a token is what it reaches downstream -- which is the question
     R2 asks, stated as a matrix.

WHAT IS NOT CLAIMED HERE. That this beats attention on the perturbation ratio is
the falsifier in tests/w2/, not an assumption of this module.
"""
from __future__ import annotations

import torch

MAD_TO_SIGMA = 1.4826   # median-absolute-deviation -> sigma for a normal
OUTLIER_MADS = 3.0      # conventional robust-outlier cut
CONTROL = 0             # channel carrying "this token is a flag / a bound"

_Q_SEED, _K_SEED = 101, 202


def _proj(d: int, dtype, device, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(d, d, generator=g, dtype=dtype).to(device) / d ** 0.5


def robust_z(v: torch.Tensor) -> torch.Tensor:
    """Standardize by median and 1.4826*MAD, never mean and sigma.

    Measured reason: on distilgpt2 the largest per-dimension sigma is 22.1
    against a median of 0.308 -- a 72x ratio concentrated in about five
    channels. Under mean/sigma those five channels set the whole scale and any
    downstream 'topological' quantity is measuring activation magnitude.
    """
    med = v.median()
    mad = (v - med).abs().median()
    return (v - med) / (MAD_TO_SIGMA * mad + torch.finfo(v.dtype).eps)


def scores(x: torch.Tensor) -> torch.Tensor:
    """Attention logits, with the control channel as an additive bias.

    The bias is present so that a softmax baseline sees exactly the signal this
    operator's gate sees. A baseline denied the module's information is not a
    baseline.
    """
    d = x.shape[-1]
    q = x @ _proj(d, x.dtype, x.device, _Q_SEED)
    k = x @ _proj(d, x.dtype, x.device, _K_SEED)
    return q @ k.transpose(-2, -1) / d ** 0.5 + robust_z(x[:, CONTROL])[None, :]


def salience(x: torch.Tensor, rho: float) -> torch.Tensor:
    """Per-SOURCE gate s_j in [0, rho].

    Gating the source rather than the reader is what lets a token's contribution
    be destroyed rather than redistributed: a row-stochastic reader that
    down-weights one source must hand that mass to another.
    """
    return rho * torch.sigmoid(robust_z(x[:, CONTROL]) - OUTLIER_MADS)


def causal_mask(n: int, device) -> torch.Tensor:
    """Strictly lower triangular: token i reads j < i, never itself.

    Excluding the diagonal is what makes A nilpotent. With the diagonal included
    the operator has a self-loop, rho(A) > 0, and the resolvent stops being a
    finite sum.
    """
    return torch.ones(n, n, dtype=torch.bool, device=device).tril(-1)


def causal_operator(x: torch.Tensor, rho: float = 0.9) -> torch.Tensor:
    """A_ij = s_j * P_ij for j < i, else 0.

    Row 0 has no predecessors. A softmax over an empty set is the all-masked-row
    case: masking with -inf gives NaN, so the mask uses the dtype's finite
    minimum and the result is re-masked to exact zero afterwards. Row 0 ends up
    identically zero, which is correct -- the first token has nothing upstream --
    and finite, which the NaN guard in tests/w2/ checks.
    """
    n = x.shape[0]
    m = causal_mask(n, x.device)
    s = scores(x).masked_fill(~m, torch.finfo(x.dtype).min)
    p = torch.softmax(s, dim=-1).masked_fill(~m, 0.0)
    return p * salience(x, rho)[None, :]


def settle_exact(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """z* = (I - A)^-1 b for nilpotent A, by a FINITE sum.

    Terminates as soon as a term is identically zero, which for a strictly
    triangular A happens within n steps and usually far sooner. No truncation
    error, no tolerance, no dense solve.
    """
    z = b.clone()
    term = b
    for _ in range(a.shape[0]):
        term = a @ term
        if not bool((term != 0).any()):
            break
        z = z + term
    return z


def spectral_radius(a: torch.Tensor) -> float:
    return float(torch.linalg.eigvals(a.to(torch.complex128)).abs().max())


def non_normality(a: torch.Tensor) -> float:
    """Henrici departure from normality, ||A A^T - A^T A||_F / ||A||_F^2.

    Zero for a normal operator. This is the quantity the whole item is about, so
    it is measured rather than asserted.
    """
    aa = a @ a.transpose(-2, -1) - a.transpose(-2, -1) @ a
    return float(aa.norm() / (a.norm() ** 2 + torch.finfo(a.dtype).eps))
