"""Eviction: forgetting that is structural rather than cosmetic.

THE DISTINCTION

A gate applied after the softmax multiplies a column by s_j ~ 0. The token's
contribution vanishes, but its score never left the softmax DENOMINATOR. Every
surviving token is still divided by a normalizer that includes the crushed
token's mass, so the crushed token goes on shadowing the state, and perturbing it
still moves every row. The mass it was denied is not given to anyone -- it leaks.

Eviction removes the token from the key set before the softmax. The survivors
renormalize over what remains. The token is not in the denominator, not in the
computation, and perturbing it cannot move anything -- not approximately,
bitwise.

That is the difference between "the weight is zero" and "the token was never
there". Only the second one is forgetting.

WHY THIS IS NOT A PRUNING HEURISTIC BOLTED ON

R4 demanded that forgetting be a property of the operator rather than a rule
outside it, and was deleted when its own controls showed the gate forgetting
nothing (a retained filler scored -0.11224 against a crushed -0.11446, a gap of
+0.00222 against a required 0.25). Eviction is a different claim: not that the
operator suppresses what does not matter, but that what does not matter is not
represented at all. The state has no slot for it.

Inspiration, not dependency. `Epsilon-Hollow`'s `foliation` treats memory as a
quotient by an equivalence relation, where a block's refcount IS the cardinality
of a fibre and eviction is the elementary collapse of a free face -- forgetting
as a property of the data structure rather than a policy running on top of it.
That is the idea taken. None of that code is imported and none of it is
load-bearing here.
"""
from __future__ import annotations

import torch

from .nonnormal import causal_mask, salience, scores, settle_exact


def _causal_softmax(s: torch.Tensor) -> torch.Tensor:
    """Softmax over strictly-earlier positions only.

    Row 0 has no predecessors. Masking with -inf would make it NaN, so the mask
    uses the dtype's finite minimum and the row is re-masked to exact zero
    afterwards -- the all-masked-row case tda-tdd names.
    """
    m = causal_mask(s.shape[0], s.device)
    return torch.softmax(s.masked_fill(~m, torch.finfo(s.dtype).min), dim=-1).masked_fill(~m, 0.0)


# ------------------------------------------------------------------ keep rules

def keep_indices(x: torch.Tensor, keep: int, rho: float) -> torch.Tensor:
    """Keep the `keep` highest-salience tokens, in position order."""
    return torch.topk(salience(x, rho), keep).indices.sort().values


def random_indices(x: torch.Tensor, keep: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed + 31337)
    return torch.randperm(x.shape[0], generator=g)[:keep].sort().values.to(x.device)


def recency_indices(x: torch.Tensor, keep: int) -> torch.Tensor:
    """Keep the last `keep` tokens. A keep-rule that cannot beat this is a
    sliding window wearing a different name."""
    return torch.arange(x.shape[0] - keep, x.shape[0], device=x.device)


def lowest_salience_token(x: torch.Tensor, rho: float, exclude: torch.Tensor) -> int:
    """The token the gate crushes hardest among those NOT kept."""
    s = salience(x, rho).clone()
    s[exclude] = float("inf")
    s[:2] = float("inf")           # corpus convention: tokens 0,1 are never targets
    return int(torch.argmin(s))


# ------------------------------------------------------------------- operators

def gated_operator(x: torch.Tensor, keep: torch.Tensor, rho: float) -> torch.Tensor:
    """Full N x N causal operator with non-kept COLUMNS zeroed after the softmax.

    The softmax still ran over every token, so the denominator still contains the
    crushed tokens and the surviving row sums fall short of rho by exactly the
    mass that leaked to the columns now zeroed.
    """
    n = x.shape[0]
    p = _causal_softmax(scores(x))
    drop = torch.ones(n, dtype=torch.bool, device=x.device)
    drop[keep] = False
    return (rho * p).masked_fill(drop[None, :], 0.0)


def evicted_operator(x: torch.Tensor, keep: torch.Tensor, rho: float) -> torch.Tensor:
    """Causal operator on the kept subsequence alone.

    Depends on `x[keep]` and nothing else. That is the whole claim: a token
    outside `keep` cannot reach this matrix by any path, so perturbing it is
    bitwise invisible. Row sums are exactly rho (row 0 excepted, which has no
    predecessors), because the survivors renormalized.
    """
    return rho * _causal_softmax(scores(x[keep]))


# --------------------------------------------------------------- settled state

def settle_gated(x: torch.Tensor, keep: torch.Tensor, rho: float) -> torch.Tensor:
    """Settle on the full sequence, then read the kept rows.

    Shape matches `settle_evicted` so the two are directly comparable; the rows
    dropped here are dropped only for the comparison, not for the computation.
    """
    a = gated_operator(x, keep, rho)
    return settle_exact(a, x)[keep]


def settle_evicted(x: torch.Tensor, keep: torch.Tensor, rho: float) -> torch.Tensor:
    """Settle on the kept subsequence. Nothing outside `keep` is ever read."""
    xk = x[keep]
    return settle_exact(evicted_operator(x, keep, rho), xk)
