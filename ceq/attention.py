"""Consequence-equilibrium attention: a signed, strictly causal path sum.

    out = v + A v + A^2 v + ... + A^K v          A strictly lower triangular, SIGNED

WHY SIGNED, AND WHY THAT IS THE WHOLE POINT

Measured, over 40 max-plus instances and 160 APPNP kernels: the minimum entry of
the influence Jacobian is **exactly 0.000e+00**. The general statement both are
special cases of is that the Kleene star of a NON-NEGATIVE matrix has a
non-negative influence Jacobian in any ordered semiring. A token can never
reduce another token's contribution, so `not`, a bound, a negated flag are all
unreachable -- by softmax attention, by APPNP, and by the max-plus star alike.

Three tiers. Every figure below is produced by
`tests/chase/test_module_prose_is_bound.py`, seed 0, S=8 D=16 hops=2, float64,
with BOTH arms fed the identical logits so the operator is the only free
variable. The three figures this header carried for the previous rounds are
WITHDRAWN: they appeared in no run log anywhere, only in a quoted agent sentence
in `house-events-round1.jsonl`. That test asserts they have not come back, so it
does not restate them; DONE.md records what they were.

    tier 1  pairwise similarity    a third token cannot move the RATIO of two
                                   weights in a row: 0.200257204381 ->
                                   0.200257204381, |delta| 1.110e-16, while the
                                   same perturbation moved the row by 5.349e-01
    tier 2  non-negative multi-hop the ratio DOES move, 0.846220 -> 0.425642
                                   (49.70%), but the minimum entry of the
                                   influence Jacobian is still exactly 0:
                                   0 of 200 draws reach a negative entry
    tier 3  SIGNED multi-hop       107 of 200 draws reach one, global minimum
                                   -1.377561e-01

THE RATIO IS NOT THE DISCRIMINATOR; THE SIGN OF THE MINIMUM ENTRY IS. The signed
arm's ratio moves 90.30% on the same draw the non-negative arm moves 49.70%, so
tier 2 and tier 3 are indistinguishable by the ratio test that separates tier 1
from tier 2. The 0/200 control is what gives the 107/200 its meaning.

Moving to the (max,+) semiring stays in tier 2 -- it was measured identical to
the APPNP of its own greedy policy at 9.95e-14, and its gradient identical at
1.65e-08, the finite-difference floor. Dropping non-negativity is what reaches
tier 3. That is the only change here, and it is the one that matters.

WHY DROPPING NON-NEGATIVITY COSTS NOTHING HERE

A signed operator has no Perron vector, so a Perron contraction certificate does
not extend to it. This operator does not need one. Strict causality -- the
diagonal EXCLUDED -- makes A nilpotent, and `CEQ.Nilpotent.pow_card_eq_zero`
proves `A ^ n = 0` over `[CommRing R]` with the single hypothesis
`forall i j, i <= j -> A i j = 0`. There is no non-negativity hypothesis in that
theorem. Nilpotency is sign-blind.

So: rho(A) = 0 structurally, the series terminates, and
`CEQ.Nilpotent.occupancy_is_exact_inverse` proves the finite sum is exactly
(I - A)^-1 v. No certificate, no power iteration, and therefore none of the
measured reducible-A failure mode where power iteration returns w with exact
zeros and the weighted sup norm becomes undefined.

WHERE THE SELF-CONTRIBUTION WENT

The identity term. Excluding the diagonal from A costs nothing because `v` is
already the k=0 term of the series. A^k v is the total weight of length-k causal
paths, so the output is a discounted sum over paths rather than a single hop.

K IS A COST KNOB WITH A STATED BOUND, NOT AN APPROXIMATION

Rows are normalized to L1 <= rho, so ||A||_inf <= rho. WHEN rho < 1 the discarded
tail is bounded by rho^(K+1)/(1-rho), and `tests/w6` checks that bound holds
rather than assuming it. WHEN rho >= 1 -- which includes the `sgate` operator
that reached parity, at rho = 1.5 -- that expression is negative and there is no
geometric bound at all; `truncation_bound` raises instead of returning -6.75.
Cost is K matvecs: K times a single attention pass.
"""
from __future__ import annotations

import math

import torch

NAME = "ceq"
#: The `signed` operator's own defaults for the AttentionInterface path.
#: DELIBERATELY NOT the shipped model's: `ceq/hf/configuration_ceq.py` ships
#: `sgate` at rho=1.5 lam=0.10 hops=2, and this module has no sgate
#: implementation at all. `DEFAULT_HOPS = 4` here also differs from the `hops=3`
#: that every measured `signed` number in `ceq/lm.py` used -- the 4 is this
#: module's interface default and was never the trained one. Stated because the
#: two were confused once already.
DEFAULT_RHO = 0.9
DEFAULT_HOPS = 4


#: Counters a caller can read back; the twin of `ceq.hf.modeling_ceq.STATS`.
#: This project has shipped two constraints that were REPORTED applied and were
#: not -- a BlockMask that `to_dense()` showed sparse while the kernel computed
#: fully dense, and an `is_causal` correct in prefill and wrong at decode -- and
#: neither raised.
#: COST OF THE COUNTER, stated rather than assumed: one reduction over the mask
#: per call. A padding mask is [B,1,1,S] and that is free; a full [B,1,S,S] mask
#: makes it O(S^2), which is 1/H of the operator's own [B,H,S,S] work and the
#: same order as the float-mask validation directly above it.
STATS = {"calls": 0, "masked_key_positions": 0}


def _keep_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Validate a padding mask and return it as a boolean "may attend" tensor.

    Legal: bool with True = attend, or additive float with 0.0 = attend and
    `<= finfo(dtype).min / 2` = masked. ANYTHING ELSE RAISES.

    The historical `-10000.0` transformers mask value is not below
    `finfo(float32).min / 2` (-1.7e38), so it used to be read as "not masked":
    the returned operator was BITWISE EQUAL to the unmasked one while the masked
    columns kept 0.48290979862213135 of weight, and the caller was told the
    constraint had been applied.

    DUPLICATED, DELIBERATELY, into `ceq/hf/modeling_ceq.py`. The Hub copies the
    modeling file and cannot import this package, so the guard has to exist in
    both; `test_an_additive_mask_that_is_not_the_min_convention_is_refused`
    parametrizes over both modules rather than trusting that they agree.
    """
    if attention_mask is None:
        return None
    m = attention_mask
    if m.dtype == torch.bool:
        keep = m
    else:
        floor = torch.finfo(m.dtype).min / 2
        if bool(((m < 0) & (m > floor)).any()):
            raise ValueError(
                "unrecognised attention mask convention: the mask contains "
                "negative values above {:.3e}, which this operator cannot "
                "distinguish from 'not masked'. Legal masks are bool (True = "
                "attend) or additive float with 0.0 for attend and <= {:.3e} "
                "for masked. The historical -10000.0 convention silently "
                "produced a DENSE operator here; min value seen {:.6g}."
                .format(floor, floor, float(m.min())))
        keep = m >= floor
    STATS["masked_key_positions"] += int((~keep).sum())
    return keep


def _strict_causal_mask(s: int, device) -> torch.Tensor:
    """Strictly lower triangular: token i reads j < i, never itself.

    Excluding the diagonal is what makes A nilpotent. Include it and the
    operator has a self-loop, rho(A) becomes the largest diagonal entry, and
    `CEQ.Nilpotent.pow_card_eq_zero` no longer applies.
    """
    return torch.ones(s, s, dtype=torch.bool, device=device).tril(-1)


def ceq_operator(query: torch.Tensor, key: torch.Tensor,
                 attention_mask: torch.Tensor | None = None,
                 *, rho: float = DEFAULT_RHO,
                 scaling: float | None = None,
                 eps: float = 0.0) -> torch.Tensor:
    """The signed, strictly causal coupling operator A.

    Signed by construction: the raw logits are used as weights and normalized by
    their row L1 norm, rather than pushed through a softmax that would force
    every entry non-negative and drop the module to tier 2.

    Row 0 has no predecessors, so its L1 norm is zero; the clamp makes that row
    exactly zero instead of NaN. That is the all-masked-row case, and it is
    handled here rather than left to the caller.

    THE CONDITIONING ROLLBACK, OFF BY DEFAULT. `eps` adds a floor to the
    denominator: `rho * w / (l1 + eps)`. The forward of this operator is exactly
    scale-invariant -- every row is renormalized to L1 = rho -- but the backward
    is not: |dA/dq| scales as 1/l1, measured at exactly 10x per decade over six
    decades on cpu and cuda, reaching 1.337e+13 at input scale 1e-12. Row L1
    already spans 1.10e-03 to 1.67e+01 at initialization, and over 800 steps on
    real text the minimum reached 9.35e-07 while the median rose to 53.07. `eps`
    bounds the Jacobian by rho/eps and leaves the sign and the strict causality
    untouched; at eps=1e-3 it moves the operator by under 1e-2 relative on rows
    with typical mass. `eps = 0.0` is BITWISE the operator that shipped, so
    turning it on is a decision rather than a default that quietly moved.
    """
    STATS["calls"] += 1
    s = query.shape[-2]
    scale = scaling if scaling is not None else 1.0 / math.sqrt(query.shape[-1])
    w = (query @ key.transpose(-2, -1)) * scale

    keep = _keep_mask(attention_mask)
    if keep is not None:
        w = w.masked_fill(~keep, 0.0)

    w = w.masked_fill(~_strict_causal_mask(s, w.device), 0.0)
    l1 = w.abs().sum(-1, keepdim=True) + eps
    return rho * w / l1.clamp_min(torch.finfo(w.dtype).tiny)


def path_sum(a: torch.Tensor, v: torch.Tensor, hops: int) -> torch.Tensor:
    """v + A v + ... + A^hops v.

    A direct method, not a fixed-point search. There is no policy to improve and
    nothing to converge to: for nilpotent A the series is finite and exact. That
    is why the measured result that Howard policy iteration beats value
    iteration 7.9x on CPU and 21.4x on CUDA does not apply -- nothing here is
    iterating toward a fixed point.
    """
    z = v
    term = v
    for _ in range(hops):
        term = a @ term
        z = z + term
    return z


def truncation_bound(rho: float, hops: int) -> float:
    """Relative bound on the discarded tail: `rho^(K+1)/(1-rho)`.

    REFUSES rho >= 1 INSTEAD OF RETURNING A NEGATIVE BOUND. The expression is
    the sum of a geometric series and needs `rho < 1`; at the operator that
    actually reached parity -- `sgate` at rho = 1.5, hops = 2 -- it evaluates to
    **-6.75**, and returning a negative error bound from a function whose whole
    job is to bound an error is a silent wrong answer in a public artifact.

    There is no substitute bound at rho >= 1. The path sum is still EXACT once
    `hops >= S - 1`, because strict causality makes A nilpotent
    (`CEQ.Nilpotent.pow_card_eq_zero`, which has no non-negativity and no
    magnitude hypothesis), but the shipped point uses hops = 2 and is therefore
    a genuine truncation with nothing bounding it. Stated rather than papered
    over. Bound by
    `tests/chase/test_hub_package_hardening.py::test_the_truncation_bound_refuses_a_rho_it_cannot_bound`.
    """
    if not rho < 1.0:
        raise ValueError(
            "no geometric truncation bound exists at rho = {!r}: "
            "rho**(K+1)/(1-rho) = {:.6g}, a NEGATIVE bound. The series converges "
            "only for rho < 1. At rho >= 1 the path sum is exact only through "
            "nilpotency, which needs hops >= S-1; the shipped sgate point is "
            "rho=1.5 hops=2 and has no bound."
            .format(rho, rho ** (hops + 1) / (1.0 - rho) if rho != 1.0 else float("inf")))
    return rho ** (hops + 1) / (1.0 - rho)


def ceq_attention(module, query: torch.Tensor, key: torch.Tensor,
                  value: torch.Tensor, attention_mask: torch.Tensor | None = None,
                  *, rho: float = DEFAULT_RHO, hops: int = DEFAULT_HOPS,
                  scaling: float | None = None, **kwargs):
    """`transformers` AttentionInterface entry point.

    Signature is `(module, query, key, value, attention_mask, **kwargs)` and the
    return is `(attn_output, attn_weights)`, per the documented contract. The
    weights are returned as None: this operator's influence is a path sum over K
    hops, not a single weight matrix, so handing back `A` alone would understate
    it and handing back the full Jacobian would cost more than the forward pass.
    """
    a = ceq_operator(query, key, attention_mask, rho=rho, scaling=scaling)
    return path_sum(a, value, hops), None


def ceq_mask(*args, **kwargs):
    """Mask builder registered alongside the attention function.

    Registering the attention function WITHOUT this makes transformers pass
    `attention_mask=None`, and causal, padding, packing and sliding-window
    constraints are then silently dropped. The causal constraint here is
    structural -- `ceq_operator` applies `tril(-1)` unconditionally -- so this
    forwards the standard sdpa mask for padding and nothing more.
    """
    from transformers.masking_utils import sdpa_mask
    return sdpa_mask(*args, **kwargs)


def register() -> None:
    """Register into BOTH interfaces. Idempotent."""
    from transformers.modeling_utils import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    AttentionInterface.register(NAME, ceq_attention)
    AttentionMaskInterface.register(NAME, ceq_mask)
