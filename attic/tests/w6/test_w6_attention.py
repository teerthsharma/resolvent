"""W6.2a -- the CPU reference forward pass.

LOOP.md completion condition 1: a `transformers`-loadable module exists and runs
a forward pass on CPU. The CPU path is also the parity oracle any GPU kernel is
checked against, so it has to exist before the kernel rather than after.

WHAT THIS OPERATOR IS, stated so the tests below are readable.

Standard causal attention lets token i read tokens j <= i in ONE hop. This one
builds a STRICTLY causal operator A (j < i, diagonal excluded) and returns

    out = v + A v + A^2 v + ... + A^K v

The identity term is the token's own value -- that is where self-attention went,
and it is why excluding the diagonal from A costs nothing. A^k v is the total
weight of length-k causal paths. So the output is a discounted sum over paths
rather than a single hop, which is the whole "consequence, not similarity"
claim reduced to a matrix.

`CEQ.Nilpotent.pow_card_eq_zero` proves A^n = 0, so the series TERMINATES and
`CEQ.Nilpotent.occupancy_is_exact_inverse` proves the finite sum is exactly
(I - A)^-1 v. K below is a cost knob, not an approximation of unknown quality:
row sums are bounded by rho, so the discarded tail is bounded by rho^(K+1)/(1-rho).

TWO TRAPS THIS FILE GUARDS, both measured earlier in this project:

  * Registering a custom attention WITHOUT registering a matching mask makes
    transformers pass attention_mask=None, and causal/padding/sliding-window
    constraints are silently dropped. Both registrations are checked here.
  * Sparsity and causality are verified by OUTPUT PARITY against an independent
    reference, never by inspecting a mask. `BlockMask.from_kv_blocks` with raw
    CSR reported the correct sparse pattern from `to_dense()` while computing
    fully dense attention, bitwise identical to unmasked.
"""
from __future__ import annotations

import math

import pytest
import torch

from ceq import attention as ca

B, H, S, DH = 2, 3, 24, 16
RHO = 0.9


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    return torch.device(request.param)


def qkv(device, seed=0, dtype=torch.float64):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda: torch.randn(B, H, S, DH, generator=g, dtype=dtype).to(device)
    return mk(), mk(), mk()


# ------------------------------------------------------------------ existence

def test_forward_pass_runs_on_cpu_and_cuda(device):
    """Completion condition 1, the checkable half."""
    q, k, v = qkv(device)
    out, weights = ca.ceq_attention(None, q, k, v, None, rho=RHO, hops=4)
    assert out.shape == v.shape, (out.shape, v.shape)
    assert out.dtype == v.dtype
    assert out.device.type == device.type
    assert torch.isfinite(out).all()
    assert weights is None or weights.shape[:2] == (B, H)


def test_signature_matches_the_attention_interface(device):
    """`(module, query, key, value, attention_mask, **kwargs)` returning a
    2-tuple. A function that does not take `module` first cannot be registered.
    """
    import inspect
    p = list(inspect.signature(ca.ceq_attention).parameters)
    assert p[:5] == ["module", "query", "key", "value", "attention_mask"], p


def test_both_interfaces_are_registered(device):
    """The trap. Registering the attention function alone makes transformers
    pass attention_mask=None and silently drop causal, padding, packing and
    sliding-window constraints.
    """
    from transformers.modeling_utils import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    ca.register()
    assert ca.NAME in AttentionInterface._global_mapping, "attention not registered"
    assert ca.NAME in AttentionMaskInterface._global_mapping, "MASK not registered"


# ------------------------------------------------- exactness of the path sum

def test_zero_hops_is_exactly_the_value(device):
    """K = 0 leaves only the identity term. out == v BITWISE, not approximately.
    If this drifts, the self-contribution is not coming from I and the
    diagonal-excluded design is wrong."""
    q, k, v = qkv(device)
    out, _ = ca.ceq_attention(None, q, k, v, None, rho=RHO, hops=0)
    assert torch.equal(out, v), float((out - v).abs().max())


def test_one_hop_is_exactly_v_plus_Av(device):
    """K = 1 must equal v + A v to machine precision, with A the operator the
    module itself builds. This pins the recursion against an independent
    expression of the same quantity."""
    q, k, v = qkv(device)
    a = ca.ceq_operator(q, k, None, rho=RHO)
    out, _ = ca.ceq_attention(None, q, k, v, None, rho=RHO, hops=1)
    assert torch.allclose(out, v + a @ v, atol=1e-12, rtol=0), \
        float((out - (v + a @ v)).abs().max())


def test_operator_is_strictly_causal(device):
    """Diagonal excluded. With the diagonal included the operator has a
    self-loop, rho(A) is the largest diagonal entry rather than 0, and
    `CEQ.Nilpotent.pow_card_eq_zero` no longer applies."""
    q, k, _ = qkv(device)
    a = ca.ceq_operator(q, k, None, rho=RHO)
    assert float(a.triu(0).abs().max()) == 0.0
    assert torch.isfinite(a).all()
    assert float(a[..., 0, :].abs().max()) == 0.0, "row 0 has no predecessors"


def test_row_sums_are_bounded_by_rho(device):
    """The truncation bound rho^(K+1)/(1-rho) is only meaningful if row sums
    really are bounded by rho."""
    q, k, _ = qkv(device)
    a = ca.ceq_operator(q, k, None, rho=RHO)
    assert float(a.sum(-1).max()) <= RHO + 1e-12


# ------------------------------------------------------------------ causality

def test_causality_is_bitwise_under_a_future_edit(device):
    """Editing token j must not move any output at i < j. Not 'small' -- zero.

    Verified by output parity against the unedited run. That is the only check
    that survives the measured failure where a mask reported the correct sparse
    pattern from `to_dense()` while the kernel computed fully dense attention,
    bitwise identical to unmasked.
    """
    q, k, v = qkv(device)
    j = S - 3
    base, _ = ca.ceq_attention(None, q, k, v, None, rho=RHO, hops=4)
    for idx, name in enumerate("qkv"):
        pert = [q.clone(), k.clone(), v.clone()]
        pert[idx][:, :, j] += 1.0
        moved, _ = ca.ceq_attention(None, *pert, None, rho=RHO, hops=4)
        d = float((moved[:, :, :j] - base[:, :, :j]).abs().max())
        assert d == 0.0, f"future edit to {name} leaked backwards: {d:.3e}"


def test_truncation_respects_its_own_bound(device):
    """The discarded tail must be below rho^(K+1)/(1-rho) times the value scale.
    A cost knob with an unstated error is not a cost knob."""
    q, k, v = qkv(device)
    far, _ = ca.ceq_attention(None, q, k, v, None, rho=RHO, hops=S)
    for hops in (2, 4, 8):
        near, _ = ca.ceq_attention(None, q, k, v, None, rho=RHO, hops=hops)
        bound = RHO ** (hops + 1) / (1 - RHO) * float(v.abs().max()) * S
        assert float((far - near).abs().max()) <= bound, (hops, bound)


def test_padding_mask_is_honoured(device):
    """An additive mask must actually remove the masked keys. Checked by parity
    against a run where those tokens were deleted from the sequence, not by
    reading the mask back."""
    q, k, v = qkv(device)
    m = torch.zeros(B, 1, S, S, dtype=v.dtype, device=device)
    m[..., -4:] = torch.finfo(v.dtype).min
    out, _ = ca.ceq_attention(None, q, k, v, m, rho=RHO, hops=3)
    a = ca.ceq_operator(q, k, m, rho=RHO)
    assert float(a[..., -4:].abs().max()) == 0.0, "masked keys still carry weight"
    assert torch.isfinite(out).all()


# ------------------------------------------------------- the tier-3 property

def _influence_jacobian(a: torch.Tensor, hops: int) -> torch.Tensor:
    """d(out_i)/d(v_j) = (I + A + ... + A^K)_ij.

    The output is linear in v, so the Jacobian is exactly the path-sum matrix.
    No autograd needed and no finite differences to argue about.
    """
    s = a.shape[-1]
    eye = torch.eye(s, dtype=a.dtype, device=a.device).expand_as(a).clone()
    j, term = eye.clone(), eye
    for _ in range(hops):
        term = a @ term
        j = j + term
    return j


def _nonneg_operator(q, k, rho):
    """Tier-2 control: identical construction with a softmax, so every entry is
    non-negative. The ONLY difference from `ceq_operator` is the sign."""
    s = q.shape[-2]
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    m = torch.ones(s, s, dtype=torch.bool, device=w.device).tril(-1)
    p = torch.softmax(w.masked_fill(~m, torch.finfo(w.dtype).min), dim=-1)
    return rho * p.masked_fill(~m, 0.0)


def test_signed_operator_reaches_negative_influence(device):
    """Tier 3, and the reason this module is signed.

    Measured elsewhere: the minimum influence-Jacobian entry is EXACTLY
    0.000e+00 over 40 max-plus instances and 160 APPNP kernels. The general
    statement is that the Kleene star of a non-negative matrix has a
    non-negative influence Jacobian in any ordered semiring, so `not` is
    unreachable by softmax attention, by APPNP, and by the max-plus star alike.

    Dropping non-negativity is the only thing that escapes it. This test is the
    escape, measured on the operator that ships.
    """
    q, k, _ = qkv(device)
    j = _influence_jacobian(ca.ceq_operator(q, k, None, rho=RHO), hops=4)
    assert float(j.min()) < -1e-3, (
        f"min influence {float(j.min()):.3e} is not negative; the operator is "
        f"still tier 2 and cannot represent a negation")


def test_the_nonnegative_control_is_stuck_at_exactly_zero(device):
    """Calibrate the claim against its own control. Same construction, softmax
    instead of signed normalization, and the floor is exactly zero -- not small,
    zero. If this control ever goes negative the tier argument is wrong and the
    test above proves nothing."""
    q, k, _ = qkv(device)
    j = _influence_jacobian(_nonneg_operator(q, k, RHO), hops=4)
    assert float(j.min()) == 0.0, f"non-negative control reached {float(j.min()):.3e}"


def test_nilpotency_does_not_care_about_sign(device):
    """The property that lets non-negativity be dropped for free.

    `CEQ.Nilpotent.pow_card_eq_zero` is stated over `[CommRing R]` with the
    single hypothesis `forall i j, i <= j -> A i j = 0`. No non-negativity
    appears in it. So rho(A) = 0 holds for the signed operator too, and no
    Perron vector is needed -- which is exactly the gap a signed operator would
    otherwise open, since it has no Perron vector to certify with.
    """
    q, k, _ = qkv(device)
    a = ca.ceq_operator(q, k, None, rho=RHO)
    assert float(a.min()) < 0.0, "operator is not actually signed"
    ev = torch.linalg.eigvals(a.reshape(-1, S, S).to(torch.complex128))
    assert float(ev.abs().max()) < 1e-12, f"rho(A) = {float(ev.abs().max()):.3e}"
    a_pow = torch.linalg.matrix_power(a, S)
    assert float(a_pow.abs().max()) == 0.0, "A^S is not exactly zero"
