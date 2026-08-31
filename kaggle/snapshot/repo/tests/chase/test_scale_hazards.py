"""Failure modes that do not appear at 3.3M and do appear at 300M.

Round 2 recorded a conditioning hazard on the `signed` operator -- row L1 of the
raw logits spanning 1.10e-03 to 1.67e+01 at init, reaching 9.35e-07 over 800
steps, with the Jacobian carrying a 1/l1 -- and shipped `eps` as the rollback,
off by default because `eps = 0.0` is bitwise the shipped operator.

Then the shipped operator CHANGED. Iteration 16 reached parity on `sgate`, a
difference of two softmaxes that divides by nothing data-dependent. Three
questions follow and each one is a test here:

  1. Does the `eps` rollback still apply? It is documented in `ceq/attention.py`
     and `ceq/hf/modeling_ceq.py`, both of which implement ONE operator.
  2. Does the hazard it was built for still exist on `sgate`?
  3. What replaces it at 24 layers and d=1024 -- what does depth and width do to
     the gradient that one global `clip_grad_norm_(1.0)` has to cover?

Every test parametrizes over cpu and cuda, skipping cuda when absent.
"""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ceq import attention, lm
from ceq.hf import modeling_ceq

#: rho=1.5 lam=0.10 hops=2, the iteration-16 parity point.
PARITY = (1.5, 0.10, 2)


@pytest.fixture
def parity_point():
    old = (lm.RHO, lm.SGATE_LAM, lm.HOPS)
    lm.RHO, lm.SGATE_LAM, lm.HOPS = PARITY
    try:
        yield
    finally:
        lm.RHO, lm.SGATE_LAM, lm.HOPS = old


def _sgate(device, *, d=256, heads=4, seq=48, batch=2, seed=0):
    torch.manual_seed(seed)
    a = lm.Attention("sgate", d, heads).to(device)
    x = torch.randn(batch, seq, d, device=device)
    q, k, _ = a.qkv_heads(x)
    return a, q, k


# --------------------------------------------- 1. does the rollback still exist


def test_the_shipped_conditioning_rollback_covers_the_parity_operator(device, parity_point):
    """RED first, and it is the one that decides whether a 300M run can be rolled
    back at all.

    `eps` is a keyword on `ceq.attention.ceq_operator` and on
    `ceq.hf.modeling_ceq.ceq_operator`. Both compute `rho * w / (l1 + eps)` --
    the L1-normalized `signed` operator, and only that one.

    STILL RED AFTER ROUND 4, FOR A NARROWER REASON. `sgate` is now implemented in
    `ceq/hf/modeling_ceq.sgate_operator` and is the shipped default, so the
    original half of this finding -- "the package computes an operator the number
    was not about" -- is FIXED and is bound by
    `test_hub_package_hardening.py::test_the_package_implements_the_operator_that_reached_parity`.
    What remains true is that `eps` does not reach `sgate` and never will: `sgate`
    divides by the constant `1 + lam` and by nothing data-dependent, so there is
    no `1/l1` for `eps` to bound. The companion test below measures that directly
    (min sgate row L1 is a constructive floor of 1.227272 at every d and S).

    A rollback that is not on the code path is not a rollback -- and a rollback
    that has nothing to roll back is not needed. This test records the first
    clause; it is the second that makes it harmless.
    """
    a, q, k = _sgate(device)
    got = a.operator(q, k)
    ref = modeling_ceq.ceq_operator(q, k, rho=lm.RHO, eps=0.0)
    rel = float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
    assert rel < 1e-5, (
        "the HF shipping path computes a DIFFERENT operator from the one that "
        "reached parity: max relative difference {:.3e}. `modeling_ceq."
        "ceq_operator` implements only `rho * w / (l1 + eps)`; `sgate` is "
        "`rho * (softmax(w) - lam * softmax(-w)) / (1 + lam)` and has no "
        "implementation outside `ceq/lm.py`. `eps` therefore rolls back an "
        "operator that a 300M run would not be training.".format(rel))


def test_the_hazard_the_eps_rollback_was_built_for_exists_on_the_parity_operator(
        device, parity_point):
    """The other half. Even if `eps` were ported, is there anything to bound?

    `signed` divides by `||w||_1`, so `|dA/dw| ~ rho / l1` and a small row L1 is
    an unbounded Jacobian. `sgate` divides by the constant `1 + lam`. The row L1
    of `sgate` is two-sidedly bounded by the knobs alone:

        rho (1 - lam) / (1 + lam)  <=  ||A_i||_1  <=  rho

    which at the parity point is [1.227273, 1.5] for every row, every head,
    every layer, every input, at every d and every S.

    This test FAILS when the hazard is absent, and that failure is the finding:
    there is nothing for `eps` to bound, so the rollback is not merely
    unimplemented on this path, it is unnecessary on it.
    """
    lo = lm.RHO * (1 - lm.SGATE_LAM) / (1 + lm.SGATE_LAM)
    worst = math.inf
    for d, heads, seq in ((256, 4, 48), (1024, 16, 48), (256, 4, 512)):
        a, q, k = _sgate(device, d=d, heads=heads, seq=seq)
        l1 = a.operator(q, k).abs().sum(-1)[..., 1:]
        worst = min(worst, float(l1.min()))
    assert worst < 0.5 * lo, (
        "smallest sgate row L1 over d=256/1024 and seq=48/512 is {:.6f}, against "
        "the constructive floor rho(1-lam)/(1+lam) = {:.6f} and ceiling rho = "
        "{:.3f}. `signed` reached 9.35e-07 on the same quantity. There is no "
        "1/l1 to bound.".format(worst, lo, lm.RHO))


# ------------------------------------------ 2. what 24 layers does to the clip


def _per_block_grad_norms(kind, device, *, d, layers, heads, seq=64, batch=2, seed=0):
    torch.manual_seed(seed)
    m = lm.TinyLM(kind, d=d, n_layers=layers, n_heads=heads, seq=seq, seed=seed).to(device)
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randint(0, lm.VOCAB, (batch, seq), generator=g).to(device)
    F.cross_entropy(m(x).reshape(-1, lm.VOCAB), x.reshape(-1)).backward()
    out = [float(torch.sqrt(sum(p.grad.pow(2).sum() for p in b.parameters())))
           for b in m.blocks]
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


@pytest.mark.parametrize("d,layers,heads", [(256, 4, 4), (256, 24, 4), (1024, 4, 16)])
def test_one_global_grad_clip_covers_the_layer_spread_at_300m_depth_and_width(
        device, parity_point, d, layers, heads):
    """`train_one` and `ceq/hf/train.py` both call `clip_grad_norm_(m.parameters(),
    1.0)` -- ONE norm over the whole model. That is safe only while the per-layer
    gradient norms are of one magnitude.

    The three points separate the axes: (256,4,4) is the size parity was measured
    at, (256,24,4) moves depth alone to the 300M value, (1024,4,16) moves width
    and head count alone to the 300M value. The bar is that `sgate`'s
    max/min-over-layers spread stays within 2x the softmax control's at the same
    shape, seed and data.
    """
    got = {}
    for kind in ("softmax", "sgate"):
        n = _per_block_grad_norms(kind, device, d=d, layers=layers, heads=heads)
        got[kind] = (max(n) / max(min(n), 1e-30), n)
    sg, sm = got["sgate"][0], got["softmax"][0]
    assert sg <= 2.0 * sm, (
        "d={} L={} H={}: per-layer grad-norm spread is {:.1f}x for sgate against "
        "{:.1f}x for softmax = {:.2f}x worse. sgate norms {}, softmax norms {}"
        .format(d, layers, heads, sg, sm, sg / sm,
                ["{:.3e}".format(v) for v in got["sgate"][1]],
                ["{:.3e}".format(v) for v in got["softmax"][1]]))


# --------------------------------------------------- 3. the numeric edge cases


@pytest.mark.parametrize("seq", [1, 2, 2048])
def test_the_strictly_causal_first_row_does_not_produce_nan(device, parity_point, seq):
    """tda-tdd contract 4, all-masked row. `sgate` masks with
    `torch.finfo(w.dtype).min` and softmaxes, and row 0 of a `tril(-1)` mask has
    ZERO valid entries. softmax over an all-equal row returns 1/S rather than
    NaN and the trailing `masked_fill(~m, 0.0)` zeroes it -- so the guard is
    real, but it is incidental rather than written down, and seq=1 is the case
    where the whole operator is that row."""
    a, q, k = _sgate(device, seq=seq, batch=1, d=64, heads=2)
    A = a.operator(q, k)
    assert torch.isfinite(A).all(), "non-finite entries at seq={}".format(seq)
    assert float(A[..., 0, :].abs().max()) == 0.0, (
        "row 0 is not exactly zero at seq={}: max |A[0,:]| = {:.3e}".format(
            seq, float(A[..., 0, :].abs().max())))


def test_sgate_is_finite_under_bf16_autocast_at_the_300m_sequence_length(
        device, parity_point):
    """A 300M run trains at seq 2048 under bf16 autocast, and neither has ever
    been combined with this operator. `finfo(bfloat16).min` is -3.39e38 and the
    negative half softmaxes `-w`, so both halves are evaluated at the extreme of
    the mask fill."""
    if device != "cuda":
        pytest.skip("autocast('cuda') needs CUDA")
    a, q, k = _sgate(device, d=64, heads=2, seq=2048, batch=1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        A = a.operator(q.bfloat16(), k.bfloat16())
    assert torch.isfinite(A).all(), "{} non-finite of {}".format(
        int((~torch.isfinite(A)).sum()), A.numel())
    l1 = A.float().abs().sum(-1)[..., 1:]
    lo = lm.RHO * (1 - lm.SGATE_LAM) / (1 + lm.SGATE_LAM)
    assert float(l1.min()) >= lo - 2e-2 and float(l1.max()) <= lm.RHO + 2e-2, (
        "bf16 row L1 spans {:.6f} to {:.6f}, outside [{:.6f}, {:.6f}] + bf16 "
        "slack".format(float(l1.min()), float(l1.max()), lo, lm.RHO))


def test_the_parity_operator_is_bitwise_deterministic(device, parity_point):
    """An A/B benchmark at 300M prices is only as good as the reproducibility of
    each arm.

    `torch.use_deterministic_algorithms(True)` is NOT used here and the reason is
    an instrument fault worth recording: on this torch/CUDA pair it raises
    outright on `F.linear` unless `CUBLAS_WORKSPACE_CONFIG` was exported BEFORE
    the interpreter started, so inside pytest it tests the environment rather
    than the operator. The claim is repeat-run bitwise equality, so that is what
    is asserted -- which still catches the atomics and split-K reductions that
    break it in practice."""
    a, q, k = _sgate(device, seq=128)
    first = a.operator(q, k)
    for _ in range(3):
        assert torch.equal(first, a.operator(q, k)), "sgate is not bitwise stable"
