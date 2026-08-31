"""tda-tdd kernel correctness contracts for the multi-zoom attention kernel.

Subject under test: `ceq.multizoom` (CPU reference, the parity oracle) and
`ceq.mz_kernel` (the Triton kernel). Written BEFORE either module exists.

R5 asks for fine-near / coarse-far reading with a stated accuracy bound. The
measured fact these tests are shaped around is REQUIREMENTS.md R5's hazard note:
an H-matrix far-field is LOW-RANK, not sparse, and a block schedule can only keep
or drop. So the mechanism here does not drop: distant keys are replaced by
MEAN-POOLED summaries carrying a `log(count)` logit so they keep their softmax
mass. The pooling factor doubles per level, which is the self-similarity.

Contracts, in the skill's stated order of bug-caught-per-line:
  1 dense parity   2 mask/plan fidelity   3 causality   4 all-masked-row guard
  5 gradient       6 determinism          7 shape and dtype edges

Every test parametrizes over device. `cuda` skips when absent.
"""

import math

import pytest
import torch

from conftest import DEVICES, requires_triton

from ceq.multizoom import (
    ZoomPlan,
    coarsening_bound,
    exact_causal_attention,
    multizoom_reference,
    plan_for,
)

BLOCK = 32
DIM = 32


def _qkv(seq, dim=DIM, device="cpu", dtype=torch.float32, seed=0, heads=1, batch=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [
        torch.randn(batch, heads, seq, dim, generator=g, dtype=torch.float64)
        .to(device=device, dtype=dtype)
        for _ in range(3)
    ]


# --------------------------------------------------------- contract 1: dense parity


@pytest.mark.parametrize("device", DEVICES)
def test_all_exact_plan_equals_dense_causal_attention(device):
    """A plan whose exact window covers the whole sequence must equal dense SDPA.

    The highest-value smoke test in the file: catches transposed strides, wrong
    scale, off-by-one block indices, and softmax over the wrong axis, before any
    coarsening is involved.
    """
    seq = 4 * BLOCK
    q, k, v = _qkv(seq, device=device)
    plan = plan_for(seq, block=BLOCK, window_blocks=seq // BLOCK, sink_blocks=0, levels=0)
    got = multizoom_reference(q, k, v, plan)
    want = exact_causal_attention(q, k, v)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_zero_level_plan_is_dense_because_the_scheme_never_drops(device):
    """levels=0 must equal DENSE causal attention, not window+sinks.

    This is the load-bearing structural difference from every schedule that scored
    on the random->oracle axis. Window+sinks OMITS the middle of the context.
    This scheme tiles the entire past gap-free and overlap-free; the ladder only
    changes the RESOLUTION at which each part is read, never whether it is read.
    So levels=0 is dense, and the ladder is the only approximation in the module.
    If this ever returns something other than dense attention, a gap has opened in
    the tiling and the coarse level has started dropping after all -- which is the
    failure mode R5 says has already lost.
    """
    seq = 8 * BLOCK
    q, k, v = _qkv(seq, device=device, seed=3)
    plan = plan_for(seq, block=BLOCK, window_blocks=2, sink_blocks=1, levels=0)
    torch.testing.assert_close(
        multizoom_reference(q, k, v, plan),
        exact_causal_attention(q, k, v),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_tiling_is_gap_free_and_overlap_free_at_every_level(device):
    """The tiling invariant, asserted directly on the plan geometry."""
    plan = plan_for(37 * BLOCK, block=BLOCK, window_blocks=2, sink_blocks=1)
    for qb in range(plan.num_blocks):
        covered = []
        for level, raw_start in plan.units(qb):
            covered.append((raw_start, raw_start + (plan.block << level)))
        covered.sort()
        assert covered[0][0] == 0, f"query block {qb} does not tile from token 0"
        for (_, end), (start, _) in zip(covered, covered[1:]):
            assert end == start, f"query block {qb}: gap or overlap at {end}/{start}"
        assert covered[-1][1] >= min((qb + 1) * plan.block, plan.seq)


@requires_triton
def test_triton_kernel_matches_the_cpu_reference():
    """Contract 1 for the kernel proper: parity against the CPU oracle."""
    from ceq.mz_kernel import multizoom_attention

    seq = 16 * BLOCK
    q, k, v = _qkv(seq, device="cuda", seed=7, heads=2, batch=2)
    plan = plan_for(seq, block=BLOCK, window_blocks=2, sink_blocks=1)
    assert plan.levels > 0, "the plan under test must actually coarsen something"

    got = multizoom_attention(q, k, v, plan)
    want = multizoom_reference(q, k, v, plan)
    torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)


# ------------------------------------------------------- contract 2: plan fidelity


@pytest.mark.parametrize("device", DEVICES)
def test_coarsening_actually_changed_the_output(device):
    """Sparsity/coarsening verified by OUTPUT PARITY, never by inspecting a plan.

    REQUIREMENTS.md R5: `BlockMask.from_kv_blocks` with raw CSR reports the sparse
    pattern via to_dense() while computing fully DENSE attention -- measured
    0.000e+00 from unmasked. A plan that claims to coarsen and silently does not
    would pass every structural assertion. So assert on the output: it must differ
    from dense by a real margin.
    """
    seq = 16 * BLOCK
    q, k, v = _qkv(seq, device=device, seed=11)
    plan = plan_for(seq, block=BLOCK, window_blocks=1, sink_blocks=1)
    got = multizoom_reference(q, k, v, plan)
    dense = exact_causal_attention(q, k, v)
    assert (got - dense).abs().max() > 1e-3, (
        "multizoom output is indistinguishable from dense attention -- the plan is "
        "being reported but not applied (the from_kv_blocks failure mode)"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_summary_carries_the_mass_of_the_tokens_it_replaces(device):
    """The log(count) logit is what makes this summarize instead of drop.

    Falsifier for the mechanism itself: build keys where every key in a far group is
    IDENTICAL. Then the mean is exact and the group's summary must reproduce dense
    attention to floating-point, because a group of n identical keys with a
    log(n) logit is algebraically the same softmax.
    """
    seq = 16 * BLOCK
    q, k, v = _qkv(seq, device=device, seed=13)
    plan = plan_for(seq, block=BLOCK, window_blocks=1, sink_blocks=1)

    # collapse every pooled group to a constant key AND value
    g = plan.max_group_tokens
    k = k.reshape(*k.shape[:-2], seq // g, g, DIM)[..., :1, :].expand(
        *k.shape[:-2], seq // g, g, DIM
    ).reshape(*q.shape[:-2], seq, DIM).contiguous()
    v = v.reshape(*v.shape[:-2], seq // g, g, DIM)[..., :1, :].expand(
        *v.shape[:-2], seq // g, g, DIM
    ).reshape(*q.shape[:-2], seq, DIM).contiguous()

    got = multizoom_reference(q, k, v, plan)
    want = exact_causal_attention(q, k, v)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


# ------------------------------------------------------------- contract 3: causality


@pytest.mark.parametrize("device", DEVICES)
def test_causality_by_perturbation_not_by_mask_inspection(device):
    """Perturb position j; every output at i < j must be bitwise unchanged."""
    seq = 8 * BLOCK
    q, k, v = _qkv(seq, device=device, seed=17)
    plan = plan_for(seq, block=BLOCK, window_blocks=1, sink_blocks=1)
    base = multizoom_reference(q, k, v, plan)

    j = 5 * BLOCK + 7
    k2, v2 = k.clone(), v.clone()
    k2[..., j, :] += 100.0
    v2[..., j, :] += 100.0
    pert = multizoom_reference(q, k2, v2, plan)

    assert torch.equal(base[..., :j, :], pert[..., :j, :]), (
        "future token leaked into an earlier output position"
    )
    assert not torch.equal(base[..., j:, :], pert[..., j:, :]), (
        "perturbation had no effect anywhere -- the test is not exercising anything"
    )


@requires_triton
def test_kernel_causality_by_perturbation():
    from ceq.mz_kernel import multizoom_attention

    seq = 8 * BLOCK
    q, k, v = _qkv(seq, device="cuda", seed=19)
    plan = plan_for(seq, block=BLOCK, window_blocks=1, sink_blocks=1)
    base = multizoom_attention(q, k, v, plan)
    j = 5 * BLOCK + 7
    k2, v2 = k.clone(), v.clone()
    k2[..., j, :] += 100.0
    v2[..., j, :] += 100.0
    pert = multizoom_attention(q, k2, v2, plan)
    assert torch.equal(base[..., :j, :], pert[..., :j, :])


# ------------------------------------------------------ contract 4: all-masked row


@pytest.mark.parametrize("device", DEVICES)
def test_no_nan_when_a_query_can_only_see_itself(device):
    """Position 0 sees exactly one key. Nothing may produce NaN anywhere."""
    seq = 4 * BLOCK
    q, k, v = _qkv(seq, device=device, seed=23)
    plan = plan_for(seq, block=BLOCK, window_blocks=1, sink_blocks=0)
    out = multizoom_reference(q, k, v, plan)
    assert torch.isfinite(out).all(), "non-finite value in multizoom output"


@pytest.mark.parametrize("device", DEVICES)
def test_large_logits_do_not_overflow(device):
    """Max-subtraction must survive scores that would overflow exp()."""
    seq = 8 * BLOCK
    q, k, v = _qkv(seq, device=device, seed=29)
    q = q * 200.0
    k = k * 200.0
    plan = plan_for(seq, block=BLOCK, window_blocks=1, sink_blocks=1)
    out = multizoom_reference(q, k, v, plan)
    assert torch.isfinite(out).all(), "overflow: online softmax is not max-subtracting"


# ---------------------------------------------------------- contract 5: gradients


@pytest.mark.parametrize("device", DEVICES)
def test_reference_is_differentiable_through_the_pooling(device):
    """The dyadic schedule 'received no gradient at all -- top-k is an argsort'.

    A mean pool is linear, so this mechanism must have a gradient. If it does not,
    it has the same defect as the schedule it replaces and R5 dies here.
    """
    seq = 8 * BLOCK
    q, k, v = _qkv(seq, device=device, seed=31)
    plan = plan_for(seq, block=BLOCK, window_blocks=1, sink_blocks=1)
    q, k, v = [x.clone().requires_grad_() for x in (q, k, v)]
    multizoom_reference(q, k, v, plan).square().sum().backward()
    for name, x in (("q", q), ("k", k), ("v", v)):
        assert x.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(x.grad).all(), f"non-finite gradient on {name}"
        assert x.grad.abs().sum() > 0, f"zero gradient on {name}"


@pytest.mark.parametrize("device", DEVICES)
def test_gradcheck_against_float64(device):
    seq = 4 * BLOCK
    q, k, v = _qkv(seq, dim=8, device=device, seed=37)
    q, k, v = [x.double().clone().requires_grad_() for x in (q, k, v)]
    plan = plan_for(seq, block=BLOCK, window_blocks=1, sink_blocks=1)
    assert torch.autograd.gradcheck(
        lambda a, b, c: multizoom_reference(a, b, c, plan),
        (q, k, v),
        eps=1e-6,
        atol=1e-6,
    )


# -------------------------------------------------------- contract 6: determinism


@pytest.mark.parametrize("device", DEVICES)
def test_reference_is_bitwise_deterministic(device):
    seq = 8 * BLOCK
    q, k, v = _qkv(seq, device=device, seed=41)
    plan = plan_for(seq, block=BLOCK, window_blocks=1, sink_blocks=1)
    a = multizoom_reference(q, k, v, plan)
    b = multizoom_reference(q, k, v, plan)
    assert torch.equal(a, b)


@requires_triton
def test_kernel_is_bitwise_deterministic():
    from ceq.mz_kernel import multizoom_attention

    seq = 16 * BLOCK
    q, k, v = _qkv(seq, device="cuda", seed=43)
    plan = plan_for(seq, block=BLOCK, window_blocks=1, sink_blocks=1)
    a = multizoom_attention(q, k, v, plan)
    for _ in range(4):
        assert torch.equal(a, multizoom_attention(q, k, v, plan))


# --------------------------------------------------- contract 7: shape/dtype edges


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seq", [1, BLOCK - 1, BLOCK, BLOCK + 1, 2 * BLOCK + 1, 5 * BLOCK + 3])
def test_tile_boundary_sequence_lengths(device, seq):
    """The tail tile is where index arithmetic fails."""
    q, k, v = _qkv(seq, device=device, seed=seq)
    plan = plan_for(seq, block=BLOCK, window_blocks=1, sink_blocks=1)
    out = multizoom_reference(q, k, v, plan)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim", [16, 32, 64, 48])
def test_head_dims_including_non_power_of_two(device, dim):
    seq = 4 * BLOCK
    q, k, v = _qkv(seq, dim=dim, device=device, seed=dim)
    plan = plan_for(seq, block=BLOCK, window_blocks=1, sink_blocks=1)
    out = multizoom_reference(q, k, v, plan)
    assert out.shape == q.shape and torch.isfinite(out).all()


@requires_triton
@pytest.mark.parametrize("seq", [BLOCK, BLOCK + 1, 2 * BLOCK + 1, 9 * BLOCK + 5])
def test_kernel_tile_boundaries_match_reference(seq):
    from ceq.mz_kernel import multizoom_attention

    q, k, v = _qkv(seq, device="cuda", seed=seq)
    plan = plan_for(seq, block=BLOCK, window_blocks=1, sink_blocks=1)
    torch.testing.assert_close(
        multizoom_attention(q, k, v, plan),
        multizoom_reference(q, k, v, plan),
        rtol=2e-3,
        atol=2e-3,
    )


# ------------------------------------- input validation: the IMA that poisons a run


@pytest.mark.parametrize("device", DEVICES)
def test_plan_rejects_impossible_geometry_before_launch(device):
    """An out-of-range offset currently produces a CUDA illegal memory access that
    poisons the process context for every subsequent test. Validation happens on the
    host, raises ValueError, and never reaches a launch."""
    with pytest.raises(ValueError):
        plan_for(128, block=0, window_blocks=1, sink_blocks=1)
    with pytest.raises(ValueError):
        plan_for(128, block=48, window_blocks=1, sink_blocks=1)  # not a power of two
    with pytest.raises(ValueError):
        plan_for(128, block=32, window_blocks=-1, sink_blocks=1)
    with pytest.raises(ValueError):
        plan_for(0, block=32, window_blocks=1, sink_blocks=1)


@requires_triton
def test_kernel_validates_inputs_on_the_host_and_never_launches():
    """No malformed input may reach a kernel launch. Each of these must raise a
    plain ValueError, and the CUDA context must still be usable afterwards."""
    from ceq.mz_kernel import multizoom_attention

    seq = 4 * BLOCK
    q, k, v = _qkv(seq, device="cuda", seed=47)
    plan = plan_for(seq, block=BLOCK, window_blocks=1, sink_blocks=1)

    bad = [
        (q[..., :-1, :], k, v),                       # mismatched seq
        (q, k[..., :8], v),                           # mismatched head dim
        (q.cpu(), k, v),                              # device mismatch
        (q, k, v.cpu()),
        (q.to(torch.int32), k, v),                    # dtype
    ]
    for args in bad:
        with pytest.raises(ValueError):
            multizoom_attention(*args, plan)

    with pytest.raises(ValueError):
        multizoom_attention(q, k, v, ZoomPlan(seq=seq * 2, block=BLOCK, window_blocks=1,
                                              sink_blocks=1, levels=1))

    torch.cuda.synchronize()
    assert torch.isfinite(multizoom_attention(q, k, v, plan)).all(), (
        "CUDA context was poisoned by a rejected input"
    )


# ------------------------------------------------------------- the bound exists


@pytest.mark.parametrize("device", DEVICES)
def test_coarsening_bound_is_returned_and_finite(device):
    seq = 8 * BLOCK
    q, k, v = _qkv(seq, device=device, seed=53)
    plan = plan_for(seq, block=BLOCK, window_blocks=1, sink_blocks=1)
    b = coarsening_bound(q, k, v, plan)
    assert math.isfinite(b) and b >= 0.0
