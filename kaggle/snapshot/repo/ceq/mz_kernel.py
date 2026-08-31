"""Triton kernel for multi-zoom attention. The mechanism is IN the kernel.

Not a schedule builder feeding somebody else's kernel: the pooled far field, the
`log n` mass correction, and the causal test on a summary's span are all
evaluated inside the fused online-softmax loop, in the same pass as the exact
near field. There is one accumulator and one softmax normalisation across both
resolutions -- which is the only way to do it, because the coarse and fine
contributions share a denominator.

Two silent failure modes of the merged `triton-lang/kernels#22` kernel are made
STRUCTURALLY impossible here rather than merely validated, because both were
measured to escape validation:

  * an empty schedule row leaves `l_i` at zero and the row returns `acc/0` = NaN
    with no error anywhere. Here the final divide is guarded.
  * an out-of-range block index loads out of bounds, because the only mask on the
    key tile is `k_pos < seq`, which a NEGATIVE index passes. That is the illegal
    memory access that poisons the CUDA context for every subsequent test in the
    process. Here every row index is clamped into the pool inside the kernel, and
    the host validates before launch as well. Two independent guards, because one
    of them failing is exactly how this bug got shipped.

Layout. All pyramid levels are concatenated into one `[BH, P, D]` array, so a
unit is a contiguous row range and the inner loop has no branch on level. Every
unit is exactly `BLOCK_N` entries by construction (see `ZoomPlan.units`), so
there is one tile shape.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .multizoom import ZoomPlan, build_pyramid

__all__ = ["multizoom_attention", "plan_tensors", "attention_flops"]

_POW2_HEAD_DIMS = (16, 32, 64, 128)
_CACHE: dict = {}


@triton.jit
def _mz_kernel(
    q_ptr, kp_ptr, vp_ptr, out_ptr,
    off_ptr, row_ptr, fac_ptr, rs_ptr, lc_ptr,
    seq, n_pool,
    sq_b, sq_m, sq_d,
    sk_b, sk_n, sk_d,
    so_b, so_m, so_d,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    qb = tl.program_id(0)
    bh = tl.program_id(1)

    offs_m = qb * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        q_ptr + bh * sq_b + offs_m[:, None] * sq_m + offs_d[None, :] * sq_d,
        mask=offs_m[:, None] < seq, other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    u = tl.load(off_ptr + qb)
    u_end = tl.load(off_ptr + qb + 1)
    while u < u_end:
        row0 = tl.load(row_ptr + u)
        fac = tl.load(fac_ptr + u)
        rs = tl.load(rs_ptr + u)
        lc = tl.load(lc_ptr + u)

        rows = row0 + offs_n
        # structural out-of-bounds guard: a clamped row is always a legal load,
        # and the causal/bounds mask below discards it anyway.
        rows = tl.maximum(tl.minimum(rows, n_pool - 1), 0)

        k_tile = tl.load(
            kp_ptr + bh * sk_b + rows[:, None] * sk_n + offs_d[None, :] * sk_d
        )
        scores = tl.dot(q, tl.trans(k_tile)) * scale + lc

        # ONE uniform causal test for both resolutions: the last raw token this
        # entry summarises must not be in the future. At level 0 (fac == 1) that
        # is the ordinary token mask; above it, it is the exact statement that the
        # whole pooled group is in the past. Causality is therefore a property of
        # the arithmetic, not of a mask someone remembered to pass.
        raw_end = rs + (offs_n + 1) * fac - 1
        ok = (
            (raw_end[None, :] <= offs_m[:, None])
            & (raw_end[None, :] < seq)
            & (offs_m[:, None] < seq)
        )
        scores = tl.where(ok, scores, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, 1))
        m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        p = tl.exp(scores - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)

        v_tile = tl.load(
            vp_ptr + bh * sk_b + rows[:, None] * sk_n + offs_d[None, :] * sk_d
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_ij
        u += 1

    # guarded divide: an empty unit list must not return NaN silently.
    # `[:, None]` is load-bearing -- without it the [BLOCK_M] normaliser
    # broadcasts along the LAST axis and divides column d by row d's
    # normaliser. That produces finite, plausible, entirely wrong output that no
    # NaN guard and no smoke test catches, only parity against a reference.
    out = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]
    tl.store(
        out_ptr + bh * so_b + offs_m[:, None] * so_m + offs_d[None, :] * so_d,
        out.to(out_ptr.dtype.element_ty),
        mask=offs_m[:, None] < seq,
    )


def plan_tensors(plan: ZoomPlan, device):
    """Flatten the plan to CSR once and cache it. Pure host arithmetic."""
    key = (plan, str(device))
    if key in _CACHE:
        return _CACHE[key]

    lens, n = [], plan.seq
    for _ in range(plan.levels + 1):
        lens.append(n)
        n = (n + 1) // 2
    base, off = [], 0
    for length in lens:
        base.append(off)
        off += length
    n_pool = off

    offsets, rows, facs, starts, lcs = [0], [], [], [], []
    ln2 = math.log(2.0)
    for qb in range(plan.num_blocks):
        for level, raw_start in plan.units(qb):
            row0 = base[level] + (raw_start >> level)
            if not (0 <= row0 <= n_pool - 1):
                raise ValueError(
                    f"plan produced pool row {row0} outside [0, {n_pool}) for query "
                    f"block {qb}, level {level}, raw_start {raw_start}"
                )
            rows.append(row0)
            facs.append(1 << level)
            starts.append(raw_start)
            lcs.append(level * ln2)
        offsets.append(len(rows))

    t = lambda a, d: torch.tensor(a, dtype=d, device=device)  # noqa: E731
    out = (
        t(offsets, torch.int32), t(rows, torch.int32), t(facs, torch.int32),
        t(starts, torch.int32), t(lcs, torch.float32), n_pool, len(rows),
    )
    _CACHE[key] = out
    return out


def attention_flops(plan: ZoomPlan, head_dim: int, batch_heads: int) -> int:
    """Useful FLOPs actually executed: 2 * (QK^T) + 2 * (PV) per tile.

    Counted over the tiles the kernel really visits, which is the only honest
    numerator for a utilisation figure on a kernel that does less work than dense.
    Per unit: QK^T is BLOCK_M x BLOCK_N x HEAD_DIM MACs and PV is the same again,
    at 2 FLOP per MAC.
    """
    units = sum(len(plan.units(qb)) for qb in range(plan.num_blocks))
    return 4 * units * plan.block * plan.block * head_dim * batch_heads


def multizoom_attention(q, k, v, plan: ZoomPlan, num_warps: int = 4,
                        num_stages: int = 2):
    """Fused multi-zoom causal attention. Validates on the host before any launch."""
    for name, x in (("q", q), ("k", k), ("v", v)):
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor, got {type(x)!r}")
    if not isinstance(plan, ZoomPlan):
        raise ValueError(f"plan must be a ZoomPlan, got {type(plan)!r}")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"q/k/v shapes must match, got {tuple(q.shape)}, {tuple(k.shape)}, "
            f"{tuple(v.shape)}"
        )
    if q.ndim != 4:
        raise ValueError(f"q/k/v must be [batch, heads, seq, dim], got {q.ndim} dims")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"q/k/v dtypes must match, got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"dtype {q.dtype} unsupported; tl.dot has no float64 path and integer "
            f"inputs are rejected. Use float16, bfloat16, or float32."
        )
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q, k and v must all be CUDA tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(
            f"q/k/v must be on one device, got {q.device}, {k.device}, {v.device}"
        )
    seq, dim = q.shape[-2], q.shape[-1]
    if seq != plan.seq:
        raise ValueError(f"plan.seq={plan.seq} but tensors have seq={seq}")
    if dim not in _POW2_HEAD_DIMS:
        raise ValueError(f"head dim must be one of {_POW2_HEAD_DIMS}, got {dim}")
    if plan.block not in _POW2_HEAD_DIMS and plan.block < 16:
        raise ValueError(f"block must be >= 16, got {plan.block}")

    offsets, rows, facs, starts, lcs, n_pool, n_units = plan_tensors(plan, q.device)

    kp = torch.cat(build_pyramid(k, plan.levels), dim=-2).contiguous()
    vp = torch.cat(build_pyramid(v, plan.levels), dim=-2).contiguous()
    if kp.shape[-2] != n_pool:
        raise ValueError(
            f"pyramid has {kp.shape[-2]} rows but the plan indexes {n_pool}"
        )

    q = q.contiguous()
    qf = q.reshape(-1, seq, dim)
    kf = kp.reshape(-1, n_pool, dim)
    vf = vp.reshape(-1, n_pool, dim)
    out = torch.empty_like(qf)

    _mz_kernel[(plan.num_blocks, qf.shape[0])](
        qf, kf, vf, out,
        offsets, rows, facs, starts, lcs,
        seq, n_pool,
        qf.stride(0), qf.stride(1), qf.stride(2),
        kf.stride(0), kf.stride(1), kf.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        1.0 / math.sqrt(dim),
        BLOCK_M=plan.block, BLOCK_N=plan.block, HEAD_DIM=dim,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out.reshape(q.shape)
