"""Multi-zoom attention: exact nearby, mean-pooled far, with an accuracy bound.

R5 asks for fine-near / coarse-far reading in one self-similar scheme, at
near-linear cost, with a stated accuracy bound for the coarsening.

The measured reason the obvious construction lost (REQUIREMENTS.md R5): an
H-matrix far-field is LOW-RANK, not sparse, and a block schedule can only keep or
drop. So this does not drop. A distant group of `n` keys is replaced by its mean
`k̄` carrying a `log n` logit, which is the algebraically exact substitution when
the group's keys are identical and a controlled approximation when they are not.

Why the mean and not a selection: sigmoid measured, on this same RTX 4060 Laptop,
that selection-based sparse attention loses because the GATHER dominates --
2.58 ms of `index_select` against 0.82 ms of attention over the same slice at
65536 positions. A mean pool is a contiguous strided reduction. Nothing is
materialised, nothing is gathered, and it is linear, so it has a gradient. The
dyadic schedule that scored 0.39 had neither property: its builder cost 53x the
attention and top-k is an argsort.

THE SCHEME
----------
Pyramid level ℓ holds the mean of every aligned group of `2**ℓ` raw tokens.
For a query block starting at `qs`, the past is tiled backwards, gap-free and
overlap-free, by units of exactly `block` pyramid entries. A unit at level ℓ
covers `block * 2**ℓ` raw tokens. The level used at distance `d` is capped by
`floor(log2(d / (window_blocks * block)))` -- that cap IS the self-similarity:
double the distance, double the pooling factor.

Every unit is chosen to divide its own start offset, so every unit is aligned to
its own group size. Consequences that matter more than the speed:

  * a pooled group NEVER straddles the causal boundary, so causality is
    structural rather than validated;
  * a pooled group is NEVER partial, so the `log n` count is exactly `ℓ*log 2`;
  * a unit is ALWAYS exactly `block` entries, so the kernel's inner loop has one
    uniform shape and no runtime branch on level.

THE ACCURACY BOUND
------------------
`coarsening_bound` returns, for scale `s = 1/sqrt(d)`:

    ||A - Ã||_inf  <=  (D_inf / 2) * (exp(δ_max) - 1)
                       + max_G [ min(2, exp(R_G) - 1) * r_G ]

    R_G   = s * ||q||_2 * diam_2(K_G)        (<= 2 * s * ||q||_2 * ρ_G)
    δ_G   <= R_G^2 / 8                        (Hoeffding's lemma)
    ρ_G   = max_{i in G} ||k_i - k̄_G||_2
    r_G   = max_{i in G} ||v_i - v̄_G||_inf
    D_inf = max_{i,j} ||v_i - v_j||_inf

Derivation. Scores are LINEAR in the keys, so `s·q·k̄_G` is exactly the mean of
the group's scores -- the mean key introduces no error in the mean logit. All the
error is the gap between the true log-sum-exp and `log n + mean(s)`, which Jensen
puts at >= 0 and Hoeffding's lemma caps at `range(s)^2/8`. That deficit `δ_G`
perturbs every softmax coefficient by a factor in `[e^-δmax, e^δmax]`, so the
coefficient vectors differ in L1 by at most `e^δmax - 1`; both are probability
vectors, so re-centering the values on their midpoint pays `D_inf/2` per unit of
L1. The second term is the within-group value error: the truth uses the group's
softmax weights, the summary uses uniform weights, and those differ in L1 by at
most `min(2, e^R_G - 1)`.

Both terms depend only on key/value GEOMETRY -- no true scores -- so the bound is
computable at pool time and cached. It is also honest about when it says nothing:
`exp` of a large `R_G` saturates the `min(2, ...)`, and the bound degrades to the
trivial convex-hull statement exactly when the far field is not tight. That is
the formal version of the objection Pyramid Sparse Attention (arXiv:2512.04025)
states empirically: "Mean logits are effective only when logits are nearly
uniform." Small `R_G` IS near-uniform logits.

No dependency on any other repository. Standalone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "ZoomPlan",
    "plan_for",
    "build_pyramid",
    "multizoom_reference",
    "exact_causal_attention",
    "coarsening_bound",
    "window_sink_reference",
    "drop_reference",
]


def _is_pow2(n: int) -> bool:
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


@dataclass(frozen=True)
class ZoomPlan:
    """Geometry of one multi-zoom read. Validated on construction, on the host."""

    seq: int
    block: int
    window_blocks: int
    sink_blocks: int
    levels: int

    def __post_init__(self) -> None:
        if not isinstance(self.seq, int) or self.seq <= 0:
            raise ValueError(f"seq must be a positive int, got {self.seq!r}")
        if not _is_pow2(self.block):
            raise ValueError(f"block must be a positive power of two, got {self.block!r}")
        for name in ("window_blocks", "sink_blocks", "levels"):
            val = getattr(self, name)
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} must be a non-negative int, got {val!r}")
        if self.window_blocks < 1:
            raise ValueError("window_blocks must be at least 1; the query block itself "
                             "is always read exactly")

    @property
    def num_blocks(self) -> int:
        return (self.seq + self.block - 1) // self.block

    @property
    def max_group_tokens(self) -> int:
        return 1 << self.levels

    def units(self, q_block: int) -> list[tuple[int, int]]:
        """Tile query block `q_block`'s past as `(level, raw_start)` units.

        Each unit is exactly `self.block` pyramid entries at its level, covering
        `self.block << level` raw tokens starting at `raw_start`. Gap-free,
        overlap-free, aligned, and entirely causal.
        """
        b, w = self.block, self.window_blocks
        qs = q_block * b
        units: list[tuple[int, int]] = []

        sink_end = min(self.sink_blocks * b, qs)
        for s in range(0, sink_end, b):
            units.append((0, s))

        exact_start = max(sink_end, qs - (w - 1) * b)
        pos = exact_start
        while pos > sink_end:
            # coarsest level that (a) the self-similar distance law permits,
            # (b) divides `pos` so the group is aligned, and (c) does not cross
            # back over the sinks.
            dist = qs - pos
            cap = 0
            if dist >= w * b:
                cap = int(math.log2(dist // (w * b)))
            cap = min(cap, self.levels)
            lvl = 0
            for cand in range(cap, 0, -1):
                g = b << cand
                if pos % g == 0 and pos - g >= sink_end:
                    lvl = cand
                    break
            g = b << lvl
            units.append((lvl, pos - g))
            pos -= g

        for s in range(exact_start, min(qs + b, self.seq), b):
            units.append((0, s))
        return units


def plan_for(seq, block=64, window_blocks=2, sink_blocks=1, levels=None) -> ZoomPlan:
    """Build a plan, rejecting impossible geometry on the host before any launch.

    `levels=None` picks the smallest ladder that can reach the start of the
    sequence, which is `ceil(log2(seq / (window_blocks * block)))`.
    """
    if not isinstance(seq, int) or seq <= 0:
        raise ValueError(f"seq must be a positive int, got {seq!r}")
    if not _is_pow2(block):
        raise ValueError(f"block must be a positive power of two, got {block!r}")
    if not isinstance(window_blocks, int) or window_blocks < 1:
        raise ValueError(f"window_blocks must be a positive int, got {window_blocks!r}")
    if not isinstance(sink_blocks, int) or sink_blocks < 0:
        raise ValueError(f"sink_blocks must be a non-negative int, got {sink_blocks!r}")
    if levels is None:
        span = max(1.0, seq / (window_blocks * block))
        levels = max(0, math.ceil(math.log2(span)))
    if not isinstance(levels, int) or levels < 0:
        raise ValueError(f"levels must be a non-negative int, got {levels!r}")
    return ZoomPlan(seq=seq, block=block, window_blocks=window_blocks,
                    sink_blocks=sink_blocks, levels=levels)


def build_pyramid(x: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """`pyr[l][..., p, :]` is the mean of raw tokens `[p<<l, (p+1)<<l)`.

    Linear, so it carries a gradient. Contiguous strided reduction, so it costs a
    read of the sequence and nothing is materialised or gathered. The tail is
    zero-padded; a padded group is never referenced, because every unit is
    aligned and lies strictly below its query.
    """
    pyr = [x]
    cur = x
    for _ in range(levels):
        n = cur.shape[-2]
        if n % 2:
            cur = torch.cat([cur, cur.new_zeros(*cur.shape[:-2], 1, cur.shape[-1])], dim=-2)
            n += 1
        cur = cur.reshape(*cur.shape[:-2], n // 2, 2, cur.shape[-1]).mean(dim=-2)
        pyr.append(cur)
    return pyr


def _unit_index(plan: ZoomPlan, q_block: int, device) -> tuple:
    """Flatten one query block's units into gather indices and mask data."""
    units = plan.units(q_block)
    b = plan.block
    lvl, pool_idx, raw_end = [], [], []
    ar = torch.arange(b)
    for level, raw_start in units:
        lvl.append(torch.full((b,), level, dtype=torch.long))
        pool_idx.append((raw_start >> level) + ar)
        raw_end.append(raw_start + (ar + 1) * (1 << level) - 1)
    return (
        torch.cat(lvl).to(device),
        torch.cat(pool_idx).to(device),
        torch.cat(raw_end).to(device),
        units,
    )


def _level_lengths(seq: int, levels: int) -> list[int]:
    lens, n = [], seq
    for _ in range(levels + 1):
        lens.append(n)
        n = (n + 1) // 2
    return lens


def multizoom_reference(q, k, v, plan: ZoomPlan):
    """CPU/GPU reference. The parity oracle for the Triton kernel.

    Deliberately unclever: it materialises each query block's effective key set
    and runs one plain softmax over it. Slow, obviously correct, differentiable.
    """
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v shapes must match, got {tuple(q.shape)}, "
                         f"{tuple(k.shape)}, {tuple(v.shape)}")
    if q.shape[-2] != plan.seq:
        raise ValueError(f"plan.seq={plan.seq} but tensors have seq={q.shape[-2]}")

    dim = q.shape[-1]
    scale = 1.0 / math.sqrt(dim)
    kp = build_pyramid(k, plan.levels)
    vp = build_pyramid(v, plan.levels)
    ln2 = math.log(2.0)

    # one flat pool array per tensor, so a unit is a contiguous row range
    lens = _level_lengths(plan.seq, plan.levels)
    base, kflat, vflat = [], [], []
    off = 0
    for level in range(plan.levels + 1):
        base.append(off)
        kflat.append(kp[level])
        vflat.append(vp[level])
        off += kp[level].shape[-2]
    kflat = torch.cat(kflat, dim=-2)
    vflat = torch.cat(vflat, dim=-2)

    out = torch.zeros_like(q)
    pos = torch.arange(plan.seq, device=q.device)
    for qb in range(plan.num_blocks):
        qs = qb * plan.block
        qe = min(qs + plan.block, plan.seq)
        lvl, pool_idx, raw_end, _ = _unit_index(plan, qb, q.device)
        lvl_l = lvl.tolist()
        # clamp INSIDE each level before adding its base. An unclamped row is an
        # out-of-bounds gather; on CUDA that is a device-side assert that poisons
        # the context for every later test in the process.
        cl = torch.tensor([min(int(p), lens[l] - 1) for p, l in
                           zip(pool_idx.tolist(), lvl_l)], device=q.device)
        rows = cl + torch.tensor([base[l] for l in lvl_l], device=q.device)
        kk = kflat.index_select(-2, rows)
        vv = vflat.index_select(-2, rows)

        qb_t = q[..., qs:qe, :]
        scores = (qb_t @ kk.transpose(-1, -2)) * scale + (lvl * ln2)
        # one uniform causal test: the LAST raw token a summary covers must not
        # be in the future. For level 0 this is the ordinary token mask; for
        # level > 0 it is always true by construction, and checking it anyway
        # makes causality structural.
        ok = (raw_end[None, :] <= pos[qs:qe, None]) & (raw_end[None, :] < plan.seq)
        scores = scores.masked_fill(~ok, float("-inf"))
        out[..., qs:qe, :] = torch.softmax(scores, dim=-1) @ vv
    return out


def exact_causal_attention(q, k, v):
    dim = q.shape[-1]
    seq = q.shape[-2]
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(dim)
    pos = torch.arange(seq, device=q.device)
    logits = logits.masked_fill(pos[None, :] > pos[:, None], float("-inf"))
    return torch.softmax(logits, dim=-1) @ v


def window_sink_reference(q, k, v, block, window_blocks, sink_blocks):
    """Sliding window + attention sinks: the baseline that scored 0.975 and beat
    the dyadic schedule 15/15. Reachable here as `plan_for(..., levels=0)`, but
    written out separately so the comparison cannot be an artefact of my own
    plan builder."""
    seq, dim = q.shape[-2], q.shape[-1]
    pos = torch.arange(seq, device=q.device)
    qb, kb = pos // block, pos // block
    allow = (kb[None, :] <= qb[:, None]) & (
        (kb[None, :] > qb[:, None] - window_blocks) | (kb[None, :] < sink_blocks)
    )
    allow &= pos[None, :] <= pos[:, None]
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(dim)
    return torch.softmax(logits.masked_fill(~allow, float("-inf")), dim=-1) @ v


def drop_reference(q, k, v, plan: ZoomPlan):
    """The same geometry, but the coarse levels DROP instead of summarising.

    This is the control that decides R5. It reads exactly the same near field and
    spends exactly the same number of attended tokens, but every coarse unit is
    replaced by its FIRST `block` raw tokens instead of by pooled means -- a
    schedule that keeps or drops, which is what an H-matrix far-field cannot be
    represented by. If summarising does not beat this, summarising bought nothing.
    """
    dim = q.shape[-1]
    scale = 1.0 / math.sqrt(dim)
    out = torch.zeros_like(q)
    pos = torch.arange(plan.seq, device=q.device)
    for qb in range(plan.num_blocks):
        qs = qb * plan.block
        qe = min(qs + plan.block, plan.seq)
        rows = []
        for level, raw_start in plan.units(qb):
            rows.append(torch.arange(raw_start, raw_start + plan.block,
                                     device=q.device))
        rows = torch.cat(rows).clamp_(max=plan.seq - 1)
        kk = k.index_select(-2, rows)
        vv = v.index_select(-2, rows)
        scores = (q[..., qs:qe, :] @ kk.transpose(-1, -2)) * scale
        ok = rows[None, :] <= pos[qs:qe, None]
        scores = scores.masked_fill(~ok, float("-inf"))
        out[..., qs:qe, :] = torch.softmax(scores, dim=-1) @ vv
    return out


# --------------------------------------------------------------- the bound


def _group_radii(k, v, levels):
    """`(rho[l], r[l])` per pyramid entry: key L2 radius and value Linf radius."""
    rho, rad = [], []
    for level in range(levels + 1):
        g = 1 << level
        n = k.shape[-2]
        pad = (-n) % g
        kk = torch.cat([k, k.new_zeros(*k.shape[:-2], pad, k.shape[-1])], -2) if pad else k
        vv = torch.cat([v, v.new_zeros(*v.shape[:-2], pad, v.shape[-1])], -2) if pad else v
        kk = kk.reshape(*kk.shape[:-2], (n + pad) // g, g, kk.shape[-1])
        vv = vv.reshape(*vv.shape[:-2], (n + pad) // g, g, vv.shape[-1])
        rho.append((kk - kk.mean(-2, keepdim=True)).norm(dim=-1).amax(-1))
        rad.append((vv - vv.mean(-2, keepdim=True)).abs().amax(-1).amax(-1))
    return rho, rad


def coarsening_bound(q, k, v, plan: ZoomPlan) -> float:
    """The stated accuracy bound of the docstring, as a single number.

    Returns `max` over queries of the per-query bound on `||A - Ã||_inf`. Uses
    only key/value geometry and `||q||`; never the true scores. `inf` is a legal
    answer and means the bound is vacuous, which is information.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    rho, rad = _group_radii(k, v, plan.levels)
    d_inf = float((v.amax(-2) - v.amin(-2)).amax())
    qnorm = q.norm(dim=-1)  # [..., seq]

    worst = 0.0
    for qb in range(plan.num_blocks):
        qs = qb * plan.block
        qe = min(qs + plan.block, plan.seq)
        qmax = float(qnorm[..., qs:qe].amax())
        coarse = [(lv, rs) for lv, rs in plan.units(qb) if lv > 0]
        if not coarse:
            continue
        rho_max = 0.0
        val_term = 0.0
        d_max = 0.0
        for level, raw_start in coarse:
            p0 = raw_start >> level
            sl_rho = rho[level][..., p0:p0 + plan.block]
            sl_rad = rad[level][..., p0:p0 + plan.block]
            r_g = scale * qmax * 2.0 * sl_rho              # diam <= 2 * radius
            d_max = max(d_max, float((r_g ** 2 / 8.0).amax()))
            l1 = torch.clamp(torch.expm1(r_g), max=2.0)
            val_term = max(val_term, float((l1 * sl_rad).amax()))
            rho_max = max(rho_max, float(sl_rho.amax()))
        try:
            mass_term = (d_inf / 2.0) * math.expm1(d_max)
        except OverflowError:
            return float("inf")
        worst = max(worst, mass_term + val_term)
    return worst
