"""Is there a cheaper structure than a bespoke H-matrix schedule builder?

THEORY.md section 3 makes `triton-lang/kernels#22` the substrate because its
attention pattern is a causal CSR block schedule -- data, not code. Risk 6 says
that kernel is forward-only, so training through it needs a backward pass that
does not exist.

torch 2.5.1 ships `torch.nn.attention.flex_attention`. Its `BlockMask` stores
`kv_num_blocks` (blocks per query-block row) and `kv_indices` (their column
indices) -- the same row-count + column-index pair a CSR schedule is. It has a
backward pass. These tests check whether it is literally the same schedule
format, and cover the correctness contracts a topology-derived schedule must
satisfy before any speedup number means anything.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

BLK = 32
SEQ = 128
NBLK = SEQ // BLK
HEAD = 16
torch.manual_seed(0)


# ---------------------------------------------------------------- helpers

def dtype_and_atol(device):
    """flex_attention does not support float64 on CUDA; scale tolerance with dtype."""
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    atol = 1e-9 if dtype is torch.float64 else 1e-4
    return dtype, atol


def qkv(dtype=torch.float64, requires_grad=False, seq=SEQ, device=None):
    g = torch.Generator().manual_seed(11)
    out = []
    for _ in range(3):
        t = torch.randn(1, 1, seq, HEAD, dtype=dtype, generator=g).to(device)
        t.requires_grad_(requires_grad)
        out.append(t)
    return out


def csr_block_table(counts, indices, device=None):
    n = len(counts)
    t = torch.zeros(n, n, dtype=torch.bool)
    for qb, c in enumerate(counts):
        for kb in indices[qb][:c]:
            t[qb, kb] = True
    return t.to(device)


def csr_to_blockmask_naive(counts, indices, blk=BLK, device=None):
    """The obvious port: hand the CSR arrays straight to BlockMask."""
    kv_num_blocks = torch.tensor(counts, dtype=torch.int32).view(1, 1, -1).to(device)
    kv_indices = torch.tensor(indices, dtype=torch.int32).view(1, 1, len(counts), -1).to(device)
    return BlockMask.from_kv_blocks(kv_num_blocks, kv_indices, BLOCK_SIZE=(blk, blk))


def csr_to_blockmask(counts, indices, seq=SEQ, blk=BLK, device=None):
    """A CSR block schedule -> a flex_attention BlockMask, the way that works.

    The schedule has to be carried as a `mask_mod` predicate. See
    test_raw_csr_arrays_alone_do_not_restrict_attention for why.
    """
    table = csr_block_table(counts, indices, device=device)

    def mod(b, h, q_idx, kv_idx):
        return table[q_idx // blk, kv_idx // blk]

    return create_block_mask(mod, 1, 1, seq, seq, device=device, BLOCK_SIZE=blk)


def blockmask_to_csr(bm):
    counts = bm.kv_num_blocks.view(-1).tolist()
    idx = bm.kv_indices.view(len(counts), -1)
    return counts, [idx[i, : counts[i]].tolist() for i in range(len(counts))]


def schedule_to_dense_mask(counts, indices, seq=SEQ, blk=BLK, device=None):
    m = torch.zeros(seq, seq, dtype=torch.bool).to(device)
    for qb, n in enumerate(counts):
        for kb in indices[qb][:n]:
            m[qb * blk:(qb + 1) * blk, kb * blk:(kb + 1) * blk] = True
    return m


def dense_masked_attention(q, k, v, mask):
    scores = (q @ k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def causal_dense_schedule(nblk=NBLK):
    counts = [i + 1 for i in range(nblk)]
    indices = [list(range(i + 1)) + [0] * (nblk - i - 1) for i in range(nblk)]
    return counts, indices


# ---------------------------------------------------------------- risk 6

def _forward_only_scheduled_attention(q, k, v, mask):
    """Faithful stand-in for a forward-only Triton kernel.

    A Triton kernel operates on raw pointers, so autograd records nothing and
    the output carries no grad_fn -- exactly what a no_grad region produces.
    """
    with torch.no_grad():
        return dense_masked_attention(q, k, v, mask)


def test_forward_only_scheduled_attention_can_be_trained_through(device):
    """Risk 6, made concrete: the merged kernel cannot be trained through."""
    q, k, v = qkv(requires_grad=True, device=device)
    counts, indices = causal_dense_schedule()
    out = _forward_only_scheduled_attention(
        q, k, v, schedule_to_dense_mask(counts, indices, device=device))
    assert out.requires_grad, "forward-only kernel output has no grad_fn"
    out.sum().backward()
    assert q.grad is not None


def test_flex_attention_backward_matches_dense_on_the_same_schedule(device):
    """The same schedule, run through flex_attention, is differentiable."""
    dtype, atol = dtype_and_atol(device)
    counts, indices = causal_dense_schedule()
    mask = schedule_to_dense_mask(counts, indices, device=device)
    bm = csr_to_blockmask(counts, indices, device=device)

    qa, ka, va = qkv(dtype=dtype, requires_grad=True, device=device)
    flex_attention(qa, ka, va, block_mask=bm).sum().backward()
    qb, kb, vb = qkv(dtype=dtype, requires_grad=True, device=device)
    dense_masked_attention(qb, kb, vb, mask).sum().backward()

    for name, ga, gb in [("q", qa.grad, qb.grad), ("k", ka.grad, kb.grad),
                         ("v", va.grad, vb.grad)]:
        assert torch.allclose(ga, gb, atol=atol), f"{name} grad mismatch"


# ------------------------------------------------- schedule format identity

def test_raw_csr_arrays_alone_do_not_restrict_attention(device):
    """The trap: BlockMask.from_kv_blocks REPORTS the schedule and IGNORES it.

    `to_dense()` returns the correct sparse block pattern, so every structural
    check passes, while flex_attention computes full dense attention. This is
    the silent-wrong-answer class: a benchmark run this way measures a dense
    kernel and reports it as sparse.
    """
    dtype, atol = dtype_and_atol(device)
    counts, indices = causal_dense_schedule()
    bm = csr_to_blockmask_naive(counts, indices, device=device)
    q, k, v = qkv(dtype=dtype, device=device)

    # structural check passes
    assert torch.equal(bm.to_dense().view(NBLK, NBLK).bool(),
                       csr_block_table(counts, indices, device=device))

    got = flex_attention(q, k, v, block_mask=bm)
    want = dense_masked_attention(q, k, v, schedule_to_dense_mask(counts, indices, device=device))
    unmasked = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(got, want, atol=atol), (
        f"schedule ignored: distance to scheduled attention "
        f"{(got - want).abs().max():.3e}, distance to FULL attention "
        f"{(got - unmasked).abs().max():.3e}"
    )


def test_blockmask_roundtrips_a_csr_block_schedule(device):
    """A CSR schedule survives a trip through BlockMask unchanged."""
    counts = [1, 2, 2, 3]
    indices = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 1, 3, 0]]
    counts2, idx2 = blockmask_to_csr(csr_to_blockmask_naive(counts, indices, device=device))
    assert counts2 == counts
    assert idx2 == [indices[i][: counts[i]] for i in range(len(counts))]


def test_mask_fidelity_realized_pattern_equals_the_schedule(device):
    """tda-tdd contract 2: the realized mask is the schedule, element-wise."""
    counts = [1, 2, 2, 3]
    indices = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 1, 3, 0]]
    bm = csr_to_blockmask(counts, indices, device=device)
    realized = bm.to_dense().view(SEQ // BLK, SEQ // BLK).bool()
    expected = torch.zeros(NBLK, NBLK, dtype=torch.bool, device=device)
    for qb, n in enumerate(counts):
        for kb in indices[qb][:n]:
            expected[qb, kb] = True
    assert torch.equal(realized, expected)


def test_dense_parity_on_a_full_mask(device):
    """tda-tdd contract 1: a schedule selecting every block equals dense SDPA."""
    dtype, atol = dtype_and_atol(device)
    q, k, v = qkv(dtype=dtype, device=device)
    counts = [NBLK] * NBLK
    indices = [list(range(NBLK)) for _ in range(NBLK)]
    got = flex_attention(q, k, v, block_mask=csr_to_blockmask(counts, indices, device=device))
    want = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(got, want, atol=atol)


def test_causality_by_perturbation(device):
    """tda-tdd contract 3: perturbing a key must not move earlier outputs."""
    dtype, _ = dtype_and_atol(device)
    q, k, v = qkv(dtype=dtype, device=device)
    bm = create_block_mask(lambda b, h, i, j: i >= j, 1, 1, SEQ, SEQ,
                           device=device, BLOCK_SIZE=BLK)
    base = flex_attention(q, k, v, block_mask=bm)
    k2, v2 = k.clone(), v.clone()
    k2[..., SEQ - 1, :] += 5.0
    v2[..., SEQ - 1, :] += 5.0
    pert = flex_attention(q, k2, v2, block_mask=bm)
    assert torch.equal(base[..., : SEQ - 1, :], pert[..., : SEQ - 1, :])


def test_empty_schedule_row_does_not_produce_nan(device):
    """tda-tdd contract 4, and a bug the hierarchical builder will hit.

    A topological partition can leave a query block with no selected key
    blocks -- an isolated point is its own component and selects nothing.
    softmax over an all -inf row is NaN.
    """
    dtype, _ = dtype_and_atol(device)
    q, k, v = qkv(dtype=dtype, device=device)
    counts = [1, 0, 2, 3]  # row 1 selects nothing
    indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0], [0, 1, 3, 0]]
    out = flex_attention(q, k, v, block_mask=csr_to_blockmask(counts, indices, device=device))
    bad = int(torch.isnan(out).any(dim=-1).sum())
    assert bad == 0, f"{bad} query positions returned NaN from an empty schedule row"


def test_determinism_across_repeated_runs(device):
    """tda-tdd contract 6: bitwise identical output on repeated calls."""
    dtype, _ = dtype_and_atol(device)
    q, k, v = qkv(dtype=dtype, device=device)
    counts, indices = causal_dense_schedule()
    bm = csr_to_blockmask(counts, indices, device=device)
    a = flex_attention(q, k, v, block_mask=bm)
    b = flex_attention(q, k, v, block_mask=bm)
    assert torch.equal(a, b)


@pytest.mark.parametrize("seq", [BLK, 2 * BLK, 3 * BLK])
def test_tail_tile_shapes(seq, device):
    """tda-tdd contract 7: index arithmetic at the tail tile."""
    dtype, _ = dtype_and_atol(device)
    q, k, v = qkv(dtype=dtype, device=device, seq=seq)
    nb = seq // BLK
    counts = [i + 1 for i in range(nb)]
    indices = [list(range(i + 1)) + [0] * (nb - i - 1) for i in range(nb)]
    out = flex_attention(q, k, v, block_mask=csr_to_blockmask(counts, indices, seq=seq, device=device))
    assert out.shape == (1, 1, seq, HEAD)
    assert torch.isfinite(out).all()
