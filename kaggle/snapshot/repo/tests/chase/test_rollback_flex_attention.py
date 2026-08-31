"""The conservative option for §8 risk 6, tested rather than asserted.

THEORY.md §8 risk 6: "kernels#22 is forward-only. Training through it requires a
backward pass that does not exist yet."

Writing a block-sparse attention backward in Triton is ~500-950 lines (measured:
torch's own flex_attention backward is ~934 lines in
torch/_inductor/kernel/flex_attention.py; the Triton tutorial's DENSE backward is 255;
flash-attn's dense Triton backward is 524 + a 118-line wrapper, and its own docstring
says "I'm not 100% sure that the backward pass doesn't have race conditions").

The alternative is to not write one. torch.nn.attention.flex_attention has a
maintained, autotuned Triton backward AND `BlockMask.from_kv_blocks` accepts
user-supplied CSR block lists -- exactly the format kernels#22 already produces.

Dead ends checked and ruled out, so nobody re-checks them:
  - xformers BlockSparseAttention: removed in xformers 0.0.29 (2024-12-27). It wrapped
    triton.ops.blocksparse, which was deleted in Triton 3.2.0. Not revivable.
  - DeepSpeed sparse_attention: has forward+backward, pinned to `triton==1.0.0`.
  - flash-attn block-sparse: fwd_block/bwd_block are FlashAttention-v1 CUDA symbols no
    longer compiled by the current setup.py. Last functional commit 2022-06-02.
  - Live third-party option: fla-org/native-sparse-attention (MIT), real Triton
    backward over user-supplied block_indices/block_counts.

These tests establish what the migration does and does not give for free.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conftest import HAS_CUDA  # noqa: E402

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")

S, D, BLK = 1024, 64, 128


def _adapter(offsets, indices, nb, device):
    """kernels#22 CSR -> flex_attention BlockMask.

    from_kv_blocks distinguishes PARTIAL blocks (kv_indices, get mask_mod applied) from
    FULL blocks (full_kv_indices, no masking). kernels#22's CSR carries no such
    distinction -- its kernel applies `k_pos <= offs_m` to every block unconditionally.
    The diagonal block is the partial one; strictly-below blocks are full.
    """
    def causal(b, h, qi, kvi):
        return qi >= kvi

    from torch.nn.attention.flex_attention import BlockMask
    pc = torch.zeros(nb, device=device, dtype=torch.int32)
    pi = torch.zeros(nb, nb, device=device, dtype=torch.int32)
    fc = torch.zeros(nb, device=device, dtype=torch.int32)
    fi = torch.zeros(nb, nb, device=device, dtype=torch.int32)
    for i in range(nb):
        row = sorted(int(x) for x in indices[offsets[i]:offsets[i + 1]])
        p = [j for j in row if j == i]
        f = [j for j in row if j < i]
        pc[i] = len(p); pi[i, :len(p)] = torch.tensor(p, device=device, dtype=torch.int32)
        fc[i] = len(f); fi[i, :len(f)] = torch.tensor(f, device=device, dtype=torch.int32)
    return BlockMask.from_kv_blocks(pc.view(1, 1, nb), pi.view(1, 1, nb, nb),
                                    fc.view(1, 1, nb), fi.view(1, 1, nb, nb),
                                    BLOCK_SIZE=BLK, mask_mod=causal)


def _setup():
    import k22
    torch.manual_seed(0)
    nb = S // BLK
    keys = torch.randn(S, D, device="cuda")
    off, idx = k22.build_topology_block_schedule(keys, BLK, 1, 1, 1)
    q, k, v = [torch.randn(S, D, device="cuda", dtype=torch.float32) for _ in range(3)]
    return k22, off, idx, nb, q, k, v


def test_flex_attention_accepts_a_kernels22_csr_schedule_and_gives_gradients():
    """The load-bearing half of the rollback: a real backward, on the real schedule."""
    from torch.nn.attention.flex_attention import flex_attention
    k22, off, idx, nb, q, k, v = _setup()
    bm = _adapter(off, idx, nb, "cuda")
    qq, kk, vv = [x.view(1, 1, S, D).clone().requires_grad_() for x in (q, k, v)]
    out = flex_attention(qq, kk, vv, block_mask=bm)
    assert out.requires_grad
    out.sum().backward()
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
               for g in (qq.grad, kk.grad, vv.grad))


def test_adapted_flex_attention_matches_the_merged_kernel():
    """The half that is NOT free. tda-tdd contract 1 (dense parity), applied to the
    migration rather than to the kernel."""
    from torch.nn.attention.flex_attention import flex_attention
    k22, off, idx, nb, q, k, v = _setup()
    ref = k22.dense_masked_attention(q, k, v, off, idx, BLK)
    kern = k22.scheduled_attention(q, k, v, off.cuda(), idx.cuda(), BLK)
    kern_err = float((kern - ref).abs().max())

    bm = _adapter(off, idx, nb, "cuda")
    qq, kk, vv = [x.view(1, 1, S, D).clone() for x in (q, k, v)]
    flex_err = float((flex_attention(qq, kk, vv, block_mask=bm).view(S, D) - ref).abs().max())

    assert flex_err <= max(kern_err * 10, 1e-4), (
        f"the merged Triton kernel reproduces its own dense reference to "
        f"max|d| = {kern_err:.2e}. The straightforward CSR -> BlockMask adapter "
        f"reproduces it only to max|d| = {flex_err:.2e} -- {flex_err / kern_err:.0f}x "
        f"worse. Schedule rows: "
        f"{[sorted(int(x) for x in idx[off[i]:off[i + 1]]) for i in range(nb)]}. "
        f"The migration is a real port with its own parity bug surface, not a "
        f"format cast. Run the tda-tdd dense-parity and mask-fidelity contracts "
        f"against it before trusting a training run to it."
    )
