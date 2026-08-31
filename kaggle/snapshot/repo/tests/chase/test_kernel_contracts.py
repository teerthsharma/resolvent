"""tda-tdd kernel correctness contracts applied to the ACTUAL merged kernel.

Subject under test: kernels/topology_sparse_attention.py from triton-lang/kernels PR #22
(merged 2026-07-28, e16236a), fetched verbatim to ./k22.py.

THEORY.md §3 declares this kernel "the base point, and it is the right one", and §0
counts it as MERGED infrastructure carrying zero remaining risk. THEORY.md §3's own
words: "Decision 3 therefore needs a *schedule builder*, not a new kernel." That
sentence is the hypothesis these tests attack: it assumes the kernel is safe for
schedules produced by builders the kernel does not own.

Contracts from anthropic-skills:tda-tdd "Kernel correctness contracts", in the
skill's stated order of bug-caught-per-line.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k22  # noqa: E402
from conftest import run_isolated  # noqa: E402

# routed through conftest: torch.cuda.is_available() alone still reports True
# under CUDA_VISIBLE_DEVICES="" while device_count() is 0, so these did not skip
# on a CPU-only box -- they ran and failed.
from conftest import HAS_CUDA  # noqa: E402

cuda = pytest.mark.skipif(not HAS_CUDA, reason="kernel is CUDA-only")

BLOCK = 64
SEQ = 256
DIM = 64


def _qkv(seq=SEQ, dim=DIM, dtype=torch.float32, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return [torch.randn(seq, dim, generator=g, device=device, dtype=dtype) for _ in range(3)]


# ---------------------------------------------------------- contract 1: dense parity

@cuda
def test_dense_parity_on_full_schedule():
    """Full (lower-triangular) schedule must equal dense causal SDPA.

    This is the harness smoke test. If this fails, every RED below is suspect.
    """
    q, k, v = _qkv()
    offsets, indices = k22.build_dense_causal_block_schedule(SEQ // BLOCK)
    offsets, indices = offsets.cuda(), indices.cuda()
    got = k22.scheduled_attention(q, k, v, offsets, indices, BLOCK)

    logits = (q @ k.T) / math.sqrt(DIM)
    pos = torch.arange(SEQ, device="cuda")
    logits = logits.masked_fill(pos[None, :] > pos[:, None], float("-inf"))
    want = torch.softmax(logits, dim=-1) @ v

    torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)


# ------------------------------------------------------- contract 5: gradient check

@cuda
def test_gradient_reaches_qkv_through_scheduled_attention():
    """THEORY.md trains the whole architecture through this kernel (§6 pipeline).

    Minimum viable requirement: d(loss)/dq must exist and be non-zero.
    Anything weaker than a raised exception is a silent training failure.
    """
    q, k, v = _qkv()
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    offsets, indices = k22.build_dense_causal_block_schedule(SEQ // BLOCK)
    out = k22.scheduled_attention(q, k, v, offsets.cuda(), indices.cuda(), BLOCK)

    assert out.requires_grad, (
        "output detached from the graph: autograd fails OPEN, not closed. "
        "No exception is raised; gradients simply never arrive."
    )
    out.sum().backward()
    assert q.grad is not None and q.grad.abs().sum() > 0


@cuda
def test_residual_path_masks_the_missing_backward():
    """The 2am version: a residual connection makes the missing backward invisible.

    x -> attn(x) + x is the standard transformer block. If attn detaches, .backward()
    does NOT raise, the loss decreases, and the attention parameters never train.
    """
    q, k, v = _qkv()
    offsets, indices = k22.build_dense_causal_block_schedule(SEQ // BLOCK)

    w = torch.eye(DIM, device="cuda", requires_grad=True)  # stands in for W_q
    out = k22.scheduled_attention(q @ w, k, v, offsets.cuda(), indices.cuda(), BLOCK) + (q @ w)
    out.sum().backward()  # does NOT raise: the residual keeps the graph alive

    w2 = torch.eye(DIM, device="cuda", requires_grad=True)
    ref = k22.dense_masked_attention(q @ w2, k, v, offsets, indices, BLOCK) + (q @ w2)
    ref.sum().backward()

    torch.testing.assert_close(
        w.grad, w2.grad, rtol=1e-3, atol=1e-3,
        msg="kernel-path gradient != reference-path gradient: the attention branch "
            "contributed nothing to dL/dW and .backward() did not complain",
    )


# ------------------------------------------------- contract 4: all-masked row guard

@cuda
def test_empty_schedule_row_does_not_produce_nan():
    """A query block with zero scheduled key blocks.

    THEORY.md §3 replaces the 0D-persistence builder with a NEW hierarchical
    near-field/far-field builder. The empty-row guard in k22 lives in
    build_topology_block_schedule (`if not allowed: allowed.add(q_block)`), NOT in
    the kernel and NOT in scheduled_attention()'s validation. A new builder does not
    inherit it.
    """
    q, k, v = _qkv()
    nb = SEQ // BLOCK
    # row 2 selects nothing; every other row is diagonal-only.
    offsets, indices = [0], []
    for i in range(nb):
        if i != 2:
            indices.append(i)
        offsets.append(len(indices))
    offsets = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    indices = torch.tensor(indices, dtype=torch.int64, device="cuda")

    out = k22.scheduled_attention(q, k, v, offsets, indices, BLOCK)
    assert torch.isfinite(out).all(), (
        f"non-finite output: {(~torch.isfinite(out)).sum().item()} elements. "
        "out = acc / l_i with l_i == 0."
    )


@cuda
def test_scheduled_attention_rejects_empty_schedule_row():
    """Failing closed would be acceptable. Validate the row is non-empty."""
    q, k, v = _qkv()
    nb = SEQ // BLOCK
    offsets, indices = [0], []
    for i in range(nb):
        if i != 2:
            indices.append(i)
        offsets.append(len(indices))
    with pytest.raises(ValueError):
        k22.scheduled_attention(q, k, v,
                                torch.tensor(offsets, dtype=torch.int64, device="cuda"),
                                torch.tensor(indices, dtype=torch.int64, device="cuda"),
                                BLOCK)


# ------------------------------------------- contract 2: mask fidelity / index bounds

# ---------------------------------------------------------------------------
# The next two probes are ISOLATED IN A SUBPROCESS. Both drive the merged kernel
# with an out-of-bounds schedule, and an illegal memory access is not recoverable
# in-process: every later CUDA launch fails whatever its own inputs are. Run
# in-process, `test_out_of_range_offsets_are_rejected` took 13 unrelated tests
# down with it. `run_isolated` gives each one its own interpreter, so the finding
# is still recorded and the blast radius is one subprocess.
#
# Because they no longer poison anything, these assert the MEASURED behaviour
# positively instead of sitting in the known-red ledger.
# ---------------------------------------------------------------------------


@cuda
def test_negative_block_index_returns_plausible_garbage_instead_of_failing():
    """indices is untrusted data that indexes GPU memory.

    k22 masks with `k_pos < seq` only. A negative k_block passes that mask and
    reads below the base pointer. On a Hub release the schedule arrives with the
    checkpoint, so this is attacker-reachable input.
    """
    rc, out, err = run_isolated(f"""
        import torch, k22
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k, v = [torch.randn({SEQ}, {DIM}, generator=g, device="cuda") for _ in range(3)]
        nb = {SEQ} // {BLOCK}
        off = torch.tensor([0] + [i + 1 for i in range(nb)], dtype=torch.int64, device="cuda")
        idx = torch.tensor([-1] + list(range(1, nb)), dtype=torch.int64, device="cuda")
        try:
            o = k22.scheduled_attention(q, k, v, off, idx, {BLOCK})
            torch.cuda.synchronize()
            print("NO_EXCEPTION finite=" + str(bool(torch.isfinite(o).all())))
        except ValueError as e:
            print("REJECTED " + str(e))
        except Exception as e:
            print("CRASHED " + type(e).__name__ + " " + str(e)[:120])
    """)
    assert "REJECTED" not in out, (
        "the merged kernel now rejects a negative block index; the finding no "
        f"longer reproduces and should be retired. stdout={out!r}"
    )
    assert "NO_EXCEPTION" in out or "CRASHED" in out, (
        f"probe produced no verdict. rc={rc} stdout={out!r} stderr={err[-800:]!r}"
    )
    print(f"\n  negative block index -> {out.strip()}")


@cuda
def test_out_of_range_offsets_reach_the_launch_and_poison_the_context():
    """offsets[i+1] beyond indices.numel() -> the kernel reads past the indices buffer.

    The subprocess is expected to die. That IS the finding: an untrusted schedule
    can take down the whole process, and no host-side check stands in the way.
    """
    rc, out, err = run_isolated(f"""
        import torch, k22
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k, v = [torch.randn({SEQ}, {DIM}, generator=g, device="cuda") for _ in range(3)]
        off = torch.tensor([0, 1, 2, 3, 999], dtype=torch.int64, device="cuda")
        idx = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda")
        try:
            k22.scheduled_attention(q, k, v, off, idx, {BLOCK})
            torch.cuda.synchronize()
            print("NO_EXCEPTION")
        except ValueError as e:
            print("REJECTED " + str(e))
        except Exception as e:
            print("CRASHED " + type(e).__name__ + " " + str(e)[:160])
        # the context is now suspect: prove it by trying an unrelated launch
        try:
            torch.randn(8, 8, device="cuda").sum().item()
            print("CONTEXT_OK")
        except Exception as e:
            print("CONTEXT_POISONED " + type(e).__name__)
    """)
    assert "REJECTED" not in out, (
        "the merged kernel now rejects out-of-range offsets before launch; the "
        f"finding no longer reproduces and should be retired. stdout={out!r}"
    )
    print(f"\n  out-of-range offsets -> rc={rc} {out.strip()!r}")
    assert "CONTEXT_POISONED" in out or "CRASHED" in out or rc != 0 or "NO_EXCEPTION" in out, (
        f"probe produced no verdict. rc={rc} stdout={out!r} stderr={err[-800:]!r}"
    )


# --------------------------------------------------------------- contract 3: causality

@cuda
def test_causality_perturbation():
    """Perturb key position j; outputs at i < j must be bitwise unchanged."""
    q, k, v = _qkv()
    offsets, indices = k22.build_dense_causal_block_schedule(SEQ // BLOCK)
    offsets, indices = offsets.cuda(), indices.cuda()
    a = k22.scheduled_attention(q, k, v, offsets, indices, BLOCK)
    j = 200
    k2 = k.clone()
    k2[j] += 10.0
    b = k22.scheduled_attention(q, k2, v, offsets, indices, BLOCK)
    assert torch.equal(a[:j], b[:j]), "future key leaked into past queries"


# --------------------------------------------------------------- contract 6: determinism

@cuda
def test_determinism_bitwise():
    q, k, v = _qkv()
    offsets, indices = k22.build_dense_causal_block_schedule(SEQ // BLOCK)
    offsets, indices = offsets.cuda(), indices.cuda()
    a = k22.scheduled_attention(q, k, v, offsets, indices, BLOCK)
    b = k22.scheduled_attention(q, k, v, offsets, indices, BLOCK)
    assert torch.equal(a, b)


# ---------------------------------------------------- contract 7: shape and dtype edges

@cuda
@pytest.mark.parametrize("seq", [BLOCK, BLOCK * 2, BLOCK * 2 + 1, BLOCK - 1, 1])
def test_tile_boundary_sequence_lengths(seq):
    """A 1B LM sees arbitrary sequence lengths at inference. The tail tile is where
    index arithmetic fails."""
    q, k, v = _qkv(seq=seq)
    nb = max(1, (seq + BLOCK - 1) // BLOCK)
    offsets, indices = k22.build_dense_causal_block_schedule(nb)
    out = k22.scheduled_attention(q, k, v, offsets.cuda(), indices.cuda(), BLOCK)
    assert torch.isfinite(out).all()


@cuda
@pytest.mark.parametrize("dim", [80, 96, 128])
def test_common_head_dims(dim):
    """Llama-3-8B head_dim=128 ok; many 1B configs use 80 or 96 (e.g. hidden 2048 /
    heads 26, or Gemma-2B's 256). k22 allows only {16,32,64,128}."""
    q, k, v = _qkv(dim=dim)
    offsets, indices = k22.build_dense_causal_block_schedule(SEQ // BLOCK)
    out = k22.scheduled_attention(q, k, v, offsets.cuda(), indices.cuda(), BLOCK)
    assert out.shape == (SEQ, dim)
