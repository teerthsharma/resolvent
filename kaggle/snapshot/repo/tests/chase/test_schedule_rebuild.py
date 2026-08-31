"""THEORY.md's single new component, attacked directly.

§0: "The single genuinely new thing is decision 3's schedule: replacing the
0D-persistence salience schedule with a hierarchical near-field/far-field block
partition, and making that partition switch per input at inference time."

§3: "A schedule can be rebuilt per input. This is the mechanism for 'internal switch
of phases or geometry in real time'. Nothing else in the stack has to move."

Three questions about the only new part:
  A. Does it receive a gradient? It is the only component being proposed; if it is
     not trainable, it is a hyperparameter search, not an architecture.
  B. What happens if the schedule differs between forward and backward? §3 rebuilds
     per input and §4 gates rebuilds on a regime detector. Neither pins the schedule
     for the duration of a training step.
  C. What does rebuilding cost, against the attention it is scheduling?
"""

import os
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theory_ref import ScheduledAttention, build_schedule, masked_attention, schedule_to_mask  # noqa: E402

BLOCK, SEQ, DIM = 16, 128, 32
NB = SEQ // BLOCK


def _qkv(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 1, SEQ, DIM, generator=g, dtype=torch.float64, requires_grad=True)
            for _ in range(3)]


def _mask(top_k, salience, seed=0):
    ip, ix = build_schedule(NB, NB, local_window=1, n_sink=1, salience=salience, top_k=top_k)
    return schedule_to_mask(ip, ix, NB, NB, BLOCK, SEQ)


# ==================================================== A. is the new component trainable?

def test_schedule_selection_receives_gradient():
    """The hierarchical partition is chosen by a salience score over key-block
    centroids. Make that score a learnable parameter and ask for its gradient."""
    q, k, v = _qkv()
    salience = torch.randn(NB, NB, dtype=torch.float64, requires_grad=True)

    ip, ix = build_schedule(NB, NB, local_window=1, n_sink=1,
                            salience=salience.detach(), top_k=2)
    m = schedule_to_mask(ip, ix, NB, NB, BLOCK, SEQ)
    out = masked_attention(q, k, v, m)
    out.sum().backward()

    assert salience.grad is not None and salience.grad.abs().sum() > 0, (
        "d(loss)/d(salience) is None. The schedule enters the computation only through "
        "a top-k over salience, which is an argsort: piecewise constant, zero gradient "
        "almost everywhere. THEORY.md's only new component cannot be trained by the "
        "gradient that trains everything around it."
    )


def test_loss_is_continuous_in_the_schedule_score():
    """Zero gradient would be tolerable if the loss were also flat. It is not: the
    loss jumps discontinuously the moment the top-k ranking flips."""
    q, k, v = _qkv()
    # Blocks 3 and 4 are genuine far-field candidates: neither is a sink block nor in
    # any row's local window, so the top-k ranking between them decides the schedule.
    base = torch.zeros(NB, NB, dtype=torch.float64)
    base[:, 3] = 1.0

    losses = []
    for eps in torch.linspace(0.9, 1.1, 41, dtype=torch.float64):
        s = base.clone()
        s[:, 4] = float(eps)              # block 4 crosses block 3 in ranking at eps=1
        m = _mask(top_k=1, salience=s)
        losses.append(float(masked_attention(q, k, v, m).pow(2).sum()))

    jumps = [abs(losses[i + 1] - losses[i]) for i in range(len(losses) - 1)]
    biggest = max(jumps)
    typical = sorted(jumps)[len(jumps) // 2]
    assert biggest < 10 * max(typical, 1e-12), (
        f"loss jumps by {biggest:.4f} at a single top-k rank flip while the median "
        f"step-to-step change is {typical:.2e}. The loss surface over the schedule is "
        f"a step function: flat everywhere the gradient exists, discontinuous exactly "
        f"where the schedule changes. Neither gradient descent nor a line search sees it."
    )


# =================================== B. forward and backward disagreeing on the schedule

@pytest.mark.parametrize("blocks_changed", [1, 2])
def test_gradient_survives_a_schedule_rebuild_between_forward_and_backward(blocks_changed):
    """§4 gates rebuilds on `stratum`, a regime detector that runs on the trajectory.
    A rebuild triggered between the forward and backward of one step gives the
    backward a different mask than the forward used."""
    q, k, v = _qkv()
    sal_fwd = torch.zeros(NB, NB, dtype=torch.float64)
    sal_fwd[:, 3] = 1.0
    sal_bwd = sal_fwd.clone()
    for j in range(4, 4 + blocks_changed):
        sal_bwd[:, j] = 2.0                       # the rebuild picks different far-field blocks

    m_fwd = _mask(top_k=1, salience=sal_fwd)
    m_bwd = _mask(top_k=1, salience=sal_bwd)
    changed = int((m_fwd != m_bwd).sum())

    out = ScheduledAttention.apply(q, k, v, m_fwd, m_bwd)
    out.sum().backward()
    got = q.grad.clone()

    q2, k2, v2 = _qkv()
    masked_attention(q2, k2, v2, m_fwd).sum().backward()
    want = q2.grad

    cos = float(torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0))
    rel = float((got - want).norm() / want.norm())
    assert rel < 1e-6, (
        f"{changed} of {SEQ * SEQ} mask entries differed between the forward and "
        f"backward schedule ({100 * changed / SEQ ** 2:.1f}%). Resulting gradient: "
        f"relative error {rel:.3f}, cosine similarity to the true gradient {cos:.4f}. "
        f"Nothing raises; the optimiser simply takes a step in a direction that is not "
        f"the gradient of anything it computed."
    )


def test_mismatched_backward_still_descends_the_loss():
    """The 2am question: is a mismatched gradient at least a descent direction?"""
    torch.manual_seed(0)
    q, k, v = _qkv()
    sal_fwd = torch.zeros(NB, NB, dtype=torch.float64); sal_fwd[:, 3] = 1.0
    sal_bwd = sal_fwd.clone(); sal_bwd[:, 4] = 2.0
    m_fwd, m_bwd = _mask(1, sal_fwd), _mask(1, sal_bwd)

    target = torch.randn(1, 1, SEQ, DIM, dtype=torch.float64)
    out = ScheduledAttention.apply(q, k, v, m_fwd, m_bwd)
    loss0 = ((out - target) ** 2).mean()
    loss0.backward()

    lr = 1e-2
    with torch.no_grad():
        qn = q - lr * q.grad
    loss1 = ((masked_attention(qn, k.detach(), v.detach(), m_fwd) - target) ** 2).mean()

    q2, k2, v2 = _qkv()
    ((masked_attention(q2, k2, v2, m_fwd) - target) ** 2).mean().backward()
    with torch.no_grad():
        qt = q2 - lr * q2.grad
    loss_true = ((masked_attention(qt, k.detach(), v.detach(), m_fwd) - target) ** 2).mean()

    assert float(loss1) <= float(loss_true) + 1e-12, (
        f"mismatched-schedule step reached loss {float(loss1):.6f}; the true gradient "
        f"step reaches {float(loss_true):.6f} from the same start ({float(loss0):.6f}). "
        f"The mismatched step is worse by {float(loss1) - float(loss_true):.2e} per step."
    )


# ============================================================== C. cost of rebuilding

def test_schedule_rebuild_is_cheaper_than_the_attention_it_schedules():
    """§4's premise: "Recomputing a hierarchical partition every token is wasteful;
    recomputing it when the regime changes is not." That only holds if a rebuild is
    cheap relative to a forward pass."""
    from conftest import HAS_CUDA
    if not HAS_CUDA:
        pytest.skip("needs the CUDA kernel")
    import k22
    # Measured (RTX 4060, warm, block_size=64, dim=64), builder vs dense causal SDPA:
    #   seq  1024 (16 blk):   1.68 ms builder | 0.043 kernel | 0.053 dense =  32x dense
    #   seq  2048 (32 blk):   2.46 ms builder | 0.051 kernel | 0.107 dense =  23x dense
    #   seq  4096 (64 blk):   8.33 ms builder | 0.038 kernel | 0.246 dense =  34x dense
    #   seq  8192 (128 blk): 42.12 ms builder | 0.080 kernel | 0.792 dense =  53x dense
    seq, dim, blk = 8192, 64, 64
    keys = torch.randn(seq, dim, device="cuda")
    q, k, v = [torch.randn(seq, dim, device="cuda", dtype=torch.float16) for _ in range(3)]

    torch.cuda.synchronize(); t = time.perf_counter()
    off, idx = k22.build_topology_block_schedule(keys, blk, 1, 1, 4)
    torch.cuda.synchronize(); build_ms = (time.perf_counter() - t) * 1e3

    off, idx = off.cuda(), idx.cuda()
    for _ in range(3):
        k22.scheduled_attention(q, k, v, off, idx, blk)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10):
        k22.scheduled_attention(q, k, v, off, idx, blk)
    torch.cuda.synchronize(); kernel_ms = (time.perf_counter() - t) / 10 * 1e3

    assert build_ms < kernel_ms, (
        f"rebuilding the schedule at seq={seq} costs {build_ms:.2f} ms; running the "
        f"attention it schedules costs {kernel_ms:.3f} ms -- {build_ms / kernel_ms:.0f}x. "
        f"_zero_dim_persistence_salience materialises all O(B^2) centroid pairs in a "
        f"Python loop, sorts them in Python, and runs union-find in Python, after a "
        f"blocking .to('cpu') of the centroids."
    )
