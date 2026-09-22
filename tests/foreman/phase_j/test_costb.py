"""Test-first, CPU, S=64: Mode B' output must equal a direct dense fold.

fold.py does not exist in this dir, so the fold is defined here directly
(RED line recorded): dense sum over j<=i of softmax(QK^T/sqrt(d)) weights
applied to conj(Pi_j)-rotated V per 4-block, then Pi_i-rotated output --
i.e. Mode B' IS its own reference formula at small S; the only thing under
test is that the batched/scan implementation matches a naive per-position
Python loop computing the same three steps.
"""
import numpy as np
import torch
import torch.nn.functional as F
import su2

torch.manual_seed(0)
D_MODEL = 512
HEADS = 8
HEAD_DIM = 64
N_BLOCKS = HEAD_DIM // 4


def quat_head(x, w):
    """x: (S, d_model) -> per-token unit quaternion (S,4), head-shared."""
    q = x @ w.T  # (S,4)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return q


def hillis_steele_prefix_quat(q):
    """q: (S,4) per-token quaternion deltas -> Pi_i = q_0 * q_1 * ... * q_i
    via log-depth Hillis-Steele scan (associative op = quaternion mult)."""
    S = q.shape[0]
    acc = q.clone()
    offset = 1
    while offset < S:
        shifted = torch.cat([
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=acc.dtype, device=acc.device).repeat(min(offset, S), 1),
            acc[:-offset] if offset < S else acc[:0],
        ], dim=0)
        acc = qmul_torch(shifted, acc)
        offset *= 2
    return acc


def qmul_torch(p, q):
    w1, x1, y1, z1 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    w2, x2, y2, z2 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def qconj_torch(q):
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)


def rotate_blocks(v, pi):
    """v: (S, head_dim) viewed as (S, N_BLOCKS, 4); pi: (S,4) quaternion,
    left-multiply each 4-block by pi. Returns same shape as v."""
    S = v.shape[0]
    vb = v.view(S, N_BLOCKS, 4)
    pi_b = pi.unsqueeze(1).expand(S, N_BLOCKS, 4)
    out = qmul_torch(pi_b, vb)
    return out.reshape(S, HEAD_DIM)


def mode_bprime_batched(q, k, v, w_head, causal=True):
    """q,k,v: (S, head_dim) single head/batch slice; w_head: (4, d_model)
    applied to a d_model-width token stream fed in separately (x)."""
    raise NotImplementedError


def naive_fold_S64(x, q_proj, k_proj, v_proj, w_head):
    """Direct dense per-position Python loop, S=64, single head, float64.
    This IS the reference fold (fold.py absent, RED-first line recorded)."""
    S = x.shape[0]
    quat = quat_head(x, w_head)  # (S,4)
    Pi = hillis_steele_prefix_quat(quat)  # (S,4), Pi[i] = q0*...*qi
    q = x @ q_proj.T
    k = x @ k_proj.T
    v = x @ v_proj.T
    v_rot = torch.empty_like(v)
    for j in range(S):
        v_rot[j] = rotate_blocks(v[j:j+1], qconj_torch(Pi[j:j+1]))[0]
    scores = (q @ k.T) / (HEAD_DIM ** 0.5)
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = attn @ v_rot  # (S, head_dim)
    out_rot = torch.empty_like(out)
    for i in range(S):
        out_rot[i] = rotate_blocks(out[i:i+1], Pi[i:i+1])[0]
    return out_rot


def test_costb_matches_naive_fold_S64():
    torch.manual_seed(0)
    S = 64
    x = torch.randn(S, D_MODEL, dtype=torch.float64)
    w_head = torch.randn(4, D_MODEL, dtype=torch.float64) * 0.1
    q_proj = torch.randn(HEAD_DIM, D_MODEL, dtype=torch.float64) * 0.1
    k_proj = torch.randn(HEAD_DIM, D_MODEL, dtype=torch.float64) * 0.1
    v_proj = torch.randn(HEAD_DIM, D_MODEL, dtype=torch.float64) * 0.1

    ref = naive_fold_S64(x, q_proj, k_proj, v_proj, w_head)

    # batched implementation under test, imported lazily so this file can be
    # run RED before costb_impl.py exists
    import costb_impl
    out = costb_impl.mode_bprime_forward(
        x, q_proj, k_proj, v_proj, w_head, causal=True
    )
    assert torch.allclose(ref, out, atol=1e-8, rtol=1e-6), \
        f"max abs diff {(ref-out).abs().max().item()}"


if __name__ == "__main__":
    test_costb_matches_naive_fold_S64()
    print("PASS")
