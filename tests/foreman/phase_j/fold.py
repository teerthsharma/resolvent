"""FOLD row: does the SU(2) composing-order gate need a custom kernel?

Direct:  out_i = sum_{j<=i} p_ij * G_ij * v_j
         G_ij = Pi_i * conj(Pi_j), applied per 4-block by left quaternion mult.
Fold:    pre-rotate v_j blocks by conj(Pi_j), run stock SDPA(is_causal=True)
         on CPU, post-rotate output blocks by Pi_i.

su2.py is a shared file in this scratchpad that another concurrent seat
overwrote with an incompatible API mid-task (qmul/prefix-scan helpers, not
this row's quat_mul/quat_conj). To avoid that race, this row's quaternion ops
are written inline here rather than imported.
"""
import numpy as np
import torch
import torch.nn.functional as F

N_BLOCKS = 4  # head_dim=16 -> four 4-blocks


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([w, x, y, z], dim=-1)


def quat_conj(a: torch.Tensor) -> torch.Tensor:
    w, x, y, z = a.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def random_unit_quats(shape, rng, dtype=torch.float64):
    q = rng.standard_normal(shape + (4,))
    q = q / (q**2).sum(-1, keepdims=True) ** 0.5
    return torch.tensor(q, dtype=dtype)


def make_inputs(B, H, S, D, seed=0, dtype=torch.float64):
    assert D % 4 == 0
    nb = D // 4
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, S, D, generator=g, dtype=dtype)
    k = torch.randn(B, H, S, D, generator=g, dtype=dtype)
    v = torch.randn(B, H, S, D, generator=g, dtype=dtype)
    rng = np.random.default_rng(seed)
    Pi = random_unit_quats((B, H, S, nb), rng, dtype=dtype)  # (B,H,S,nb,4)
    return q, k, v, Pi


def _blocks(x):
    # (B,H,S,D) -> (B,H,S,nb,4)
    B, H, S, D = x.shape
    return x.view(B, H, S, D // 4, 4)


def _unblocks(x):
    B, H, S, nb, _ = x.shape
    return x.reshape(B, H, S, nb * 4)


def direct_out(q, k, v, Pi):
    """Reference O(S^2) implementation, vectorized (no python loop)."""
    D = q.shape[-1]
    scale = D ** -0.5
    scores = torch.einsum("bhid,bhjd->bhij", q, k) * scale
    S = q.shape[2]
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
    scores = scores.masked_fill(~causal, float("-inf"))
    p = torch.softmax(scores, dim=-1)  # (B,H,S,S)

    vb = _blocks(v)  # (B,H,S,nb,4)
    Pi_i = Pi.unsqueeze(3)  # (B,H,S,1,nb,4)  broadcast over j
    Pi_j = Pi.unsqueeze(2)  # (B,H,1,S,nb,4)  broadcast over i
    G = quat_mul(Pi_i, quat_conj(Pi_j))  # (B,H,S,S,nb,4)
    v_j = vb.unsqueeze(2)  # (B,H,1,S,nb,4) broadcast over i
    rotated = quat_mul(G, v_j)  # (B,H,S,S,nb,4)

    p5 = p.unsqueeze(-1).unsqueeze(-1)  # (B,H,S,S,1,1)
    out_blocks = (p5 * rotated).sum(dim=3)  # (B,H,S,nb,4)
    return _unblocks(out_blocks)


def fold_out(q, k, v, Pi):
    D = q.shape[-1]
    vb = _blocks(v)
    w = quat_mul(quat_conj(Pi), vb)  # pre-rotate, (B,H,S,nb,4)
    w = _unblocks(w)
    attn = F.scaled_dot_product_attention(q, k, w, is_causal=True)
    attn_b = _blocks(attn)
    out_b = quat_mul(Pi, attn_b)  # post-rotate
    return _unblocks(out_b)
