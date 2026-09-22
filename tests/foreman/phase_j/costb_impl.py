"""Mode B' (quaternion head + prefix scan + value pre-rotation + SDPA +
output post-rotation), batched/vectorized. Supports (S,d) single-head input
(used by the CPU correctness test) and (B,H,S,head_dim) for GPU timing.
"""
import torch
import torch.nn.functional as F

HEAD_DIM = 64
N_BLOCKS = HEAD_DIM // 4
IDENTITY_Q = (1.0, 0.0, 0.0, 0.0)


def qmul(p, q):
    w1, x1, y1, z1 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    w2, x2, y2, z2 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def qconj(q):
    sign = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)
    return q * sign


def hillis_steele_prefix_quat(q):
    """q: (..., S, 4) per-token quaternion deltas -> Pi[..., i, :] =
    q[...,0,:] * q[...,1,:] * ... * q[...,i,:], log-depth Hillis-Steele scan
    (associative op = quaternion multiplication), batched over leading dims.
    """
    *lead, S, _ = q.shape
    acc = q.clone()
    identity = torch.tensor(IDENTITY_Q, dtype=q.dtype, device=q.device)
    offset = 1
    while offset < S:
        pad = identity.expand(*lead, offset, 4)
        shifted = torch.cat([pad, acc[..., :-offset, :]], dim=-2)
        acc = qmul(shifted, acc)
        offset *= 2
    return acc


def rotate_blocks(v, pi):
    """v: (..., S, head_dim); pi: (..., S, 4). Left-multiply each 4-block of
    v by pi (broadcast over the N_BLOCKS axis)."""
    *lead, S, hd = v.shape
    vb = v.view(*lead, S, N_BLOCKS, 4)
    pib = pi.unsqueeze(-2).expand(*lead, S, N_BLOCKS, 4)
    out = qmul(pib, vb)
    return out.reshape(*lead, S, hd)


def mode_bprime_forward(x, q_proj, k_proj, v_proj, w_head, causal=True):
    """CPU-correctness path: x (S, d_model) -> out (S, head_dim), single head.
    Mirrors the timed GPU path's math exactly (float64 here for the RED
    fold-parity test; the GPU timing path below uses float32 shapes)."""
    quat = x @ w_head.T
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    Pi = hillis_steele_prefix_quat(quat)

    q = x @ q_proj.T
    k = x @ k_proj.T
    v = x @ v_proj.T

    v_rot = rotate_blocks(v, qconj(Pi))

    S = x.shape[0]
    scores = (q @ k.T) / (HEAD_DIM ** 0.5)
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = attn @ v_rot

    out_rot = rotate_blocks(out, Pi)
    return out_rot


def mode_bprime_gpu(x, w_head, causal=True):
    """Timing path: x (B, S, d_model), single quaternion head (head-shared,
    Linear d_model->4), heads=8 * head_dim=64 = 512 = d_model, so Q/K/V are
    just x split into heads (no separate q/k/v Linear -- SDPA cost-bar row
    only needs a plain attention call plus the rotation overhead, matching
    the contract's ratio-of-SDPA framing). Returns (out, timings_dict)."""
    raise NotImplementedError("use the explicit staged timing script instead")
