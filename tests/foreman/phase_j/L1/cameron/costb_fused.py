"""Mode B' with the scan and the two rotations as single Triton launches,
around stock SDPA (attention pattern untouched). Contract order:
Pi_i = q_i * ... * q_0, so Pi_i * conj(Pi_j) = q_i ... q_{j+1} = G_ij.
Head is head-shared (Linear(d,4)), as in costb_timing.py.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _qcomb(aw, ax, ay, az, bw, bx, by, bz):
    # a = earlier prefix, b = later element -> b * a (contract order)
    w = bw * aw - bx * ax - by * ay - bz * az
    x = bw * ax + bx * aw + by * az - bz * ay
    y = bw * ay - bx * az + by * aw + bz * ax
    z = bw * az + bx * ay - by * ax + bz * aw
    return w, x, y, z


@triton.jit
def _scan_kernel(R, P, S, BLOCK: tl.constexpr, NORM: tl.constexpr = True):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < S
    base = R + row * S * 4 + offs * 4
    w = tl.load(base + 0, mask=m, other=1.0)
    x = tl.load(base + 1, mask=m, other=0.0)
    y = tl.load(base + 2, mask=m, other=0.0)
    z = tl.load(base + 3, mask=m, other=0.0)
    n = 1.0
    if NORM:
        n = tl.maximum(tl.sqrt(w * w + x * x + y * y + z * z), 1e-8)  # head normalize, fused
    w, x, y, z = tl.associative_scan((w / n, x / n, y / n, z / n), 0, _qcomb)
    out = P + row * S * 4 + offs * 4
    tl.store(out + 0, w, mask=m)
    tl.store(out + 1, x, mask=m)
    tl.store(out + 2, y, mask=m)
    tl.store(out + 3, z, mask=m)


@triton.jit
def _rot_kernel(V, P, O, n_rows, H, S, sb, sh, ss,
                CONJ: tl.constexpr, BR: tl.constexpr, NB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rm = rows < n_rows
    s = rows % S
    h = (rows // S) % H
    b = rows // (S * H)
    prow = (b * S + s) * 4
    pw = tl.load(P + prow + 0, mask=rm, other=1.0)[:, None]
    px = tl.load(P + prow + 1, mask=rm, other=0.0)[:, None]
    py = tl.load(P + prow + 2, mask=rm, other=0.0)[:, None]
    pz = tl.load(P + prow + 3, mask=rm, other=0.0)[:, None]
    if CONJ:
        px = -px
        py = -py
        pz = -pz
    blk = tl.arange(0, NB)[None, :] * 4
    vin = V + (b * sb + h * sh + s * ss)[:, None] + blk
    m2 = rm[:, None]
    vw = tl.load(vin + 0, mask=m2, other=0.0)
    vx = tl.load(vin + 1, mask=m2, other=0.0)
    vy = tl.load(vin + 2, mask=m2, other=0.0)
    vz = tl.load(vin + 3, mask=m2, other=0.0)
    o = O + (rows * (NB * 4))[:, None] + blk
    tl.store(o + 0, pw * vw - px * vx - py * vy - pz * vz, mask=m2)
    tl.store(o + 1, pw * vx + px * vw + py * vz - pz * vy, mask=m2)
    tl.store(o + 2, pw * vy - px * vz + py * vw + pz * vx, mask=m2)
    tl.store(o + 3, pw * vz + px * vy - py * vx + pz * vw, mask=m2)


def prefix_scan(r, norm=True):
    """r: (B, S, 4) raw head output -> Pi (B, S, 4), normalized per token, one launch."""
    B, S, _ = r.shape
    r = r.contiguous()
    P = torch.empty_like(r)
    _scan_kernel[(B,)](r, P, S, BLOCK=triton.next_power_of_2(S), NORM=norm, num_warps=8)
    return P


def rotate(v, P, conj):
    """v: (B, H, S, D) any strides with unit last stride; P: (B, S, 4). Left-multiply
    every 4-block of v by P (or conj(P)). Returns contiguous (B, H, S, D)."""
    B, H, S, D = v.shape
    assert v.stride(3) == 1 and D % 4 == 0
    out = torch.empty((B, H, S, D), device=v.device, dtype=v.dtype)
    n_rows, BR = B * H * S, 64
    _rot_kernel[(triton.cdiv(n_rows, BR),)](
        v, P, out, n_rows, H, S, v.stride(0), v.stride(1), v.stride(2),
        CONJ=conj, BR=BR, NB=D // 4, num_warps=4)
    return out


def mode_bprime_fused(x, w_head, q, k, v):
    """x (B,S,d), w_head (4,d), q/k/v (B,H,S,D) -> out (B,H,S,D). Forward only."""
    P = prefix_scan(x @ w_head.T)
    out = F.scaled_dot_product_attention(q, k, rotate(v, P, True), is_causal=True)
    return rotate(out, P, False)


# ---------------------------------------------------------------- backward
@triton.jit
def _qmul(pw, px, py, pz, qw, qx, qy, qz):
    return (pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw)


@triton.jit
def _scan_bwd_kernel(P, G, DQ, S, BLOCK: tl.constexpr):
    # Pi_i = q_i Pi_{i-1}, unit q: dL/dq_k = Pi_k [sum_{i>=k} conj(Pi_i) g_i] conj(Pi_{k-1})
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < S
    pb = P + row * S * 4 + offs * 4
    gb = G + row * S * 4 + offs * 4
    pw = tl.load(pb + 0, mask=m, other=1.0)
    px = tl.load(pb + 1, mask=m, other=0.0)
    py = tl.load(pb + 2, mask=m, other=0.0)
    pz = tl.load(pb + 3, mask=m, other=0.0)
    gw = tl.load(gb + 0, mask=m, other=0.0)
    gx = tl.load(gb + 1, mask=m, other=0.0)
    gy = tl.load(gb + 2, mask=m, other=0.0)
    gz = tl.load(gb + 3, mask=m, other=0.0)
    tw, tx, ty, tz = _qmul(pw, -px, -py, -pz, gw, gx, gy, gz)
    uw = tl.cumsum(tw, 0, reverse=True)
    ux = tl.cumsum(tx, 0, reverse=True)
    uy = tl.cumsum(ty, 0, reverse=True)
    uz = tl.cumsum(tz, 0, reverse=True)
    mp = m & (offs > 0)
    rb = P + row * S * 4 + (offs - 1) * 4
    rw = tl.load(rb + 0, mask=mp, other=1.0)
    rx = tl.load(rb + 1, mask=mp, other=0.0)
    ry = tl.load(rb + 2, mask=mp, other=0.0)
    rz = tl.load(rb + 3, mask=mp, other=0.0)
    aw, ax, ay, az = _qmul(pw, px, py, pz, uw, ux, uy, uz)
    dw, dx, dy, dz = _qmul(aw, ax, ay, az, rw, -rx, -ry, -rz)
    ob = DQ + row * S * 4 + offs * 4
    tl.store(ob + 0, dw, mask=m)
    tl.store(ob + 1, dx, mask=m)
    tl.store(ob + 2, dy, mask=m)
    tl.store(ob + 3, dz, mask=m)


class _Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qn):
        P = prefix_scan(qn, norm=False)
        ctx.save_for_backward(P)
        return P

    @staticmethod
    def backward(ctx, g):
        (P,) = ctx.saved_tensors
        B, S, _ = P.shape
        dq = torch.empty_like(P)
        _scan_bwd_kernel[(B,)](P, g.contiguous(), dq, S, BLOCK=triton.next_power_of_2(S), num_warps=8)
        return dq


_CT = {}


def _conj_struct(dev, dt):
    """C[a, c, :] = e_a * conj(e_c), so (g * conj(v))_d = sum_ac g_a v_c C[a,c,d]."""
    key = (dev, dt)
    if key not in _CT:
        e = torch.eye(4, dtype=torch.float64)
        C = torch.zeros(4, 4, 4, dtype=torch.float64)
        sign = torch.tensor([1.0, -1, -1, -1], dtype=torch.float64)
        for a in range(4):
            for c in range(4):
                p, q = e[a], e[c] * sign
                C[a, c] = torch.stack([p[0]*q[0]-p[1]*q[1]-p[2]*q[2]-p[3]*q[3],
                                       p[0]*q[1]+p[1]*q[0]+p[2]*q[3]-p[3]*q[2],
                                       p[0]*q[2]-p[1]*q[3]+p[2]*q[0]+p[3]*q[1],
                                       p[0]*q[3]+p[1]*q[2]-p[2]*q[1]+p[3]*q[0]])
        _CT[key] = C.to(dev, dt)
    return _CT[key]


class _Rot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, P, conj):
        ctx.save_for_backward(v, P)
        ctx.conj = conj
        return rotate(v, P, conj)

    @staticmethod
    def backward(ctx, g):
        v, P = ctx.saved_tensors
        g = g.contiguous()  # out.sum() hands an expanded (stride-0) grad
        B, H, S, D = v.shape
        gv = rotate(g, P, not ctx.conj)  # L(p)^T g = conj(p) g
        M = torch.einsum("bhsna,bhsnc->bsac", g.reshape(B, H, S, D // 4, 4), v.reshape(B, H, S, D // 4, 4))
        gp = torch.einsum("bsac,acd->bsd", M, _conj_struct(v.device, v.dtype))  # sum g * conj(v)
        if ctx.conj:
            gp = gp * torch.tensor([1.0, -1, -1, -1], device=gp.device, dtype=gp.dtype)
        return gv, gp, None


def mode_bprime_train(x, w_head, q, k, v):
    """Differentiable Mode B' (forward identical math to mode_bprime_fused)."""
    P = _Scan.apply(F.normalize(x @ w_head.T, dim=-1))
    out = F.scaled_dot_product_attention(q, k, _Rot.apply(v, P, True), is_causal=True)
    return _Rot.apply(out, P, False)
