"""f-lite: the gated family with phase off, as one fused causal attention.

theta == 0 turns arm_smprime's operator into
    O_i = sum_j R_ij e^{s_ij} v_j / Z_i^beta,   R_ij = prod_{k=j+1..i} m_k,  Z_i = sum_j R_ij e^{s_ij}
        = softmax(s + log R)_i V * exp((1 - beta) * LSE_i)
with log R_ij = L_i - L_j (L = cumsum of log m over nonzero gates) when no exact
zero lies in (j, i], and -inf when one does (segment id = running zero count).
The kernel never writes an S x S tensor: flash-style online softmax forward,
two-kernel backward, and the LSE gradient enters as D_i -> D_i - dLSE_i.
L is built in float64 and handed to the kernel as fp32 (hi, lo) pairs, so L_i - L_j keeps
fp32 accuracy relative to the bias at any S.
"""
import torch
import triton
import triton.language as tl

ZETA, GAMMA = 1.1, -0.1  # hard-concrete stretch, same constants as r1_gate.magnitude_hardconcrete


def hard_concrete(u):
    return torch.clamp(torch.sigmoid(u) * (ZETA - GAMMA) + GAMMA, 0.0, 1.0)


@triton.jit
def _bias_mask(offs_m, offs_n, Li, Si, Lg, Sg, b_off, S):
    # L is (hi, lo) fp32 pairs: hi_i - hi_j rounds relative to the bias, lo restores what hi lost.
    Lj = tl.load(Lg + 2 * (b_off + offs_n), mask=offs_n < S, other=0.0)
    Lj_lo = tl.load(Lg + 2 * (b_off + offs_n) + 1, mask=offs_n < S, other=0.0)
    Li_lo = tl.load(Lg + 2 * (b_off + offs_m) + 1, mask=offs_m < S, other=0.0)
    Sj = tl.load(Sg + b_off + offs_n, mask=offs_n < S, other=-2)
    bias = (Li[:, None] - Lj[None, :]) + (Li_lo[:, None] - Lj_lo[None, :])
    valid = (offs_m[:, None] >= offs_n[None, :]) & (Si[:, None] == Sj[None, :]) & (offs_n[None, :] < S)
    return bias, valid


@triton.jit
def _fwd(Q, K, V, Lg, Sg, O, LSE, S, H,
         D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b_off = (bh // H) * S
    base = bh.to(tl.int64) * S * D
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mrow = offs_m < S
    q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :], mask=mrow[:, None], other=0.0)
    Li = tl.load(Lg + 2 * (b_off + offs_m), mask=mrow, other=0.0)
    Si = tl.load(Sg + b_off + offs_m, mask=mrow, other=-1)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    # ponytail: no segment-skip of dead key blocks; add when gates close often enough to pay.
    hi = tl.minimum((pid_m + 1) * BLOCK_M, S)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < S
        k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :], mask=nmask[:, None], other=0.0)
        v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :], mask=nmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision=PREC)
        bias, valid = _bias_mask(offs_m, offs_n, Li, Si, Lg, Sg, b_off, S)
        a = tl.where(valid, s + bias, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(a, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(a - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision=PREC)
        m_i = m_new
    tl.store(O + base + offs_m[:, None] * D + offs_d[None, :], acc / l_i[:, None], mask=mrow[:, None])
    tl.store(LSE + bh.to(tl.int64) * S + offs_m, m_i + tl.log(l_i), mask=mrow)


@triton.jit
def _bwd_dkdv(Q, K, V, Lg, Sg, DO, LSE, Dl, DK, DV, CS, S, H,
              D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    b_off = (bh // H) * S
    base = bh.to(tl.int64) * S * D
    rbase = bh.to(tl.int64) * S
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    nmask = offs_n < S
    k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :], mask=nmask[:, None], other=0.0)
    v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :], mask=nmask[:, None], other=0.0)
    dk = tl.zeros([BLOCK_N, D], tl.float32)
    dv = tl.zeros([BLOCK_N, D], tl.float32)
    cs = tl.zeros([BLOCK_N], tl.float32)
    lo = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M
    for start_m in range(lo, S, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mrow = offs_m < S
        q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :], mask=mrow[:, None], other=0.0)
        do = tl.load(DO + base + offs_m[:, None] * D + offs_d[None, :], mask=mrow[:, None], other=0.0)
        lse = tl.load(LSE + rbase + offs_m, mask=mrow, other=0.0)
        Di = tl.load(Dl + rbase + offs_m, mask=mrow, other=0.0)
        Li = tl.load(Lg + 2 * (b_off + offs_m), mask=mrow, other=0.0)
        Si = tl.load(Sg + b_off + offs_m, mask=mrow, other=-1)
        s = tl.dot(q, tl.trans(k), input_precision=PREC)
        bias, valid = _bias_mask(offs_m, offs_n, Li, Si, Lg, Sg, b_off, S)
        p = tl.where(valid & mrow[:, None], tl.exp(s + bias - lse[:, None]), 0.0)
        dv += tl.dot(tl.trans(p), do, input_precision=PREC)
        dp = tl.dot(do, tl.trans(v), input_precision=PREC)
        ds = p * (dp - Di[:, None])
        dk += tl.dot(tl.trans(ds), q, input_precision=PREC)
        cs += tl.sum(ds, 0)
    tl.store(DK + base + offs_n[:, None] * D + offs_d[None, :], dk, mask=nmask[:, None])
    tl.store(DV + base + offs_n[:, None] * D + offs_d[None, :], dv, mask=nmask[:, None])
    tl.store(CS + rbase + offs_n, cs, mask=nmask)


@triton.jit
def _bwd_dq(Q, K, V, Lg, Sg, DO, LSE, Dl, DQ, RS, S, H,
            D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b_off = (bh // H) * S
    base = bh.to(tl.int64) * S * D
    rbase = bh.to(tl.int64) * S
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mrow = offs_m < S
    q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :], mask=mrow[:, None], other=0.0)
    do = tl.load(DO + base + offs_m[:, None] * D + offs_d[None, :], mask=mrow[:, None], other=0.0)
    lse = tl.load(LSE + rbase + offs_m, mask=mrow, other=0.0)
    Di = tl.load(Dl + rbase + offs_m, mask=mrow, other=0.0)
    Li = tl.load(Lg + 2 * (b_off + offs_m), mask=mrow, other=0.0)
    Si = tl.load(Sg + b_off + offs_m, mask=mrow, other=-1)
    dq = tl.zeros([BLOCK_M, D], tl.float32)
    rs = tl.zeros([BLOCK_M], tl.float32)
    hi = tl.minimum((pid_m + 1) * BLOCK_M, S)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < S
        k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :], mask=nmask[:, None], other=0.0)
        v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :], mask=nmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision=PREC)
        bias, valid = _bias_mask(offs_m, offs_n, Li, Si, Lg, Sg, b_off, S)
        p = tl.where(valid & mrow[:, None], tl.exp(s + bias - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v), input_precision=PREC)
        ds = p * (dp - Di[:, None])
        dq += tl.dot(ds, k, input_precision=PREC)
        rs += tl.sum(ds, 1)
    tl.store(DQ + base + offs_m[:, None] * D + offs_d[None, :], dq, mask=mrow[:, None])
    tl.store(RS + rbase + offs_m, rs, mask=mrow)


#: (BLOCK_M, BLOCK_N, num_warps, num_stages); set from tune.py on the 4060 at d_head 64.
FWD = (32, 64, 4, 1)
#: 'tf32x3' = three TF32 tensor-core passes (fp32-grade error); 'ieee' = SIMT fp32.
PREC = "tf32x3"
BWD = (32, 32, 4, 2)


class _BiasedAttn(torch.autograd.Function):
    """(q, k, v, L) -> (softmax(qk^T + L_i - L_j, segment/causal masked) v, LSE).
    q arrives pre-scaled; seg carries no gradient."""

    @staticmethod
    def forward(ctx, q, k, v, L, seg):
        B, H, S, D = q.shape
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        hi = L.float()
        L2 = torch.stack([hi, (L - hi.double()).float()], -1).contiguous()
        o = torch.empty_like(q)
        lse = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
        bm, bn, w, st = FWD
        _fwd[(triton.cdiv(S, bm), B * H)](q, k, v, L2, seg, o, lse, S, H, D=D, BLOCK_M=bm, BLOCK_N=bn,
                                           PREC=PREC, num_warps=w, num_stages=st)
        ctx.save_for_backward(q, k, v, L2, seg, o, lse)
        return o, lse

    @staticmethod
    def backward(ctx, do, dlse):
        q, k, v, L2, seg, o, lse = ctx.saved_tensors
        B, H, S, D = q.shape
        do = torch.zeros_like(o) if do is None else do.contiguous()
        Dl = (do * o).sum(-1)
        if dlse is not None:
            Dl = Dl - dlse
        Dl = Dl.contiguous()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        rs = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
        cs = torch.empty_like(rs)
        bm, bn, w, st = BWD
        _bwd_dkdv[(triton.cdiv(S, bn), B * H)](q, k, v, L2, seg, do, lse, Dl, dk, dv, cs, S, H,
                                                 D=D, BLOCK_M=bm, BLOCK_N=bn, PREC=PREC, num_warps=w, num_stages=st)
        _bwd_dq[(triton.cdiv(S, bm), B * H)](q, k, v, L2, seg, do, lse, Dl, dq, rs, S, H,
                                              D=D, BLOCK_M=bm, BLOCK_N=bn, PREC=PREC, num_warps=w, num_stages=st)
        dL = (rs - cs).sum(1).double()  # L is shared across heads
        return dq, dk, dv, dL, None


def flite_attention(q, k, v, u, beta, qk=1.0, g=1.0, magnitude=hard_concrete):
    """q, k, v: [B, H, S, D]; u: [B, S] gate logits (shared over heads). Returns [B, H, S, D].
    Same parametrisation as arm_smprime.readout(..., theta=0).real: m = magnitude(lerp(1, u, g)),
    s = qk * q.k / sqrt(D), O = sum_j R_ij e^{s_ij} v_j / Z_i^beta."""
    m = magnitude(torch.lerp(torch.ones_like(u), u, g))
    live = m > 0
    L = torch.cumsum(torch.log(torch.where(live, m, torch.ones_like(m))).double(), -1).contiguous()
    seg = torch.cumsum((~live).int(), -1).int().contiguous()
    qs = q * (qk / q.shape[-1] ** 0.5)
    o, lse = _BiasedAttn.apply(qs, k, v, L, seg)
    return o * torch.exp((1 - beta) * lse).unsqueeze(-1)


def flite_forward(self, x, attention_mask=None):
    """Drop-in for CEQAttention.forward on an operator='smprime' build: uses m_head, beta, qk, g;
    theta_head is left unused (phase off), so its parameters receive no gradient."""
    if attention_mask is not None:
        raise NotImplementedError("f-lite: no padding-mask path")
    b, s, d = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)

    def shape(t):
        return t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

    u = self.m_head(x).squeeze(-1)
    o = flite_attention(shape(q), shape(k), shape(v), u, self.beta, self.qk, self.g)
    return self.o_proj(o.transpose(1, 2).reshape(b, s, d))
