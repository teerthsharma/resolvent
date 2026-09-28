"""Resolvent attention read for arm (f_R): x = (1-g)(I - g W)^-1 V, W the causal head at beta = 1.
Logits z_ij = s_i q_i.k_j / sqrt(D) with the SSMax law s_i = a ln(i+1) + b (arXiv 2501.19399; causal n = i+1),
applied by pre-scaling q (ssmax_q), so a, b get gradients through ordinary autograd.
W = softmax(z) or sparsemax(z). Gradients by the adjoint: with A = I - gW, x = A^-1 (1-g)V,
  u = A^-T gx (transposed = upper triangular solve), dV = (1-g) u, dW = g u x^T,
then the ordinary softmax/sparsemax Jacobian. Nothing is unrolled.
Paths:
  dense : W materialized (any device/dtype; the float64 reference and the gradcheck path; softmax or sparsemax)
  fused : softmax only, CUDA fp32. Forward = Cameron's Phase J fs5 (tests/foreman/phase_j/N1/cameron/cost_n.py,
          coarse blocked forward substitution c=256, fused efficient-attention pushes), reused unmodified via its
          module globals. Backward = blocked right-looking adjoint solve (row-blocks of W materialized c x hi,
          upper trsm on the diagonal block) + ONE efficient-attention backward with value = x, out = W x,
          grad_out = g u, which yields dq, dk exactly (dW = g u x^T is the attention dP with dO = g u, V = x).
"""
import math, sys
import torch
from torch.autograd import Function

CAMERON = "C:/Users/seal/Desktop/New folder (32)/tests/foreman/phase_j/N1/cameron"


def ssmax_q(q, a, b):
    """q_i * (a ln(i+1) + b). a learned, b learned."""
    n = torch.arange(1, q.shape[-2] + 1, device=q.device, dtype=q.dtype)
    return q * (a * torch.log(n) + b)[:, None]


def _mask(S, dev):
    return torch.ones(S, S, dtype=torch.bool, device=dev).triu(1)


def _logits(qs, k, alibi=False):
    S, D = qs.shape[-2], qs.shape[-1]
    z = (qs @ k.transpose(-1, -2)) / math.sqrt(D)
    if alibi:
        H = qs.shape[-3]
        sl = torch.tensor([2.0 ** (-8.0 * h / H) for h in range(1, H + 1)], device=qs.device, dtype=qs.dtype)
        i = torch.arange(S, device=qs.device)
        z = z - sl[:, None, None] * (i[:, None] - i[None, :]).clamp(min=0).to(qs.dtype)
    return z.masked_fill(_mask(S, qs.device), float("-inf"))


def sparsemax(z):
    zs, _ = torch.sort(z, dim=-1, descending=True)
    kk = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    cs = zs.cumsum(-1)
    supp = (1 + kk * zs) > cs                                # -inf entries give False
    ksz = supp.sum(-1, keepdim=True)
    tau = (cs.gather(-1, ksz - 1) - 1) / ksz.to(z.dtype)
    return torch.clamp(z - tau, min=0)


def weights(qs, k, kind="softmax", alibi=False):
    z = _logits(qs, k, alibi)
    return torch.softmax(z, -1) if kind == "softmax" else sparsemax(z)


def diag_weights(qs, k, alibi=False):
    """W_ii for every row (the eigenvalues of the triangular W), memory-light: row blocks."""
    out = []
    S = qs.shape[-2]
    for lo in range(0, S, 512):
        hi = min(S, lo + 512)
        W = torch.softmax(_logits_rows(qs, k, lo, hi, alibi), -1)
        out.append(torch.diagonal(W[..., lo:hi], dim1=-2, dim2=-1))
    return torch.cat(out, -1)


def _logits_rows(qs, k, lo, hi, alibi=False):
    D, H = qs.shape[-1], qs.shape[-3]
    z = (qs[..., lo:hi, :] @ k[..., :hi, :].transpose(-1, -2)) / math.sqrt(D)
    i = torch.arange(lo, hi, device=qs.device)[:, None]
    j = torch.arange(hi, device=qs.device)[None, :]
    if alibi:
        sl = torch.tensor([2.0 ** (-8.0 * h / H) for h in range(1, H + 1)], device=qs.device, dtype=qs.dtype)
        z = z - sl[:, None, None] * (i - j).clamp(min=0).to(qs.dtype)
    return z.masked_fill(j > i, float("-inf"))


class _Dense(Function):
    @staticmethod
    def forward(ctx, qs, k, v, g, kind, pivot=False):
        W = weights(qs, k, kind)
        S = W.shape[-1]
        A = torch.eye(S, dtype=W.dtype, device=W.device) - g * W
        if pivot:   # 1 - g W_ii = (1-g) + g * sum_{j != i} W_ij (complementary mass, no cancellation near W_ii = 1)
            A.diagonal(dim1=-2, dim2=-1).copy_((1 - g) + g * W.masked_fill(torch.eye(S, dtype=torch.bool, device=W.device), 0.0).sum(-1))
        x = torch.linalg.solve_triangular(A, (1 - g) * v, upper=False)
        ctx.save_for_backward(qs, k, W, A, x)
        ctx.g, ctx.kind = g, kind
        return x

    @staticmethod
    def backward(ctx, gx):
        qs, k, W, A, x = ctx.saved_tensors
        g = ctx.g
        u = torch.linalg.solve_triangular(A.transpose(-1, -2), gx, upper=True)   # adjoint: A^T u = gx
        dW = g * (u @ x.transpose(-1, -2))
        if ctx.kind == "softmax":
            dz = W * (dW - (dW * W).sum(-1, keepdim=True))
        else:
            s = (W > 0).to(dW.dtype)
            dz = s * (dW - (dW * s).sum(-1, keepdim=True) / s.sum(-1, keepdim=True))
        D = qs.shape[-1]
        dz = dz / math.sqrt(D)
        return dz @ k, dz.transpose(-1, -2) @ qs, (1 - g) * u, None, None, None


_cn = {}


def _cost_n():
    if "m" not in _cn:
        sys.path.insert(0, CAMERON)
        import cost_n
        _cn["m"] = cost_n
    return _cn["m"]


def fused_forward(qs, k, v, g, c=256):
    cn = _cost_n()
    B, H, S, D = qs.shape
    cn.B, cn.H, cn.S, cn.D, cn.GAMMA, cn.SCALE = B, H, S, D, g, 1.0 / math.sqrt(D)
    return cn.fs5(qs, k, v, c)


def adjoint_blocked(qs, k, lse, gx, g, c=256):
    """u = (I - g W^T)^-1 gx, right-looking from the last row block. W_{J,:hi} = exp(z - lse_J)."""
    B, H, S, D = qs.shape
    U = torch.empty_like(gx)
    acc = torch.zeros_like(gx)
    tri = _mask(c, qs.device)
    eye = torch.eye(c, dtype=qs.dtype, device=qs.device)
    for J in reversed(range(S // c)):
        lo, hi = J * c, (J + 1) * c
        z = (qs[:, :, lo:hi] @ k[:, :, :hi].transpose(-1, -2)) / math.sqrt(D)
        z[..., lo:hi].masked_fill_(tri, float("-inf"))
        P = torch.exp(z - lse[:, :, lo:hi, None])
        AT = eye - g * P[..., lo:hi].transpose(-1, -2)
        U[:, :, lo:hi] = torch.linalg.solve_triangular(AT, gx[:, :, lo:hi] + g * acc[:, :, lo:hi], upper=True)
        if lo:
            acc[:, :, :lo] += P[..., :lo].transpose(-1, -2) @ U[:, :, lo:hi]
    return U


class _Fused(Function):
    @staticmethod
    def forward(ctx, qs, k, v, g):
        x = fused_forward(qs, k, v, g)
        _, lse, seed, off = torch.ops.aten._scaled_dot_product_efficient_attention(qs, k, v, None, True, 0.0, True)
        ctx.save_for_backward(qs, k, v, x, lse, seed, off)
        ctx.g = g
        return x

    @staticmethod
    def backward(ctx, gx):
        qs, k, v, x, lse, seed, off = ctx.saved_tensors
        g = ctx.g
        gx = gx.contiguous()
        u = adjoint_blocked(qs, k, lse[..., :qs.shape[-2]], gx, g)
        out = (x - (1 - g) * v) / g                              # = W x, the attention output on values x
        dq, dk, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
            (g * u).contiguous(), qs, k, x, None, out, lse, seed, off, 0.0, [True, True, False, False], True)
        return dq, dk, (1 - g) * u, None


def fs5c(qs, k, v, g, c=256):
    """Cameron's fs5 (cost_n.py, lines 244-293) copied with ONE change: the diagonal pivot of the c x c block is
    (1-g) + g * exp(l_off - l_tot) (the complementary, off-diagonal row mass) instead of 1 - g*exp(s_ii - l_tot),
    so a near-unit self-weight (pivot -> 1-g) keeps its relative accuracy. Everything else as fs5."""
    cn = _cost_n()
    B, H, S, D = qs.shape
    cn.B, cn.H, cn.S, cn.D, cn.GAMMA, cn.SCALE = B, H, S, D, g, 1.0 / math.sqrt(D)
    _eff, trsm9, dev = cn._eff, cn.trsm9, qs.device
    X = torch.empty_like(v)
    acc_o = torch.zeros_like(v)
    acc_l = torch.full(v.shape[:-1], float("-inf"), dtype=v.dtype, device=dev)
    tri = torch.ones(c, c, dtype=torch.bool, device=dev).triu(1)
    dmask = torch.eye(c, dtype=torch.bool, device=dev)
    eye = torch.eye(c, dtype=v.dtype, device=dev)
    main = torch.cuda.current_stream()
    side = cn._side.setdefault("s", torch.cuda.Stream())
    side.wait_stream(main)
    nb = S // c
    done_bulk = [None] * nb
    for C in range(nb):
        lo, hi = C * c, (C + 1) * c
        if C >= 2:
            main.wait_event(done_bulk[C - 2])
        sCC = ((qs[:, :, lo:hi] @ k[:, :, lo:hi].transpose(-1, -2)) / math.sqrt(D)).masked_fill_(tri, float("-inf"))
        l_far, o_far = acc_l[:, :, lo:hi], acc_o[:, :, lo:hi]
        l_near_off = torch.logsumexp(sCC.masked_fill(dmask, float("-inf")), -1)
        l_off = torch.logaddexp(l_far, l_near_off)
        l_tot = torch.logaddexp(l_far, torch.logsumexp(sCC, -1))
        if C >= 1:
            o_u, l_u = _eff(qs[:, :, lo:hi], k[:, :, lo - c:lo], X[:, :, lo - c:lo])
            l_tot = torch.logaddexp(l_tot, l_u)
            l_off = torch.logaddexp(l_off, l_u)
            off = torch.exp(l_far - l_tot).unsqueeze(-1) * o_far + torch.exp(l_u - l_tot).unsqueeze(-1) * o_u
        else:
            off = torch.exp(l_far - l_tot).unsqueeze(-1) * o_far
        rhs = (1 - g) * v[:, :, lo:hi] + g * off
        A = eye - g * torch.exp(sCC - l_tot.unsqueeze(-1))
        A.diagonal(dim1=-2, dim2=-1).copy_((1 - g) + g * torch.exp(l_off - l_tot))
        X[:, :, lo:hi] = trsm9(A, rhs)
        top = hi + c
        if top < S:
            ev = torch.cuda.Event(); ev.record(main)
            side.wait_event(ev)
            with torch.cuda.stream(side):
                o_f, l_f = _eff(qs[:, :, top:], k[:, :, lo:hi], X[:, :, lo:hi])
                la = acc_l[:, :, top:]
                ln = torch.logaddexp(la, l_f)
                acc_o[:, :, top:] = (torch.exp(la - ln).unsqueeze(-1) * acc_o[:, :, top:]
                                     + torch.exp(l_f - ln).unsqueeze(-1) * o_f)
                acc_l[:, :, top:] = ln
                done_bulk[C] = torch.cuda.Event(); done_bulk[C].record(side)
    main.wait_stream(side)
    return X


def backward_blocked(qs, k, x, gx, g, c=256):
    """Exact materialized adjoint + gradient, one reverse pass over row blocks J (c rows, keys <= hi):
    P = W_{J,:hi} = softmax over the materialized row block (max-shifted); pivot of the transposed diagonal block by the complementary mass;
    u_J = (I - g W_JJ^T)^-1 (gx_J + g acc_J); acc_{<lo} += W_{J,<lo}^T u_J;
    dP = g u_J x_{:hi}^T; dz = P * (dP - rowsum(P*dP)); dq_J = dz k / sqrt(D); dk_{:hi} += dz^T qs_J / sqrt(D)."""
    B, H, S, D = qs.shape
    sc = 1.0 / math.sqrt(D)
    U = torch.empty_like(gx); acc = torch.zeros_like(gx)
    dq = torch.empty_like(qs); dk = torch.zeros_like(k)
    tri = _mask(c, qs.device)
    dmask = torch.eye(c, dtype=torch.bool, device=qs.device)
    eye = torch.eye(c, dtype=qs.dtype, device=qs.device)
    for J in reversed(range(S // c)):
        lo, hi = J * c, (J + 1) * c
        z = (qs[:, :, lo:hi] @ k[:, :, :hi].transpose(-1, -2)) * sc
        z[..., lo:hi].masked_fill_(tri, float("-inf"))
        P = torch.softmax(z, -1)          # v2: max-shifted normalization (v1 used exp(z - lse): lse ulp ~4e-6 at |z|~37)
        comp = P[..., :lo].sum(-1) + P[..., lo:hi].masked_fill(dmask, 0.0).sum(-1)   # 1 - W_ii as the off-diagonal sum
        AT = eye - g * P[..., lo:hi].transpose(-1, -2)
        AT.diagonal(dim1=-2, dim2=-1).copy_((1 - g) + g * comp)
        uJ = torch.linalg.solve_triangular(AT, gx[:, :, lo:hi] + g * acc[:, :, lo:hi], upper=True)
        U[:, :, lo:hi] = uJ
        if lo:
            acc[:, :, :lo] += P[..., :lo].transpose(-1, -2) @ uJ
        dP = g * (uJ @ x[:, :, :hi].transpose(-1, -2))
        dz = P * (dP - (P * dP).sum(-1, keepdim=True))
        dq[:, :, lo:hi] = (dz @ k[:, :, :hi]) * sc
        dk[:, :, :hi] += (dz.transpose(-1, -2) @ qs[:, :, lo:hi]) * sc
    return dq, dk, (1 - g) * U


class _FusedC(Function):
    @staticmethod
    def forward(ctx, qs, k, v, g):
        x = fs5c(qs, k, v, g)
        ctx.save_for_backward(qs, k, x)
        ctx.g = g
        return x

    @staticmethod
    def backward(ctx, gx):
        qs, k, x = ctx.saved_tensors
        dq, dk, dv = backward_blocked(qs, k, x, gx.contiguous(), ctx.g)
        return dq, dk, dv, None


_graphs = {}


def _graph_call(key, fn, *inputs):
    """CUDA-graph replay with static input buffers (training-legal: inputs are copied in, outputs cloned out).
    The eager fs5c/backward_blocked are launch-bound on this box (CPU launch time == GPU time, diag_cost.py)."""
    if key not in _graphs:
        st = [t.detach().clone() for t in inputs]
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn(*st)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            out = fn(*st)
        _graphs[key] = (gr, st, out if isinstance(out, tuple) else (out,))
    gr, st, out = _graphs[key]
    for a, b in zip(st, inputs):
        a.copy_(b)
    gr.replay()
    return tuple(o.clone() for o in out)


class _FusedCG(Function):
    @staticmethod
    def forward(ctx, qs, k, v, g):
        (x,) = _graph_call(("f", tuple(qs.shape), g), lambda a, b, c: fs5c(a, b, c, g), qs, k, v)
        ctx.save_for_backward(qs, k, x)
        ctx.g = g
        return x

    @staticmethod
    def backward(ctx, gx):
        qs, k, x = ctx.saved_tensors
        g = ctx.g
        dq, dk, dv = _graph_call(("b", tuple(qs.shape), g), lambda a, b, c, d: backward_blocked(a, b, c, d, g),
                                 qs, k, x, gx.contiguous())
        return dq, dk, dv, None


def resolvent(qs, k, v, g, kind="softmax", path="dense"):
    if path == "fusedcg":
        assert kind == "softmax" and qs.is_cuda and qs.dtype == torch.float32
        return _FusedCG.apply(qs.contiguous(), k.contiguous(), v.contiguous(), g)
    if path == "fusedc":
        assert kind == "softmax" and qs.is_cuda and qs.dtype == torch.float32
        return _FusedC.apply(qs.contiguous(), k.contiguous(), v.contiguous(), g)
    if path == "fused":
        assert kind == "softmax" and qs.is_cuda and qs.dtype == torch.float32
        return _Fused.apply(qs.contiguous(), k.contiguous(), v.contiguous(), g)
    return _Dense.apply(qs, k, v, g, kind, path == "densec")
