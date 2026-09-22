"""R-COST-N: causal resolvent read r = (1-g)(I - gW)^-1 V at B1 H8 S4096 D64 fp32,
W = causal softmax(q k^T / sqrt(D)) from random q, k (numpy default_rng seed).
Arms are timed against ONE SDPA pass of the same shape (default dispatch).
Usage: python cost_n.py <impl> [<impl> ...]  -> prints one JSON line per impl.
impls: hop_cheb (Golub-Varga Chebyshev semi-iteration, K=130, each hop = one SDPA),
       hop_neumann (K=130 hops, each hop = one SDPA),
       fs_block<b> (blocked forward substitution, block b), fs_dense (one trsm).
"""
import json, sys, math, re
import numpy as np
import torch
import torch.nn.functional as F

B, H, S, D = 1, 8, 4096, 64
GAMMA = 0.99
K_CHEB = 130          # contract §4 C4 / PHASE_J C11: degree 130 at gamma = 0.99
SCALE = 1.0 / math.sqrt(D)
dev = "cuda"


def inputs(seed=0):
    rng = np.random.default_rng(seed)
    q, k, v = (torch.from_numpy(rng.standard_normal((B, H, S, D), dtype=np.float32)).to(dev)
               for _ in range(3))
    return q, k, v


def reference_f64(q, k, v):
    """float64 dense causal solve, one head at a time (GPU fp64)."""
    out = torch.empty(B, H, S, D, dtype=torch.float64, device=dev)
    mask = torch.ones(S, S, dtype=torch.bool, device=dev).triu(1)
    eye = torch.eye(S, dtype=torch.float64, device=dev)
    for h in range(H):
        s = (q[0, h].double() @ k[0, h].double().T) * SCALE
        W = torch.softmax(s.masked_fill(mask, float("-inf")), -1)
        out[0, h] = torch.linalg.solve_triangular(eye - GAMMA * W, (1 - GAMMA) * v[0, h].double(),
                                                  upper=False)
        del s, W
    return out


def sdpa(q, k, x):
    return F.scaled_dot_product_attention(q, k, x, is_causal=True)


def hop_neumann(q, k, v, K=K_CHEB):
    b = (1 - GAMMA) * v
    x = b
    for _ in range(K):
        x = b + GAMMA * sdpa(q, k, x)
    return x


def hop_cheb(q, k, v, K=K_CHEB):
    """Chebyshev semi-iteration (Golub-Varga) for x = gW x + b, spectrum of gW in [-g, g]."""
    b = (1 - GAMMA) * v
    rho2 = GAMMA * GAMMA
    x_prev = b
    x = b + GAMMA * sdpa(q, k, b)          # step 1, omega_1 = 1
    omega = 2.0 / (2.0 - rho2)             # omega_2
    for _ in range(K - 1):
        y = b + GAMMA * sdpa(q, k, x)
        x, x_prev = omega * (y - x_prev) + x_prev, x
        omega = 1.0 / (1.0 - rho2 * omega / 4.0)
    return x


def fs_block(q, k, v, bs):
    """Blocked forward substitution on (I - gW) x = (1-g) v.
    Step I materializes the softmax rows of block I over keys <= block end (full row
    is available, so W is exactly normalized), then solves the b x b lower-triangular
    diagonal block. Sequential steps: S / bs."""
    X = torch.empty_like(v)
    tri = torch.ones(bs, bs, dtype=torch.bool, device=dev).triu(1)
    eye = torch.eye(bs, dtype=v.dtype, device=dev)
    for I in range(S // bs):
        lo, hi = I * bs, (I + 1) * bs
        s = (q[:, :, lo:hi] @ k[:, :, :hi].transpose(-1, -2)) * SCALE
        s[..., lo:hi].masked_fill_(tri, float("-inf"))
        P = torch.softmax(s, -1)
        rhs = (1 - GAMMA) * v[:, :, lo:hi]
        if I:
            rhs = torch.baddbmm(rhs.view(B * H, bs, D), P[..., :lo].reshape(B * H, bs, lo),
                                X[:, :, :lo].reshape(B * H, lo, D), alpha=GAMMA).view(B, H, bs, D)
        X[:, :, lo:hi] = torch.linalg.solve_triangular(eye - GAMMA * P[..., lo:hi], rhs, upper=False)
    return X


def fs_sdpa(q, k, v, bs):
    """Blocked forward substitution, left-looking, W never materialized off the diagonal.
    Step I: the fused efficient-attention kernel over keys < block I returns
    out_off = softmax_off @ X and lse_off; the b x b diagonal block is explicit.
    Full-row normalization: lse = logaddexp(lse_off, lse_II), so
    W_{I,<I} X = exp(lse_off - lse) * out_off and W_II = exp(s_II - lse)."""
    X = torch.empty_like(v)
    tri = torch.ones(bs, bs, dtype=torch.bool, device=dev).triu(1)
    eye = torch.eye(bs, dtype=v.dtype, device=dev)
    for I in range(S // bs):
        lo, hi = I * bs, (I + 1) * bs
        qI = q[:, :, lo:hi]
        sII = ((qI @ k[:, :, lo:hi].transpose(-1, -2)) * SCALE).masked_fill_(tri, float("-inf"))
        lse = torch.logsumexp(sII, -1)
        rhs = (1 - GAMMA) * v[:, :, lo:hi]
        if I:
            out, lse_off = torch.ops.aten._scaled_dot_product_efficient_attention(
                qI, k[:, :, :lo], X[:, :, :lo], None, True, 0.0, False)[:2]
            lse_off = lse_off[..., :bs]
            lse = torch.logaddexp(lse_off, lse)
            rhs = rhs + GAMMA * torch.exp(lse_off - lse).unsqueeze(-1) * out
        PII = torch.exp(sII - lse.unsqueeze(-1))
        X[:, :, lo:hi] = torch.linalg.solve_triangular(eye - GAMMA * PII, rhs, upper=False)
    return X


def _eff(q, k, v):
    o, l = torch.ops.aten._scaled_dot_product_efficient_attention(q, k, v, None, True, 0.0, False)[:2]
    return o, l[..., :q.shape[-2]]


def tri_solve_doubling(A, rhs):
    """Exact lower-triangular solve with no sequential trsm: A = Dg (I - N), N strictly
    lower and >= 0 elementwise here, so (I - N)^-1 = prod_k (I + N^(2^k)) over
    ceil(log2 b) factors (N is nilpotent of index <= b; exact, no cancellation since N >= 0)."""
    d = torch.diagonal(A, dim1=-2, dim2=-1)
    N = -torch.tril(A, -1) / d.unsqueeze(-1)
    y = rhs / d.unsqueeze(-1)
    for _ in range(max(1, (A.shape[-1] - 1).bit_length())):
        y = y + N @ y
        N = N @ N
    return y


SOLVE_TRSM = True


def fs2(q, k, v, c, bs):
    """Two-level forward substitution. Coarse blocks of c rows; after coarse block C is
    solved, ONE fused efficient-attention call (queries = every later row, keys = block C)
    pushes its contribution to all later rows as a (partial out, partial lse) pair, merged
    flash-decoding style. Inside a coarse block, fine steps of bs rows are left-looking
    over keys in the same coarse block only. Sequential fine steps: S / bs; every key
    block is visited once, so total work is one causal attention pass plus the
    bs x bs diagonal solves."""
    X = torch.empty_like(v)
    acc_o = torch.zeros_like(v)
    acc_l = torch.full(v.shape[:-1], float("-inf"), dtype=v.dtype, device=dev)
    tri = torch.ones(bs, bs, dtype=torch.bool, device=dev).triu(1)
    eye = torch.eye(bs, dtype=v.dtype, device=dev)
    for C in range(S // c):
        base = C * c
        for f in range(c // bs):
            lo, hi = base + f * bs, base + (f + 1) * bs
            qI = q[:, :, lo:hi]
            sII = ((qI @ k[:, :, lo:hi].transpose(-1, -2)) * SCALE).masked_fill_(tri, float("-inf"))
            l_far, o_far = acc_l[:, :, lo:hi], acc_o[:, :, lo:hi]
            l_tot = torch.logaddexp(l_far, torch.logsumexp(sII, -1))
            off = torch.exp(l_far - l_tot).unsqueeze(-1) * o_far
            if f:
                o_n, l_n = _eff(qI, k[:, :, base:lo], X[:, :, base:lo])
                l_tot = torch.logaddexp(l_tot, l_n)
                off = torch.exp(l_far - l_tot).unsqueeze(-1) * o_far + torch.exp(l_n - l_tot).unsqueeze(-1) * o_n
            rhs = (1 - GAMMA) * v[:, :, lo:hi] + GAMMA * off
            PII = torch.exp(sII - l_tot.unsqueeze(-1))
            A = eye - GAMMA * PII
            X[:, :, lo:hi] = (torch.linalg.solve_triangular(A, rhs, upper=False) if SOLVE_TRSM
                              else tri_solve_doubling(A, rhs))
        top = base + c
        if top < S:
            o_f, l_f = _eff(q[:, :, top:], k[:, :, base:top], X[:, :, base:top])
            la = acc_l[:, :, top:]
            ln = torch.logaddexp(la, l_f)
            acc_o[:, :, top:] = (torch.exp(la - ln).unsqueeze(-1) * acc_o[:, :, top:]
                                 + torch.exp(l_f - ln).unsqueeze(-1) * o_f)
            acc_l[:, :, top:] = ln
    return X


def fs4(q, k, v, c):
    """Coarse-only forward substitution. A row's log-normalizer never depends on X, so
    for coarse block C the full-row lse is known before C is solved: far part from the
    merged right-looking pushes, near part from the explicit c x c causal block. Then
    (I - g W_CC) X_C = (1-g) V_C + g W_{C,<C} X_{<C} is ONE c x c triangular solve.
    Sequential steps: S / c; each is one explicit c x c block + trsm + one fused push."""
    X = torch.empty_like(v)
    acc_o = torch.zeros_like(v)
    acc_l = torch.full(v.shape[:-1], float("-inf"), dtype=v.dtype, device=dev)
    tri = torch.ones(c, c, dtype=torch.bool, device=dev).triu(1)
    eye = torch.eye(c, dtype=v.dtype, device=dev)
    for C in range(S // c):
        lo, hi = C * c, (C + 1) * c
        sCC = ((q[:, :, lo:hi] @ k[:, :, lo:hi].transpose(-1, -2)) * SCALE).masked_fill_(tri, float("-inf"))
        l_far = acc_l[:, :, lo:hi]
        l_tot = torch.logaddexp(l_far, torch.logsumexp(sCC, -1))
        rhs = (1 - GAMMA) * v[:, :, lo:hi] + GAMMA * torch.exp(l_far - l_tot).unsqueeze(-1) * acc_o[:, :, lo:hi]
        A = eye - GAMMA * torch.exp(sCC - l_tot.unsqueeze(-1))
        X[:, :, lo:hi] = torch.linalg.solve_triangular(A, rhs, upper=False)
        if hi < S:
            o_f, l_f = _eff(q[:, :, hi:], k[:, :, lo:hi], X[:, :, lo:hi])
            la = acc_l[:, :, hi:]
            ln = torch.logaddexp(la, l_f)
            acc_o[:, :, hi:] = (torch.exp(la - ln).unsqueeze(-1) * acc_o[:, :, hi:]
                                + torch.exp(l_f - ln).unsqueeze(-1) * o_f)
            acc_l[:, :, hi:] = ln
    return X


def trsm9(A, rhs):
    """solve_triangular with the batch padded 8 -> 9 by one identity system: this torch
    build loops per-matrix trsm at batch <= 8 and calls the batched cuBLAS path above it
    (measured 0.908 -> 0.516 ms at 512 x 512 x 64). Pure dispatch, same arithmetic."""
    Bh, c = A.shape[0] * A.shape[1], A.shape[-1]
    Ap = torch.cat([A.reshape(Bh, c, c), torch.eye(c, dtype=A.dtype, device=A.device)[None]])
    rp = torch.cat([rhs.reshape(Bh, c, rhs.shape[-1]), rhs.new_zeros(1, c, rhs.shape[-1])])
    return torch.linalg.solve_triangular(Ap, rp, upper=False)[:Bh].view_as(rhs)


def tri_solve_blocked(A, rhs, sb=64):
    """Exact lower-triangular solve without trsm. The c/sb diagonal sub-blocks are
    inverted all at once (batched) by the doubling product (I - N)^-1 = prod_k (I + N^(2^k)),
    N = strictly-lower part scaled by the diagonal (N >= 0 here: no cancellation);
    then c/sb sequential GEMM steps. Every op is a batched GEMM or elementwise."""
    c = A.shape[-1]; m = c // sb
    Ad = torch.stack([A[..., i * sb:(i + 1) * sb, i * sb:(i + 1) * sb] for i in range(m)], dim=-3)
    d = torch.diagonal(Ad, dim1=-2, dim2=-1)
    N = -torch.tril(Ad, -1) / d.unsqueeze(-1)
    Minv = N + torch.eye(sb, dtype=A.dtype, device=A.device)
    Nk = N
    for _ in range((sb - 1).bit_length() - 1):
        Nk = Nk @ Nk
        Minv = Minv + Minv @ Nk
    Minv = Minv / d.unsqueeze(-2)
    X = torch.empty_like(rhs)
    for i in range(m):
        r = rhs[..., i * sb:(i + 1) * sb, :]
        if i:
            r = r - A[..., i * sb:(i + 1) * sb, :i * sb] @ X[..., :i * sb, :]
        X[..., i * sb:(i + 1) * sb, :] = Minv[..., i, :, :] @ r
    return X


DIAG_SOLVE = {"f": None}


def fs5(q, k, v, c):
    """fs4 with one-block lookahead on two streams. Step C (main stream) reads the bulk
    accumulator (pushes from blocks <= C-2), makes the urgent push from block C-1 inline,
    builds the c x c diagonal block and solves it; the bulk push of block C to rows
    >= (C+2)c runs on a side stream concurrently with step C+1's solve."""
    X = torch.empty_like(v)
    acc_o = torch.zeros_like(v)
    acc_l = torch.full(v.shape[:-1], float("-inf"), dtype=v.dtype, device=dev)
    tri = torch.ones(c, c, dtype=torch.bool, device=dev).triu(1)
    eye = torch.eye(c, dtype=v.dtype, device=dev)
    main = torch.cuda.current_stream()
    side = _side.setdefault("s", torch.cuda.Stream())
    side.wait_stream(main)
    nb = S // c
    done_bulk = [None] * nb
    for C in range(nb):
        lo, hi = C * c, (C + 1) * c
        if C >= 2:
            main.wait_event(done_bulk[C - 2])
        sCC = ((q[:, :, lo:hi] @ k[:, :, lo:hi].transpose(-1, -2)) * SCALE).masked_fill_(tri, float("-inf"))
        l_far, o_far = acc_l[:, :, lo:hi], acc_o[:, :, lo:hi]
        l_tot = torch.logaddexp(l_far, torch.logsumexp(sCC, -1))
        if C >= 1:
            o_u, l_u = _eff(q[:, :, lo:hi], k[:, :, lo - c:lo], X[:, :, lo - c:lo])
            l_tot = torch.logaddexp(l_tot, l_u)
            off = torch.exp(l_far - l_tot).unsqueeze(-1) * o_far + torch.exp(l_u - l_tot).unsqueeze(-1) * o_u
        else:
            off = torch.exp(l_far - l_tot).unsqueeze(-1) * o_far
        rhs = (1 - GAMMA) * v[:, :, lo:hi] + GAMMA * off
        A = eye - GAMMA * torch.exp(sCC - l_tot.unsqueeze(-1))
        X[:, :, lo:hi] = trsm9(A, rhs) if DIAG_SOLVE["f"] is None else DIAG_SOLVE["f"](A, rhs)
        top = hi + c
        if top < S:
            ev = torch.cuda.Event(); ev.record(main)
            side.wait_event(ev)
            with torch.cuda.stream(side):
                o_f, l_f = _eff(q[:, :, top:], k[:, :, lo:hi], X[:, :, lo:hi])
                la = acc_l[:, :, top:]
                ln = torch.logaddexp(la, l_f)
                acc_o[:, :, top:] = (torch.exp(la - ln).unsqueeze(-1) * acc_o[:, :, top:]
                                     + torch.exp(l_f - ln).unsqueeze(-1) * o_f)
                acc_l[:, :, top:] = ln
                done_bulk[C] = torch.cuda.Event(); done_bulk[C].record(side)
    main.wait_stream(side)
    return X


def fs7(q, k, v, c, Wn=2):
    """Inverse-ahead forward substitution. A row's normalizer never depends on X, so the
    diagonal-block matrix A_C = I - g W_CC and its explicit inverse are X-independent once
    the far lse of block C is final. Three streams:
      inv : after the bulk push of block C-Wn-1 lands, build A_C, invert it (batched trsm
            against I), and pre-form base_C = (1-g) V_C + g W_{C,far} X_far;
      main: the only X-dependent chain: one fused attention call over the Wn previous
            blocks, one c x c GEMM with A_C^-1;
      side: bulk push of block C to rows >= (C+Wn+1)c (flash-decoding merge)."""
    nb = S // c
    X = torch.empty_like(v)
    acc_o = torch.zeros_like(v)
    acc_l = torch.full(v.shape[:-1], float("-inf"), dtype=v.dtype, device=dev)
    base = torch.empty_like(v)
    w_u = torch.zeros(v.shape[:-1], dtype=v.dtype, device=dev)
    Ainv = torch.empty(B, H, nb, c, c, dtype=v.dtype, device=dev)
    tri = torch.ones(c, c, dtype=torch.bool, device=dev).triu(1)
    eye = torch.eye(c, dtype=v.dtype, device=dev)
    eyeB = eye.expand(B, H, c, c)
    sdiag = ((q.view(B, H, nb, c, D) @ k.view(B, H, nb, c, D).transpose(-1, -2)) * SCALE).masked_fill_(tri, float("-inf"))
    l_near = torch.logsumexp(sdiag, -1)
    main = torch.cuda.current_stream()
    side = _side.setdefault("s", torch.cuda.Stream())
    inv = _side.setdefault("i", torch.cuda.Stream())
    side.wait_stream(main); inv.wait_stream(main)
    done_bulk, ev_inv = [None] * nb, [None] * nb
    ninf = torch.full((B, H, c), float("-inf"), dtype=v.dtype, device=dev)

    def launch_inv(C):
        lo, hi = C * c, (C + 1) * c
        if C - Wn - 1 >= 0:
            inv.wait_event(done_bulk[C - Wn - 1])
        with torch.cuda.stream(inv):
            wlo = max(0, lo - Wn * c)
            l_u = (torch.logsumexp((q[:, :, lo:hi] @ k[:, :, wlo:lo].transpose(-1, -2)) * SCALE, -1)
                   if C else ninf)
            l_far = acc_l[:, :, lo:hi]
            l_tot = torch.logaddexp(torch.logaddexp(l_far, l_u), l_near[:, :, C])
            A = eye - GAMMA * torch.exp(sdiag[:, :, C] - l_tot.unsqueeze(-1))
            Ainv[:, :, C] = trsm9(A, eyeB)
            base[:, :, lo:hi] = (1 - GAMMA) * v[:, :, lo:hi] + GAMMA * torch.exp(l_far - l_tot).unsqueeze(-1) * acc_o[:, :, lo:hi]
            w_u[:, :, lo:hi] = GAMMA * torch.exp(l_u - l_tot)
            ev_inv[C] = torch.cuda.Event(); ev_inv[C].record(inv)

    for C in range(min(nb, Wn + 1)):
        launch_inv(C)
    for C in range(nb):
        lo, hi = C * c, (C + 1) * c
        main.wait_event(ev_inv[C])
        rhs = base[:, :, lo:hi]
        if C:
            wlo = max(0, lo - Wn * c)
            o_u, _ = _eff(q[:, :, lo:hi], k[:, :, wlo:lo], X[:, :, wlo:lo])
            rhs = rhs + w_u[:, :, lo:hi].unsqueeze(-1) * o_u
        X[:, :, lo:hi] = Ainv[:, :, C] @ rhs
        top = hi + Wn * c
        if top < S:
            ev = torch.cuda.Event(); ev.record(main)
            side.wait_event(ev)
            with torch.cuda.stream(side):
                o_f, l_f = _eff(q[:, :, top:], k[:, :, lo:hi], X[:, :, lo:hi])
                la = acc_l[:, :, top:]
                ln = torch.logaddexp(la, l_f)
                acc_o[:, :, top:] = (torch.exp(la - ln).unsqueeze(-1) * acc_o[:, :, top:]
                                     + torch.exp(l_f - ln).unsqueeze(-1) * o_f)
                acc_l[:, :, top:] = ln
                done_bulk[C] = torch.cuda.Event(); done_bulk[C].record(side)
            if C + Wn + 1 < nb:
                launch_inv(C + Wn + 1)
    main.wait_stream(side); main.wait_stream(inv)
    return X


_side = {}
_graphs = {}


def graphed(fn, q, k, v, *a):
    key = (fn.__name__, a, q.data_ptr(), k.data_ptr(), v.data_ptr())
    if key not in _graphs:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn(q, k, v, *a)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn(q, k, v, *a)
        _graphs[key] = (g, out)
    g, out = _graphs[key]
    g.replay()
    return out


def fs_sdpa_graph(q, k, v, bs):
    """fs_sdpa captured once as a CUDA graph (removes the S/bs x ~10 launch overheads).
    Replays read the static q, k, v it was captured on (same tensors in this bench)."""
    key = (bs, q.data_ptr(), k.data_ptr(), v.data_ptr())
    if key not in _graphs:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fs_sdpa(q, k, v, bs)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fs_sdpa(q, k, v, bs)
        _graphs[key] = (g, out)
    g, out = _graphs[key]
    g.replay()
    return out


def fs_dense(q, k, v):
    s = (q @ k.transpose(-1, -2)) * SCALE
    s.masked_fill_(torch.ones(S, S, dtype=torch.bool, device=dev).triu(1), float("-inf"))
    W = torch.softmax(s, -1)
    del s
    A = torch.eye(S, device=dev) - GAMMA * W
    del W
    return torch.linalg.solve_triangular(A, (1 - GAMMA) * v, upper=False)


def get_impl(name):
    if name == "hop_cheb":
        return hop_cheb
    if name == "hop_neumann":
        return hop_neumann
    if name == "fs_dense":
        return fs_dense
    m = re.fullmatch(r"fs2(g?)_c(\d+)_b(\d+)", name)      # e.g. fs2g_c1024_b256 (g = CUDA graph)
    m7 = re.fullmatch(r"fs7(g?)_c(\d+)_w(\d+)", name)    # inverse-ahead, 3 streams
    if m7:
        c, wn = int(m7.group(2)), int(m7.group(3))
        if m7.group(1):
            return lambda q, k, v: graphed(fs7, q, k, v, c, wn)
        return lambda q, k, v: fs7(q, k, v, c, wn)
    m6 = re.fullmatch(r"fs6g_c(\d+)_s(\d+)", name)       # fs5 graphed + blocked doubling diag solve
    if m6:
        c, sb = int(m6.group(1)), int(m6.group(2))
        DIAG_SOLVE["f"] = lambda A, r: tri_solve_blocked(A, r, sb)
        return lambda q, k, v: graphed(fs5, q, k, v, c)
    m5 = re.fullmatch(r"fs5(g?)_c(\d+)", name)            # lookahead two-stream, graph if g
    if m5:
        c = int(m5.group(2))
        if m5.group(1):
            return lambda q, k, v: graphed(fs5, q, k, v, c)
        return lambda q, k, v: fs5(q, k, v, c)
    m4 = re.fullmatch(r"fs4(g?)_c(\d+)", name)            # coarse-only, graph if g
    if m4:
        c = int(m4.group(2))
        if m4.group(1):
            return lambda q, k, v: graphed(fs4, q, k, v, c)
        return lambda q, k, v: fs4(q, k, v, c)
    m3 = re.fullmatch(r"fs3(g?)_c(\d+)_b(\d+)", name)     # fs2 with the doubling diagonal solve
    if m3:
        global SOLVE_TRSM
        SOLVE_TRSM = False
        m = m3
    if m:
        c, bs = int(m.group(2)), int(m.group(3))
        if m.group(1):
            return lambda q, k, v: graphed(fs2, q, k, v, c, bs)
        return lambda q, k, v: fs2(q, k, v, c, bs)
    for pre, fn in (("fs_sdpa_graph", fs_sdpa_graph), ("fs_sdpa", fs_sdpa), ("fs_block", fs_block)):
        if name.startswith(pre):
            bs = int(name[len(pre):])
            return lambda q, k, v, fn=fn, bs=bs: fn(q, k, v, bs)
    raise SystemExit("unknown impl " + name)


def sdpa_backend(q, k, v):
    """Which backend does default dispatch pick? Probe by disabling each in turn."""
    from torch.nn.attention import sdpa_kernel, SDPBackend
    ok = {}
    for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
        try:
            with sdpa_kernel([be]):
                sdpa(q, k, v)
            ok[be.name] = True
        except RuntimeError:
            ok[be.name] = False
    # default dispatch priority is flash > efficient > math; report the first that runs
    for nm in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH"):
        if ok[nm]:
            return nm, ok
    return None, ok


def time_arms(fns, reps, q, k, v):
    """Alternate arms inside one loop so clock drift cancels; CUDA events."""
    for f in fns.values():
        f(q, k, v)
    torch.cuda.synchronize()
    ts = {n: [] for n in fns}
    for _ in range(reps):
        for n, f in fns.items():
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); f(q, k, v); b.record(); torch.cuda.synchronize()
            ts[n].append(a.elapsed_time(b))
    return {n: float(np.median(t)) for n, t in ts.items()}, ts


def main(names, reps=15):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    q, k, v = inputs(0)
    backend, avail = sdpa_backend(q, k, v)
    ref = reference_f64(q, k, v)
    rows = []
    for name in names:
        f = get_impl(name)
        with torch.no_grad():
            x = f(q, k, v)
            torch.cuda.synchronize()
            err = (x.double() - ref).abs().max().item()
            rel = err / ref.abs().max().item()
            r = 3 if name.startswith("hop") else reps
            med, raw = time_arms({"sdpa": lambda q, k, v: sdpa(q, k, v), name: f}, r, q, k, v)
        row = dict(impl=name, S=S, B=B, H=H, D=D, gamma=GAMMA, dtype="fp32", tf32=False,
                   sdpa_backend_default=backend, sdpa_backends_available=avail,
                   ms_sdpa=med["sdpa"], ms_impl=med[name], ratio=med[name] / med["sdpa"],
                   max_abs_err_vs_f64=err, rel_err_vs_f64=rel, reps=r,
                   sequential_steps=(K_CHEB if name.startswith("hop") else
                                     1 if name == "fs_dense" else S // int(re.findall(r'\d+', name)[-1])),
                   torch=torch.__version__, gpu=torch.cuda.get_device_name())
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


if __name__ == "__main__":
    main(sys.argv[1:] or ["hop_cheb"])
