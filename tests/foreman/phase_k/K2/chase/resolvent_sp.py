"""(f_R_sp) attention hook, Chase K2: the K1 resolvent layer with sparsemax weights (exact zeros), blocked.
  rdepth.py:            --arm fR --hook <this file>:ResolventAttention   (--gamma_sched works: it calls layer(..., g=))
  train_ladder.py:      --attn <this file>:FRSP   (L-1 ALiBi layers from K0 train_ladder + this layer last, as K1's FR)
ResolventAttention: q,k,v,out projections (no bias), per-head logit scale s_i = a_h ln(i+1) + b_h (a = 1, b = 0 at init,
  learned), W = causal sparsemax(s_i q_i.k_j / sqrt(D)) at beta = 1, read x = (1-g)(I - gW)^-1 V, g = self.gamma
  (GAMMA = 0.999 unless a trainer sets it). The layer runs in fp32 (float64 if the weights are) with autocast off.
Blocked path (every device and dtype; nothing S x S is ever formed), row blocks of c = 256 rows:
  forward  : pass 1 (last block first): logits z (c x hi) against keys <= hi; tau_i by K0 R.sparsemax's sort rule on
             that block (the whole row is present, so tau is exact). Pass 2: z again, W = max(z - tau, 0) (exact zeros)
             in place in one flat workspace (see _workspace); forward substitution
             x_J = (I - g W_JJ)^-1 [(1-g) v_J + g W_{J,<lo} x_{<lo}], pivot (1-g) + g * off-diagonal mass. Saves tau (B,H,S,1).
  backward : reverse over blocks, W rebuilt from the saved tau; adjoint u_J = (I - g W_JJ^T)^-1 (gx_J + g acc_J),
             acc_{<lo} += W_{J,<lo}^T u_J; dW = g u_J (x - mean_S x)^T (the shift is exact: the sparsemax Jacobian
             ignores a per-row constant); dz = s (dW - mean_s dW), s = [W > 0]; dq, dk from dz; dv = (1-g) u.
Memory is O(c * S) per head. Not for torch.compile (untested).
"""
import functools, importlib.util, math, sys
sys.dont_write_bytecode = True
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

TL_PATH = "C:/Users/seal/Desktop/New folder (32)/tests/foreman/phase_k/K0/wilson/train_ladder.py"
GAMMA = 0.999
C = 256


def _tau(z):
    """Sparsemax threshold per row, K0 R.sparsemax's arithmetic (sort, cumsum, support size, tau)."""
    zs, _ = torch.sort(z, dim=-1, descending=True)
    kk = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    cs = zs.cumsum(-1)
    ksz = ((1 + kk * zs) > cs).sum(-1, keepdim=True)
    return (cs.gather(-1, ksz - 1) - 1) / ksz.to(z.dtype)


def _jac(W, dW):
    """Sparsemax VJP: dz = s (dW - mean over the support of dW), s = [W > 0]."""
    s = (W > 0).to(dW.dtype)
    return s * (dW - (dW * s).sum(-1, keepdim=True) / s.sum(-1, keepdim=True))


def _workspace(qs, c):
    """One flat buffer for a row block's logits. The row blocks widen as hi grows; allocated per block, the caching
    allocator could not reuse the narrower freed segments and reserved their sum (6.36 GiB reserved for 0.68 GiB
    allocated at S 16384, B1 H4 D32, hook 98259dff)."""
    B, H, S, _ = qs.shape
    return torch.empty(B * H * min(c, S) * S, dtype=qs.dtype, device=qs.device)


def _z(qs, k, lo, hi, ws):
    """Logits of rows lo:hi against keys :hi (causal -inf above the diagonal), written into ws."""
    B, H, _, D = qs.shape
    z = ws[:B * H * (hi - lo) * hi].view(B, H, hi - lo, hi)
    torch.matmul(qs[:, :, lo:hi], k[:, :, :hi].transpose(-1, -2), out=z)
    z.div_(math.sqrt(D))
    z[..., lo:hi].masked_fill_(torch.ones(hi - lo, hi - lo, dtype=torch.bool, device=qs.device).triu(1), float("-inf"))
    return z


def _rows(qs, k, lo, hi, tau, ws):
    """Sparsemax weights of rows lo:hi against keys :hi, from their saved tau, in place in ws."""
    return _z(qs, k, lo, hi, ws).sub_(tau).clamp_(min=0)


def _pivot_block(W, lo, g, transpose=False):
    """I - g W_JJ (or its transpose) with the diagonal written as (1-g) + g * sum_{j != i} W_ij (no cancellation)."""
    n = W.shape[-2]
    Wd = W[..., lo:]
    dmask = torch.eye(n, dtype=torch.bool, device=W.device)
    A = torch.eye(n, dtype=W.dtype, device=W.device) - g * (Wd.transpose(-1, -2) if transpose else Wd)
    A.diagonal(dim1=-2, dim2=-1).copy_((1 - g) + g * (W[..., :lo].sum(-1) + Wd.masked_fill(dmask, 0.0).sum(-1)))
    return A


def forward_blocked(qs, k, v, g, c=C):
    S = qs.shape[-2]
    ws = _workspace(qs, c)
    tau = torch.empty(*qs.shape[:-1], 1, dtype=qs.dtype, device=qs.device)
    for lo in reversed(range(0, S, c)):                    # widest blocks first, so the sort's shrinking transients reuse segments
        hi = min(S, lo + c)
        tau[:, :, lo:hi] = _tau(_z(qs, k, lo, hi, ws))
    X = torch.empty_like(v)
    for lo in range(0, S, c):
        hi = min(S, lo + c)
        W = _rows(qs, k, lo, hi, tau[:, :, lo:hi], ws)
        rhs = (1 - g) * v[:, :, lo:hi]
        if lo:
            rhs = rhs + g * (W[..., :lo] @ X[:, :, :lo])
        X[:, :, lo:hi] = torch.linalg.solve_triangular(_pivot_block(W, lo, g), rhs, upper=False)
    return X, tau


def backward_blocked(qs, k, x, tau, gx, g, c=C):
    S = qs.shape[-2]
    sc = 1.0 / math.sqrt(qs.shape[-1])
    xc = x - x.mean(-2, keepdim=True)
    U, acc = torch.empty_like(gx), torch.zeros_like(gx)
    dq, dk = torch.empty_like(qs), torch.zeros_like(k)
    ws = _workspace(qs, c)
    for lo in reversed(range(0, S, c)):
        hi = min(S, lo + c)
        W = _rows(qs, k, lo, hi, tau[:, :, lo:hi], ws)
        uJ = torch.linalg.solve_triangular(_pivot_block(W, lo, g, transpose=True), gx[:, :, lo:hi] + g * acc[:, :, lo:hi], upper=True)
        U[:, :, lo:hi] = uJ
        if lo:
            acc[:, :, :lo] += W[..., :lo].transpose(-1, -2) @ uJ
        dz = _jac(W, g * (uJ @ xc[:, :, :hi].transpose(-1, -2))) * sc
        dq[:, :, lo:hi] = dz @ k[:, :, :hi]
        dk[:, :, :hi] += dz.transpose(-1, -2) @ qs[:, :, lo:hi]
    return dq, dk, (1 - g) * U


class _Read(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, qs, k, v, g, c):
        x, tau = forward_blocked(qs, k, v, g, c)
        ctx.save_for_backward(qs, k, x, tau)
        ctx.g, ctx.c = g, c
        return x

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gx):
        qs, k, x, tau = ctx.saved_tensors
        return (*backward_blocked(qs, k, x, tau, gx.contiguous(), ctx.g, ctx.c), None, None)


def read(qs, k, v, g=GAMMA, c=C):
    """x = (1-g)(I - g W)^-1 v, W = causal sparsemax of qs.k / sqrt(D); qs pre-scaled; [B,H,S,D] each."""
    return _Read.apply(qs.contiguous(), k.contiguous(), v.contiguous(), g, c)


def weights_blocked(qs, k, c=C):
    """The forward's own per-block W assembled into [B,H,S,S] (tests only)."""
    S = qs.shape[-2]
    ws = _workspace(qs, c)
    W = torch.zeros(*qs.shape[:-1], S, dtype=qs.dtype, device=qs.device)
    for lo in range(0, S, c):
        hi = min(S, lo + c)
        W[..., lo:hi, :hi] = _rows(qs, k, lo, hi, _tau(_z(qs, k, lo, hi, ws)), ws)
    return W


def logit_scale(a, b, T):
    """s[h, i] = a_h ln(i+1) + b_h."""
    return a[:, None] * torch.log(torch.arange(1, T + 1, device=a.device, dtype=a.dtype))[None, :] + b[:, None]


def layer(x, w_qkv, w_out, a, b, n_heads, g=GAMMA, c=C, capture=None):
    B, T, Dm = x.shape
    q, k, v = F.linear(x, w_qkv).split(Dm, dim=2)
    q, k, v = (t.view(B, T, n_heads, Dm // n_heads).transpose(1, 2) for t in (q, k, v))
    qs = q * logit_scale(a, b, T)[None, :, :, None]
    if capture is not None:
        capture.append({"qs": qs.detach(), "k": k.detach(), "v": v.detach()})
    y = read(qs, k, v, g, c)
    return F.linear(y.transpose(1, 2).reshape(B, T, Dm), w_out)


class ResolventAttention(nn.Module):
    def __init__(self, d_model, n_heads, layer_idx=0, n_layers=1, ctx=None):
        super().__init__()
        self.n_heads = n_heads
        self.gamma = GAMMA                                 # schedule by setting this, or call layer(..., g=)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.a = nn.Parameter(torch.ones(n_heads))
        self.b = nn.Parameter(torch.zeros(n_heads))
        self.capture = None                                # set to a list to record qs, k, v per call

    def forward(self, x):
        dt = torch.float64 if self.qkv.weight.dtype == torch.float64 else torch.float32
        with torch.autocast(x.device.type, enabled=False):
            return layer(x.to(dt), self.qkv.weight.to(dt), self.out.weight.to(dt), self.a.to(dt), self.b.to(dt),
                         self.n_heads, g=self.gamma, capture=self.capture)


@functools.cache
def _alibi():
    spec = importlib.util.spec_from_file_location("k0_train_ladder", TL_PATH)
    tl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tl)
    return tl.AlibiAttention


class FRSP:
    """Arm (f_R_sp) for train_ladder.py --attn: L-1 ALiBi layers (K0 train_ladder, unchanged) + this layer last."""
    def __new__(cls, d_model, n_heads, layer_idx, n_layers, ctx):
        make = ResolventAttention if layer_idx == n_layers - 1 else _alibi()
        return make(d_model, n_heads, layer_idx, n_layers, ctx)
