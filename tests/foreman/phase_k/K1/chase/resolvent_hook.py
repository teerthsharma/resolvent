"""(f_R) attention hook, Chase K1. Load with  train_ladder.py --attn <this file>:FR
(any trainer that builds attn_cls(d_model, n_heads, layer_idx, n_layers, ctx) and calls .forward(x[B,T,d])).

FR: layers 0..L-2 are train_ladder's AlibiAttention (K0 harness, unchanged); layer L-1 is ResolventAttention.
  The resolvent is the LAST layer: its q/k/v are then built by the L-1 ALiBi layers (a root has to learn to
  point at itself, the absorber, and V has to carry the root id; both need the content mixed first), and no
  later attention can re-mix its read, so depth past the ALiBi stack's 2^(L-1) is attributable to it.
ResolventAttention: q,k,v,out projections (no bias, as ALiBi's), per-head logit scale s_i = a_h ln(i+1) + b_h
  (a = 1, b = 0 at init, both learned, SSMax law), W = causal softmax(s_i q_i.k_j / sqrt(D)) at beta = 1,
  read x = (1-g)(I - gW)^-1 V at g = GAMMA = 0.999 fixed.
  CUDA fp32: forward fs5c (K0 resolvent.py), backward backward_blocked (K0's, with x centred in dP; below),
  both CUDA-graphed per shape.
  Otherwise (CPU, float64): dense triangular solve forward, the same backward_blocked with row block c.
  The whole layer runs with autocast disabled in the weights' dtype (fp32 in training); the solve's
  Function is also custom_fwd/custom_bwd, so its backward never runs under autocast either.
  S is zero-padded to a multiple of c (causal: padded rows are never read by real rows).
  Not for torch.compile (the solve captures its own CUDA graphs inside an autograd Function; untested under compile).
"""
import importlib.util, math, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

REPO = "C:/Users/seal/Desktop/New folder (32)"
sys.path.insert(0, REPO + "/tests/foreman/phase_k/K0/chase")
import resolvent as R                                     # K0 lane, committed at e5c3cc7

_spec = importlib.util.spec_from_file_location("k0_train_ladder", REPO + "/tests/foreman/phase_k/K0/wilson/train_ladder.py")
TL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TL)
ALIBI = TL.AlibiAttention

GAMMA = 0.999
C = 256


def backward_blocked(qs, k, x, gx, g, c=256):
    """K0 resolvent.backward_blocked copied with ONE change: dP is formed from x - mean_S(x) instead of x.
    Exact, since dz = P * (dP - rowsum(P dP)) is invariant to any shift of x shared by all keys (rows of P sum to 1).
    Why (diag_dq.py): at g = .999, |u| ~ 2e4 |gx| and the reads x_j are nearly equal, so g u x^T - rowsum cancels in
    fp32; with x uncentred, dq's error is 2.7e-4 (u itself is exact to 2.8e-7)."""
    B, H, S, D = qs.shape
    sc = 1.0 / math.sqrt(D)
    xc = x - x.mean(-2, keepdim=True)
    U = torch.empty_like(gx); acc = torch.zeros_like(gx)
    dq = torch.empty_like(qs); dk = torch.zeros_like(k)
    tri = R._mask(c, qs.device)
    dmask = torch.eye(c, dtype=torch.bool, device=qs.device)
    eye = torch.eye(c, dtype=qs.dtype, device=qs.device)
    for J in reversed(range(S // c)):
        lo, hi = J * c, (J + 1) * c
        z = (qs[:, :, lo:hi] @ k[:, :, :hi].transpose(-1, -2)) * sc
        z[..., lo:hi].masked_fill_(tri, float("-inf"))
        P = torch.softmax(z, -1)
        comp = P[..., :lo].sum(-1) + P[..., lo:hi].masked_fill(dmask, 0.0).sum(-1)
        AT = eye - g * P[..., lo:hi].transpose(-1, -2)
        AT.diagonal(dim1=-2, dim2=-1).copy_((1 - g) + g * comp)
        uJ = torch.linalg.solve_triangular(AT, gx[:, :, lo:hi] + g * acc[:, :, lo:hi], upper=True)
        U[:, :, lo:hi] = uJ
        if lo:
            acc[:, :, :lo] += P[..., :lo].transpose(-1, -2) @ uJ
        dP = g * (uJ @ xc[:, :, :hi].transpose(-1, -2))
        dz = P * (dP - (P * dP).sum(-1, keepdim=True))
        dq[:, :, lo:hi] = (dz @ k[:, :, :hi]) * sc
        dk[:, :, :hi] += (dz.transpose(-1, -2) @ qs[:, :, lo:hi]) * sc
    return dq, dk, (1 - g) * U


def _dense_fwd(qs, k, v, g):
    A = torch.eye(qs.shape[-2], dtype=qs.dtype, device=qs.device) - g * R.weights(qs, k)
    return torch.linalg.solve_triangular(A, (1 - g) * v, upper=False)


class _Read(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, qs, k, v, g, c):
        if qs.is_cuda and qs.dtype == torch.float32:
            (x,) = R._graph_call(("f", str(qs.device), tuple(qs.shape), g, c), lambda a, b, d: R.fs5c(a, b, d, g, c), qs, k, v)
        else:
            x = _dense_fwd(qs, k, v, g)
        ctx.save_for_backward(qs, k, x)
        ctx.g, ctx.c = g, c
        return x

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gx):
        qs, k, x = ctx.saved_tensors
        g, c = ctx.g, ctx.c
        if qs.is_cuda and qs.dtype == torch.float32:
            dq, dk, dv = R._graph_call(("b", str(qs.device), tuple(qs.shape), g, c),
                                       lambda a, b, d, e: backward_blocked(a, b, d, e, g, c), qs, k, x, gx.contiguous())
        else:
            dq, dk, dv = backward_blocked(qs, k, x, gx.contiguous(), g, c)
        return dq, dk, dv, None, None


def read(qs, k, v, g=GAMMA, c=C):
    """x = (1-g)(I - g W)^-1 v for pre-scaled queries qs; [B,H,S,D] each."""
    S = qs.shape[-2]
    p = -S % c
    if p:
        qs, k, v = (F.pad(t, (0, 0, 0, p)) for t in (qs, k, v))
    x = _Read.apply(qs.contiguous(), k.contiguous(), v.contiguous(), g, c)
    return x[..., :S, :] if p else x


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
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.a = nn.Parameter(torch.ones(n_heads))
        self.b = nn.Parameter(torch.zeros(n_heads))
        self.capture = None                                # set to a list to record qs, k, v per call

    def forward(self, x):
        dt = torch.float64 if self.qkv.weight.dtype == torch.float64 else torch.float32   # fp32 even if the model is cast down
        with torch.autocast(x.device.type, enabled=False):
            return layer(x.to(dt), self.qkv.weight.to(dt), self.out.weight.to(dt), self.a.to(dt), self.b.to(dt),
                         self.n_heads, capture=self.capture)


class FR:
    """Arm (f_R): L-1 ALiBi layers + the resolvent layer last. A factory with the harness's class signature."""
    def __new__(cls, d_model, n_heads, layer_idx, n_layers, ctx):
        make = ResolventAttention if layer_idx == n_layers - 1 else ALIBI
        return make(d_model, n_heads, layer_idx, n_layers, ctx)
