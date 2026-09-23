"""RED stub for test_chase_k2.py: the resolvent_sp.py API with a read that ignores the weights, x = (1-g) v,
and identity weights. Every bar must fail on it."""
import torch
import torch.nn as nn
import torch.nn.functional as F

GAMMA = 0.999
C = 256


def _tau(z):
    return z.max(-1, keepdim=True).values - 1


def _jac(W, dW):
    return dW


def read(qs, k, v, g=GAMMA, c=C):
    return (1 - g) * v


def weights_blocked(qs, k, c=C):
    B, H, S, D = qs.shape
    return torch.eye(S, dtype=qs.dtype, device=qs.device).expand(B, H, S, S).clone()


def logit_scale(a, b, T):
    return a[:, None] * torch.log(torch.arange(1, T + 1, device=a.device, dtype=a.dtype))[None, :] + b[:, None]


def layer(x, w_qkv, w_out, a, b, n_heads, g=GAMMA, c=C, capture=None):
    B, T, Dm = x.shape
    q, k, v = F.linear(x, w_qkv).split(Dm, dim=2)
    q, k, v = (t.view(B, T, n_heads, Dm // n_heads).transpose(1, 2) for t in (q, k, v))
    qs = q * logit_scale(a, b, T)[None, :, :, None]
    y = read(qs, k, v, g, c)
    return F.linear(y.transpose(1, 2).reshape(B, T, Dm), w_out)


class ResolventAttention(nn.Module):
    def __init__(self, d_model, n_heads, layer_idx=0, n_layers=1, ctx=None):
        super().__init__()
        self.n_heads, self.gamma = n_heads, GAMMA
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.a = nn.Parameter(torch.ones(n_heads))
        self.b = nn.Parameter(torch.zeros(n_heads))
        self.capture = None

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return layer(x.float(), self.qkv.weight, self.out.weight, self.a, self.b, self.n_heads, g=self.gamma)
