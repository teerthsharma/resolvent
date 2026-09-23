"""RED stub for test_chase_k1.py: resolvent_hook.py's interface with no resolvent (the read is v, no solve;
a, b unused). Used only for the RED runs (CHASE_HOOK=<this file>)."""
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = "C:/Users/seal/Desktop/New folder (32)"
_spec = importlib.util.spec_from_file_location("k0_train_ladder", REPO + "/tests/foreman/phase_k/K0/wilson/train_ladder.py")
TL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TL)
ALIBI = TL.AlibiAttention
GAMMA, C = 0.999, 256
backward_blocked = None


def read(qs, k, v, g=GAMMA, c=C):
    return v


def layer(x, w_qkv, w_out, a, b, n_heads, g=GAMMA, c=C, capture=None):
    Dm = x.shape[-1]
    return F.linear(F.linear(x, w_qkv)[..., 2 * Dm:], w_out)


class ResolventAttention(nn.Module):
    def __init__(self, d_model, n_heads, layer_idx=0, n_layers=1, ctx=None):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.a = nn.Parameter(torch.ones(n_heads))
        self.b = nn.Parameter(torch.zeros(n_heads))
        self.capture = None

    def forward(self, x):
        return layer(x, self.qkv.weight, self.out.weight, self.a, self.b, self.n_heads)


class FR:
    def __new__(cls, d_model, n_heads, layer_idx, n_layers, ctx):
        make = ResolventAttention if layer_idx == n_layers - 1 else ALIBI
        return make(d_model, n_heads, layer_idx, n_layers, ctx)
