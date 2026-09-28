"""Plain resolvent arm: causal softmax attention is replaced by the repo's signed
path-sum operator (ceq/attention.py ceq_operator + path_sum, inlined -- candidates
are pure modules, no imports of ceq). No ALiBi, no position table: the only signal
in the mixer is the strictly-causal path sum itself. 2 layers, 4 heads, d=48, FFN
width solved to the param budget (structure copied from controls/alibi_twin)."""
import math
import torch
import torch.nn as nn

RHO = 0.9
HOPS = 3


class Block(nn.Module):
    def __init__(self, d, h, f):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, f), nn.GELU(), nn.Linear(f, d))
        self.last_A = None  # last block's A, stored as A / rho (row L1 <= 1)

    def attn(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        dh = D // self.h
        w = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(-1)
        w = w.masked_fill(~mask, 0.0)
        l1 = w.abs().sum(-1, keepdim=True)
        A = RHO * w / l1.clamp_min(torch.finfo(w.dtype).tiny)
        self.last_A = A / RHO
        z = v
        term = v
        for _ in range(HOPS):
            term = A @ term
            z = z + term
        return self.o(z.transpose(1, 2).reshape(B, T, D))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class Twin(nn.Module):
    def __init__(self, vocab, n_classes, d, h, f, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, h, f) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        x = self.emb(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def _n(m):
    return sum(p.numel() for p in m.parameters())


def build(spec, d=48, h=4):
    V, C, P = spec["vocab"], spec["n_classes"], spec["param_budget"]
    base = _n(Twin(V, C, d, h, 1))
    per_f = _n(Twin(V, C, d, h, 2)) - base
    f = max(1, round((P - base) / per_f) + 1)
    m = Twin(V, C, d, h, f)
    assert abs(_n(m) - P) <= 0.02 * P, _n(m)
    return m


def resolvent_params(model, tokens):
    model(tokens)
    return torch.stack([b.last_A for b in model.blocks]), RHO
