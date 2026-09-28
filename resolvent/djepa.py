"""D-JEPA's relational candidate operator, reimplemented from the paper's equations (no code copied).

Source: Liu et al., "D-JEPA: A Decision-Aligned Latent World Model", arXiv 2609.24749v1 (CC BY 4.0),
Sec. 3 and App. B.1. The authors' code (github.com/NEBULIS-Lab/D-JEPA, Apache-2.0) was not cloned,
downloaded or read; see NOTICE.

    h_i     = Transformer(enc(v_i))              no candidate-order positional encoding
    delta_i = eps * tanh(W_up tanh(W_down h_i))  zero-init head, so delta = 0 at initialisation
    s_i     = b_i + delta_i                      lower is better; b_i = normalised base rank

|delta_i| <= eps, so any preference across a base gap > 2 eps survives and the selected candidate lies
within 2 eps of the base minimum (the paper's Prop 2 / Cor 1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS, TEMP, MARGIN = 0.2, 0.05, 0.02


class RelationalOperator(nn.Module):
    def __init__(self, nin, width=64, eps=EPS):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(nin, width), nn.LayerNorm(width), nn.GELU())
        layer = nn.TransformerEncoderLayer(width, 4, 128, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.down, self.up = nn.Linear(width, 8), nn.Linear(8, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.eps = eps

    def forward(self, v, base, pad=None):
        """v: (B, K, nin) tokens; base: (B, K); pad: (B, K) bool, True = padding. Returns (s, delta)."""
        h = self.tf(self.enc(v), src_key_padding_mask=pad)
        delta = self.eps * torch.tanh(self.up(torch.tanh(self.down(h)))).squeeze(-1)
        return base + delta, delta


def rank01(cost):
    """(rank(c) - 1) / (K - 1) along the candidate axis, rank 1 = lowest cost."""
    return cost.argsort(-1).argsort(-1).to(torch.float32) / max(cost.shape[-1] - 1, 1)


def dj_loss(s, delta, y, T=TEMP, margin=MARGIN, local=0.25, trust=0.1):
    """-log sum_{i: y_i} softmax(-s/T)_i + local * pairwise margin + trust * mean delta^2.
    Rows without a positive candidate contribute only to the trust term."""
    lp = torch.log_softmax(-s / T, 1)
    has = y.any(1)
    l_set = -(torch.logsumexp(lp.masked_fill(~y, -1e9), 1))[has].mean() if has.any() else s.sum() * 0
    pair = (y[:, :, None] & ~y[:, None, :]).float()
    l_loc = (F.softplus((margin + s[:, :, None] - s[:, None, :]) / T) * pair).sum() / pair.sum().clamp(min=1)
    return l_set + local * l_loc + trust * (delta ** 2).mean()
