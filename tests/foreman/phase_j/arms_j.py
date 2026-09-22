"""Phase J: arm (a_T) [temperature head] and arm (a'') [FoX forget gate],
both built as (a)'s exact SDPA softmax twin plus one extra piece, reusing
design4x5's train_arm/train_with_eval machinery (same CRN, same steps, same
eval subsample). See RECORD.md for the contract this implements.

(a_T): model.lm_head replaced by TempLMHead(orig_linear, temp_head). Extra
params: nn.Linear(hidden, 1) -- 129 (128 weight + 1 bias). Init weight 0,
bias ln(e-1) so softplus(bias) == 1 and T starts at 1.0 exactly.

(a''): FoX forget gate (arXiv 2503.02130). Per layer, nn.Linear(hidden,
n_heads) -> f_t = sigmoid(.); score bias D_ij = sum_{k=j+1..i} log f_k per
head, causal. Implemented as D_ij = C_i - C_j with C = cumsum(log f), log f
via -softplus(-x) for stability, then masked -inf for j>i and added as an
explicit SDPA attn_mask (replacing is_causal=True). Extra params per layer:
nn.Linear(hidden, n_heads), same shape as design4x5's o_gate_head -> 3,096
total, matching arm (a2). Any init choice (here: PyTorch's default Linear
init, no special init applied) is this file's own, not FoX's; only the D_ij
score-bias construction is cited to the paper.
"""
from __future__ import annotations

import math
import os
import sys

SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
REPO = r"C:\Users\seal\Desktop\New folder (32)"
PHASE_J = os.path.join(SCRATCH, "phase_j")
sys.path.insert(0, SCRATCH)
sys.path.insert(0, REPO)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import design4x5 as D  # noqa: E402 -- REUSED, not edited
import r3_eval as RE  # noqa: E402 -- REUSED, not edited (LIVE r3_eval.py)
from ceq.hf.modeling_ceq import CEQAttention  # noqa: E402
from q2_certificate import softmax_forward  # noqa: E402 -- (a)'s forward, REUSED

_ORIG_BUILD = RE.build
_ORIG_ATTN_FORWARD = CEQAttention.forward

OUT_DIR_TMPL = os.path.join(PHASE_J, "ckpt_{arm}_ss{ss}")


# ------------------------------------------------------------- (a_T) wiring

class TempLMHead(nn.Module):
    """Wraps the original lm_head Linear; logits_t = orig(h) / T_t,
    T_t = softplus(lin(h)) + 1e-3. init weight 0, bias ln(e-1) -> T==1.0."""

    def __init__(self, orig_linear, hidden):
        super().__init__()
        self.orig = orig_linear
        self.temp = nn.Linear(hidden, 1)
        nn.init.zeros_(self.temp.weight)
        nn.init.constant_(self.temp.bias, math.log(math.e - 1.0))

    @property
    def weight(self):
        # CEQForCausalLM.forward reads self.lm_head.weight.is_meta before
        # calling; delegate so that check still sees the tied embedding.
        return self.orig.weight

    def forward(self, h):
        logits = self.orig(h)
        T = F.softplus(self.temp(h)) + 1e-3
        return logits / T


def _attach_temp_head(model):
    model.lm_head = TempLMHead(model.lm_head, D.HIDDEN)
    return model


def temp_build(**kw):
    return _attach_temp_head(_ORIG_BUILD(**kw))


def temp_head_numel(model):
    lm = model.lm_head
    return sum(p.numel() for p in lm.temp.parameters())


# ------------------------------------------------------------- (a'') FoX wiring

def _attach_forget_heads(model):
    for layer in model.model.layers:
        layer.self_attn.forget_head = nn.Linear(D.HIDDEN, D.HEADS)
    return model


def fox_forward(self, x, attention_mask=None):
    """(a)'s softmax_forward plus a FoX (arXiv 2503.02130) per-head forget
    score-bias D_ij = sum_{k=j+1..i} log f_k, f_t = sigmoid(forget_head(x)),
    added to SDPA as an explicit float attn_mask together with the causal
    -inf mask (replacing is_causal=True). Attention weights are the softmax
    of QK^T/sqrt(d) + D, not merely reweighted afterward -- this changes the
    attention pattern itself, unlike (a2)'s output gate."""
    if attention_mask is not None:
        raise NotImplementedError("FoX twin: no padding-mask path needed")
    b, s, d = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)

    def shape(t):
        return t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

    gate_logits = self.forget_head(x)                       # [B, S, H]
    log_f = -F.softplus(-gate_logits)                        # log(sigmoid(.)), stable
    c = log_f.cumsum(dim=1).transpose(1, 2)                   # [B, H, S]
    bias = c.unsqueeze(-1) - c.unsqueeze(-2)                   # [B, H, S, S]: C_i - C_j
    causal = torch.triu(torch.ones(s, s, dtype=torch.bool, device=x.device), diagonal=1)
    bias = bias.masked_fill(causal, float("-inf"))
    o = F.scaled_dot_product_attention(shape(q), shape(k), shape(v),
                                        attn_mask=bias, is_causal=False)
    return self.o_proj(o.transpose(1, 2).reshape(b, s, d))


def fox_head_numel(model):
    n = 0
    for layer in model.model.layers:
        n += sum(p.numel() for p in layer.self_attn.forget_head.parameters())
    return n


# ------------------------------------------------------------- shared trainer

def _common_kwargs(out_dir, seed, split_seed, steps):
    return dict(out_dir=out_dir, steps=steps, batch=D.BATCH, seq=D.SEQ,
                hidden_size=D.HIDDEN, n_layers=D.LAYERS, n_heads=D.HEADS,
                device=D.DEVICE, vocab_size=D.VOCAB, lr=D.LR, seed=seed,
                split_seed=split_seed, eval_every=steps,
                eval_batches=D.EVAL_BATCHES, log_every=50, save_model=True)


def train_a_T(seed, split_seed, steps):
    out_dir = OUT_DIR_TMPL.format(arm="a_T", ss=split_seed)
    CEQAttention.forward = softmax_forward
    RE.build = temp_build
    try:
        rec = RE.train_with_eval(operator="sgate",
                                  **_common_kwargs(out_dir, seed, split_seed, steps))
        torch.manual_seed(0)
        probe = temp_build(hidden_size=D.HIDDEN, n_layers=D.LAYERS, n_heads=D.HEADS,
                            seq=D.SEQ, vocab_size=D.VOCAB, operator="sgate")
        rec["extra_params"] = temp_head_numel(probe)
        del probe
    finally:
        CEQAttention.forward = _ORIG_ATTN_FORWARD
        RE.build = _ORIG_BUILD
    rec["arm"] = "a_T"
    rec["out_dir"] = out_dir
    return rec


def train_a2F(seed, split_seed, steps):
    out_dir = OUT_DIR_TMPL.format(arm="a2F", ss=split_seed)
    CEQAttention.forward = fox_forward
    RE.build = lambda **kw: _attach_forget_heads(_ORIG_BUILD(**kw))
    try:
        rec = RE.train_with_eval(operator="sgate",
                                  **_common_kwargs(out_dir, seed, split_seed, steps))
        torch.manual_seed(0)
        probe = _attach_forget_heads(
            _ORIG_BUILD(hidden_size=D.HIDDEN, n_layers=D.LAYERS, n_heads=D.HEADS,
                        seq=D.SEQ, vocab_size=D.VOCAB, operator="sgate"))
        rec["extra_params"] = fox_head_numel(probe)
        del probe
    finally:
        CEQAttention.forward = _ORIG_ATTN_FORWARD
        RE.build = _ORIG_BUILD
    rec["arm"] = "a2F"
    rec["out_dir"] = out_dir
    return rec


def build_for_eval(arm, split_seed):
    """Rebuild the arm's architecture (untrained) so a checkpoint can be
    loaded onto it, and return the CEQAttention.forward it must run under."""
    common = dict(hidden_size=D.HIDDEN, n_layers=D.LAYERS, n_heads=D.HEADS,
                  seq=D.SEQ, vocab_size=D.VOCAB)
    if arm == "a_T":
        model = temp_build(operator="sgate", **common)
        fwd = softmax_forward
    elif arm == "a2F":
        model = _attach_forget_heads(_ORIG_BUILD(operator="sgate", **common))
        fwd = fox_forward
    else:
        raise ValueError(arm)
    return model, fwd
