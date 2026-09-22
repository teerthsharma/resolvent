# -*- coding: utf-8 -*-
"""Eval-only hard causal window (keys i-w+1..i) on (a), (f), FoX; mean held-out
NLL on the same 64x512 eval windows. Bars in test_window.py (written first)."""
import json, os, sys, time
SP = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(SP, "phase_j"), SP, r"C:\Users\seal\Desktop\New folder (32)"):
    sys.path.insert(0, p)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import design4x5 as D  # noqa: E402
from abstention_deciles import build_arm, eval_batches, per_token_nll  # noqa: E402
import arms_j  # noqa: E402
from ceq import arm_smprime as A  # noqa: E402
from ceq.hf.modeling_ceq import CEQAttention  # noqa: E402

t0 = time.time()
print("gpu free:", D.poll_until_free(timeout_s=900), flush=True)
m_a, _ = build_arm("a")
m_f, _ = build_arm("f")
m_x, fox_fwd = arms_j.build_for_eval("a2F", split_seed=0)
m_x.load_state_dict(torch.load(os.path.join(SP, "phase_j", "ckpt_a2F_ss0", "model.pt"),
                               map_location="cpu", weights_only=False)["state_dict"], strict=True)
m_x.to(D.DEVICE).eval()
batches = eval_batches()
S = D.SEQ
W = {"w": None}


def band(n, dev):
    i = torch.arange(n, device=dev)
    d = i.unsqueeze(-1) - i.unsqueeze(0)
    return (d >= 0) & (d < W["w"])


def a_win(self, x, attention_mask=None):
    b, s, dm = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    sh = lambda t: t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
    o = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), attn_mask=band(s, x.device))
    return self.o_proj(o.transpose(1, 2).reshape(b, s, dm))


def fox_win(self, x, attention_mask=None):
    b, s, dm = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    sh = lambda t: t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
    c = (-F.softplus(-self.forget_head(x))).cumsum(1).transpose(1, 2)
    bias = (c.unsqueeze(-1) - c.unsqueeze(-2)).masked_fill(~band(s, x.device), float("-inf"))
    o = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), attn_mask=bias)
    return self.o_proj(o.transpose(1, 2).reshape(b, s, dm))


_num = A.numerator


def num_win(q, k, *a, **kw):
    num, mod = _num(q, k, *a, **kw)
    keep = band(q.shape[-2], q.device)
    return num * keep, mod * keep


def nll(model, fwd, x):
    ctx = CEQAttention.forward
    if fwd is not None:
        CEQAttention.forward = fwd
    try:
        return float(per_token_nll(model, x, softmax_twin=False).mean())
    finally:
        CEQAttention.forward = ctx


res = {}
for w in (None, 4, 16, 64):
    W["w"] = w
    tag = "full" if w is None else "w{}".format(w)
    if w is not None:
        A.numerator = num_win
    try:
        va = [nll(m_a, D.softmax_forward if w is None else a_win, x) for x in batches]
        vf = [nll(m_f, None, x) for x in batches]
        vx = [nll(m_x, fox_fwd if w is None else fox_win, x) for x in batches]
    finally:
        A.numerator = _num
    res["a_" + tag], res["f_" + tag], res["fox_" + tag] = (sum(v) / len(v) for v in (va, vf, vx))
    print(tag, res["a_" + tag], res["f_" + tag], res["fox_" + tag], flush=True)
res["C_win_here"] = res["a_full"] - res["f_full"]
res["seconds"] = time.time() - t0
json.dump(res, open(os.path.join(HERE, "window.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
