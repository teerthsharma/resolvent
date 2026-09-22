# -*- coding: utf-8 -*-
"""Eval-only: (f) with m_t -> layer geometric mean; FoX with log f_t -> (layer,
head) mean. Same 64x512 eval windows. Bars in test_const.py (written first)."""
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
m_f, _ = build_arm("f")
m_x, fox_fwd = arms_j.build_for_eval("a2F", split_seed=0)
m_x.load_state_dict(torch.load(os.path.join(SP, "phase_j", "ckpt_a2F_ss0", "model.pt"),
                               map_location="cpu", weights_only=False)["state_dict"], strict=True)
m_x.to(D.DEVICE).eval()
batches = eval_batches()
L = D.LAYERS
CUR = {"l": 0}
logm = [[] for _ in range(L)]
logf = [[] for _ in range(L)]


def hook_f(i):
    def h(mod, inp):
        CUR["l"] = i
        x = inp[0]
        m, _ = A.blend(mod.m_head(x).squeeze(-1), mod.theta_head(x).squeeze(-1), mod.g)
        logm[i].append(m.double().clamp_min(1e-30).log().flatten().cpu())
    return h


def hook_x(i):
    def h(mod, inp):
        logf[i].append((-F.softplus(-mod.forget_head(inp[0]))).double().reshape(-1, D.HEADS).cpu())
    return h


def nll(model, fwd, x):
    ctx = CEQAttention.forward
    if fwd is not None:
        CEQAttention.forward = fwd
    try:
        return float(per_token_nll(model, x, softmax_twin=False).mean())
    finally:
        CEQAttention.forward = ctx


hs = [b.self_attn.register_forward_pre_hook(hook_f(i)) for i, b in enumerate(m_f.model.layers)]
hs += [b.self_attn.register_forward_pre_hook(hook_x(i)) for i, b in enumerate(m_x.model.layers)]
f_full = sum(nll(m_f, None, x) for x in batches) / len(batches)
x_full = sum(nll(m_x, fox_fwd, x) for x in batches) / len(batches)
for h in hs[L:]:
    h.remove()
gm = [float(torch.cat(v).mean().exp()) for v in logm]            # per-layer geometric mean m
lf = [torch.cat(v).mean(0).float().to(D.DEVICE) for v in logf]  # [H] per layer
for i, b in enumerate(m_x.model.layers):
    b.self_attn._logf_const = lf[i]

_mag = A.magnitude
A.magnitude = lambda u: torch.full_like(u, gm[CUR["l"]])


def fox_const(self, x, attention_mask=None):
    b, s, dm = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    sh = lambda t: t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
    c = self._logf_const.view(1, 1, -1).expand(b, s, -1).cumsum(1).transpose(1, 2)
    causal = torch.triu(torch.ones(s, s, dtype=torch.bool, device=x.device), 1)
    bias = (c.unsqueeze(-1) - c.unsqueeze(-2)).masked_fill(causal, float("-inf"))
    o = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), attn_mask=bias)
    return self.o_proj(o.transpose(1, 2).reshape(b, s, dm))


try:
    f_const = sum(nll(m_f, None, x) for x in batches) / len(batches)
finally:
    A.magnitude = _mag
for h in hs[:L]:
    h.remove()
x_const = sum(nll(m_x, fox_const, x) for x in batches) / len(batches)
res = dict(f_full=f_full, f_const=f_const, fox_full=x_full, fox_const=x_const,
           f_geomean_m=gm, fox_mean_logf=[v.tolist() for v in lf],
           fox_mean_f=[v.exp().tolist() for v in lf], seconds=time.time() - t0)
json.dump(res, open(os.path.join(HERE, "const.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
