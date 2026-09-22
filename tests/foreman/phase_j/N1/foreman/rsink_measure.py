# -*- coding: utf-8 -*-
"""R-SINK P1-P4 + first-token mass + (f)/FoX gate colocation, existing
checkpoints, no training. Bars live in test_rsink.py (written first).
Reuses abstention_deciles (build_arm, eval_batches) and arms_j (FoX) as
r_xfer.py did. Writes rsink.json next to this file."""
import json
import os
import sys
import time

SP = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SP, "phase_j"))
sys.path.insert(0, SP)
sys.path.insert(0, r"C:\Users\seal\Desktop\New folder (32)")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

import design4x5 as D  # noqa: E402
from abstention_deciles import build_arm, eval_batches  # noqa: E402
import arms_j  # noqa: E402
from ceq import arm_smprime as A  # noqa: E402
from ceq.hf.modeling_ceq import CEQAttention  # noqa: E402

THR, K, DELIM = 0.05, 64, (10, 46)
rng = np.random.default_rng(0)
t0 = time.time()
print("gpu free:", D.poll_until_free(timeout_s=900), flush=True)

m_a, _ = build_arm("a")
m_f, _ = build_arm("f")
m_x, fox_fwd = arms_j.build_for_eval("a2F", split_seed=0)
m_x.load_state_dict(torch.load(os.path.join(SP, "phase_j", "ckpt_a2F_ss0", "model.pt"),
                               map_location="cpu", weights_only=False)["state_dict"], strict=True)
m_x.to(D.DEVICE).eval()
batches = eval_batches()
S, L = D.SEQ, D.LAYERS


def capture(model, x, fwd):
    """Inputs to every self_attn, under the forward the arm was trained with."""
    caps, hs = {}, []
    for i, blk in enumerate(model.model.layers):
        hs.append(blk.self_attn.register_forward_pre_hook(
            (lambda i: lambda mod, inp: caps.__setitem__(i, inp[0].detach()))(i)))
    ctx = CEQAttention.forward
    if fwd is not None:
        CEQAttention.forward = fwd
    try:
        with torch.no_grad():
            model(input_ids=x)
    finally:
        CEQAttention.forward = ctx
        for h in hs:
            h.remove()
    return caps


def qkv(at, h):
    b, s, _ = h.shape
    q, k, v = at.qkv(h).chunk(3, dim=-1)
    sh = lambda t: t.view(b, s, at.n_heads, at.d_head).transpose(1, 2)
    return sh(q), sh(k), sh(v)


causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device=D.DEVICE), 1)


def softmax_attn(at, h, fox=False):
    q, k, v = qkv(at, h)
    b = torch.zeros(1, 1, S, S, device=h.device)
    fg = None
    if fox:
        lf = -F.softplus(-at.forget_head(h))                  # [B,S,H]
        c = lf.cumsum(1).transpose(1, 2)
        b = c.unsqueeze(-1) - c.unsqueeze(-2)
        fg = lf.exp().mean(-1)                                # head-mean forget f_t
    b = b.masked_fill(causal, float("-inf"))
    a = torch.softmax((q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5) + b, -1)
    # instrument check: my attention reproduces SDPA's output on the same q,k,v
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=b.expand(q.shape[0], q.shape[1], S, S))
    return a, fg, float((a @ v - o).abs().max())


def f_attn(at, h):
    """(f)'s |W_ij| at beta=1: R_ij e^{s_ij} / Z_i (the modulus row Z sums)."""
    q, k, v = qkv(at, h)
    u = at.m_head(h).squeeze(-1)
    th = at.theta_head(h).squeeze(-1)
    m, _ = A.blend(u, th, at.g)                               # [B,S]
    out = []
    for bi in range(h.shape[0]):
        _, mod = A.numerator(q[bi:bi + 1], k[bi:bi + 1], u[bi:bi + 1].unsqueeze(-2),
                             th[bi:bi + 1].unsqueeze(-2), qk=at.qk, g=at.g,
                             route="product", phase_route="gate")
        out.append((mod / mod.sum(-1, keepdim=True)).float())
    vn = v.float().norm(dim=-1).mean(1)                       # [B,S] head-mean ||v||
    return torch.cat(out, 0), m.float(), vn


X, M, VN, FG = [], [], [], []
FT = {k: [] for k in ("a", "f", "fox")}
DIST = {k: [] for k in ("a", "f", "fox")}
RECV = []                    # (a) received attention, [L,B,S]
P2_f, P2_a = [], []          # per (closure, query) attention on the closure token
errs = []
idx = torch.arange(S, device=D.DEVICE, dtype=torch.float32)
dist_mat = (idx.unsqueeze(-1) - idx.unsqueeze(0)).clamp_min(0)
band = ((dist_mat >= 1) & (dist_mat <= K)).float()      # [i,t]: t < i <= t+K
beta = float(m_f.model.layers[0].self_attn.beta)

for x in batches:
    ca = capture(m_a, x, D.softmax_forward)
    cf = capture(m_f, x, None)
    cx = capture(m_x, x, fox_fwd)
    Ml, VNl, FGl, Rl = [], [], [], []
    for l in range(L):
        with torch.no_grad():
            aa, _, e1 = softmax_attn(m_a.model.layers[l].self_attn, ca[l])
            ax, fg, e2 = softmax_attn(m_x.model.layers[l].self_attn, cx[l], fox=True)
            af, mf, vn = f_attn(m_f.model.layers[l].self_attn, cf[l])
        errs += [e1, e2]
        for key, att in (("a", aa), ("f", af), ("fox", ax)):
            FT[key].append((l, att[:, :, 1:, 0].mean((1, 2)).cpu()))       # [B]
            DIST[key].append((l, (att * dist_mat).sum(-1)[:, :, 1:].mean((1, 2)).cpu()))
        # received attention in (a) from the next K queries, head mean
        am = aa.mean(1)                                                      # [B,S,S]
        Rl.append((am * band).sum(1).cpu())                                  # [B,S]
        # P2: queries in closure c's segment (up to next closure) -> weight on c
        afm, aam = af.mean(1).cpu().numpy(), am.cpu().numpy()
        mc = (mf < THR).cpu().numpy()
        for bi in range(x.shape[0]):
            cl = np.flatnonzero(mc[bi])
            for j, c in enumerate(cl):
                end = cl[j + 1] if j + 1 < len(cl) else S
                if end - c < 2:
                    continue
                P2_f.append(afm[bi, c + 1:end, c])
                P2_a.append(aam[bi, c + 1:end, c])
        Ml.append(mf.cpu()); VNl.append(vn.cpu()); FGl.append(fg.cpu())
    X.append(x.cpu()); M.append(torch.stack(Ml)); VN.append(torch.stack(VNl))
    FG.append(torch.stack(FGl)); RECV.append(torch.stack(Rl))
    print("batch done {:.0f}s".format(time.time() - t0), flush=True)

X = torch.cat(X).numpy()                     # [W,S]
M = torch.cat(M, 1).numpy()                  # [L,W,S]
VN = torch.cat(VN, 1).numpy()
FG = torch.cat(FG, 1).numpy()
RECV = torch.cat(RECV, 1).numpy()
W = X.shape[0]
delim = np.isin(X, DELIM)
closed = M < THR


def auc(score, label):
    r = rankdata(score)
    n1 = label.sum(); n0 = label.size - n1
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def doc_segments(xw):
    """Within-window document spans, split after each '\\n\\n'."""
    cuts = [i + 2 for i in range(len(xw) - 1) if xw[i] == 10 and xw[i + 1] == 10]
    return np.split(np.arange(len(xw)), [c for c in cuts if c < len(xw)])


segs = [doc_segments(X[w]) for w in range(W)]


def shuffle_within_docs(Mw):
    out = Mw.copy()
    for l in range(L):
        for w in range(W):
            for sg in segs[w]:
                out[l, w, sg] = Mw[l, w, rng.permutation(sg)]
    return out


def p1_stats(Mx, win=None):
    win = np.arange(W) if win is None else win
    c, d = (Mx[:, win] < THR), np.broadcast_to(delim[win], Mx[:, win].shape)
    pd = d.mean()
    lift = float(d[c].mean() / pd) if c.any() else float("nan")
    return lift, auc(-Mx[:, win].ravel(), d.ravel())


lift, a_ = p1_stats(M)
null_auc = [p1_stats(shuffle_within_docs(M))[1] for _ in range(100)]
boot = [p1_stats(M, rng.integers(0, W, W)) for _ in range(300)]
lift_off = {}
for off in (-1, 1, 2):     # closure at the byte `off` after a delimiter
    dsh = np.zeros_like(delim)
    if off > 0:
        dsh[:, off:] = delim[:, :-off]
    else:
        dsh[:, :off] = delim[:, -off:]
    dd = np.broadcast_to(dsh, M.shape)
    lift_off[off] = float(dd[closed].mean() / dd.mean()) if closed.any() else None
P1 = dict(lift=lift, auc=a_, null_auc_mean=float(np.mean(null_auc)),
          null_auc_p95=float(np.percentile(null_auc, 95)),
          lift_ci=[float(np.nanpercentile([b[0] for b in boot], q)) for q in (2.5, 97.5)],
          auc_ci=[float(np.percentile([b[1] for b in boot], q)) for q in (2.5, 97.5)],
          per_layer=[dict(closure_rate=float(closed[l].mean()),
                          lift=float(delim[closed[l]].mean() / delim.mean()) if closed[l].any() else None,
                          auc=auc(-M[l].ravel(), delim.ravel())) for l in range(L)],
          lift_by_offset=lift_off, p_delim=float(delim.mean()),
          n_closed=int(closed.sum()), n_site_layers=int(closed.size))

p2f = np.concatenate(P2_f) if P2_f else np.array([np.nan])
p2a = np.concatenate(P2_a) if P2_a else np.array([np.nan])
P2 = dict(mean_attn_on_closure=float(p2f.mean()) if P2_f else None,
          twin_same_pairs=float(p2a.mean()) if P2_a else None,
          closure_mean=float(np.mean([v.mean() for v in P2_f])) if P2_f else None,
          n_pairs=int(p2f.size), n_closures=len(P2_f))

med_all = np.median(VN, axis=(1, 2))
P3 = dict(median_ratio=float(np.median(VN[closed]) / np.median(VN)) if closed.any() else None,
          mean_ratio=float(VN[closed].mean() / np.median(VN)) if closed.any() else None,
          per_layer=[float(np.median(VN[l][closed[l]]) / med_all[l]) if closed[l].any() else None
                     for l in range(L)])

rate = closed.mean(0)                        # [W,S]
recv = RECV.mean(0)
sl = slice(1, S - K)
r_, rv = rate[:, sl], recv[:, sl]


def sp(a, b):
    v = spearmanr(a.ravel(), b.ravel()).statistic
    return float(v) if np.isfinite(v) else None


def shuf_rows(a):
    return np.stack([row[rng.permutation(row.size)] for row in a])


rho = sp(r_, rv)
null = [abs(sp(shuf_rows(r_), rv) or 0.0) for _ in range(100)]
bw = [rng.integers(0, W, W) for _ in range(200)]
rho_b = [sp(r_[b], rv[b]) for b in bw]
P4 = dict(rho=rho, null_abs_rho_p95=float(np.percentile(null, 95)),
          rho_ci=[float(np.nanpercentile([v if v is not None else np.nan for v in rho_b], q))
                  for q in (2.5, 97.5)],
          rho_per_layer_twin=[sp(r_, RECV[l][:, sl]) for l in range(L)],
          rho_continuous_minm=sp(-M.min(0)[:, sl], rv),
          rate_nonzero_sites=float((r_ > 0).mean()))


def per_layer_ci(lst):
    out = []
    for l in range(L):
        v = torch.cat([t for (ll, t) in lst if ll == l]).numpy()
        bs = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(1000)]
        out.append(dict(mean=float(v.mean()), ci=[float(np.percentile(bs, 2.5)),
                                                  float(np.percentile(bs, 97.5))]))
    return out


FTs = {k: per_layer_ci(v) for k, v in FT.items()}
DISTs = {k: per_layer_ci(v) for k, v in DIST.items()}
uniform_ft = float(np.mean(1.0 / (np.arange(1, S) + 1)))

col = [sp(M[l], FG[l]) for l in range(L)]
null_c = [abs(sp(shuf_rows(M[1]), FG[1]) or 0.0) for _ in range(100)]
cross = [[sp(M[i], FG[j]) for j in range(L)] for i in range(L)]
fg_delim_lift = [float(delim[FG[l] < np.percentile(FG[l], 5)].mean() / delim.mean()) for l in range(L)]

res = dict(
    P1=P1, P2=P2, P3=P3, P4=P4,
    FT=dict(a=[d["mean"] for d in FTs["a"]], f=[d["mean"] for d in FTs["f"]],
            fox=[d["mean"] for d in FTs["fox"]], detail=FTs, uniform=uniform_ft),
    attn_distance=DISTs,
    X_colocate=dict(rho_per_layer=col, null_abs_rho_p95=float(np.percentile(null_c, 95)),
                    cross_layer=cross, fox_bottom5pct_forget_delim_lift=fg_delim_lift),
    m_quantiles=[[float(np.quantile(M[l], q)) for q in (0.001, 0.01, 0.05, 0.25, 0.5)] for l in range(L)],
    fox_f_quantiles=[[float(np.quantile(FG[l], q)) for q in (0.001, 0.01, 0.05, 0.25, 0.5)] for l in range(L)],
    beta_f=beta, instrument_max_err=max(errs), n_windows=W, S=S, thr=THR, K=K,
    seconds=time.time() - t0)
with open(os.path.join(HERE, "rsink.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res, indent=1))
