"""R-DIAG': is the Mode B equilibrium read of the trained family well-posed at
gamma = 0.99, or past a pole?  CPU only.

Checkpoint d45_ckpt_f_ss0_pair/model.pt, rebuilt by abstention_deciles.build_arm("f")
(hard-concrete magnitude, repaired init), read on abstention_deciles.eval_batches()
(8 batches x 8 windows x 512 bytes). Per layer and head, W = G e^{s} / Z^beta is formed
by ceq/arm_smprime.py::operator with the layer's own q, k, u, theta, beta, qk, g --
the call CEQAttention._smprime makes -- in float32 (the model's dtype) and again in
float64 from the same float32 inputs. W is lower triangular, so its spectrum is its
diagonal and the resolvent (I - gamma W)^{-1} has a pole at gamma = 1 / W_ii.
"""
import json
import math
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU only: Foreman holds the GPU
HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPO = r"C:\Users\seal\Desktop\New folder (32)"
sys.path.insert(0, SP)
sys.path.insert(0, REPO)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import abstention_deciles as AD  # noqa: E402
import design4x5 as D  # noqa: E402

D.DEVICE = "cpu"   # the env var alone did not hide the GPU on this box (run 1 asserted)
from ceq import arm_smprime as A  # noqa: E402

OUT = os.path.join(HERE, "rdiag_prime.json")
GAMMA = 0.99
WMAX = 1.0 / GAMMA
TOL64 = 1e-6


def _capture(model, x):
    layers = model.model.layers
    ins, outs, hs = {}, {}, []
    for i, blk in enumerate(layers):
        hs.append(blk.self_attn.register_forward_pre_hook(
            lambda m, a, i=i: ins.__setitem__(i, a[0].detach())))
        hs.append(blk.self_attn.register_forward_hook(
            lambda m, a, o, i=i: outs.__setitem__(i, o.detach())))
    try:
        with torch.no_grad():
            logits = model(input_ids=x).logits
    finally:
        for h in hs:
            h.remove()
    return logits, ins, outs


def _heads(at, h):
    b, s, _ = h.shape
    q, k, v = at.qkv(h).chunk(3, dim=-1)
    shp = lambda t: t.view(b, s, at.n_heads, at.d_head).transpose(1, 2)
    u = at.m_head(h).squeeze(-1).unsqueeze(-2)
    th = at.theta_head(h).squeeze(-1).unsqueeze(-2)
    return shp(q), shp(k), shp(v), u, th


def _offdiag_zero(M):
    """[.., S] bool: row i has every entry j < i exactly zero."""
    s = M.shape[-1]
    lo = torch.ones(s, s, dtype=torch.bool).tril(-1)
    return ((M == 0) | ~lo).all(-1)


def _new_acc():
    return dict(gstar=[], argmax_i=[], top=(-1.0, None), top_excl0=(-1.0, None),
                unit=0, n_diag=0, n_gt1=0, n_gtWMAX=0, iso_excl0=0,
                absorbers_excl_pos0=0, gstar_excl0=[])


def _fold(acc, d, iso, unit_mask, w_index0):
    """d: [B,H,S] real diagonal; iso: [B,H,S] off-diagonal-zero rows."""
    B, H, S = d.shape
    acc["n_diag"] += d.numel()
    acc["unit"] += int(unit_mask.sum())
    acc["n_gt1"] += int((d > 1).sum())
    acc["n_gtWMAX"] += int((d > WMAX).sum())
    mx, am = d.max(-1)
    mx0, am0 = d[..., 1:].max(-1)
    acc["gstar"] += (1.0 / mx.double()).reshape(-1).tolist()
    acc["gstar_excl0"] += (1.0 / mx0.double()).reshape(-1).tolist()
    acc["argmax_i"] += am.reshape(-1).tolist()
    iso0 = iso.clone()
    iso0[..., 0] = False
    acc["iso_excl0"] += int(iso0.sum())
    acc["absorbers_excl_pos0"] += int((iso0 & unit_mask).sum())
    flat = int(d.reshape(-1).argmax())
    b, h, i = flat // (H * S), (flat // S) % H, flat % S
    if float(d[b, h, i]) > acc["top"][0]:
        acc["top"] = (float(d[b, h, i]), dict(window=w_index0 + b, head=h, pos=i))
    flat0 = int(d[..., 1:].reshape(-1).argmax())
    b0, h0, i0 = flat0 // (H * (S - 1)), (flat0 // (S - 1)) % H, flat0 % (S - 1) + 1
    if float(d[b0, h0, i0]) > acc["top_excl0"][0]:
        acc["top_excl0"] = (float(d[b0, h0, i0]), dict(window=w_index0 + b0, head=h0, pos=i0))


def _summ(acc):
    g = torch.tensor(acc["gstar"], dtype=torch.float64)
    g0 = torch.tensor(acc["gstar_excl0"], dtype=torch.float64)
    am = torch.tensor(acc["argmax_i"])
    return dict(
        n_pairs=g.numel(), n_diag=acc["n_diag"], unit_diagonals=acc["unit"],
        max_Wii=acc["top"][0], max_Wii_at=acc["top"][1],
        gstar_min=float(g.min()), gstar_median=float(g.median()),
        n_pairs_gstar_lt_099=int((g < GAMMA).sum()),
        frac_pairs_gstar_lt_099=float((g < GAMMA).double().mean()),
        frac_pairs_argmax_at_pos0=float((am == 0).double().mean()),
        excl_pos0=dict(max_Wii=acc["top_excl0"][0], max_Wii_at=acc["top_excl0"][1],
                       gstar_min=float(g0.min()),
                       n_pairs_gstar_lt_099=int((g0 < GAMMA).sum()),
                       frac_pairs_gstar_lt_099=float((g0 < GAMMA).double().mean())),
        n_positions_Wii_gt_1=acc["n_gt1"], n_positions_Wii_gt_1_over_099=acc["n_gtWMAX"],
        isolated_rows_excl_pos0=acc["iso_excl0"],
        absorbers_excl_pos0=acc["absorbers_excl_pos0"],
        gstar_all=[round(x, 7) for x in acc["gstar"]],
    )


def _neumann_growth(W, v, steps=400):
    """Power/Neumann iterate x <- gamma W x on one head (float64), return the
    per-step growth ratio over the last 50 steps."""
    x = v.to(W.dtype)
    norms = []
    for _ in range(steps):
        x = GAMMA * (W @ x)
        n = float(x.abs().norm())
        norms.append(n)
        if n > 0:
            x = x / n                             # renormalise; ratios are what we read
    return float(torch.tensor(norms[-50:], dtype=torch.float64).log().mean().exp())


def run():
    t0 = time.time()
    model, blob = AD.build_arm("f")
    assert next(model.parameters()).device.type == "cpu"
    layers = model.model.layers
    L = len(layers)
    sd = blob["state_dict"]
    beta_sd = [float(sd["model.layers.{}.self_attn.beta".format(i)]) for i in range(L)]

    acc = {k: [_new_acc() for _ in range(L)] for k in ("f32", "f64", "ReW")}
    mstat = [dict(n=0, zeros_all=0, zeros_excl0=0, m_min=1e9, m_min_excl0=1e9) for _ in range(L)]
    diag_imag_nonzero = [0] * L
    rew_diag_eq = [True] * L
    nll_sum, nll_n, attn_diff = 0.0, 0, 0.0
    worst = dict(val=-1.0)

    for bidx, x in enumerate(AD.eval_batches()):
        x = x.cpu()
        logits, ins, outs = _capture(model, x)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                             x[:, 1:].reshape(-1), reduction="sum")
        nll_sum += float(ce)
        nll_n += x[:, 1:].numel()
        w0 = bidx * x.shape[0]
        for li, blk in enumerate(layers):
            at = blk.self_attn
            with torch.no_grad():
                q, k, v, u, th = _heads(at, ins[li])
                m, _ = A.blend(u, th, at.g)                                   # [B,1,S]
                st = mstat[li]
                st["n"] += m.numel()
                st["zeros_all"] += int((m == 0).sum())
                st["zeros_excl0"] += int((m[..., 1:] == 0).sum())
                st["m_min"] = min(st["m_min"], float(m.min()))
                st["m_min_excl0"] = min(st["m_min_excl0"], float(m[..., 1:].min()))

                W = A.operator(q, k, u, th, beta=at.beta, qk=at.qk, g=at.g)  # model's call
                if bidx == 0:
                    o = (W @ v.to(W.dtype)).real
                    b, s = x.shape
                    re = at.o_proj(o.transpose(1, 2).reshape(b, s, -1))
                    attn_diff = max(attn_diff, float((re - outs[li]).abs().max()))
                dg = W.diagonal(dim1=-2, dim2=-1)
                diag_imag_nonzero[li] += int((dg.imag != 0).sum())
                d32 = dg.real
                rw = W.real
                rd = rw.diagonal(dim1=-2, dim2=-1)
                rew_diag_eq[li] &= bool(torch.equal(rd, d32))
                iso32 = _offdiag_zero(W)
                _fold(acc["f32"][li], d32, iso32, d32 == 1.0, w0)
                _fold(acc["ReW"][li], rd, _offdiag_zero(rw), rd == 1.0, w0)
                del rw, iso32

                beta64, qk64, g64 = (at.beta.double(), at.qk.double(), at.g.double())
                d64s, iso64s = [], []
                for bi in range(x.shape[0]):
                    W64 = A.operator(q[bi].double(), k[bi].double(), u[bi].double(),
                                     th[bi].double(), beta=beta64, qk=qk64, g=g64)
                    d = W64.diagonal(dim1=-2, dim2=-1).real
                    d64s.append(d)
                    iso64s.append(_offdiag_zero(W64))
                    mv, fl = d.reshape(-1).max(0)
                    if float(mv) > worst["val"]:
                        h, i = int(fl) // d.shape[-1], int(fl) % d.shape[-1]
                        wii = float(qk64 * (q[bi, h, i].double() @ k[bi, h, i].double())
                                    / math.sqrt(q.shape[-1]))
                        lnZ = (wii - math.log(float(mv))) / float(beta64)
                        worst = dict(val=float(mv), layer=li, window=w0 + bi, head=h, pos=i,
                                     s_ii=wii, lnZ=lnZ, p_ii=math.exp(wii - lnZ),
                                     one_minus_beta_s_ii=(1 - float(beta64)) * wii,
                                     neumann_growth_per_step=_neumann_growth(
                                         W64[h], v[bi, h, :, 0].double()),
                                     rho_gammaW=GAMMA * float(mv))
                    del W64
                d64 = torch.stack(d64s)
                _fold(acc["f64"][li], d64, torch.stack(iso64s),
                      (d64 - 1).abs() <= TOL64, w0)
                del W
        print("[rdiag'] batch {} done  {:.0f}s".format(bidx, time.time() - t0), flush=True)

    out = dict(
        checkpoint=AD.ckpt_path("f"), gamma=GAMMA, tol64=TOL64,
        eval_nll=nll_sum / nll_n, eval_tokens=nll_n,
        attn_out_max_abs_diff=attn_diff,
        n_windows=len(acc["f32"][0]["gstar"]) // layers[0].self_attn.n_heads,
        worst_f64=worst, seconds=time.time() - t0,
        layers=[dict(layer=li, beta=beta_sd[li],
                     beta_param=float(layers[li].self_attn.beta),
                     qk=float(layers[li].self_attn.qk), g=float(layers[li].self_attn.g),
                     gate=mstat[li], diag_imag_nonzero=diag_imag_nonzero[li],
                     ReW_diag_equals_W_diag_bitwise=rew_diag_eq[li],
                     f32=_summ(acc["f32"][li]), f64=_summ(acc["f64"][li]),
                     ReW=_summ(acc["ReW"][li]))
                for li in range(L)],
    )
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def results():
    return run()


if __name__ == "__main__":
    r = run()
    for Ld in r["layers"]:
        s = {k: v for k, v in Ld.items() if k not in ("f32", "f64", "ReW")}
        print(json.dumps(s))
        for k in ("f32", "f64", "ReW"):
            print(k, json.dumps({a: b for a, b in Ld[k].items() if a != "gstar_all"}))
    print(json.dumps({k: v for k, v in r.items() if k != "layers"}, indent=1))
