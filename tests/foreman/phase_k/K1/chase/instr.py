"""Chase K1 instruments on trained (f_R) / (a_L) checkpoints. Measures only; the bars live in test_chase_k1_ckpt.py.
Capture: the resolvent layer's own fp32 qs (= s_i q_i), k, v via ResolventAttention.capture, from a forward run under the
trainer's bf16 autocast; ALiBi heads' q, k recomputed in fp32 from each attention's input (forward hook).
Checkpoint formats: Cameron rdepth.py (<run>/model.pt + log.jsonl config, finished = result.json) and train_ladder
(<run>/ckpt.pt with args, finished = an "end" record in log.jsonl)."""
import glob, hashlib, importlib.util, json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F

REPO = "C:/Users/seal/Desktop/New folder (32)"
SP = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad"
K1 = SP + "/phase_k/K1"
sys.path.insert(0, REPO + "/tests/foreman/phase_k/K0/chase")
import resolvent as R


def load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


hk = load(K1 + "/chase/resolvent_hook.py", "chase_hook_instr")
GAMMA, C = hk.GAMMA, hk.C
DELTA = 32


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ---------------- discovery ----------------

def find_ckpts():
    """[{lane, arm, path, cfg}] for finished runs. arm in {fR, aL}; fR only at R0 (4 layers, width 128)."""
    out = []
    for p in sorted(glob.glob(K1 + "/cameron/**/model.pt", recursive=True)):
        d = os.path.dirname(p)
        if not os.path.exists(os.path.join(d, "result.json")):
            continue
        cfg = json.loads(open(os.path.join(d, "log.jsonl")).readline())
        if cfg.get("arm") == "fR" and cfg.get("layers") == 4 or cfg.get("arm") == "aL":
            out.append({"lane": "cameron", "arm": cfg["arm"], "path": p, "cfg": cfg})
    for p in sorted(glob.glob(K1 + "/wilson/**/ckpt.pt", recursive=True)):
        d = os.path.dirname(p)
        lg = os.path.join(d, "log.jsonl")
        if not os.path.exists(lg) or not any('"t": "end"' in l for l in open(lg)):
            continue
        args = torch.load(p, map_location="cpu", weights_only=False)["args"]
        attn = args["attn"]
        arm = "fR" if ("resolvent_hook" in attn and attn.endswith(":FR")) else ("aL" if attn == "alibi" else None)
        r0 = args["rung"] == "R0" and args.get("layers") in (None, 4) and args.get("width") in (None, 128)
        if arm and (r0 or arm == "aL"):
            out.append({"lane": "wilson", "arm": arm, "path": p, "cfg": args})
    return out


# ---------------- capture ----------------

def _alibi_hooks(model, store):
    hs = []
    for li, blk in enumerate(model.blocks):
        a = blk.attn
        if hasattr(a, "slopes"):
            def fh(mod, inp, out, li=li):
                x = inp[0].float()
                B, T, Dm = x.shape
                q, k, _ = F.linear(x, mod.qkv.weight.float()).split(Dm, 2)
                q, k = (t.view(B, T, mod.n_heads, Dm // mod.n_heads).transpose(1, 2) for t in (q, k))
                store.append({"layer": li, "q": q, "k": k, "slopes": mod.slopes.float()})
            hs.append(a.register_forward_hook(fh))
    return hs


def _res_layer(model):
    rl = [b.attn for b in model.blocks if hasattr(b.attn, "capture") and hasattr(b.attn, "a")]
    return rl[0] if rl else None


@torch.no_grad()
def capture(ck, lengths):
    """{S: {"res": dict(qs,k,v) or None, "alibi": [..], "bed": bed or None}} for each eval length."""
    dev = torch.device("cuda")
    res = {}
    if ck["lane"] == "cameron":
        sys.path.insert(0, K1 + "/cameron")
        rd = load(K1 + "/cameron/rdepth.py", "cam_rdepth")
        cfg = ck["cfg"]
        hook_cls = rd.load_hook(cfg["hook"])[0] if cfg["arm"] == "fR" else None
        model = rd.Chain(cfg["arm"], cfg["layers"], cfg["dff"], cfg["loops"], hook_cls)
        model.load_state_dict(torch.load(ck["path"], map_location="cpu"))
        model = model.to(dev).eval()
        for S in lengths:
            bed = rd.make_train(np.random.default_rng([13, 0])) if S == 1024 else \
                rd.make_test(np.random.default_rng([12, S, 0]), S)
            ids = torch.from_numpy(bed["ids"].astype(np.int64))[None].to(dev)
            pids = torch.from_numpy(bed["pids"].astype(np.int64))[None].to(dev)
            res[S] = _run(model, lambda: model(ids, pids))
            res[S]["bed"] = bed
            rd._BIAS_CACHE.clear()
            torch.cuda.empty_cache()
    else:
        a = ck["cfg"]
        TL = load(REPO + "/tests/foreman/phase_k/K0/wilson/train_ladder.py", "k0_tl_instr")
        if ck["arm"] == "fR":
            attn_cls = load(a["attn"].rsplit(":", 1)[0], "wil_hook").FR
        else:
            attn_cls = TL.AlibiAttention
        L, W = TL.RUNGS[a["rung"]]
        L, W = a.get("layers") or L, a.get("width") or W
        model = TL.GPT(L, W, a["ctx"], attn_cls)
        model.load_state_dict(TL.strip_prefix(torch.load(ck["path"], map_location="cpu", weights_only=False)["model"]))
        model = model.to(dev).eval()
        es = a.get("eval_seed")
        x, _ = TL.build_eval_batches(TL.load_shards(a["data_dir"], "val_*.bin"), a["seed"] if es is None else es,
                                     1, 2, a["ctx"])[0]
        res[a["ctx"]] = _run(model, lambda: model(x.to(dev)))
        res[a["ctx"]]["bed"] = None
    return res


def _run(model, fwd):
    rl = _res_layer(model)
    store = []
    hs = _alibi_hooks(model, store)
    if rl is not None:
        rl.capture = []
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fwd()
    finally:
        for h in hs:
            h.remove()
    out = {"res": rl.capture[0] if rl is not None else None, "alibi": store,
           "a": rl.a.detach().cpu().tolist() if rl is not None else None,
           "b": rl.b.detach().cpu().tolist() if rl is not None else None}
    if rl is not None:
        rl.capture = None
    return out


# ---------------- numerics (clause 6) ----------------

def rowsums(qs, k, c=C):
    """max_i rowsum of P (fp32 max-shifted softmax row blocks, as backward_blocked forms them) and of the mutant P rounded
    to bf16; count of mutant rows with GAMMA * rowsum >= 1. Never a full W."""
    S = qs.shape[-2]
    mx, mxb, nbad = 0.0, 0.0, 0
    for lo in range(0, S, c):
        hi = min(S, lo + c)
        P = torch.softmax(R._logits_rows(qs, k, lo, hi), -1)
        mx = max(mx, float(P.sum(-1).max()))
        rb = P.bfloat16().float().sum(-1)
        mxb = max(mxb, float(rb.max()))
        nbad += int((GAMMA * rb >= 1).sum())
    return {"gamma_max_rowsum": GAMMA * mx, "gamma_max_rowsum_bf16": GAMMA * mxb, "rows_bf16_ge1": nbad,
            "rows": int(qs.shape[0] * qs.shape[1] * S)}


def _pad(t, c=C):
    p = -t.shape[-2] % c
    return F.pad(t, (0, 0, 0, p)) if p else t


def read1(qs, k):
    """max |x - 1| for V = 1: the hook's read (fs5c, graphed, fp32) and the mutant K0 fs5 (naive pivot)."""
    S = qs.shape[-2]
    one = torch.ones_like(k)
    x = hk.read(qs, k, one)
    x5 = R.fused_forward(_pad(qs), _pad(k), _pad(one), GAMMA)[..., :S, :]
    return {"read1_dev": float((x - 1).abs().max()), "read1_dev_fs5": float((x5 - 1).abs().max()),
            "finite": bool(torch.isfinite(x).all())}


# ---------------- BOS share (clause 7) ----------------

def bos_share(qs, k, g=GAMMA):
    """R(i, 0) = (1-g) G(i, 0) per head: the read with V = e_0 (position 0), fused path. Mean over batch and rows
    i in [S/2, S), and the last row."""
    S = qs.shape[-2]
    V = torch.zeros_like(k)
    V[..., 0, :] = 1.0
    x = hk.read(qs, k, V, g=g)[..., 0]                   # [B, H, S]
    return {"mean_2nd_half": x[..., S // 2:].mean((0, 2)).tolist(), "last_row": x[..., -1].mean(0).tolist()}


# ---------------- range (R-RANGE) ----------------

def rows_for(S, T=64):
    return np.unique(S - 1 - np.round(np.arange(T) * (S / 2) / T).astype(int))[::-1].copy()


def r_for(S):
    return np.arange(16, S // 2 - DELTA)


def resolvent_rows(qs, k, rows, g=GAMMA):
    """R(i_t, j) = (1-g) G(i_t, j) for the given rows (<= D of them), by the adjoint: backward_blocked with gx = one-hots."""
    S = qs.shape[-2]
    qp, kp = _pad(qs), _pad(k)
    gx = torch.zeros_like(qp)
    t = torch.arange(len(rows), device=qs.device)
    gx[:, :, torch.as_tensor(rows, device=qs.device), t] = 1.0
    _, _, dv = R.backward_blocked(qp, kp, torch.zeros_like(qp), gx, g)
    return dv[..., :S, :len(rows)].transpose(-1, -2)     # [B, H, T, S]


def hop_rows(q, k, rows, slopes=None):
    """single-hop W rows: softmax(q_i.k_j / sqrt(D) - slope (i - j)) over j <= i."""
    S, D = k.shape[-2], k.shape[-1]
    ri = torch.as_tensor(rows, device=q.device)
    z = (q[..., ri, :] @ k.transpose(-1, -2)) / math.sqrt(D)
    dist = ri[:, None] - torch.arange(S, device=q.device)[None, :]
    if slopes is not None:
        z = z - slopes.view(-1, 1, 1) * dist.clamp(min=0).to(z.dtype)
    return torch.softmax(z.masked_fill(dist < 0, float("-inf")), -1)


def band_mass(K, rows, r, delta=DELTA):
    """B(r) = mean over batch, rows of sum_{i-r-delta < j <= i-r} K(i, j): direct window sums (no cumsum, no
    cancellation), j >= 1 always (position 0, the absorber, excluded). K: [B, H, T, S] torch (any device). -> [H, len(r)]."""
    Bn, H, T, S = K.shape
    win = F.avg_pool1d(K.reshape(-1, 1, S).double(), delta, stride=1).reshape(Bn, H, T, -1) * delta
    idx = torch.as_tensor(np.asarray(rows)[:, None] - np.asarray(r)[None, :] - delta + 1, device=K.device)
    assert int(idx.min()) >= 1, "band reaches position 0"
    return win.gather(-1, idx.expand(Bn, H, T, -1)).mean((0, 2)).cpu().numpy()


def fit_mass(Bm, r):
    """m = -slope of ln B(r), least squares weighted by B (np.polyfit w = sqrt(B)); points with B > 0. range = 1/m."""
    ok = Bm > 0
    if ok.sum() < 10:
        return float("inf")
    return float(-np.polyfit(np.asarray(r)[ok], np.log(Bm[ok]), 1, w=np.sqrt(Bm[ok]))[0])


def rng_of(m):
    return 1.0 / m if m > 0 else float("inf")


def gamma_h(lam, Dt):
    """Y2 inverted: dressed range = Dt  <=>  g = (e^lam - e^(1/Dt)) / (e^lam - 1); clipped to [0, 0.999]."""
    if not lam > 0 or not math.isfinite(lam):
        return 0.0
    return float(min(max((math.exp(lam) - math.exp(1.0 / Dt)) / (math.exp(lam) - 1.0), 0.0), 0.999))


# ---------------- synthetic Toeplitz heads (CPU float64, K0 rrange) ----------------

def toeplitz_band_mass(lam, g, sigma=0.0, n=3000, rows=np.arange(2000, 2901, 100), r=np.arange(30, 400)):
    import rrange as RR
    W = RR.toeplitz_head(lam, n)
    if sigma:
        W = (1 - sigma) * W
        W[:, 0] += sigma
    K = (1 - g) * RR.resolvent_rows(W, g, rows)            # [T, n]
    Bm = band_mass(torch.from_numpy(K)[None, None], rows, r)[0]
    return fit_mass(Bm, r), W


def toeplitz_tail_mass(W, g, rows=np.arange(2000, 2901, 100), r=np.arange(30, 400)):
    import rrange as RR
    return RR.head_mass_tail(W, g, rows, r)
