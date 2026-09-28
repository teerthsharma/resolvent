"""BAR.md runs: ranking bed (quality, 3 seeds, cpu) then per-decision cost (gpu). One process.

    python sun/rjepa/chase/bench.py quality   -> results_quality.json
    python sun/rjepa/chase/bench.py speed     -> results_speed.json
"""
import json
import math
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rjepa_ops import set_resolvent, neumann_resolvent, stochastic_resolvent  # noqa: E402

DZ, DA, K, S_ERR = 16, 8, 16, 0.8


# ------------------------------------------------------------------ bed
def bed(n, seed):
    B = torch.randn(DA, DZ, generator=torch.Generator().manual_seed(1234)) / math.sqrt(DA)
    g_ = torch.Generator().manual_seed(seed)
    g = torch.randn(n, 1, DZ, generator=g_)
    z = g + 1.5 * torch.randn(n, K, DZ, generator=g_)
    a = torch.randn(n, K, DA, generator=g_)
    zhat = z + S_ERR * torch.tanh(a @ B)
    x = torch.cat([zhat - g, a], -1)
    y = -(z - g).norm(dim=-1)
    return x, y


def regret(y, pick):
    yb, ym = y.max(-1).values, y.mean(-1)
    return float(((yb - y.gather(-1, pick[:, None])[:, 0]) / (yb - ym)).mean())


# ------------------------------------------------------------------ arms
class Arm(nn.Module):
    def __init__(self, kind, h=64, d_in=DZ + DA, hops=None):
        super().__init__()
        self.kind, self.h, self.hops = kind, h, hops
        self.enc = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, h))
        if kind == "deepsets":
            self.mix = nn.Sequential(nn.Linear(2 * h, 2 * h), nn.GELU(), nn.Linear(2 * h, h))
        elif kind in ("settf", "resolvent"):
            self.qkv, self.o = nn.Linear(h, 3 * h), nn.Linear(h, h)
        elif kind == "pointwise":
            self.mix = nn.Sequential(nn.Linear(h, 4 * h), nn.GELU(), nn.Linear(4 * h, h))
        self.ln = nn.LayerNorm(h)
        self.ff = nn.Sequential(nn.Linear(h, 2 * h), nn.GELU(), nn.Linear(2 * h, h))
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        e = self.enc(x)
        if self.kind == "deepsets":
            e = e + self.mix(torch.cat([e, e.mean(-2, keepdim=True).expand_as(e)], -1))
        elif self.kind == "pointwise":
            e = e + self.mix(e)
        else:
            q, k, v = self.qkv(self.ln(e)).chunk(3, -1)
            if self.kind == "settf":
                sh = lambda t: t.unflatten(-1, (4, self.h // 4)).transpose(-2, -3)
                m = F.scaled_dot_product_attention(sh(q), sh(k), sh(v)).transpose(-2, -3).flatten(-2)
            elif self.hops is None:
                m = set_resolvent(q, k, v, rho=0.9)
            else:
                m = neumann_resolvent(q, k, v, rho=0.9, hops=self.hops)
            e = e + self.o(m)
        e = e + self.ff(e)
        return self.head(e)[..., 0]


class DJ(nn.Module):
    """D-JEPA operator shell (Wilson's spec, BAR.md A1); `mixer` is the only free variable."""

    def __init__(self, mixer, h=64, nh=4, d_in=DZ + DA + 1):
        super().__init__()
        self.mixer, self.h, self.nh = mixer, h, nh
        self.lnz = nn.LayerNorm(DZ)
        self.enc = nn.Sequential(nn.Linear(d_in, h), nn.LayerNorm(h), nn.GELU())
        self.layers = nn.ModuleList(nn.ModuleDict(dict(
            ln1=nn.LayerNorm(h), qkv=nn.Linear(h, 3 * h), o=nn.Linear(h, h),
            ln2=nn.LayerNorm(h), ff=nn.Sequential(nn.Linear(h, 128), nn.GELU(), nn.Linear(128, h))))
            for _ in range(2))
        self.down, self.up = nn.Linear(h, 8), nn.Linear(8, 1)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, x):  # returns s (lower = better) and delta
        Kc = x.shape[-2]
        c = x[..., :DZ].norm(dim=-1)
        r = c.argsort(-1).argsort(-1).to(x.dtype) / max(Kc - 1, 1)
        e = self.enc(torch.cat([self.lnz(x[..., :DZ]), x[..., DZ:], r[..., None]], -1))
        sh = lambda t: t.unflatten(-1, (self.nh, self.h // self.nh)).transpose(-2, -3)
        for L in self.layers:
            q, k, v = map(sh, L["qkv"](L["ln1"](e)).chunk(3, -1))
            if self.mixer == "sdpa":
                m = F.scaled_dot_product_attention(q, k, v)
            elif self.mixer == "signed":
                m = set_resolvent(q, k, v, rho=0.9)
            else:
                m = stochastic_resolvent(q, k, v, g=0.9)
            e = e + L["o"](m.transpose(-2, -3).flatten(-2))
            e = e + L["ff"](L["ln2"](e))
        delta = 0.2 * torch.tanh(self.up(torch.tanh(self.down(e))))[..., 0]
        return r + delta, delta


def dj_loss(s, delta, y, T=0.05, mu=0.02):
    best = y.argmax(-1)
    ce = F.cross_entropy(-s / T, best)
    sb = s.gather(-1, best[:, None])
    local = F.softplus((mu + sb - s) / T)
    local = (local.sum(-1) - F.softplus(torch.tensor(mu / T))) / (s.shape[-1] - 1)  # drop the i=j pair
    return ce + 0.25 * local.mean() + 0.1 * (delta ** 2).mean()


def quality_dj():
    torch.set_num_threads(4)
    xe, ye = bed(4000, 99)
    out = {"floor_dist": regret(ye, (-xe[..., :DZ].norm(dim=-1)).argmax(-1)), "arms": {}}
    for mixer in ["sdpa", "signed", "stoch"]:
        rows = []
        for seed in range(3):
            torch.manual_seed(seed)
            xt, yt = bed(20000, 1000 + seed)
            m = DJ(mixer)
            opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=1e-4)
            t0, r = time.time(), {"seed": seed}
            for step in range(1, 3001):
                i = torch.randint(0, 20000, (256,))
                loss = dj_loss(*m(xt[i]), yt[i])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                if step in (1000, 3000):
                    with torch.no_grad():
                        r[f"regret_{step}"] = regret(ye, m(xe)[0].argmin(-1))
            r["train_s"] = round(time.time() - t0, 1)
            with torch.no_grad():
                p64 = m.double()(xe.double())[0].argmin(-1)
                pb = m.to(torch.bfloat16)(xe.bfloat16())[0].argmin(-1)
                r["bf16_argmax_flip_vs_f64"] = float((pb != p64).float().mean())
            rows.append(r)
            print(mixer, r, flush=True)
        out["arms"][mixer] = {"params": nparams(DJ(mixer)), "runs": rows,
                              "mean_regret_3000": sum(r["regret_3000"] for r in rows) / 3,
                              "mean_regret_1000": sum(r["regret_1000"] for r in rows) / 3}
    json.dump(out, open(os.path.join(HERE, "results_quality_dj.json"), "w"), indent=1)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def quality():
    torch.set_num_threads(4)
    xe, ye = bed(4000, 99)
    out = {"floor_dist": regret(ye, (-xe[..., :DZ].norm(dim=-1)).argmax(-1)),
           "oracle": regret(ye, ye.argmax(-1)), "random": regret(ye, torch.zeros(4000, dtype=torch.long)),
           "arms": {}}
    for kind in ["pointwise", "deepsets", "settf", "resolvent"]:
        rows = []
        for seed in range(3):
            torch.manual_seed(seed)
            xt, yt = bed(20000, 1000 + seed)
            m = Arm(kind)
            opt = torch.optim.Adam(m.parameters(), 3e-3)
            t0 = time.time()
            for step in range(3000):
                i = torch.randint(0, 20000, (256,))
                loss = F.cross_entropy(m(xt[i]), yt[i].argmax(-1))
                opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                r = {"seed": seed, "regret": regret(ye, m(xe).argmax(-1)), "train_s": round(time.time() - t0, 1)}
                if kind in ("resolvent", "settf"):
                    m64 = m.double(); p64 = m64(xe.double()).argmax(-1)
                    mb = m.to(torch.bfloat16); pb = mb(xe.bfloat16()).argmax(-1)
                    r["bf16_argmax_flip_vs_f64"] = float((pb != p64).float().mean())
                    m = m.float()
                if kind == "resolvent":
                    for h in (1, 2, 4, 8):
                        m.hops = h
                        r[f"regret_neumann_h{h}"] = regret(ye, m(xe).argmax(-1))
                    m.hops = None
            rows.append(r)
            print(kind, r, flush=True)
        out["arms"][kind] = {"params": nparams(Arm(kind)), "runs": rows,
                             "mean_regret": sum(r["regret"] for r in rows) / 3}
    json.dump(out, open(os.path.join(HERE, "results_quality.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}))


# ------------------------------------------------------------------ speed
def predictor(d=64, w=256):  # registered toy rollout step, deliberately small
    return nn.Sequential(nn.Linear(d, w), nn.GELU(), nn.Linear(w, w), nn.GELU(),
                         nn.Linear(w, w), nn.GELU(), nn.Linear(w, d))


def timeit(fn, reps=30, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    ts.sort()
    return {"median_us": round(ts[len(ts) // 2], 1), "p25": round(ts[len(ts) // 4], 1),
            "p75": round(ts[3 * len(ts) // 4], 1)}


def speed():
    dev, H, d = "cuda", 8, 64
    load = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    P = predictor(d).to(dev)
    p_pred = nparams(P)
    out = {"gpu_load_at_start": load, "H": H, "pred_params": p_pred, "rows": []}
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        for Kc in (4, 16, 64, 256, 1024):
            for B in (1, 64):
                z = torch.randn(B, Kc, d, device=dev); g = torch.randn(B, 1, d, device=dev)
                x = torch.randn(B, Kc, DZ + DA, device=dev)

                def rollout():
                    s = z
                    for _ in range(H):
                        s = P(s)
                    return s
                arms = {"dist": lambda: (z - g).norm(dim=-1).argmax(-1)}
                for kind, hops in (("deepsets", None), ("settf", None), ("resolvent", None), ("resolvent", 4)):
                    m = Arm(kind, hops=hops).to(dev).eval()
                    arms[kind + ("" if hops is None else f"_h{hops}")] = (lambda m=m: m(x).argmax(-1))
                for mixer in ("sdpa", "signed", "stoch"):
                    m = DJ(mixer).to(dev).eval()
                    arms["dj_" + mixer] = (lambda m=m: m(x)[0].argmin(-1))
                row = {"K": Kc, "B": B, "rollout": timeit(rollout)}
                for n, f in arms.items():
                    row[n] = timeit(f)
                h = 64
                row["flops"] = {
                    "rollout": B * Kc * H * 2 * p_pred,
                    "dist": B * Kc * d * 3,
                    "resolvent_mixer": B * (2 * Kc * Kc * h + 2 * Kc ** 3 / 3 + 2 * Kc * Kc * h),
                    "settf_mixer": B * (4 * Kc * Kc * h),
                    "resolvent_neumann_h4_mixer": B * (2 * Kc * Kc * h + 4 * 2 * Kc * Kc * h)}
                print(json.dumps(row), flush=True)
                out["rows"].append(row)
    out["gpu_load_at_end"] = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    json.dump(out, open(os.path.join(HERE, "results_speed.json"), "w"), indent=1)


if __name__ == "__main__":
    {"quality": quality, "quality_dj": quality_dj, "speed": speed}[sys.argv[1]]()
