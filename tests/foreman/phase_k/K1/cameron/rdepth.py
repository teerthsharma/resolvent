"""R-DEPTH at R0 (Cameron, it.K1). --bed kp (default): bed_k' (bed_kp.py) chains, trained at n in {256, 512, 1024}
(M = 32 lanes, depth <= 32), read at n = 1024 (64 held-out train beds default_rng([21, k])) and 4096 / 8192 / 16384
(M = 8 lanes; the K1.F' beds default_rng([22, n, k])). --bed k: the voided bed_k with the K1.F seeds.
Arms (width d = 128, 2 heads of 64, the train_ladder.py copy's AlibiAttention and slopes; no position table):
  aL     ALiBi twin, --layers L; --dff sets the MLP width (L=7 is parameter-matched to L=4 at dff 183)
  ass    the twin with SSMax: content logits x s_h * ln(i+1), s_h learned (init 1/ln 1024)
  aloop  one ALiBi block, weight-shared, applied --loops T times
  fR     3 ALiBi blocks + 1 block whose attention is Chase's resolvent hook (--hook path:Class), run in fp32 outside
         autocast (RECORD_K clause 6)
Input: x = E[id] + P(E[pid]) (one id table, NULL = V for a root's parent, P a learned d x d map); readout tied to E.
Seeds: model torch.manual_seed(seed); train beds default_rng([1000 + seed, step]); eval beds fixed (above)."""
import argparse, contextlib, hashlib, importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from train_ladder import AlibiAttention, get_lr, _BIAS_CACHE
import bed_k, bed_kp
from bed_k import V, NULL
BED = {"k": (bed_k, 13, 12), "kp": (bed_kp, 21, 22), "kpf": (bed_kp, 21, 31)}   # module, held-out train seed, test seed
BANDS = {"band4": 96, "band5": 224, "band7": 896}   # Amendment K1-a far bands: depth > 2 c_max 2^L
OUTDIR = {"dir": None}
CUR = {"bed": "kp", "train_M": None, "tok": 8192, "train_ns": (256, 512, 1024)}

D, H = 128, 2
TRAIN_NS = (256, 512, 1024)
TOK_PER_STEP = 8192
EDGES = [0, 17, 33, 65, 129, 257, 513, 1025, 10 ** 9]


class ChunkAlibi(AlibiAttention):
    """The harness's AlibiAttention; in eval at T > 4096 the same bias is built per 2048-query block (memory)."""

    def _chunked(self, q, k, v):
        T, outs = q.shape[2], []
        for t0 in range(0, T, 2048):
            t1 = min(T, t0 + 2048)
            i = torch.arange(t0, t1, device=q.device, dtype=torch.float32).view(-1, 1)
            j = torch.arange(0, t1, device=q.device, dtype=torch.float32).view(1, -1)
            bias = (-self.slopes.to(q.device).view(-1, 1, 1) * (i - j)).masked_fill(j > i, float("-inf"))
            outs.append(F.scaled_dot_product_attention(q[:, :, t0:t1], k[:, :, :t1], v[:, :, :t1],
                                                       attn_mask=bias.unsqueeze(0).to(q.dtype)))
        return torch.cat(outs, 2)

    def qkv_heads(self, x):
        B, T, Dm = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=2)
        return [t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in (q, k, v)]

    def attend(self, q, k, v, x):
        B, T, Dm = x.shape
        if not self.training and T > 4096:
            y = self._chunked(q, k, v)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=self._bias(T, x.device, q.dtype))
        return self.out(y.transpose(1, 2).contiguous().view(B, T, Dm))

    def forward(self, x):
        q, k, v = self.qkv_heads(x)
        return self.attend(q, k, v, x)


class SSMaxAlibi(ChunkAlibi):
    def __init__(self, d_model, n_heads, layer_idx, n_layers, ctx):
        super().__init__(d_model, n_heads, layer_idx, n_layers, ctx)
        self.s = nn.Parameter(torch.full((n_heads,), 1.0 / math.log(1024)))

    def forward(self, x):
        q, k, v = self.qkv_heads(x)
        T = x.shape[1]
        ln = torch.log(torch.arange(1, T + 1, device=x.device, dtype=torch.float32)).view(1, 1, T, 1)
        q = q * (self.s.view(1, -1, 1, 1) * ln).to(q.dtype)
        return self.attend(q, k, v, x)


class Blk(nn.Module):
    def __init__(self, attn, dff, fp32_attn=False):
        super().__init__()
        self.ln1, self.attn, self.ln2, self.fp32_attn = nn.LayerNorm(D), attn, nn.LayerNorm(D), fp32_attn
        self.mlp = nn.Sequential(nn.Linear(D, dff, bias=False), nn.GELU(approximate="tanh"), nn.Linear(dff, D, bias=False))

    def forward(self, x):
        if self.fp32_attn:
            with torch.autocast("cuda", enabled=False):
                x = x + self.attn(self.ln1(x.float()))
        else:
            x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class Chain(nn.Module):
    def __init__(self, arm, layers, dff, loops, hook_cls, emb="learned", par_init="normal"):
        super().__init__()
        self.wte, self.par, self.ln_f = nn.Embedding(V + 1, D), nn.Linear(D, D, bias=False), nn.LayerNorm(D)
        mk = {"aL": ChunkAlibi, "ass": SSMaxAlibi, "aloop": ChunkAlibi, "fR": ChunkAlibi}[arm]
        n = 1 if arm == "aloop" else layers
        blocks = [Blk(mk(D, H, i, n, 1024), dff) for i in range(n)]
        if arm == "fR":
            blocks[-1] = Blk(hook_cls(D, H, n - 1, n, 1024), dff, fp32_attn=True)
        self.blocks, self.loops = nn.ModuleList(blocks), (loops if arm == "aloop" else 1)
        self.apply(self._init)
        if emb == "frozen":  # fixed random unit-norm id codes (input and tied readout); nothing id-specific is learned
            g = torch.Generator().manual_seed(12345)
            w = torch.randn(V + 1, D, generator=g)
            self.wte.weight.data.copy_(w / w.norm(dim=1, keepdim=True))
            self.wte.weight.requires_grad_(False)
        if par_init == "orth":  # the parent id enters at full strength in a rotated frame
            nn.init.orthogonal_(self.par.weight)
        for b in self.blocks:  # harness init: scaled residual projections
            nn.init.normal_(b.mlp[2].weight, std=0.02 / math.sqrt(2 * n * self.loops))
            if isinstance(getattr(b.attn, "out", None), nn.Linear):
                nn.init.normal_(b.attn.out.weight, std=0.02 / math.sqrt(2 * n * self.loops))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, pids):
        x = self.wte(ids) + self.par(self.wte(pids))
        for _ in range(self.loops):
            for b in self.blocks:
                x = b(x)
        return self.ln_f(x)

    def logits(self, h):
        return h @ self.wte.weight[:V].T


def load_hook(spec):
    path, cls = spec.rsplit(":", 1)
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    s = importlib.util.spec_from_file_location("resolvent_hook", path)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    HOOK["mod"] = mod
    return getattr(mod, cls), sha


HOOK = {"mod": None, "gamma": None}   # gamma override for the pilot schedule (None = the hook's fixed GAMMA)


def gamma_wrap(cls):
    """Pilot only: the hook's ResolventAttention with gamma read from HOOK['gamma'] (same parameters, same layer())."""
    class G(cls):
        def forward(self, x):
            if HOOK["gamma"] is None:
                return super().forward(x)
            with torch.autocast(x.device.type, enabled=False):
                return HOOK["mod"].layer(x.to(self.qkv.weight.dtype), self.qkv.weight, self.out.weight, self.a, self.b,
                                         self.n_heads, g=HOOK["gamma"], capture=self.capture)
    return G


def gamma_at(sched, step):
    for g, until in sched:
        if until is None or step < until:
            return g


def batch(seed, step, dev):
    rng = np.random.default_rng([1000 + seed, step])
    ns = CUR["train_ns"]
    n = ns[step % len(ns)]
    if CUR["train_M"]:  # pilot override of the training lane count (bed_k generator, cap 32)
        mk = lambda: bed_k._pack(*bed_k.make_bed(n, CUR["train_M"], rng, cap=32), rng)
    else:
        mk = lambda: BED[CUR["bed"]][0].make_train(rng, n=n)
    bs = [mk() for _ in range(CUR["tok"] // n)]
    t = lambda key: torch.from_numpy(np.stack([b[key] for b in bs]).astype(np.int64)).to(dev)
    return t("ids"), t("pids"), t("target")


@torch.no_grad()
def predict(model, b, dev, amp):
    ids = torch.from_numpy(b["ids"].astype(np.int64))[None].to(dev)
    pids = torch.from_numpy(b["pids"].astype(np.int64))[None].to(dev)
    with (torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()):
        h = model(ids, pids)[0]
        pred = torch.cat([model.logits(h[i:i + 2048]).float().argmax(1) for i in range(0, h.shape[0], 2048)])
    return pred.cpu().numpy()


def score(oks, depths, Larm):
    ok, d = np.concatenate(oks), np.concatenate(depths)
    f = lambda m: (float(ok[m].mean()) if m.any() else None)
    per = lambda thr: [float(o[x > thr].mean()) if (x > thr).any() else None for o, x in zip(oks, depths)]
    return {"acc": float(ok.mean()), "acc_le16": f(d <= 16), "acc_le32": f(d <= 32), "beyond16": f(d > 16), "beyond128": f(d > 128), f"beyond_2^{Larm}": f(d > 2 ** Larm),
            "n_beyond16": int((d > 16).sum()), "n_beyond128": int((d > 128).sum()),
            **{b: f(d > t) for b, t in BANDS.items()}, **{f"n_{b}": int((d > t).sum()) for b, t in BANDS.items()},
            **{f"per_bed_{b}": [float(o[x > t].mean()) if (x > t).any() else None for o, x in zip(oks, depths)] for b, t in BANDS.items()},
            "per_bed_acc": [float(o.mean()) for o in oks], "per_bed_beyond16": per(16), "per_bed_beyond128": per(128),
            "by_depth": {f"{lo}-{hi - 1}": f((d >= lo) & (d < hi)) for lo, hi in zip(EDGES[:-1], EDGES[1:]) if ((d >= lo) & (d < hi)).any()},
            "by_depth_fine": {f"{lo}-{hi - 1}": f((d >= lo) & (d < hi)) for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 9), (9, 17), (17, 33)) if ((d >= lo) & (d < hi)).any()}}


def evaluate(model, dev, amp, Larm, ns=(1024, 4096, 8192, 16384)):
    model.eval()
    out = {}
    for n in ns:
        mod, s_tr, s_te = BED[CUR["bed"]]
        beds = [mod.make_train(np.random.default_rng([s_tr, k])) for k in range(CUR.get("eval_beds", 64))] if n == 1024 else \
               [mod.make_test(np.random.default_rng([s_te, n, k]), n) for k in range(8)]
        t0 = time.time()
        oks = [predict(model, b, dev, amp) == b["target"] for b in beds]
        out[str(n)] = score(oks, [b["depth"] for b in beds], Larm)
        if OUTDIR.get("dir") is not None and n > 1024:  # per-token record so any later band can be re-scored
            np.savez_compressed(OUTDIR["dir"] / f"ok_{n}.npz", ok=np.stack(oks), depth=np.stack([b["depth"] for b in beds]))
        out[str(n)]["eval_s"] = time.time() - t0
        if dev.type == "cuda":
            out[str(n)]["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        _BIAS_CACHE.clear()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  eval n={n}: acc {out[str(n)]['acc']:.4f} beyond16 {out[str(n)]['beyond16']} beyond128 {out[str(n)]['beyond128']} band4 {out[str(n)]['band4']} band7 {out[str(n)]['band7']}", flush=True)
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["aL", "ass", "aloop", "fR"], required=True)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dff", type=int, default=512)
    p.add_argument("--loops", type=int, default=5)
    p.add_argument("--hook", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval_ns", default="1024,4096,8192,16384")
    p.add_argument("--probe_every", type=int, default=500)
    p.add_argument("--out", required=True)
    p.add_argument("--bed", choices=["k", "kp", "kpf"], default="kpf")
    p.add_argument("--emb", choices=["learned", "frozen"], default="learned")
    p.add_argument("--par_init", choices=["normal", "orth"], default="normal")
    p.add_argument("--train_M", type=int, default=None)
    p.add_argument("--tok", type=int, default=8192)
    p.add_argument("--train_ns", default="256,512,1024")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--eval_beds", type=int, default=64)
    p.add_argument("--gamma_sched", default=None)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    CUR["bed"] = a.bed
    CUR["eval_beds"] = a.eval_beds
    CUR["train_M"], CUR["tok"], CUR["train_ns"] = a.train_M, a.tok, tuple(int(x) for x in a.train_ns.split(","))
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    hook_cls, hook_sha = load_hook(a.hook) if a.arm == "fR" else (None, None)
    sched = None
    if a.gamma_sched:  # e.g. "0.5:2000,0.9:4000,0.99:6000,0.999"; eval always at the hook's fixed gamma
        sched = [(float(x.split(":")[0]), int(x.split(":")[1]) if ":" in x else None) for x in a.gamma_sched.split(",")]
        hook_cls = gamma_wrap(hook_cls)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    model = Chain(a.arm, a.layers, a.dff, a.loops, hook_cls, a.emb, a.par_init).to(dev)
    params = sum(p.numel() for p in model.parameters())
    params_nonemb = params - model.wte.weight.numel()
    Larm = {"aL": a.layers, "ass": a.layers, "fR": a.layers, "aloop": a.loops}[a.arm]
    decay = [q for q in model.parameters() if q.dim() >= 2 and q.requires_grad]
    nodecay = [q for q in model.parameters() if q.dim() < 2 and q.requires_grad]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8, **({"fused": True} if dev.type == "cuda" else {}))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    OUTDIR["dir"] = out
    log = open(out / "log.jsonl", "a")
    cfg = {"t": "config", **vars(a), "params": params, "params_nonemb": params_nonemb, "hook_sha256": hook_sha,
           "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}
    log.write(json.dumps(cfg) + "\n"); log.flush()
    print(json.dumps(cfg), flush=True)
    train_s, t_start = 0.0, time.time()
    for step in range(1, a.steps + 1):
        ids, pids, tgt = batch(a.seed, step, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = get_lr(step, a.steps, a.warmup, a.lr)
        if sched:
            HOOK["gamma"] = gamma_at(sched, step)
        opt.zero_grad(set_to_none=True)
        with (torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()):
            h = model(ids, pids)
            logits = model.logits(h)
        loss = F.cross_entropy(logits.float().view(-1, V), tgt.view(-1))
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            tacc = (logits.argmax(-1) == tgt).float().mean().item()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        train_s += time.time() - t0
        rec = {"t": "step", "step": step, "n": ids.shape[1], "loss": loss.item(), "acc": tacc, "gn": float(gn), "train_s": train_s}
        if not math.isfinite(rec["loss"]):
            log.write(json.dumps({"t": "abort", "step": step, "reason": "non-finite loss"}) + "\n"); log.close()
            sys.exit(3)
        if step % 50 == 0 or step == 1:
            log.write(json.dumps(rec) + "\n"); log.flush()
        if step % a.probe_every == 0 and step < a.steps:
            ev = evaluate(model, dev, amp, Larm, ns=(1024,))
            log.write(json.dumps({"t": "probe", "step": step, "train_s": train_s, "eval": ev}) + "\n"); log.flush()
            print(f"step {step} loss {rec['loss']:.4f} acc {tacc:.4f} train_s {train_s:.0f}", flush=True)
    torch.save(model.state_dict(), out / "model.pt")
    HOOK["gamma"] = None
    ev = evaluate(model, dev, amp, Larm, ns=tuple(int(x) for x in a.eval_ns.split(",")))
    res = {"config": cfg, "Larm": Larm, "train_s": train_s, "wall_s": time.time() - t_start, "final_loss": rec["loss"],
           "eval": ev, "peak_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0.0}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    log.write(json.dumps({"t": "end", "train_s": train_s}) + "\n"); log.close()
    print(json.dumps({k: res[k] for k in ("Larm", "train_s", "wall_s", "final_loss")}), flush=True)


if __name__ == "__main__":
    main()
