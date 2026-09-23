"""K2.0 producer (Cameron): evaluation only on K1/cameron/runs/far_fR_ga_s0/model.pt, through rdepth.evaluate (the K1
eval path, amp on CUDA, the hook's fp32 resolvent). Writes the layout test_k2_leak.py reads. Usage: k2_leak.py <runs>"""
import sys
sys.dont_write_bytecode = True
import hashlib, json, time
from pathlib import Path
HERE = Path(__file__).parent
K1C = HERE.parent.parent / "K1" / "cameron"
sys.path.insert(0, str(K1C))
import numpy as np
import torch
import rdepth, bed_kp

RUN = K1C / "runs" / "far_fR_ga_s0"
OUT = HERE / sys.argv[1]
KAPPAS = ("1", "1.5", "2", "3", "5")
dev = torch.device("cuda")
torch.manual_seed(0)
cfg = json.loads((RUN / "result.json").read_text())["config"]
hook_cls, hook_sha = rdepth.load_hook(cfg["hook"])
assert hook_sha == cfg["hook_sha256"], hook_sha
R = rdepth.HOOK["mod"].R
rdepth.CUR["bed"], rdepth.CUR["eval_beds"] = cfg["bed"], cfg["eval_beds"]
m = rdepth.Chain("fR", cfg["layers"], cfg["dff"], cfg["loops"], rdepth.gamma_wrap(hook_cls), cfg["emb"], cfg["par_init"]).to(dev)
m.load_state_dict(torch.load(RUN / "model.pt", map_location=dev))
m.eval()
sha = hashlib.sha256((RUN / "model.pt").read_bytes()).hexdigest()
att = m.blocks[-1].attn
a0, b0 = att.a.data.clone(), att.b.data.clone()
print(f"a0 {a0.tolist()} b0 {b0.tolist()} sha {sha[:12]} hook {hook_sha[:8]}", flush=True)


def setk(kappa, gamma):
    att.a.data, att.b.data = a0 * kappa, b0 * kappa
    rdepth.HOOK["gamma"] = None if gamma == 0.999 else gamma   # None = the hook's fixed GAMMA 0.999, as K1's eval


def run(name, kappa, gamma):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    setk(kappa, gamma)
    rdepth.OUTDIR["dir"] = d
    t0 = time.time()
    print(f"{name}: kappa {kappa} gamma {gamma}", flush=True)
    ev = rdepth.evaluate(m, dev, True, 4, ns=(1024, 4096, 16384))
    m.eval()
    (d / "result.json").write_text(json.dumps({"kappa": kappa, "gamma": gamma, "ckpt_sha256": sha, "hook_sha256": hook_sha,
                                               "a": att.a.data.tolist(), "b": att.b.data.tolist(), "wall_s": time.time() - t0,
                                               "eval": ev}, indent=1))
    R._graphs.clear()
    torch.cuda.empty_cache()
    z = np.load(d / "ok_4096.npz")
    return float(z["ok"][z["depth"] > 160].mean())


b4 = {t: run(f"k{t}", float(t), 0.999) for t in KAPPAS}
best = max(b4, key=lambda t: (b4[t], -float(t)))
print(f"best kappa {best}: " + " / ".join(f"x{t} {b4[t]:.4f}" for t in KAPPAS), flush=True)
run("g0.9999", float(best), 0.9999)

# numerics: fp32 GPU read (fs5c, CUDA-graphed) vs float64 dense solve on the same captured qs, k, v
setk(float(best), 0.999)
b = bed_kp.make_test(np.random.default_rng([31, 4096, 0]), 4096)
att.capture = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    m(torch.from_numpy(b["ids"].astype(np.int64))[None].to(dev), torch.from_numpy(b["pids"].astype(np.int64))[None].to(dev))
c = att.capture[0]
att.capture = None
qs, k, v = c["qs"].float(), c["k"].float(), c["v"].float()
rel = {}
for g in (0.999, 0.9999):
    x32 = rdepth.HOOK["mod"].read(qs, k, v, g).double().cpu()
    x64 = rdepth.HOOK["mod"].read(qs.double().cpu(), k.double().cpu(), v.double().cpu(), g)
    rel[str(g)] = float((x32 - x64).abs().max() / v.abs().max().double().cpu())
    print(f"numerics gamma {g}: max|dx|/max|v| {rel[str(g)]:.3e}", flush=True)
(OUT / "numerics.json").write_text(json.dumps({"kappa": best, "relerr": rel, "bed": [31, 4096, 0]}))
print("done", flush=True)
