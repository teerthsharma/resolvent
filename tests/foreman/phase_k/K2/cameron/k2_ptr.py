"""Pointer producer (Cameron K2): per-head argmax of the resolvent layer's logits at the trained logit scale, captured on
the K1 eval path (amp on CUDA, fp32 resolvent). Modes:
  k2_ptr.py walk <run dir>   -> <run dir>/walk/ptr_{4096,16384}.npz + meta.json (far10 beds [31, n, k]); for
                                test_k2_pilot.py / test_k2_grid.py
  k2_ptr.py lanes <runs dir> -> <runs dir>/lanes/ptr_M{8,32}.npz + meta.json on K1's far_fR_ga_s0; for test_k2_lanes.py"""
import sys
sys.dont_write_bytecode = True
import hashlib, json, math
from pathlib import Path
HERE = Path(__file__).parent
K1C = HERE.parent.parent / "K1" / "cameron"
sys.path.insert(0, str(K1C))
import numpy as np
import torch
import rdepth, bed_k, bed_kp

mode, arg = sys.argv[1], Path(sys.argv[2])
RUN = K1C / "runs" / "far_fR_ga_s0" if mode == "lanes" else arg
OUT = (HERE / arg / "lanes") if mode == "lanes" else (arg / "walk")
OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda")
torch.manual_seed(0)
cfg = json.loads((RUN / "result.json").read_text())["config"]
hook_cls, hook_sha = rdepth.load_hook(cfg["hook"])
assert hook_sha == cfg["hook_sha256"]
m = rdepth.Chain(cfg["arm"], cfg["layers"], cfg["dff"], cfg["loops"], rdepth.gamma_wrap(hook_cls), cfg["emb"], cfg["par_init"]).to(dev)
m.load_state_dict(torch.load(RUN / "model.pt", map_location=dev))
m.eval()
att = m.blocks[-1].attn


@torch.no_grad()
def pointer(b):
    att.capture = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m(torch.from_numpy(b["ids"].astype(np.int64))[None].to(dev), torch.from_numpy(b["pids"].astype(np.int64))[None].to(dev))
    c = att.capture[0]
    att.capture = None
    qs, k = c["qs"][0].float(), c["k"][0].float()          # [H, S, D]
    S, D = qs.shape[1], qs.shape[2]
    out = torch.empty(qs.shape[0], S, dtype=torch.long, device=dev)
    for t0 in range(0, S, 2048):
        t1 = min(S, t0 + 2048)
        z = (qs[:, t0:t1] @ k[:, :t1].transpose(-1, -2)) / math.sqrt(D)
        z = z.masked_fill(torch.arange(t1, device=dev)[None, None, :] > torch.arange(t0, t1, device=dev)[None, :, None], float("-inf"))
        out[:, t0:t1] = z.argmax(-1)
    return out.cpu().numpy()


def save(name, beds):
    np.savez_compressed(OUT / f"{name}.npz", ptr=np.stack([pointer(b) for b in beds]),
                        root=np.stack([b["root"] for b in beds]), depth=np.stack([b["depth"] for b in beds]))
    print(f"{name}: {len(beds)} beds", flush=True)


if mode == "lanes":
    N = 16384
    save("ptr_M8", [bed_kp.make_test(np.random.default_rng([31, N, k]), N) for k in range(8)])
    save("ptr_M32", [bed_k._pack(*bed_k.make_bed(N, 32, rng), rng) for rng in (np.random.default_rng([41, N, k]) for k in range(8))])
else:
    for n in (4096, 16384):
        save(f"ptr_{n}", [bed_kp.make_test(np.random.default_rng([31, n, k]), n) for k in range(8)])
sha = hashlib.sha256((RUN / "model.pt").read_bytes()).hexdigest()
(OUT / "meta.json").write_text(json.dumps({"ckpt_sha256": sha, "hook_sha256": hook_sha, "run": str(RUN)}))
print("done", flush=True)
