"""Producer for test_k2_walk.py (Cameron): the argmax pointer of the trained resolvent's W per head, at kappa = 3, on
the far10 beds, captured on the K1 eval path (amp on CUDA, fp32 resolvent). Usage: k2_walk.py <runs dir>"""
import sys
sys.dont_write_bytecode = True
import hashlib, json, math
from pathlib import Path
HERE = Path(__file__).parent
K1C = HERE.parent.parent / "K1" / "cameron"
sys.path.insert(0, str(K1C))
import numpy as np
import torch
import rdepth, bed_kp

RUN = K1C / "runs" / "far_fR_ga_s0"
WD = HERE / sys.argv[1] / "walk"
WD.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda")
torch.manual_seed(0)
cfg = json.loads((RUN / "result.json").read_text())["config"]
hook_cls, hook_sha = rdepth.load_hook(cfg["hook"])
assert hook_sha == cfg["hook_sha256"]
m = rdepth.Chain("fR", cfg["layers"], cfg["dff"], cfg["loops"], rdepth.gamma_wrap(hook_cls), cfg["emb"], cfg["par_init"]).to(dev)
m.load_state_dict(torch.load(RUN / "model.pt", map_location=dev))
m.eval()
att = m.blocks[-1].attn
a0, b0 = att.a.data.clone(), att.b.data.clone()


@torch.no_grad()
def pointer(b, kappa):
    att.a.data, att.b.data = a0 * kappa, b0 * kappa
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


kinv = None
for n in (4096, 16384):
    beds = [bed_kp.make_test(np.random.default_rng([31, n, k]), n) for k in range(8)]
    ptr = np.stack([pointer(b, 3.0) for b in beds])
    if n == 4096:
        kinv = float((pointer(beds[0], 1.0) == ptr[0]).mean())
    np.savez_compressed(WD / f"ptr_{n}.npz", ptr=ptr, root=np.stack([b["root"] for b in beds]), depth=np.stack([b["depth"] for b in beds]))
    print(f"n={n}: ptr {ptr.shape}", flush=True)
sha = hashlib.sha256((RUN / "model.pt").read_bytes()).hexdigest()
(WD / "meta.json").write_text(json.dumps({"kappa": "3", "ckpt_sha256": sha, "hook_sha256": hook_sha, "kinv_agree": kinv}))
print(f"kinv_agree {kinv}", flush=True)
