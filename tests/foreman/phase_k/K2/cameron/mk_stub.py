# Stub for test_k2_leak.py's RED: the null world where kappa and gamma do nothing (every dir = K1's kappa-1 arrays).
import hashlib, json, shutil
from pathlib import Path
import torch
HERE = Path(__file__).parent
K1R = HERE.parent.parent / "K1" / "cameron" / "runs" / "far_fR_ga_s0"
S = HERE / "stub"
sha = hashlib.sha256((K1R / "model.pt").read_bytes()).hexdigest()
sd = torch.load(K1R / "model.pt", map_location="cpu")
a0, b0 = sd["blocks.3.attn.a"].double(), sd["blocks.3.attn.b"].double()
ev = json.loads((K1R / "result.json").read_text())["eval"]
for name, k, g in [(f"k{t}", float(t), 0.999) for t in ("1", "1.5", "2", "3", "5")] + [("g0.9999", 1.0, 0.9999)]:
    d = S / name
    d.mkdir(parents=True, exist_ok=True)
    for n in ("4096", "16384"):
        shutil.copy(K1R / f"ok_{n}.npz", d / f"ok_{n}.npz")
    (d / "result.json").write_text(json.dumps({"kappa": k, "gamma": g, "ckpt_sha256": sha, "a": (k * a0).tolist(),
                                               "b": (k * b0).tolist(), "eval": {"1024": ev["1024"]}, "stub": True}))
(S / "numerics.json").write_text(json.dumps({"kappa": "1", "relerr": {"0.999": 1.0, "0.9999": 1.0}, "stub": True}))
