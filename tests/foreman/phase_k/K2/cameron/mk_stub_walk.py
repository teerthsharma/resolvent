# Stub for test_k2_walk.py's RED: the null pointer (ptr[i] = i, every token absorbs at itself); ok arrays = K2.0's k3.
import hashlib, json, shutil, sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
K1C = HERE.parent.parent / "K1" / "cameron"
sys.path.insert(0, str(K1C))
import bed_kp
S = HERE / "stub_walk"
(S / "walk").mkdir(parents=True, exist_ok=True)
(S / "k3").mkdir(exist_ok=True)
for n in (4096, 16384):
    beds = [bed_kp.make_test(np.random.default_rng([31, n, k]), n) for k in range(8)]
    np.savez_compressed(S / "walk" / f"ptr_{n}.npz", ptr=np.tile(np.arange(n), (8, 2, 1)),
                        root=np.stack([b["root"] for b in beds]), depth=np.stack([b["depth"] for b in beds]))
    shutil.copy(HERE / "runs" / "k3" / f"ok_{n}.npz", S / "k3" / f"ok_{n}.npz")
sha = hashlib.sha256((K1C / "runs" / "far_fR_ga_s0" / "model.pt").read_bytes()).hexdigest()
(S / "walk" / "meta.json").write_text(json.dumps({"kappa": "3", "ckpt_sha256": sha, "kinv_agree": 1.0, "stub": True}))
