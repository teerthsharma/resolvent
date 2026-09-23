# Stub for test_k2_grid_kaggle.py's RED (stub dir only): the null world "Kaggle changes nothing". Every id_<arm>_s<seed> is
# K1's far_<arm>_s<seed % 2> relabelled with the registered id config. The 5 runs that landed locally keep local hook paths
# and the RTX 4060 device; the other 13 carry Kaggle-style hook paths (/kaggle/working/...) and device "Tesla T4", with the
# registered hook shas; id_aL7_s2 records an empty device (exercises the hardware field). No walk dirs, no model.pt.
import json, shutil
from pathlib import Path
HERE = Path(__file__).parent
K1C = HERE.parent.parent / "K1" / "cameron"
S = HERE / "stub_kaggle"
SCHED = "0.5:6000,0.9:12000,0.99:18000,0.999"
LOCAL = {"id_aL4_s0", "id_aL7_s0", "id_ass4_s0", "id_aloop5_s0", "id_fR_ga_s0"}
SP = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k"
HOOK = {"fR_ga": ("K1/chase/resolvent_hook.py", "3855288d7da7acdebf54b9b1979d9b0b5afebb3dca6d8ccefd684812d33451e2"),
        "fR_sp": ("K2/chase/resolvent_sp.py", "36039c3a4189189840d04b345fa3c54d43e6d32bcc8a176ce5e8aac135bfd7bc")}
K1ARM = {"aL4": "aL4", "aL7": "aL7", "ass4": "ass4", "aloop5": "aloop5", "fR_ga": "fR_ga", "fR_sp": "fR_ga"}
for s in (0, 1, 2):
    for a, k in K1ARM.items():
        name = f"id_{a}_s{s}"
        src = K1C / "runs" / f"far_{k}_s{s % 2}"
        d = S / name
        d.mkdir(parents=True, exist_ok=True)
        r = json.loads((src / "result.json").read_text())
        c = r["config"]
        c.update(steps=24000, train_ns="256,512,1024,2048,4096", seed=s, stub=True)
        c["device"] = "NVIDIA GeForce RTX 4060 Laptop GPU" if name in LOCAL else ("" if name == "id_aL7_s2" else "Tesla T4")
        if a.startswith("fR"):
            rel, sha = HOOK[a]
            c["gamma_sched"] = SCHED
            c["hook"] = (f"{SP}/{rel}" if name in LOCAL else f"/kaggle/working/phase_k/{rel}") + ":ResolventAttention"
            c["hook_sha256"] = sha
        (d / "result.json").write_text(json.dumps(r))
        for n in (4096, 8192, 16384):
            shutil.copy(src / f"ok_{n}.npz", d / f"ok_{n}.npz")
