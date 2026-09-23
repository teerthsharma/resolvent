# Stubs for the K2.1 registrations' RED 2 (null worlds; stub dirs only, never runs/):
#   stub_pilot: "sparsemax adds nothing to K1-b": the K1-b checkpoint, arrays and pointer relabelled as the pilot
#               (registered config, hook = Chase's current resolvent_sp.py; READY_SP absent in reality).
#   stub_grid : "24k and longer contexts change nothing": id_<arm>_s<0,1> = K1's far_<arm>_s<0,1> relabelled; seed 2 absent;
#               fR_sp = K1's fR_ga; no walk dirs.
#   stub_lanes: the null pointer ptr[i] = i on both bed sets.
import hashlib, json, shutil, sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
K1C = HERE.parent.parent / "K1" / "cameron"
sys.path.insert(0, str(K1C))
import bed_k, bed_kp
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
SCHED = "0.5:6000,0.9:12000,0.99:18000,0.999"
SPHOOK = str((HERE.parent / "chase" / "resolvent_sp.py").resolve()).replace("\\", "/") + ":ResolventAttention"
K1HOOK = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K1/chase/resolvent_hook.py:ResolventAttention"


def relabel(src, dst, **cfg):
    dst.mkdir(parents=True, exist_ok=True)
    r = json.loads((src / "result.json").read_text())
    r["config"].update(cfg, stub=True)
    if "hook" in cfg:
        r["config"]["hook_sha256"] = sha(cfg["hook"].rsplit(":", 1)[0])
    (dst / "result.json").write_text(json.dumps(r))
    for n in (4096, 8192, 16384):
        shutil.copy(src / f"ok_{n}.npz", dst / f"ok_{n}.npz")


src = K1C / "runs" / "far_fR_ga_s0"
d = HERE / "stub_pilot" / "pilot_fR_sp_s0"
relabel(src, d, steps=24000, gamma_sched=SCHED, hook=SPHOOK)
shutil.copy(src / "model.pt", d / "model.pt")
(d / "walk").mkdir(exist_ok=True)
for n in (4096, 16384):
    shutil.copy(HERE / "runs" / "walk" / f"ptr_{n}.npz", d / "walk" / f"ptr_{n}.npz")
(d / "walk" / "meta.json").write_text(json.dumps({"ckpt_sha256": sha(d / "model.pt"), "stub": True}))

K1ARM = {"aL4": "aL4", "aL7": "aL7", "ass4": "ass4", "aloop5": "aloop5", "fR_ga": "fR_ga", "fR_sp": "fR_ga"}
for a, k in K1ARM.items():
    for s in (0, 1):
        extra = {"gamma_sched": SCHED, "hook": SPHOOK if a == "fR_sp" else K1HOOK} if a.startswith("fR") else {}
        relabel(K1C / "runs" / f"far_{k}_s{s}", HERE / "stub_grid" / f"id_{a}_s{s}", steps=24000,
                train_ns="256,512,1024,2048,4096", **extra)

L = HERE / "stub_lanes" / "lanes"
L.mkdir(parents=True, exist_ok=True)
N = 16384
b8 = [bed_kp.make_test(np.random.default_rng([31, N, k]), N) for k in range(8)]
b32 = [bed_k._pack(*bed_k.make_bed(N, 32, rng), rng) for rng in (np.random.default_rng([41, N, k]) for k in range(8))]
for M, beds in ((8, b8), (32, b32)):
    np.savez_compressed(L / f"ptr_M{M}.npz", ptr=np.tile(np.arange(N), (8, 2, 1)), root=np.stack([b["root"] for b in beds]),
                        depth=np.stack([b["depth"] for b in beds]))
(L / "meta.json").write_text(json.dumps({"ckpt_sha256": sha(K1C / "runs" / "far_fR_ga_s0" / "model.pt"), "stub": True}))
