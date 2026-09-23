# K2.1 grids (Cameron; Dispatcher order after K2.0). Written 2026-09-23 BEFORE any grid run and before the pilot.
# Never edited after its RED. Which grid launches is decided by test_k2_pilot.py's BRANCH line; this file holds the bars
# for both, so neither is written after the pilot result.
#   branch sp = K2.1 as registered in RECORD_K (train_ns 256,512,1024), plus the walk-ceiling rows.
#   branch id = the pointer-identity grid: train_ns 256,512,1024,2048,4096 (bed_kp.make_train: 32 lanes, depth cap 32,
#               so training depth is unchanged and only the context grows); the same schedule for every arm.
# Runs <runs>/<branch>_<arm>_s<seed>/ (K1/cameron/rdepth.py, --out there), seeds 0, 1, 2, every arm with
#   --bed kpf --emb frozen --par_init orth --lr 3e-3 --steps 24000 --probe_every 2000 --eval_ns 1024,4096,8192,16384
#   --train_ns <branch's> and
#   aL4    --arm aL --layers 4                 aL7   --arm aL --layers 7 --dff 183 (parameter-matched, as K1)
#   ass4   --arm ass --layers 4                aloop5 --arm aloop --loops 5
#   fR_ga  --arm fR --layers 4 --hook K1/chase/resolvent_hook.py:ResolventAttention (sha 3855288d...)
#          --gamma_sched 0.5:6000,0.9:12000,0.99:18000,0.999
#   fR_sp  --arm fR --layers 4 --hook <the READY_SP hook in SP/phase_k/K2/chase/> --gamma_sched as fR_ga
# fR runs also get <run>/walk/ (ptr_4096.npz, ptr_16384.npz, meta.json {"ckpt_sha256"}), as test_k2_pilot.py.
# Rows (far10 bars as test_rdepth_far10.py, three seeds; an arm failing its gate is UNREAD, and UNREAD fails):
#   cameron.k2.grid_<b>_config      each run's config = the registration (hook sha on disk = config's; fR_ga's = 3855288d).
#   cameron.k2.grid_<b>_trained     gate L-TRAINED: held-out n = 1024 acc on depth <= min(2^Larm, 32) >= 0.9.
#   cameron.k2.grid_<b>_learnable   fR arms: held-out n = 1024 acc >= 0.8 (contract kill), each seed.
#   cameron.k2.grid_<b>_r1          RECORD_K K2 rule: fR_ga or fR_sp learnable (>= 0.8) on >= 2 of 3 seeds.
#   cameron.k2.grid_<b>_prediction  fR arms >= 0.95 on depth > 160 at 4096 / 8192 / 16384 and depth > 1280 at 16384.
#   cameron.k2.grid_<b>_multilen16k fR arms acc (all tokens) at n = 16384 >= 0.95.
#   cameron.k2.grid_<b>_walk_ceiling fR arms: OR-walk (fixed point in own chain at depth <= 8) >= 0.95 on the three
#                                   far10 cells (depth > 160 at 4096 and 16384, depth > 1280 at 16384).
#   cameron.k2.grid_<b>_counter_quiet   aL4 on depth > 160 (every n) and aL7 on depth > 1280 (16384) <= 0.5.
#   cameron.k2.grid_<b>_twin_under_line aL4 / aL7 on those bands <= K1.F''' line (K1/cameron/k1fppp.json) + 3 SE.
#   cameron.k2.grid_<b>_opponents_quiet ass4 on depth > 160 and aloop5 on depth > 320 <= 0.5, every n.
# Usage: python test_k2_grid.py <sp|id> [runs dir, default runs]. Exit 1 if any row fails.
import hashlib, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
CHASE = HERE.parent / "chase"
K1C = HERE.parent.parent / "K1" / "cameron"
BR = sys.argv[1]
assert BR in ("sp", "id")
RUNS = HERE / (sys.argv[2] if len(sys.argv) > 2 else "runs")
SCHED = "0.5:6000,0.9:12000,0.99:18000,0.999"
K1HOOK = "3855288d7da7acdebf54b9b1979d9b0b5afebb3dca6d8ccefd684812d33451e2"
COMMON = {"bed": "kpf", "emb": "frozen", "par_init": "orth", "lr": 0.003, "warmup": 200, "steps": 24000, "tok": 8192,
          "eval_ns": "1024,4096,8192,16384", "eval_beds": 64, "train_M": None,
          "train_ns": {"sp": "256,512,1024", "id": "256,512,1024,2048,4096"}[BR]}
ARMS = {"aL4": {"arm": "aL", "layers": 4, "dff": 512, "gamma_sched": None},
        "aL7": {"arm": "aL", "layers": 7, "dff": 183, "gamma_sched": None},
        "ass4": {"arm": "ass", "layers": 4, "dff": 512, "gamma_sched": None},
        "aloop5": {"arm": "aloop", "loops": 5, "dff": 512, "gamma_sched": None},
        "fR_ga": {"arm": "fR", "layers": 4, "dff": 512, "gamma_sched": SCHED},
        "fR_sp": {"arm": "fR", "layers": 4, "dff": 512, "gamma_sched": SCHED}}
LARM = {"aL4": 4, "aL7": 7, "ass4": 4, "aloop5": 5, "fR_ga": 4, "fR_sp": 4}
SEEDS = (0, 1, 2)
FR = ("fR_ga", "fR_sp")
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok is True else ("UNREAD " if ok is None else "FAIL ")) + name + " : " + str(got))
    if ok is not True:
        fails.append(name)


def land(ptr):
    L = ptr.copy()
    while True:
        L2 = L[L]
        if np.array_equal(L2, L):
            return L
        L = L2


def sha_of(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def config_bad(a, s, c):
    bad = [k for k, v in {**COMMON, **ARMS[a], "seed": s}.items() if c.get(k) != v]
    if a in FR:
        hp = Path(str(c.get("hook", ":")).rsplit(":", 1)[0])
        try:
            hs = sha_of(hp)
        except OSError:
            hs = None
        if hs is None or hs != c.get("hook_sha256"): bad.append("hook sha")
        if a == "fR_ga" and hs != K1HOOK: bad.append("fR_ga hook is not 3855288d")
        if a == "fR_sp":
            try:
                if hp.resolve().parent != CHASE.resolve(): bad.append("fR_sp hook not in K2/chase")
            except OSError:
                bad.append("fR_sp hook path")
    return bad


R = {}
for a in ARMS:
    for s in SEEDS:
        d = RUNS / f"{BR}_{a}_s{s}"
        try:
            r = json.loads((d / "result.json").read_text())
        except (OSError, ValueError):
            r = None
        bad = ["missing"] if r is None else config_bad(a, s, r["config"])
        check(f"cameron.k2.grid_{BR}_config {a} s{s}", not bad, "ok" if not bad else "bad: " + ", ".join(bad))
        key = "acc_le16" if LARM[a] == 4 else "acc_le32"
        tr = None if (r is None or bad) else r["eval"]["1024"][key]
        check(f"cameron.k2.grid_{BR}_trained {a} s{s} held-out {key} >= 0.9", tr is not None and tr >= 0.9,
              "missing/misconfigured" if tr is None else f"{tr:.4f}")
        R[(a, s)] = (d, r) if (tr is not None and tr >= 0.9) else (d, None)
        if a in FR:
            v = None if (r is None or bad) else r["eval"]["1024"]["acc"]
            check(f"cameron.k2.grid_{BR}_learnable {a} s{s} held-out acc >= 0.8", v is not None and v >= 0.8,
                  "missing/misconfigured" if v is None else f"{v:.4f}")
            R[(a, s, "learn")] = v is not None and v >= 0.8

cnt = {a: sum(R[(a, s, "learn")] for s in SEEDS) for a in FR}
check(f"cameron.k2.grid_{BR}_r1 fR_ga or fR_sp learnable on >= 2 of 3 seeds", max(cnt.values()) >= 2,
      " / ".join(f"{a} {cnt[a]}/3" for a in FR))


def band(d, n, thr):
    z = np.load(d / f"ok_{n}.npz")
    m = z["depth"] > thr
    return float(z["ok"][m].mean()), int(m.sum())


K1 = {n: np.load(K1C / "runs" / "far_fR_ga_s0" / f"ok_{n}.npz") for n in ("4096", "16384")}
for a in FR:
    for s in SEEDS:
        d, r = R[(a, s)]
        for n, thr in (("4096", 160), ("8192", 160), ("16384", 160), ("16384", 1280)):
            v = None if r is None else band(d, n, thr)[0]
            check(f"cameron.k2.grid_{BR}_prediction {a} s{s} n={n} depth>{thr} >= 0.95", None if v is None else v >= 0.95,
                  "unread/missing" if v is None else f"{v:.4f}")
        v = None if r is None else r["eval"]["16384"]["acc"]
        check(f"cameron.k2.grid_{BR}_multilen16k {a} s{s} acc n=16384 >= 0.95", None if v is None else v >= 0.95,
              "unread/missing" if v is None else f"{v:.4f}")
        P = None
        if r is not None:
            try:
                meta = json.loads((d / "walk" / "meta.json").read_text())
                P = {n: np.load(d / "walk" / f"ptr_{n}.npz") for n in ("4096", "16384")}
                if meta.get("ckpt_sha256") != sha_of(d / "model.pt"): P = None
                elif not all(np.array_equal(P[n]["depth"], K1[n]["depth"]) and (P[n]["ptr"] <= np.arange(int(n))).all()
                             and (P[n]["ptr"] >= 0).all() for n in P): P = None
            except (OSError, ValueError):
                P = None
        for n, thr in (("4096", 160), ("16384", 160), ("16384", 1280)):
            name = f"cameron.k2.grid_{BR}_walk_ceiling {a} s{s} n={n} depth>{thr} OR-walk >= 0.95"
            if P is None:
                check(name, None, "unread (gate or walk machinery)")
                continue
            ptr, root, dep = P[n]["ptr"], P[n]["root"], P[n]["depth"]
            ok = np.stack([[(root[b][land(ptr[b, h])] == root[b]) & (dep[b][land(ptr[b, h])] <= 8) for h in range(ptr.shape[1])]
                           for b in range(ptr.shape[0])]).any(1)
            m = dep > thr
            v = float(ok[m].mean())
            check(name, v >= 0.95, f"{v:.4f} (N {int(m.sum())})")

LINE = json.loads((K1C / "k1fppp.json").read_text())["cells"]
for a, thr, kind, ns in (("aL4", 160, "counter_quiet", ("4096", "8192", "16384")), ("aL7", 1280, "counter_quiet", ("16384",)),
                         ("ass4", 160, "opponents_quiet", ("4096", "8192", "16384")), ("aloop5", 320, "opponents_quiet", ("4096", "8192", "16384"))):
    for s in SEEDS:
        d, r = R[(a, s)]
        for n in ns:
            v, N = (None, None) if r is None else band(d, n, thr)
            check(f"cameron.k2.grid_{BR}_{kind} {a} s{s} n={n} depth>{thr} <= 0.5", None if v is None else v <= 0.5,
                  "unread/missing" if v is None else f"{v:.4f} (N {N})")
            if a in ("aL4", "aL7"):
                line = LINE[f"{n}_{4 if a == 'aL4' else 7}"]["line"]
                bar = None if N is None else line + 3 * math.sqrt(line * (1 - line) / N)
                check(f"cameron.k2.grid_{BR}_twin_under_line {a} s{s} n={n} depth>{thr} <= line+3SE", None if v is None else v <= bar,
                      "unread/missing" if v is None else f"{v:.4f} vs {bar:.4f} (line {line:.4f})")
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
