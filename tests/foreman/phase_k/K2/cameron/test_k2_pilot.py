# K2.1 pilot gate (Cameron; Dispatcher order after K2.0). Written 2026-09-23 BEFORE the pilot runs and before Chase's
# READY_SP exists. Never edited after its RED.
# Run pilot_fR_sp_s<seed> (seed 0; seed 1 only under the REPEAT_S1 branch below), under <runs>/pilot_fR_sp_s<seed>/:
#   K1/cameron/rdepth.py --arm fR --layers 4 --hook <the hook file READY_SP names, in SP/phase_k/K2/chase/>:<its class>
#     --bed kpf --emb frozen --par_init orth --lr 3e-3 --steps 24000 --probe_every 2000 --eval_ns 1024,4096,8192,16384
#     --gamma_sched 0.5:6000,0.9:12000,0.99:18000,0.999 --seed <seed> --out <run dir>
#   i.e. K1-b's recipe with the sparsemax resolvent hook; the K1-b anneal is kept at the same fractions of the budget
#   (gamma 0.5 / 0.9 / 0.99 for the first three quarters, 0.999 = the evaluated value for the last quarter).
#   Then the walk producer writes <run dir>/walk/{ptr_4096.npz, ptr_16384.npz, meta.json}: the per-head argmax pointer
#   of the resolvent's weights (argmax of sparsemax = argmax of its logits) on the far10 beds default_rng([31, n, k]),
#   k 0..7, at the trained logit scale; meta.json {"ckpt_sha256": sha256 of <run dir>/model.pt}.
# Hard walk (as test_k2_walk.py): follow ptr to its fixed point L(i); right iff root[L] == root[i] and depth[L] <= 8.
# OR over the two heads.
# Rows (GATE marks the rows the grid launch reads):
#   cameron.k2.pilot_config          GATE: config = the registration above; hook file inside SP/phase_k/K2/chase/, its
#                                    sha256 on disk = config hook_sha256, and READY_SP exists and names the hook
#                                    (its basename or the first 8 hex of its sha256 appear in READY_SP).
#   cameron.k2.pilot_trained         GATE: learned-the-task, held-out n = 1024 acc on depth <= 16 >= 0.9.
#   cameron.k2.pilot_learnable       contract kill, held-out n = 1024 acc >= 0.8 (read as usual, not a gate).
#   cameron.k2.pilot_walk_machinery  GATE: walk files present; ckpt sha = the run's model.pt; beds = far10; ptr causal.
#   cameron.k2.pilot_walk_ceiling    GATE: OR-walk >= 0.95 on depth > 160 at 4096 and 16384 and depth > 1280 at 16384.
#   cameron.k2.pilot_far10           far10 bars as usual (read, not a gate): >= 0.95 on depth > 160 at 4096 / 8192 /
#                                    16384 and depth > 1280 at 16384; UNREAD when pilot_trained fails.
#   cameron.k2.pilot_multilen16k     acc (all tokens) at n = 16384 >= 0.95 (read, not a gate).
#   cameron.k2.pilot_grid_launch     PASS iff every GATE row passes.
# Branch rule, registered here before the pilot (printed as BRANCH):
#   GRID      every GATE row GREEN: K2.1 as registered launches (6 arms x 3 seeds, 24k steps, train_ns 256/512/1024;
#             bars test_k2_grid.py sp). The far10 rows do not gate: the grid carries the R1 criterion (fR learnable on
#             >= 2 of 3 seeds), which the pilot's far10 cannot decide.
#   IDENTITY  config, trained and walk machinery GREEN, and any walk-ceiling cell < 0.95. This covers the 16k ceiling
#             below 0.5 AND the 0.5-0.95 range: below 0.95 the exact-zero ceiling of the pilot's own pointer already
#             fails the band, so the as-registered grid cannot pass its prediction whatever its leak, and the identity
#             grid (test_k2_grid.py id: train_ns 256/512/1024/2048/4096, depth cap 32, every arm, fR_ga and fR_sp both
#             in it) is the only registered grid that attacks the remaining wall.
#   REPEAT_S1 seed 0 fails pilot_trained (sparsemax did not learn): pilot_fR_sp_s1 runs once under this same file
#             (argument 1). If seed 1 also fails: SP_RETIRED, the sparsemax arm is retired at learnability and the
#             identity grid runs with its fR_sp rows left UNREAD.
#   NONE      pilot missing or misconfigured: nothing launches.
# Usage: python test_k2_pilot.py [seed, default 0] [runs dir, default runs]. Exit 1 if any row fails.
import hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
CHASE = HERE.parent / "chase"
READY = CHASE / "READY_SP"
K1R = HERE.parent.parent / "K1" / "cameron" / "runs" / "far_fR_ga_s0"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
RUNS = HERE / (sys.argv[2] if len(sys.argv) > 2 else "runs")
D = RUNS / f"pilot_fR_sp_s{SEED}"
REG = {"arm": "fR", "layers": 4, "dff": 512, "bed": "kpf", "emb": "frozen", "par_init": "orth", "lr": 0.003, "warmup": 200,
       "steps": 24000, "tok": 8192, "train_ns": "256,512,1024", "eval_ns": "1024,4096,8192,16384", "eval_beds": 64,
       "gamma_sched": "0.5:6000,0.9:12000,0.99:18000,0.999", "seed": SEED, "train_M": None}
CELLS = (("4096", 160), ("16384", 160), ("16384", 1280))
fails, gate = [], {}


def check(name, ok, got="", g=False):
    print(("PASS " if ok is True else ("UNREAD " if ok is None else "FAIL ")) + name + " : " + str(got))
    if ok is not True:
        fails.append(name)
    if g:
        gate[name] = ok is True


def land(ptr):
    L = ptr.copy()
    while True:
        L2 = L[L]
        if np.array_equal(L2, L):
            return L
        L = L2


def sha_of(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


try:
    r = json.loads((D / "result.json").read_text())
except (OSError, ValueError):
    r = None

bad = ["missing"] if r is None else []
if r is not None:
    c = r["config"]
    bad += [k for k, v in REG.items() if c.get(k) != v]
    hp = Path(str(c.get("hook", ":")).rsplit(":", 1)[0])
    try:
        in_chase = hp.resolve().parent == CHASE.resolve()
        hs = sha_of(hp)
    except OSError:
        in_chase, hs = False, None
    if not in_chase: bad.append("hook not in K2/chase")
    if hs is None or hs != c.get("hook_sha256"): bad.append("hook sha")
    try:
        rt = READY.read_text(errors="replace")
        if hs is None or not (hp.name in rt or hs[:8] in rt): bad.append("READY_SP does not name the hook")
    except OSError:
        bad.append("READY_SP missing")
check(f"cameron.k2.pilot_config s{SEED}", not bad, "ok" if not bad else "bad: " + ", ".join(bad), g=True)

tr = None if r is None else r["eval"]["1024"]["acc_le16"]
check(f"cameron.k2.pilot_trained s{SEED} held-out depth<=16 >= 0.9", tr is not None and tr >= 0.9, "missing" if tr is None else f"{tr:.4f}", g=True)
v = None if r is None else r["eval"]["1024"]["acc"]
check(f"cameron.k2.pilot_learnable s{SEED} held-out acc >= 0.8", v is not None and v >= 0.8, "missing" if v is None else f"{v:.4f}")
trained = tr is not None and tr >= 0.9

K1 = {n: np.load(K1R / f"ok_{n}.npz") for n in ("4096", "16384")}
try:
    meta = json.loads((D / "walk" / "meta.json").read_text())
    P = {n: np.load(D / "walk" / f"ptr_{n}.npz") for n in ("4096", "16384")}
    msha = sha_of(D / "model.pt")
except (OSError, ValueError):
    meta = P = msha = None
wb = ["missing"] if P is None else []
if P is not None:
    if meta.get("ckpt_sha256") != msha: wb.append("sha")
    for n in P:
        if not np.array_equal(P[n]["depth"], K1[n]["depth"]): wb.append(f"beds {n}")
        idx = np.arange(int(n))
        if not ((P[n]["ptr"] <= idx).all() and (P[n]["ptr"] >= 0).all()): wb.append(f"causal {n}")
check(f"cameron.k2.pilot_walk_machinery s{SEED}", not wb, "ok" if not wb else ", ".join(wb), g=True)

ceil = {}
for n, thr in CELLS:
    name = f"cameron.k2.pilot_walk_ceiling s{SEED} n={n} depth>{thr} OR-walk >= 0.95"
    if wb or not trained:
        check(name, None, "unread (walk machinery or pilot_trained)", g=True)
        continue
    ptr, root, dep = P[n]["ptr"], P[n]["root"], P[n]["depth"]
    ok = np.stack([[(root[b][land(ptr[b, h])] == root[b]) & (dep[b][land(ptr[b, h])] <= 8) for h in range(ptr.shape[1])]
                   for b in range(ptr.shape[0])]).any(1)
    m = dep > thr
    ceil[(n, thr)] = float(ok[m].mean())
    check(name, ceil[(n, thr)] >= 0.95, f"{ceil[(n, thr)]:.4f} (N {int(m.sum())})", g=True)

for n, thr in (("4096", 160), ("8192", 160), ("16384", 160), ("16384", 1280)):
    name = f"cameron.k2.pilot_far10 s{SEED} n={n} depth>{thr} >= 0.95"
    if not trained:
        check(name, None, "unread/missing")
        continue
    z = np.load(D / f"ok_{n}.npz")
    m = z["depth"] > thr
    v = float(z["ok"][m].mean())
    check(name, v >= 0.95, f"{v:.4f} (N {int(m.sum())})")
v = r["eval"]["16384"]["acc"] if trained else None
check(f"cameron.k2.pilot_multilen16k s{SEED} acc n=16384 >= 0.95", None if v is None else v >= 0.95, "unread/missing" if v is None else f"{v:.4f}")

launch = bool(gate) and all(gate.values())
if launch:
    branch = "GRID"
elif gate.get(f"cameron.k2.pilot_config s{SEED}") and trained and not wb and len(ceil) == 3:
    branch = "IDENTITY (walk ceiling " + " / ".join(f"{ceil[c]:.4f}" for c in CELLS) + ")"
elif gate.get(f"cameron.k2.pilot_config s{SEED}") and r is not None and not trained:
    branch = "REPEAT_S1" if SEED == 0 else "SP_RETIRED -> IDENTITY with fR_sp rows UNREAD"
else:
    branch = "NONE"
check(f"cameron.k2.pilot_grid_launch s{SEED} every GATE row GREEN", launch, "BRANCH: " + branch)
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
