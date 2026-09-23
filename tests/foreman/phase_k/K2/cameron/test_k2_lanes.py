# K2 position-vs-lanes wrong-link test (Cameron; Dispatcher order after K2.0; the replacement for walk_link_grows).
# Written 2026-09-23 BEFORE its producer and before any number it reads. Never edited after its RED. Eval only.
# Question: are the trained pointer's off-chain links driven by position (competitor count, the position-dependent logit
# scale) or by depth? On the far10 beds (8 lanes) depth ~ position / 8, so the two are confounded; at 32 lanes depth ~
# position / 32. The identity grid (test_k2_grid.py id) grows the training context at depth cap 32, so it is aimed at
# the position account; this test checks that premise on the K1-b checkpoint.
# Checkpoint K1/cameron/runs/far_fR_ga_s0/model.pt at its trained logit scale (the argmax pointer is kappa-invariant).
# Producer (written after this file): k2_ptr.py lanes <runs dir> writes <runs>/lanes/:
#   ptr_M8.npz  {ptr [8, H, 16384], root, depth}  beds bed_kp.make_test(default_rng([31, 16384, k]), 16384), k 0..7
#   ptr_M32.npz {ptr [8, H, 16384], root, depth}  beds bed_k._pack(*bed_k.make_bed(16384, 32, rng), rng),
#                                                rng = default_rng([41, 16384, k]), k 0..7 (no depth cap)
#   meta.json {"ckpt_sha256"}
# Off-chain link of token i in head h: root[ptr[h, i]] != root[i]. Rates pool both heads and the 8 beds.
#   r_pos = rate(M32) / rate(M8) over positions 4096-16383 (equal position; M32 is ~4x shallower there)
#   r_dep = rate(M32) / rate(M8) over depth 129-512   (equal depth; M32 sits ~4x later in position)
# Position-driven predicts r_pos ~ 1 and r_dep > 1; depth-driven predicts r_pos < 1 and r_dep ~ 1.
# Rows:
#   cameron.k2.lanes_machinery         files; sha = the K1-b checkpoint; M8 depth = K1's ok_16384 depth; M32 root/depth =
#                                      the beds regenerated from their seeds; ptr causal.
#   cameron.k2.lanes_events            each of the four rates rests on >= 30 off-chain links (else the ratios are unread).
#   cameron.k2.lanes_position_driven   prediction: r_pos >= 0.67 and r_dep >= 1.5.
#   cameron.k2.lanes_depth_counter_quiet counter "r_pos <= 0.33" (depth drives the links, and longer training contexts
#                                      at depth cap 32 are mis-aimed); PASS iff r_pos > 0.33.
# Usage: python test_k2_lanes.py [runs dir, default runs]. Exit 1 if any row fails.
import hashlib, json, sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
K1C = HERE.parent.parent / "K1" / "cameron"
sys.path.insert(0, str(K1C))
import bed_k

K1R = K1C / "runs" / "far_fR_ga_s0"
RUNS = HERE / (sys.argv[1] if len(sys.argv) > 1 else "runs")
LD = RUNS / "lanes"
N = 16384
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok is True else "FAIL ") + name + " : " + str(got))
    if ok is not True:
        fails.append(name)


try:
    meta = json.loads((LD / "meta.json").read_text())
    P = {M: np.load(LD / f"ptr_M{M}.npz") for M in (8, 32)}
except (OSError, ValueError):
    meta = P = None
bad = ["missing"] if P is None else []
if P is not None:
    if meta.get("ckpt_sha256") != hashlib.sha256((K1R / "model.pt").read_bytes()).hexdigest(): bad.append("sha")
    if not np.array_equal(P[8]["depth"], np.load(K1R / "ok_16384.npz")["depth"]): bad.append("M8 beds")
    for k in range(8):
        par, dep, root = bed_k.make_bed(N, 32, np.random.default_rng([41, N, k]))
        if not (np.array_equal(P[32]["depth"][k], dep) and np.array_equal(P[32]["root"][k], root)):
            bad.append(f"M32 bed {k}")
            break
    for M in P:
        if not ((P[M]["ptr"] <= np.arange(N)).all() and (P[M]["ptr"] >= 0).all()): bad.append(f"causal M{M}")
check("cameron.k2.lanes_machinery", not bad, "ok" if not bad else ", ".join(bad))

if bad:
    for r in ("lanes_events", "lanes_position_driven", "lanes_depth_counter_quiet"):
        check(f"cameron.k2.{r}", False, "unread (machinery)")
    print(f"RED {len(fails)} failing rows")
    sys.exit(1)

rate, ev = {}, {}
pos = np.arange(N)
for M in (8, 32):
    ptr, root, dep = P[M]["ptr"], P[M]["root"], P[M]["depth"]
    off = np.stack([[root[b][ptr[b, h]] != root[b] for h in range(ptr.shape[1])] for b in range(ptr.shape[0])])  # [8, H, N]
    for key, m in (("pos", np.broadcast_to(pos >= 4096, dep.shape)), ("dep", (dep >= 129) & (dep <= 512))):
        sel = off[:, :, :][np.broadcast_to(m[:, None, :], off.shape)]
        rate[(M, key)], ev[(M, key)] = float(sel.mean()) if sel.size else float("nan"), int(sel.sum())
ok_ev = all(v >= 30 for v in ev.values())
check("cameron.k2.lanes_events each rate >= 30 off-chain links", ok_ev,
      "; ".join(f"M{M} {k}: {rate[(M, k)]:.5f} ({ev[(M, k)]} links)" for M in (8, 32) for k in ("pos", "dep")))
r_pos = rate[(32, "pos")] / rate[(8, "pos")] if rate[(8, "pos")] > 0 else float("nan")
r_dep = rate[(32, "dep")] / rate[(8, "dep")] if rate[(8, "dep")] > 0 else float("nan")
check("cameron.k2.lanes_position_driven r_pos >= 0.67 and r_dep >= 1.5", ok_ev and r_pos >= 0.67 and r_dep >= 1.5,
      f"r_pos {r_pos:.3f}, r_dep {r_dep:.3f}")
check("cameron.k2.lanes_depth_counter_quiet r_pos > 0.33 (FAIL = counter fires)", ok_ev and r_pos > 0.33, f"r_pos {r_pos:.3f}")
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
