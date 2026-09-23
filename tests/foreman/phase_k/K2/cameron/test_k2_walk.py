# K2.0 replacement route (Cameron), registered 2026-09-23 AFTER test_k2_leak.py's measurement and BEFORE any number
# this file reads exists. Never edited after its RED.
# Why it exists: K2.0's prediction failed (max over kappa 0.8750 < 0.9 at kappa = 3) with the counter quiet, and the
# gamma-horizon counter fired (lift +0.0166 <= 0.02 on depth > 1280 at 16k). Hypothesis H_walk: what remains is the
# trained pointer's identity, not its temperature. Each row i of the resolvent's W (per head) has an argmax
# ptr[i] = argmax_{j <= i} qs_i . k_j, which is invariant to kappa (kappa scales a row's logits by a positive number).
# Follow ptr from token i to its fixed point L(i) (ptr[L] = L); the hard walk is right iff L(i) lies in i's own chain
# at depth <= 8 (root[L(i)] == root[i] and depth[L(i)] <= 8 = 2^(L-1), the reach of the 3 ALiBi layers below the
# resolvent; a deep token that absorbs at itself or at a deep ancestor has not reached a node that carries the root).
# That is the kappa -> infinity limit of this checkpoint's pointer, and the ceiling any exact-zero weighting
# (sparsemax, hard pointer) reaches on these logits.
# Producer (written after this file): k2_walk.py <runs dir> writes <runs dir>/walk/:
#   ptr_<n>.npz {ptr [8, H, n] int, root [8, n], depth [8, n]} at kappa = 3, far10 beds default_rng([31, n, k]), k 0..7
#   meta.json {"kappa": "3", "ckpt_sha256", "kinv_agree": agreement of ptr at kappa 1 vs 3 on bed k = 0, n = 4096}
# OR over heads: a token's walk is right if either head's walk is right (the optimistic ceiling).
# Rows:
#   cameron.k2.walk_machinery        : files present; kappa 3; sha = K1 checkpoint's; depth arrays = K1's ok_<n>.npz;
#                                      ptr causal (ptr[i] <= i); kinv_agree >= 0.9999.
#   cameron.k2.walk_sane             : OR-walk accuracy on depth <= 128 at n = 4096 >= 0.98 (the walk describes a
#                                      working pointer where the soft model works).
#   cameron.k2.walk_explains         : per-token agreement between the soft model's ok (runs/k3, K2.0) and the OR-walk
#                                      on depth > 160 >= 0.9, at n = 4096 and at 16384.
#   cameron.k2.walk_explains_counter_quiet : counter "agreement at 16384 <= 0.7" (the errors sit downstream of W,
#                                      in the read or readout); PASS iff agreement > 0.7.
#   cameron.k2.walk_ceiling          : OR-walk accuracy on the three far10 cells (depth > 160 at 4096 and 16384,
#                                      depth > 1280 at 16384) >= 0.95. PASS = exact zeros on this pointer clear the band.
#   cameron.k2.walk_link_grows       : pointer head (the head with the higher walk accuracy on depth > 160 at 16384,
#                                      ties to head 0): the off-chain link rate (root[ptr[i]] != root[i]) at positions
#                                      4096-16383 >= 2 x the rate at positions 0-1023 (n = 16384, 8 beds pooled).
# Usage: python test_k2_walk.py [runs dir, default runs]. Exit 1 if any row fails.
import hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
K1R = HERE.parent.parent / "K1" / "cameron" / "runs" / "far_fR_ga_s0"
RUNS = HERE / (sys.argv[1] if len(sys.argv) > 1 else "runs")
WD = RUNS / "walk"
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok is True else "FAIL ") + name + " : " + str(got))
    if ok is not True:
        fails.append(name)


def land(ptr):
    L = ptr.copy()
    while True:
        L2 = L[L]
        if np.array_equal(L2, L):
            return L
        L = L2


sha = hashlib.sha256((K1R / "model.pt").read_bytes()).hexdigest()
K1 = {n: np.load(K1R / f"ok_{n}.npz") for n in ("4096", "16384")}
try:
    meta = json.loads((WD / "meta.json").read_text())
    P = {n: np.load(WD / f"ptr_{n}.npz") for n in ("4096", "16384")}
    OK = {n: np.load(RUNS / "k3" / f"ok_{n}.npz")["ok"] for n in ("4096", "16384")}
except (OSError, ValueError, KeyError):
    meta = P = OK = None

bad = ["missing"] if P is None else []
if P is not None:
    if meta.get("kappa") != "3": bad.append("kappa")
    if meta.get("ckpt_sha256") != sha: bad.append("sha")
    if not (meta.get("kinv_agree", 0) >= 0.9999): bad.append(f"kinv_agree {meta.get('kinv_agree')}")
    for n in P:
        if not np.array_equal(P[n]["depth"], K1[n]["depth"]): bad.append(f"beds {n}")
        idx = np.arange(int(n))
        if not (P[n]["ptr"] <= idx).all() or not (P[n]["ptr"] >= 0).all(): bad.append(f"causal {n}")
check("cameron.k2.walk_machinery", not bad, "ok" if not bad else ", ".join(bad))

if bad:
    for r in ("walk_sane", "walk_explains", "walk_explains_counter_quiet", "walk_ceiling", "walk_link_grows"):
        check(f"cameron.k2.{r}", False, "unread (machinery)")
    print(f"RED {len(fails)} failing rows")
    sys.exit(1)

W, HW = {}, {}
for n in P:
    ptr, root = P[n]["ptr"], P[n]["root"]
    dep = P[n]["depth"]
    per = np.stack([[(root[b][land(ptr[b, h])] == root[b]) & (dep[b][land(ptr[b, h])] <= 8) for h in range(ptr.shape[1])]
                    for b in range(ptr.shape[0])])  # [8, H, n]
    HW[n], W[n] = per, per.any(1)

d4, d16 = P["4096"]["depth"], P["16384"]["depth"]
v = float(W["4096"][d4 <= 128].mean())
check("cameron.k2.walk_sane OR-walk acc depth<=128 n=4096 >= 0.98", v >= 0.98, f"{v:.4f} (N {int((d4 <= 128).sum())})")

agree = {}
for n, d in (("4096", d4), ("16384", d16)):
    m = d > 160
    agree[n] = float((OK[n][m] == W[n][m]).mean())
    both_wrong = float((~OK[n][m] & ~W[n][m]).mean()); soft_only = float((OK[n][m] & ~W[n][m]).mean()); walk_only = float((~OK[n][m] & W[n][m]).mean())
    check(f"cameron.k2.walk_explains n={n} depth>160 agreement(ok, OR-walk) >= 0.9", agree[n] >= 0.9,
          f"{agree[n]:.4f} (N {int(m.sum())}; both wrong {both_wrong:.4f}, soft right walk wrong {soft_only:.4f}, walk right soft wrong {walk_only:.4f})")
check("cameron.k2.walk_explains_counter_quiet n=16384 agreement > 0.7 (FAIL = counter fires)", agree["16384"] > 0.7, f"{agree['16384']:.4f}")

for n, thr in (("4096", 160), ("16384", 160), ("16384", 1280)):
    d = P[n]["depth"]
    m = d > thr
    v = float(W[n][m].mean())
    ph = " / ".join(f"head {h} {float(HW[n][:, h][m].mean()):.4f}" for h in range(HW[n].shape[1]))
    check(f"cameron.k2.walk_ceiling n={n} depth>{thr} OR-walk >= 0.95", v >= 0.95, f"{v:.4f} (N {int(m.sum())}; {ph})")

m = d16 > 160
hs = [float(HW["16384"][:, h][m].mean()) for h in range(HW["16384"].shape[1])]
ph = int(np.argmax(hs))
ptr, root = P["16384"]["ptr"][:, ph], P["16384"]["root"]
off = np.stack([root[b][ptr[b]] != root[b] for b in range(ptr.shape[0])])
pos = np.arange(off.shape[1])
early, late = float(off[:, pos < 1024].mean()), float(off[:, pos >= 4096].mean())
mid = float(off[:, (pos >= 1024) & (pos < 4096)].mean())
check(f"cameron.k2.walk_link_grows head {ph} off-chain rate pos 4096-16383 >= 2 x pos 0-1023", late > 0 and late >= 2 * early,
      f"{early:.5f} (0-1023) / {mid:.5f} (1024-4095) / {late:.5f} (4096-16383)")

print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
