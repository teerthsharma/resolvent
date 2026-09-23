# R-DEPTH at R0 on bed_k', FAR-BAND scoring (Amendment K1-a). Written 2026-09-23 after K1.F'' bound GREEN and BEFORE
# any R-DEPTH arm trained. Never edited after its RED. Replaces test_rdepth.py (bed_k, retired unrun), test_rdepth_kp.py
# and test_rdepth_kp_line2.py (beyond-2^L on bed_k', retired unrun when bed_k' was voided).
# Runs: runs/far_<arm>_s<seed>/result.json from rdepth.py --bed kpf (train: bed_kp.make_train, M = 32, depth <= 32,
# n in {256, 512, 1024}; eval: 64 held-out train beds default_rng([21, k]) and the K1.F'' beds default_rng([31, n, k]),
# n in {4096, 8192, 16384}); config bed must be "kpf". Seeds 0 and 1.
# Arms: aL4 (4 ALiBi layers), aL7 (7 layers, dff 183, non-embedding params matched to aL4), ass4 (aL4 + SSMax), aloop5
# (one ALiBi block x 5; FLOP/clock-matched to fR by Chase's 2.27x resolvent cost), fR (3 ALiBi + 1 resolvent layer).
# Bands: band4 = depth > 96, band7 = depth > 896 (2 c_max 2^L, c_max 3.0 / 3.5), band5 = depth > 224 (2 * 3.5 * 2^5).
#
# GATE L-TRAINED: arm read only if held-out n=1024 accuracy on depth <= min(2^Larm, 32) >= 0.9 (Larm = layers; aloop 5).
#   Unread rows print UNREAD and count as failing.
# cameron.k1.rdepth_far_trained : every arm/seed passes L-TRAINED.
# cameron.k1.rdepth_far_learnable : contract kill. fR held-out n=1024 accuracy (all tokens) >= 0.8, each seed.
# cameron.k1.rdepth_far_prediction : fR >= 0.95 on band4 at n in {4096, 8192, 16384} and on band7 at n in {8192, 16384},
#   each seed (K1-a: "(f_R) >= 0.95 there").
# cameron.k1.rdepth_far_multilen16k : RECORD_K clause 4. fR accuracy (all tokens) at n = 16384 >= 0.95, each seed.
# cameron.k1.rdepth_far_counter_quiet : the contract counter does NOT fire: aL4 on band4 (every n) and aL7 on band7
#   (n in {8192, 16384}) <= 0.5, each seed. A row > 0.5 = a shortcut the floor row did not find; the bed is void.
# cameron.k1.rdepth_far_opponents_quiet : ass4 on band4 and aloop5 on band5 <= 0.5 at every n, each seed.
# cameron.k1.rdepth_far_twin_under_line : aL4 band4 and aL7 band7 <= the K1.F'' line (k1fpp.json, same beds) + 3 SE,
#   SE = sqrt(line (1 - line) / N_band), each seed and scored n.
# Exit 1 if any row fails.
import json, math, sys
from pathlib import Path
HERE = Path(__file__).parent
fails = []


def check(name, ok, got=""):
    tag = "PASS " if ok is True else ("UNREAD " if ok is None else "FAIL ")
    print(tag + name + " : " + str(got))
    if ok is not True:
        fails.append(name)


LARM = {"aL4": 4, "aL7": 7, "ass4": 4, "aloop5": 5, "fR": 4}
NS = ("4096", "8192", "16384")
R = {}
for a in LARM:
    for s in (0, 1):
        p = HERE / "runs" / f"far_{a}_s{s}" / "result.json"
        r = json.loads(p.read_text()) if p.exists() else None
        R[(a, s)] = r if (r is None or r["config"].get("bed") == "kpf") else None


def tr(a, s):
    r = R[(a, s)]
    return None if r is None else r["eval"]["1024"]["acc_le16" if LARM[a] == 4 else "acc_le32"]


def read(a, s):
    v = tr(a, s)
    return R[(a, s)] if (v is not None and v >= 0.9) else None


for (a, s) in R:
    v = tr(a, s)
    check(f"cameron.k1.rdepth_far_trained {a} s{s} held-out depth<=min(2^{LARM[a]},32) >= 0.9", False if v is None else v >= 0.9,
          "missing" if v is None else f"{v:.4f}")
for s in (0, 1):
    r = R[("fR", s)]
    v = None if r is None else r["eval"]["1024"]["acc"]
    check(f"cameron.k1.rdepth_far_learnable fR s{s} held-out acc >= 0.8", False if v is None else v >= 0.8, "missing" if v is None else f"{v:.4f}")
    r = read("fR", s)
    for n in NS:
        for band in (("band4",) if n == "4096" else ("band4", "band7")):
            v = None if r is None else r["eval"][n][band]
            check(f"cameron.k1.rdepth_far_prediction fR s{s} n={n} {band} >= 0.95", None if v is None else v >= 0.95,
                  "unread/missing" if v is None else f"{v:.4f}")
    v = None if r is None else r["eval"]["16384"]["acc"]
    check(f"cameron.k1.rdepth_far_multilen16k fR s{s} acc n=16384 >= 0.95", None if v is None else v >= 0.95,
          "unread/missing" if v is None else f"{v:.4f}")
K = json.loads((HERE / "k1fpp.json").read_text())["cells"]
for a, band, name, ns in (("aL4", "band4", "counter_quiet", NS), ("aL7", "band7", "counter_quiet", NS[1:]),
                          ("ass4", "band4", "opponents_quiet", NS), ("aloop5", "band5", "opponents_quiet", NS)):
    for s in (0, 1):
        r = read(a, s)
        for n in ns:
            v = None if r is None else r["eval"][n][band]
            check(f"cameron.k1.rdepth_far_{name} {a} s{s} n={n} {band} <= 0.5", None if v is None else v <= 0.5,
                  "unread/missing" if v is None else f"{v:.4f}")
            if a in ("aL4", "aL7"):
                line = K[f"{n}_{4 if a == 'aL4' else 7}"]["line"]
                N = None if r is None else r["eval"][n][f"n_{band}"]
                bar = None if N is None else line + 3 * math.sqrt(line * (1 - line) / N)
                check(f"cameron.k1.rdepth_far_twin_under_line {a} s{s} n={n} {band} <= line+3SE", None if v is None else v <= bar,
                      "unread/missing" if v is None else f"{v:.4f} vs {bar:.4f} (line {line:.4f})")
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
