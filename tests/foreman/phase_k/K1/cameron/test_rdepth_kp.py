# R-DEPTH at R0 on bed_k' (Cameron, it.K1). Written 2026-09-23 after K1.F' bound bed_k' GREEN and BEFORE any arm trained
# on it. Never edited after its RED. Replaces test_rdepth.py (registered on bed_k, retired unrun when the Dispatcher voided
# bed_k): same bars, same arms and gate, new names cameron.k1.rdepth_kp_*, runs in runs/kp_<arm>_s<seed>/ (rdepth.py
# --bed kp), eval beds = the K1.F' beds (train-dist n=1024: 64 beds default_rng([21, k]); test: default_rng([22, n, k])),
# and the (a_L') line read from k1fp.json. Every run must carry config bed == "kp".
# Runs: runs/<arm>_s<seed>/result.json from rdepth.py, seeds 0 and 1. Arms: aL4 (4 ALiBi layers, dff 512), aL7 (7 layers,
# dff 183: non-embedding params 806,912 vs 805,120), ass4 (aL4 + SSMax), aloop5 (one ALiBi block x 5, FLOP/clock-matched
# to fR by Chase's 2.27x resolvent cost), fR (3 ALiBi + 1 resolvent). Train n in {256,512,1024}, depth <= 32.
# Eval: see header (K1.F' beds).
#
# GATE L-TRAINED (the "check that could not pass" rule): arm read only if its held-out n=1024 accuracy on tokens of
#   depth <= min(2^Larm, 32) is >= 0.9 (Larm = layers; aloop: 5 applications). An unread arm's rows print UNREAD and
#   count as failing (an arm that did not learn cannot pass or fire a bar).
# cameron.k1.rdepth_kp_trained      : every arm/seed passes L-TRAINED.
# cameron.k1.rdepth_kp_learnable    : contract kill. fR held-out n=1024 accuracy (all tokens) >= 0.8, each seed.
# cameron.k1.rdepth_kp_prediction   : fR accuracy beyond depth 2^4 = 16 at n=16384 >= 0.95, each seed.
# cameron.k1.rdepth_kp_multilen16k  : RECORD_K clause 4. fR accuracy (all tokens) at n=16384 >= 0.95, each seed.
# cameron.k1.rdepth_kp_counter_quiet: contract counter does NOT fire. aL4 and aL7 accuracy beyond 2^L <= 0.5 at every
#   test n, each seed. A row > 0.5 = the counter fires (a shortcut the author did not find; the bed is void).
# cameron.k1.rdepth_kp_opponents_quiet: same line for ass4 (beyond 2^4) and aloop5 (beyond 2^5 = 32) at every test n.
# cameron.k1.rdepth_kp_twin_under_line: aL4/aL7 beyond 2^L <= the K1.F' (a_L') line (k1fp.json, same beds) + 3 SE,
#   SE = sqrt(line (1 - line) / N_beyond), at every test n, each seed (replaces "stays under 0.254").
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


def load(arm, s):
    p = HERE / "runs" / f"kp_{arm}_s{s}" / "result.json"
    r = json.loads(p.read_text()) if p.exists() else None
    return r if (r is None or r["config"].get("bed") == "kp") else None


LARM = {"aL4": 4, "aL7": 7, "ass4": 4, "aloop5": 5, "fR": 4}
TEST_N = ("4096", "8192", "16384")
R = {(a, s): load(a, s) for a in LARM for s in (0, 1)}


def trained(a, s):
    r = R[(a, s)]
    if r is None:
        return None
    key = "acc_le16" if LARM[a] == 4 else "acc_le32"
    return r["eval"]["1024"][key]


for (a, s), r in R.items():
    v = trained(a, s)
    check(f"cameron.k1.rdepth_kp_trained {a} s{s} held-out acc depth<=min(2^{LARM[a]},32) >= 0.9",
          False if v is None else v >= 0.9, "missing" if v is None else f"{v:.4f}")


def read(a, s):
    v = trained(a, s)
    return R[(a, s)] if (v is not None and v >= 0.9) else None


for s in (0, 1):
    r = R[("fR", s)]
    v = None if r is None else r["eval"]["1024"]["acc"]
    check(f"cameron.k1.rdepth_kp_learnable fR s{s} held-out n=1024 acc >= 0.8", False if v is None else v >= 0.8,
          "missing" if v is None else f"{v:.4f}")
    r = read("fR", s)
    check(f"cameron.k1.rdepth_kp_prediction fR s{s} beyond16 at n=16384 >= 0.95",
          None if r is None else r["eval"]["16384"]["beyond16"] >= 0.95, "unread/missing" if r is None else r["eval"]["16384"]["beyond16"])
    check(f"cameron.k1.rdepth_kp_multilen16k fR s{s} acc at n=16384 >= 0.95",
          None if r is None else r["eval"]["16384"]["acc"] >= 0.95, "unread/missing" if r is None else r["eval"]["16384"]["acc"])

K1F = json.loads((HERE / "k1fp.json").read_text())
for a, L in (("aL4", 4), ("aL7", 7), ("ass4", 4), ("aloop5", 5)):
    for s in (0, 1):
        r = read(a, s)
        for n in TEST_N:
            key = "beyond16" if L == 4 else ("beyond128" if L == 7 else "beyond_2^5")
            v = None if r is None else r["eval"][n][key]
            name = "cameron.k1.rdepth_kp_counter_quiet" if a in ("aL4", "aL7") else "cameron.k1.rdepth_kp_opponents_quiet"
            check(f"{name} {a} s{s} n={n} beyond 2^{L} <= 0.5", None if v is None else v <= 0.5, "unread/missing" if v is None else f"{v:.4f}")
            if a in ("aL4", "aL7"):
                line = K1F["rows"][n][f"L{L}"]["line"]["beyond"]
                N = None if r is None else r["eval"][n][f"n_beyond{16 if L == 4 else 128}"]
                bar = None if N is None else line + 3 * math.sqrt(line * (1 - line) / N)
                check(f"cameron.k1.rdepth_kp_twin_under_line {a} s{s} n={n} beyond 2^{L} <= line+3SE",
                      None if v is None else v <= bar, "unread/missing" if v is None else f"{v:.4f} vs {bar:.4f} (line {line:.4f})")
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
