# R-DEPTH at R0 on bed_k', far band moved out to 10 2^L (in case Foreman's W=62 build binds c_max = 5.0). Written
# 2026-09-23 after K1.F''' bound GREEN and BEFORE any R-DEPTH arm trained. Never edited after its RED.
# Same runs as test_rdepth_far.py (runs/far_<arm>_s<seed>/, config bed "kpf", eval beds default_rng([31, n, k])); the band
# accuracies are recomputed here from the per-token arrays runs/far_<arm>_s<seed>/ok_<n>.npz (ok, depth; 8 beds).
# Bands: b4 = depth > 160 (10 * 2^4), b7 = depth > 1280 (10 * 2^7, n = 16384 only), b5 = depth > 320 (10 * 2^5, aloop5).
# Gate L-TRAINED as test_rdepth_far.py (held-out n=1024 acc on depth <= min(2^Larm, 32) >= 0.9; else UNREAD = failing).
# cameron.k1.rdepth_far10_prediction : fR >= 0.95 on b4 at n in {4096, 8192, 16384} and on b7 at n = 16384, each seed.
# cameron.k1.rdepth_far10_counter_quiet : aL4 on b4 (every n) and aL7 on b7 (n = 16384) <= 0.5, each seed.
# cameron.k1.rdepth_far10_opponents_quiet : ass4 on b4 and aloop5 on b5 <= 0.5 at every n, each seed.
# cameron.k1.rdepth_far10_twin_under_line : aL4 b4 / aL7 b7 <= K1.F''' line (k1fppp.json; its beds are [33, n, k], i.i.d.
#   with the arms' [31, n, k]) + 3 SE, SE = sqrt(line (1 - line) / N_band).
# Exit 1 if any row fails.
import json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
fails = []


def check(name, ok, got=""):
    tag = "PASS " if ok is True else ("UNREAD " if ok is None else "FAIL ")
    print(tag + name + " : " + str(got))
    if ok is not True:
        fails.append(name)


LARM = {"aL4": 4, "aL7": 7, "ass4": 4, "aloop5": 5, "fR": 4}
B = {"b4": 160, "b5": 320, "b7": 1280}


def read(a, s):
    d = HERE / "runs" / f"far_{a}_s{s}"
    p = d / "result.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    if r["config"].get("bed") != "kpf":
        return None
    tr = r["eval"]["1024"]["acc_le16" if LARM[a] == 4 else "acc_le32"]
    return d if tr >= 0.9 else None


def band(d, n, b):
    z = np.load(d / f"ok_{n}.npz")
    m = z["depth"] > B[b]
    return float(z["ok"][m].mean()), int(m.sum())


K = json.loads((HERE / "k1fppp.json").read_text())["cells"]
for s in (0, 1):
    d = read("fR", s)
    for n, b in (("4096", "b4"), ("8192", "b4"), ("16384", "b4"), ("16384", "b7")):
        v = None if d is None else band(d, n, b)[0]
        check(f"cameron.k1.rdepth_far10_prediction fR s{s} n={n} {b} >= 0.95", None if v is None else v >= 0.95,
              "unread/missing" if v is None else f"{v:.4f}")
for a, b, name, ns in (("aL4", "b4", "counter_quiet", ("4096", "8192", "16384")), ("aL7", "b7", "counter_quiet", ("16384",)),
                       ("ass4", "b4", "opponents_quiet", ("4096", "8192", "16384")), ("aloop5", "b5", "opponents_quiet", ("4096", "8192", "16384"))):
    for s in (0, 1):
        d = read(a, s)
        for n in ns:
            v, N = (None, None) if d is None else band(d, n, b)
            check(f"cameron.k1.rdepth_far10_{name} {a} s{s} n={n} {b} <= 0.5", None if v is None else v <= 0.5,
                  "unread/missing" if v is None else f"{v:.4f} (N {N})")
            if a in ("aL4", "aL7"):
                line = K[f"{n}_{4 if a == 'aL4' else 7}"]["line"]
                bar = None if N is None else line + 3 * math.sqrt(line * (1 - line) / N)
                check(f"cameron.k1.rdepth_far10_twin_under_line {a} s{s} n={n} {b} <= line+3SE", None if v is None else v <= bar,
                      "unread/missing" if v is None else f"{v:.4f} vs {bar:.4f} (line {line:.4f})")
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
