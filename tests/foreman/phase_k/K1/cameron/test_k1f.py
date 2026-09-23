# K1.F bars (Cameron, it.K1). Written 2026-09-23 before k1f.py existed or ran. Never edited after its RED.
# Bed: bed_k (n distinct ids from V = 32,768, parents by id). Beds: train n=1024 default_rng([11, k]),
# test n in {4096, 8192, 16384} default_rng([12, n, k]), k = 0..7. Width: R0 (4 layers x d 128); L in {4, 7}.
# C2 hybrid = Foreman's shortcut.hybrid (copied unchanged, sha 16a4a913...) GRANTED a free exact positional channel
# (the oracle; an ALiBi twin has none), chance-credited like the doubling ceiling (on an unresolved token: the most
# recent root at or before its final pointer). Window priced at R0 width as Foreman priced it (K0 C3):
# k=3 attention 12*W <= 128 -> W <= 10; k=4 MLP 12*W^2 <= 4*128 -> W <= 6; modes anchored and centered.
# "best window" = max over those configs, per (L, n, metric).
#
# cameron.k1.f_machinery : hybrid with k=2 equals pure doubling on every bed (0 pointer mismatches); credited >=
#     zero-credit on every config and bed; and 3 per-bed numbers recomputed here from bed_k/shortcut match the json
#     to 1e-12 (recency all, credited doubling L=4 beyond, C2 anchored k=3 W=8 L=4 credited beyond; test n=4096 k=0).
# cameron.k1.f_not_void  : the registered kill (RECORD_K clause 2). For L in {4,7} and every TEST n, the best-window
#     C2 hybrid's credited accuracy beyond depth 2^L (mean of 8 beds) is <= 0.5. Any row > 0.5 -> bed void.
# cameron.k1.f_line      : the (a_L) line, recomputed here as max(recency, credited doubling, best C2) per
#     (L, n, metric in {all, beyond}), equals the json's line to 1e-12 and lies in [0, 1] (beyond where defined).
# Exit 1 if any row fails.
import json, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

fails = []
def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)

R = json.loads((HERE / "k1f.json").read_text())
TEST_N = ("4096", "8192", "16384")

# ---- f_machinery
m = R["machinery"]
check("cameron.k1.f_machinery k2==doubling mismatches", m["k2_vs_doubling_mismatches"] == 0, m["k2_vs_doubling_mismatches"])
check("cameron.k1.f_machinery credited>=zero violations", m["credit_below_zero_violations"] == 0, m["credit_below_zero_violations"])
from bed_k import make_test, floor_recency, ceiling_credit, _last_root
from shortcut import hybrid
b = make_test(np.random.default_rng([12, 4096, 0]), 4096)
p, d, r = b["parent"], b["depth"], b["root"]
deep = d > 16
rec = floor_recency(p, r)[0]
dbl = ceiling_credit(p, d, r, 4, deep)
c2 = float((_last_root(p)[hybrid(p, d, 4, 8, 3, "anchored")] == r)[deep].mean())
ref = R["probe_n4096_k0"]
got = (abs(rec - ref["recency_all"]), abs(dbl - ref["doubling_credited_L4_beyond"]), abs(c2 - ref["c2_anchored_k3_W8_L4_credited_beyond"]))
check("cameron.k1.f_machinery recomputed probes match json", max(got) < 1e-12, got)

# ---- f_not_void
worst = max(R["rows"][n][f"L{L}"]["c2_best"]["beyond"][0] for n in TEST_N for L in (4, 7))
for n in TEST_N:
    for L in (4, 7):
        v = R["rows"][n][f"L{L}"]["c2_best"]["beyond"]
        check(f"cameron.k1.f_not_void n={n} L={L} C2 best credited beyond 2^L <= 0.5", v[0] <= 0.5,
              f"mean {v[0]:.4f} [min {v[1]:.4f}, max {v[2]:.4f}] cfg {R['rows'][n][f'L{L}']['c2_best']['beyond_cfg']}")

# ---- f_line
for n, row in R["rows"].items():
    for L in (4, 7):
        c = row[f"L{L}"]
        for met in ("all", "beyond"):
            comps = [c["recency"][met], c["doubling_credited"][met], c["c2_best"][met]]
            if any(x is None for x in comps):
                check(f"cameron.k1.f_line n={n} L={L} {met} undefined together", all(x is None for x in comps) and c["line"][met] is None, comps)
                continue
            want = max(x[0] for x in comps)
            got = c["line"][met]
            check(f"cameron.k1.f_line n={n} L={L} {met}", got is not None and abs(got - want) < 1e-12 and 0.0 <= want <= 1.0,
                  f"line {got} recomputed {want:.4f}")
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
