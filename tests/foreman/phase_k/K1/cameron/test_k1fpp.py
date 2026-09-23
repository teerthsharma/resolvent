# K1.F'' bars (Cameron, it.K1): the floor row on the FAR BAND of bed_k' (Amendment K1-a, RECORD_K commit 49f5a2e).
# Written 2026-09-23 after the Dispatcher's ruling that bed_k' is void and BEFORE k1fpp.py exists or runs. Never edited
# after its RED. Reads k1fpp.json.
# Bed: bed_kp.make_test (8 lanes, V = 32,768), fresh seeds default_rng([31, n, k]), calibration default_rng([32, n, k]),
# k = 0..7. Far band: depth > 2 c_max(L) 2^L = 96 at L = 4 (c_max 3.0) and 896 at L = 7 (c_max 3.5).
# Scored cells: L = 4 at n in {4096, 8192, 16384}; L = 7 at n in {8192, 16384} (n = 4096 has no token past 896).
# Families, all chance-credited (unresolved -> most recent root at or before the final pointer), scored on the band:
#   recency; credited doubling; K1.F' windows (anchored, centered, hpd_analytic, hpd_empirical, local) at k3 W<=10 and
#   k4 W<=6, plus k3 at W in {14, 30}; Foreman's sc-HPD (kp_attack.step/calibrate, imported read-only) at (W, 0) for
#   W in {1..10, 14, 30} and his split configs (W1, 10 - W1). The 14- and 30-slot windows are the bound real-heads
#   builds (kp_w14, kp_next). Unbarred field: sc-HPD at W = 62 (2-dim slots, open in K1-a).
#
# cameron.k1.fpp_machinery : (a) Foreman's sc-HPD W10 at (n=4096, L=7) on HIS beds [22, 4096, k] / [23, 4096, k] scores
#     0.43982539164490864 beyond 2^7 (his kp_attack.json) to 1e-12; (b) credited >= zero-credit on every family/bed;
#     (c) band thresholds are 96 (L4) and 896 (L7); (d) every bed has 8 roots and distinct ids.
# cameron.k1.fpp_band_size : >= 500 band tokens pooled over the 8 beds at every scored cell.
# cameron.k1.fpp_not_void  : the far-band line (max over all families above, 8-bed mean) < 0.5 at every scored cell.
# cameron.k1.fpp_line      : the line recomputed here from the json's family means equals the json's line (1e-12).
# cameron.k1.fpp_resolvent_feasible : hand-set resolvent (gamma 0.999, logit scale 20) >= 0.99 on the band, every cell.
# Exit 1 if any row fails.
import json, sys
from pathlib import Path
HERE = Path(__file__).parent
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


R = json.loads((HERE / "k1fpp.json").read_text())
m = R["machinery"]
check("cameron.k1.fpp_machinery (a) Foreman sc-HPD W10 (4096, L7) == 0.43982539164490864",
      abs(m["foreman_schpd_W10_4096_L7"] - 0.43982539164490864) < 1e-12, m["foreman_schpd_W10_4096_L7"])
check("cameron.k1.fpp_machinery (b) credited >= zero", m["credit_below_zero_violations"] == 0, m["credit_below_zero_violations"])
check("cameron.k1.fpp_machinery (c) band thresholds", m["band"] == {"4": 96, "7": 896}, m["band"])
check("cameron.k1.fpp_machinery (d) 8 roots, distinct ids", m["roots_all_8"] is True and m["ids_distinct"] is True, m)
CELLS = [("4096", 4), ("8192", 4), ("16384", 4), ("8192", 7), ("16384", 7)]
for n, L in CELLS:
    c = R["cells"][f"{n}_{L}"]
    check(f"cameron.k1.fpp_band_size n={n} L={L} >= 500", c["band_tokens"] >= 500, c["band_tokens"])
    want = max(v[0] for v in c["families"].values())
    check(f"cameron.k1.fpp_line n={n} L={L}", abs(c["line"] - want) < 1e-12, f"{c['line']} vs {want}")
    arg = max(c["families"], key=lambda k: c["families"][k][0])
    check(f"cameron.k1.fpp_not_void n={n} L={L} far-band line < 0.5", c["line"] < 0.5, f"{c['line']:.4f} (argmax {arg})")
    check(f"cameron.k1.fpp_resolvent_feasible n={n} L={L} >= 0.99", c["resolvent"][0] >= 0.99, c["resolvent"])
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
