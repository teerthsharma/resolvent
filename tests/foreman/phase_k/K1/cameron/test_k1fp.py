# K1.F' bars (Cameron, it.K1): the floor row of the replacement bed bed_k' (bed_kp.py). Written 2026-09-23 after the
# Dispatcher's ruling voided bed_k, BEFORE k1fp.py exists or runs. Never edited after its RED. Reads k1fp.json.
# The design (M_test = 8, M_train = 32) was chosen on UNBARRED exploration seeds [91..94, ...] (explore_kp*.log);
# every bed here is fresh: train-dist n=1024 default_rng([21, k]); test n in {4096, 8192, 16384} default_rng([22, n, k]),
# k = 0..7; empirical-HPD calibration beds [24, k] (train) and [23, n, k] (test), k = 0..7.
# Families at R0 (d 128) pricing, as K1.F: k=3 W<=10, k=4 W<=6; all chance-credited (unresolved -> most recent root at
# or before the final pointer). Window families: anchored, centered (K0 shortcut.hybrid shapes, centre v1 - M*J),
# hpd_analytic (Foreman's k1b.hpd_hybrid, NegBin(J, 1/M) law, ported with P = 1/M), hpd_empirical (the HPD window of the
# generator's measured J-hop offset law), local (no oracle: the token's own W predecessors). "window best" = max over
# the 5 families and the budget, per (n, L), by beyond-2^L mean over 8 beds.
#
# cameron.k1.fp_machinery : (a) hpd_analytic with M = 16 on K1.F's beds [12, 4096, k], L = 7, k3 W10 reproduces
#     Foreman's k1b.json value 0.5642224409448819 to 1e-12; (b) k=2 windows equal doubling (0 mismatches); (c) credited
#     >= zero-credit everywhere; (d) every test bed has exactly 8 roots, every train bed max depth <= 32, ids distinct.
# cameron.k1.fp_not_void : the registered kill, re-run on bed_k'. At every scored cell - (n=1024 train-dist, L=4) and
#     (n in {4096, 8192, 16384}) x (L in {4, 7}) - the (a_L') line (max of recency, credited doubling, window best)
#     beyond 2^L is < 0.5. Any cell >= 0.5 -> bed_k' void.
# cameron.k1.fp_line : the line recomputed here from the json's components equals the json's line (1e-12), all/beyond.
# cameron.k1.fp_deep : the beyond-2^L set is not a sliver: mean fraction of tokens with depth > 2^L >= 0.25 per cell.
# cameron.k1.fp_resolvent_feasible : the hand-set resolvent (gamma = 0.999, softmax logit scale 20, parent pointer and
#     root self-pointer, independent rebuild of wald.py's algebra) scores >= 0.99 overall and >= 0.99 beyond 2^4 (mean
#     of 8 beds) at every test n: the bed does not rule out (f_R)'s prediction by construction.
# (n = 1024, L = 7) has no token deeper than 32 and must be undefined in every component.
# Exit 1 if any row fails.
import json, sys
from pathlib import Path
HERE = Path(__file__).parent
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


R = json.loads((HERE / "k1fp.json").read_text())
m = R["machinery"]
check("cameron.k1.fp_machinery (a) hpd port == Foreman 0.5642224409448819",
      abs(m["hpd_port_M16_L7_n4096_k3W10"] - 0.5642224409448819) < 1e-12, m["hpd_port_M16_L7_n4096_k3W10"])
check("cameron.k1.fp_machinery (b) k2 == doubling", m["k2_vs_doubling_mismatches"] == 0, m["k2_vs_doubling_mismatches"])
check("cameron.k1.fp_machinery (c) credited >= zero", m["credit_below_zero_violations"] == 0, m["credit_below_zero_violations"])
check("cameron.k1.fp_machinery (d) bed invariants", m["test_roots_all_8"] is True and m["train_max_depth"] <= 32 and m["ids_distinct"] is True,
      {k: m[k] for k in ("test_roots_all_8", "train_max_depth", "ids_distinct")})
FAM = ("anchored", "centered", "hpd_analytic", "hpd_empirical", "local")
CELLS = [("1024", 4)] + [(n, L) for n in ("4096", "8192", "16384") for L in (4, 7)]
for n, L in CELLS:
    c = R["rows"][n][f"L{L}"]
    comps = {"recency": c["recency"], "doubling_credited": c["doubling_credited"], **{f: c["families"][f] for f in FAM}}
    for met in ("all", "beyond"):
        want = max(v[met][0] for v in comps.values())
        check(f"cameron.k1.fp_line n={n} L={L} {met}", abs(c["line"][met] - want) < 1e-12, f"{c['line'][met]} vs {want}")
    arg = max(comps, key=lambda k: comps[k]["beyond"][0])
    check(f"cameron.k1.fp_not_void n={n} L={L} line beyond 2^L < 0.5", c["line"]["beyond"] < 0.5,
          f"{c['line']['beyond']:.4f} (argmax {arg} {comps[arg].get('cfg', '')}; per-bed max {comps[arg]['beyond'][2]:.4f})")
    check(f"cameron.k1.fp_deep n={n} L={L} frac(depth > 2^L) >= 0.25", c["frac_beyond"][0] >= 0.25, c["frac_beyond"])
c = R["rows"]["1024"]["L7"]
check("cameron.k1.fp_line n=1024 L=7 undefined", c["line"]["beyond"] is None and c["frac_beyond"][2] == 0.0, c["frac_beyond"])
for n in ("4096", "8192", "16384"):
    r = R["resolvent"][n]
    check(f"cameron.k1.fp_resolvent_feasible n={n} acc >= 0.99 and beyond16 >= 0.99", r["acc"][0] >= 0.99 and r["beyond16"][0] >= 0.99, r)
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
