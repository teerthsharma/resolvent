# Foreman K1 second bar set. Written 2026-09-23 03:08 IST after hunt.json (whose wide local-window sweep is an
# UNBARRED field) and BEFORE k1b.py exists or runs. Never edited after its RED. Reads k1b.json. Exit 1 if any row fails.
#
# (1) HPD window. K1.F's "best window" is the best of two FIXED placements (anchored [v1-W, v1) or centered on
#     v1 - 16*hops). The positional offset v1 - v2 is a sum of J(t) Geometric(1/16) gaps (J(t) = the token's current
#     hop count, which it can track), i.e. NegBin(J, 1/16), so the best W-slot window is its highest-density interval:
#     anchored at J = 1 (gap mode 1), near the mode for large J. Same oracle, same pricing, same credit, same beds
#     (Cameron's default_rng([12, n, k])).
# foreman.k1.hpd_machinery : for J = 1 the HPD window equals the anchored window; for every J in 1..400 and W in 1..10
#     the HPD mass >= the centered window's mass (both computed from the same pmf).
# foreman.k1.hpd_beats_line : HPD best over R0 configs (k3 W<=10, k4 W<=6), L=7, n=4096, credited beyond 2^L
#     > 0.4085875984251969 + 0.01 (K1.F's measured line).
# foreman.k1.hpd_not_void : RECORD_K clause 2 applied to it: for L in {4, 7}, n in {4096, 8192, 16384}, HPD best
#     credited beyond 2^L <= 0.5.
# (2) No-oracle local window, width-priced. Fresh beds default_rng([52, 4096, k]), k = 0..7 (disjoint from hunt.json's).
# foreman.k1.local_wide_fires : the ALiBi-local window (the W positions before the token itself, fresh pointers, no
#     position oracle), k3 W=64, L=7, n=4096: credited beyond 2^L > 0.5.
# foreman.k1.local_crossing : the smallest W in 11..128 at which that construction exceeds 0.5 (L=7, n=4096) lies in [33, 64].
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "k1b.json")).read_text())
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


m = R["hpd_machinery"]
check("foreman.k1.hpd_machinery J=1 HPD == anchored", m["j1_equals_anchored"] is True, m)
check("foreman.k1.hpd_machinery HPD mass >= centered mass", m["hpd_ge_centered_violations"] == 0, m)
c = R["hpd"]["4096"]["7"]
check("foreman.k1.hpd_beats_line L7 n4096 > 0.4186", c["best"] > 0.4085875984251969 + 0.01, f"{c['best']:.4f} [{c['best_cfg']}]")
for n in ("4096", "8192", "16384"):
    for L in ("4", "7"):
        c = R["hpd"][n][L]
        check(f"foreman.k1.hpd_not_void n={n} L={L} <= 0.5", c["best"] <= 0.5, f"{c['best']:.4f} [{c['best_cfg']}]")
lw = R["local_wide"]
check("foreman.k1.local_wide_fires k3 W64 L7 n4096 > 0.5", lw["W64"] > 0.5, lw["W64"])
check("foreman.k1.local_crossing smallest W > 0.5 in [33, 64]", lw["crossing_W"] is not None and 33 <= lw["crossing_W"] <= 64,
      lw["crossing_W"])
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
