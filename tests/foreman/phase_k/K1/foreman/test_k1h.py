# Foreman K1 eighth bar set: test_k1g.py's reach factor re-registered with a definition that the depth tail cannot
# fool. Written 2026-09-23 03:27 IST (clock read 03:27:10), after k1g's RED at L = 4 (its reach took the LARGEST band
# >= 0.5, and a 2-depth band in the tail with a handful of tokens scored 0.5 by the credit guess: c = 131.875), and
# BEFORE k1h.py exists or runs. Never edited after its RED. Reads k1h.json. Exit 1 on any fail.
#
# Same beds (default_rng([55, 32768, k]), k = 0..3), pricing, families and credit as test_k1g.py. New reach:
# D = the largest depth d (a multiple of w = max(1, 2^L/8)) such that EVERY band (d' - w, d'] with d' <= d that holds
# >= 200 pooled tokens has accuracy >= 0.5; c = D / 2^L (max over the family's configs).
# foreman.k1.reach2_doubling_is_one : c(doubling, L) = 1.0 for L in {4, 6, 7}.
# foreman.k1.reach2_constant_factor : c(HPD, L, pricing) in [1.2, 3.0] for (4,R0), (6,R0), (7,R0), (6,R1).
# foreman.k1.reach2_grows_with_width : c(HPD, 6, R1) - c(HPD, 6, R0) >= 0.2.
# foreman.k1.far_band2_defeats_all (the property): kappa = 2 * max(c(HPD), c(local)) at (L, pricing); every config of
#     every family scores <= 0.0925 on depth > kappa * 2^L, with >= 500 such tokens, for (4,R0), (7,R0), (6,R1).
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "k1h.json")).read_text())
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


for L in ("4", "6", "7"):
    c = R["reach"][f"doubling_{L}"]
    check(f"foreman.k1.reach2_doubling_is_one L={L}", c == 1.0, c)
for key in ("4_R0", "6_R0", "7_R0", "6_R1"):
    c = R["reach"][f"hpd_{key}"]
    check(f"foreman.k1.reach2_constant_factor HPD {key} in [1.2, 3.0]", 1.2 <= c <= 3.0, c)
check("foreman.k1.reach2_grows_with_width", R["reach"]["hpd_6_R1"] - R["reach"]["hpd_6_R0"] >= 0.2,
      (R["reach"]["hpd_6_R1"], R["reach"]["hpd_6_R0"]))
for key in ("4_R0", "7_R0", "6_R1"):
    f = R["far"][key]
    check(f"foreman.k1.far_band2_defeats_all {key}", f["worst"] <= 0.0925 and f["tokens"] >= 500, f)
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
