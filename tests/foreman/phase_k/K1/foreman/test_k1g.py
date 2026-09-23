# Foreman K1 seventh bar set: the property a replacement bed must have to defeat every window I know. Written
# 2026-09-23 03:26 IST (clock read 03:25:51), after the Dispatcher's K1 reconcile and BEFORE k1g.py exists or runs.
# Never edited after its RED. Reads k1g.json. Exit 1 on any fail.
#
# Beds: bed_k.make_test(default_rng([55, 32768, k]), 32768), k = 0..3 (deep enough to see where each reach ends).
# Pricing: R0 = {k3 W<=10, k4 W<=6} (d 128); R1 = {k3 W<=26, k4 W<=10} (d 320). Families: HPD (oracle C2 with the
# highest-density window, k1b.py), local (no-oracle window over the token's own W predecessors, hunt.py), doubling.
# All credited (unresolved -> most recent root at or before the final pointer).
# Reach factor c(family, L, pricing) = max over the family's configs of D/2^L, D = the largest depth d (a multiple of
# 2^L/8) such that pooled accuracy on depth in (d - 2^L/8, d] is >= 0.5.
#
# foreman.k1.reach_doubling_is_one : c(doubling, L) = 1.0 exactly for L in {4, 6, 7} (machinery: the known wall).
# foreman.k1.reach_is_constant_factor : c(HPD, L, pricing) lies in [1.2, 3.0] for (L, pricing) in {(4,R0), (6,R0),
#     (7,R0), (6,R1)}: a constant factor above doubling, not a change of the exponent.
# foreman.k1.reach_grows_with_width : c(HPD, 6, R1) - c(HPD, 6, R0) >= 0.2 (the factor grows with the rung's width).
# foreman.k1.far_band_defeats_all (the property): with kappa = 2 * max(c(HPD), c(local)) at that (L, pricing), every
#     family (best config) scores <= 0.0925 (chance 1/16 + 0.03) on tokens with depth > kappa * 2^L, and there are
#     >= 500 such tokens, for (L, pricing) in {(4,R0), (7,R0), (6,R1)}.
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "k1g.json")).read_text())
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


for L in ("4", "6", "7"):
    c = R["reach"][f"doubling_{L}"]
    check(f"foreman.k1.reach_doubling_is_one L={L}", c == 1.0, c)
for key in ("4_R0", "6_R0", "7_R0", "6_R1"):
    c = R["reach"][f"hpd_{key}"]
    check(f"foreman.k1.reach_is_constant_factor HPD {key} in [1.2, 3.0]", 1.2 <= c <= 3.0, c)
g = R["reach"]["hpd_6_R1"] - R["reach"]["hpd_6_R0"]
check("foreman.k1.reach_grows_with_width c(HPD,6,R1) - c(HPD,6,R0) >= 0.2", g >= 0.2,
      (R["reach"]["hpd_6_R1"], R["reach"]["hpd_6_R0"]))
for key in ("4_R0", "7_R0", "6_R1"):
    f = R["far"][key]
    check(f"foreman.k1.far_band_defeats_all {key} kappa={f['kappa']}", f["worst"] <= 0.0925 and f["tokens"] >= 500, f)
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
