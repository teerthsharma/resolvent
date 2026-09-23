# Foreman K1 ninth bar set: the L = 4 reach, re-registered a second time. Written 2026-09-23 03:28 IST (clock read
# 03:27:57), after test_k1h's RED at L = 4 (its 200-token floor exceeds a 2-depth band's 4 beds x 16 lanes x 2 = 128
# tokens, so every band was skipped and c = 0; far_band2 4_R0 then passed vacuously at kappa = 0 -- not a finding),
# and BEFORE k1i.py exists or runs. Never edited after its RED. Reads k1i.json. Exit 1 on any fail.
#
# Same beds, pricing (R0), families and credit as test_k1h.py, L = 4 only. Band w = 2 depths; bands need >= 100 tokens.
# D = the largest multiple of 2 such that every band up to it scores >= 0.5; c = D / 16 (max over configs).
# foreman.k1.reach3_doubling_is_one_L4 : c(doubling, 4) = 1.0.
# foreman.k1.reach3_constant_factor_L4 : c(HPD, 4, R0) in [1.2, 3.0].
# foreman.k1.far_band3_defeats_all_L4 : kappa = 2 * max(c(HPD), c(local)) >= 2; every config of every family scores
#     <= 0.0925 on depth > kappa * 16, with >= 500 such tokens.
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "k1i.json")).read_text())
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


check("foreman.k1.reach3_doubling_is_one_L4", R["doubling"] == 1.0, R["doubling"])
check("foreman.k1.reach3_constant_factor_L4 HPD in [1.2, 3.0]", 1.2 <= R["hpd"] <= 3.0, R["hpd"])
f = R["far"]
check("foreman.k1.far_band3_defeats_all_L4", f["kappa"] >= 2 and f["worst"] <= 0.0925 and f["tokens"] >= 500, f)
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
