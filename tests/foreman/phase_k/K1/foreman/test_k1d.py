# Foreman K1 fourth bar set: does the replacement rule (n >= 64 * 2^L) hold one rung up? Written 2026-09-23 03:08 IST
# (clock read 03:08:14), BEFORE k1d.py exists or runs. Never edited after its RED. Reads k1d.json. Exit 1 on any fail.
#
# R1 pricing (Cameron's K1.F rule at d = 320): k3 12W <= 320 -> W <= 26; k4 12W^2 <= 4*320 -> W <= 10. The R1 twin is
# L = 6 and its L + 3 arm is L = 9. HPD = the oracle C2 with the highest-density window (k1b.py), credited, bed_k
# test beds default_rng([54, n, k]), k = 0..7. n = 32768 is bed_k's ceiling (ids are distinct draws from V = 32768).
# foreman.k1.r1_rule_holds : HPD best at R1 pricing, credited beyond 2^L <= 0.5 at (L, n) in
#     {(6, 4096), (6, 8192), (6, 16384), (9, 32768)}.
# foreman.k1.r1_short_fires : at the shallow ratio n = 32 * 2^L, (L, n) = (6, 2048), HPD best at R1 pricing > 0.5
#     (the same constant-factor shortcut as (7, 4096) at R0).
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "k1d.json")).read_text())
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


for key in ("6_4096", "6_8192", "6_16384", "9_32768"):
    c = R[key]
    check(f"foreman.k1.r1_rule_holds L_n={key} HPD <= 0.5", c["best"] <= 0.5, f"{c['best']:.4f} [{c['cfg']}] dbl {c['dbl']:.4f}")
c = R["6_2048"]
check("foreman.k1.r1_short_fires L_n=6_2048 HPD > 0.5", c["best"] > 0.5, f"{c['best']:.4f} [{c['cfg']}] dbl {c['dbl']:.4f}")
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
