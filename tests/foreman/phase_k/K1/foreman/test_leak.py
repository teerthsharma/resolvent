# Foreman K1 bars on what bed_k leaks (candidate (d)). Written 2026-09-23 02:58 IST, BEFORE leak.py exists or runs.
# Never edited after its RED. Reads leak.json. Exit 1 if any row fails.
#
# Probe: a pairwise count table. For each token t and each of the bed's 16 roots r, a discrete feature tuple
#   (rank of r among roots, position bucket of t (t // 256, cap 15), gap bucket of t (t - pos(parent), cap 63, // 8),
#    low-4-bits(id t) == low-4-bits(id r), low-4-bits(pid t) == low-4-bits(id r), low-4-bits(id r))
# never follows the chain past the parent's position. P(r is t's root | tuple) is counted on 40 fitting beds and the
# probe predicts argmax_r on 8 disjoint evaluation beds (n = 4096). Seeds: fit default_rng([71, k]), eval [72, k],
# planted fit [73, k], planted eval [74, k].
#
# foreman.k1.leak_probe_detects (positive control, so the next row can fail): on PLANTED beds -- bed_k structure but
#     each token's id drawn with id % 16 == its lane -- the probe scores >= 0.9 on tokens deeper than 16.
# foreman.k1.no_leak : on real bed_k beds the probe scores <= 1/16 + 3*SE on tokens deeper than 16 and deeper than
#     128, SE = sqrt((1/16)(15/16)/N) with N the evaluated token count; and ids are distinct inside every bed.
import json, math, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "leak.json")).read_text())
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


p = R["planted"]
check("foreman.k1.leak_probe_detects planted depth>16 >= 0.9", p["acc_deep16"] >= 0.9, p)
r = R["real"]
for key, nkey in (("acc_deep16", "n_deep16"), ("acc_deep128", "n_deep128")):
    bar = 1 / 16 + 3 * math.sqrt((1 / 16) * (15 / 16) / r[nkey])
    check(f"foreman.k1.no_leak real {key} <= {bar:.4f}", r[key] <= bar, f"{r[key]:.4f} (N={r[nkey]})")
check("foreman.k1.no_leak ids distinct in every bed", R["ids_distinct"] is True, R["ids_distinct"])
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
