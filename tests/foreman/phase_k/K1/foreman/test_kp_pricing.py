# Foreman K1: is the R0 slot pricing load-bearing for bed_k'? Written 2026-09-23 03:40 IST (clock read 03:40:08), after
# kp_attack.json (sc-HPD k3 at (4096, L7) rises ~0.025 per slot up to W = 10: 0.4147 at W9, 0.4398 at W10) and BEFORE
# kp_pricing.py exists or runs. Never edited after its RED. Reads kp_pricing.json. Exit 1 on any fail.
# Same construction (kp_attack.py sc-HPD, k3), Cameron's beds [22, 4096, k] / calibration [23, 4096, k], L = 7.
# foreman.k1.kp_pricing_crossing : the smallest W in 11..20 at which sc-HPD scores > 0.5 beyond 2^7 lies in [11, 14]
#     (i.e. bed_k' at (4096, L7) survives by 1-4 slots of the 12-dims-per-slot convention: at 10 dims per slot the R0
#     budget would be W = 12).
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "kp_pricing.json")).read_text())
ok = R["crossing_W"] is not None and 11 <= R["crossing_W"] <= 14
print(("PASS " if ok else "FAIL ") + "foreman.k1.kp_pricing_crossing smallest W > 0.5 in [11, 14] : " + str(R))
print(("GREEN " if ok else "RED ") + f"{0 if ok else 1} failing rows")
sys.exit(0 if ok else 1)
