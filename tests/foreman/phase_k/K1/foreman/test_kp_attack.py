# Foreman K1 attack on Cameron's replacement bed bed_k' (K1/cameron/bed_kp.py, sha256 20d50b9d71248cd3..., imported
# read-only). Written 2026-09-23 03:38 IST (clock read 03:37:40), BEFORE kp_attack.py exists or runs. Never edited after
# its RED. Reads kp_attack.json. Exit 1 on any fail.
#
# Beds: Cameron's K1.F' beds exactly -- test bed_kp.make_test(default_rng([22, n, k]), n), calibration [23, n, k];
# train bed_kp.make_train(default_rng([21, k])), calibration [24, k]; k = 0..7. R0 pricing as K1.F': k3 attention 12W
# <= 128 (W <= 10); a two-level multiplexer costs 12 * W1 * W2 <= 512 MLP units. Credit as K1.F'.
# New constructions (the oracle his line grants):
#   sc-HPD  : self-consistent HPD. The window for v2 is the HPD of the offset pos(v1) - pos(A(v1)) measured on the
#             calibration beds for THIS construction at THIS layer, conditioned on the token's hop count J(t) -- it
#             prices the heterogeneous jumps (triples) that a pure J-hop law ignores. k3, W <= 10.
#   split-4 : two windows, W1 for v2 (sc-HPD at J(t)) and W2 for v3 = A(v2) (sc-HPD of pos(v1) - pos(A(A(v1)))),
#             W1 + W2 <= 10, W1 * W2 <= 42: up to four hops per layer.
# foreman.k1.kp_attack_line_beaten : the best new construction exceeds K1.F''s own line at (n=4096, L=7): > 0.4085 + 0.01.
# foreman.k1.kp_attack_void : the best new construction scores > 0.5 beyond 2^L (mean of 8 beds) at some scored cell
#     of K1.F' -- (1024 train, L=4) or (n in {4096, 8192, 16384}) x (L in {4, 7}) -- i.e. bed_k' is void.
# foreman.k1.kp_no_leak : my pairwise count probe (leak.py features) on bed_k' test beds (n = 4096, fit on 40 beds
#     [75, k], scored on 8 beds [76, k]) scores <= 1/8 + 3 SE on depth > 16; its planted control scores >= 0.9.
import json, math, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "kp_attack.json")).read_text())
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


c = R["cells"]["4096_7"]
check("foreman.k1.kp_attack_line_beaten n=4096 L=7 > 0.4185", c["best"] > 0.4085 + 0.01, c)
worst = max(R["cells"].items(), key=lambda kv: kv[1]["best"])
check("foreman.k1.kp_attack_void some cell > 0.5", worst[1]["best"] > 0.5, worst)
lk = R["leak"]
bar = 1 / 8 + 3 * math.sqrt((1 / 8) * (7 / 8) / lk["n_deep16"])
check(f"foreman.k1.kp_no_leak real depth>16 <= {bar:.4f} and planted >= 0.9", lk["acc_deep16"] <= bar and lk["planted"] >= 0.9, lk)
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
