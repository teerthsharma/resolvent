# Foreman K1: is the W = 14 window that crosses 0.5 on bed_k' realizable at R0 width (d = 128) with real softmax heads?
# Written 2026-09-23 04:03 IST (clock read 04:02:35), per the Dispatcher's PASS2 ruling, BEFORE kp_w14.py exists or
# runs. Never edited after its RED. Reads kp_w14.json. Exit 1 on any fail.
#
# Build (as K0's softmax_heads_exact, same scope): per layer 15 causal softmax heads of d_head = 8 (15 x 8 = 120 <= 128):
# 1 content head that fetches v1's pointer, and 14 window heads that fetch the pointers of positions lo(t) .. lo(t)+13.
# Keys = the token's own position code under the oracle K1.F' grants, as 4 rotary pairs (periods 8, 64, 512, 4096:
# the adjacent-position logit gap is beta * (1 - cos(pi/4)) with beta = 100); values = the addressed token's pointer
# as one scalar position. Multiplexer = a real ReLU MLP: per slot a triangle bump on x2 - lo - j (3 units) and a gate
# relu(s_j - B(1 - bump)) (1 unit), plus a fallback unit: 57 units <= 4 x 128. Idealized, as in K0 and stated: the
# per-token arithmetic that turns a scalar position into its rotary query code and computes lo(t) from the
# self-consistent HPD table (kp_attack.calibrate on Cameron's calibration beds [23, 4096, k]).
# Beds: Cameron's K1.F' test beds default_rng([22, 4096, k]), k = 0..7, L = 7.
# foreman.k1.kp_w14_heads_exact_f64 : in float64 the real heads + ReLU multiplexer reproduce the idealized sc-HPD W14
#     pointers exactly (0 mismatches over 8 beds x 7 layers), and the credited beyond-2^7 mean equals the idealized one
#     (1e-12) and is > 0.5.
# foreman.k1.kp_w14_heads_exact_f32 : the same with every head and MLP op in float32: 0 mismatches and mean > 0.5.
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "kp_w14.json")).read_text())
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


for dt in ("f64", "f32"):
    r = R[dt]
    check(f"foreman.k1.kp_w14_heads_exact_{dt} mismatches == 0", r["mismatches"] == 0, r["mismatches"])
    check(f"foreman.k1.kp_w14_heads_exact_{dt} mean > 0.5 and == idealized",
          r["mean"] > 0.5 and abs(r["mean"] - R["idealized_mean"]) < 1e-12, (r["mean"], R["idealized_mean"]))
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
