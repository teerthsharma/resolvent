# Foreman K1: the next bed property after bed_k' is void. Written 2026-09-23 04:06 IST (clock read 04:06:25), after
# kp_w14 GREEN and BEFORE kp_next.py exists or runs. Never edited after its RED. Reads kp_next.json. Exit 1 on any fail.
#
# (1) The densest window I can build at d = 128. Same build as test_kp_w14.py but 31 heads x d_head 4 = 124 <= 128
#     (1 content + 30 window heads; rotary periods 64 and 4096, beta 6000; 121 ReLU units <= 512), float32, on Cameron's
#     beds [22, 4096, k] / calibration [23, 4096, k], L = 7.
# foreman.k1.w30_heads_exact_f32 : 0 pointer mismatches against the idealized sc-HPD W30, and the realized mean equals
#     the idealized one (1e-12).
# (2) The property at that width, on bed_k''s own test generator (bed_kp.make_test, 8 lanes), long beds n = 32768:
#     4 beds default_rng([57, k]), calibration default_rng([58, k]). Reach c = the largest depth d (multiple of
#     w = max(1, 2^L/8)) such that every band (d' - w, d'], d' <= d, holding >= 50 pooled tokens scores >= 0.5 credited;
#     c = d / 2^L, max over sc-HPD k3 at W in {10, 14, 30} and local k3 at W = 30.
# foreman.k1.next_reach_bounded : c in [1.5, 6.0] at L = 4 and at L = 7 (a constant factor, not a new exponent).
# foreman.k1.next_far_band : with kappa = 2c, every construction above plus credited doubling scores <= 1/8 + 0.03 on
#     tokens with depth > kappa * 2^L, and there are >= 500 such tokens, at L = 4 and L = 7.
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "kp_next.json")).read_text())
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


w = R["w30"]
check("foreman.k1.w30_heads_exact_f32 mismatches == 0 and mean == idealized",
      w["mismatches"] == 0 and abs(w["mean"] - w["idealized_mean"]) < 1e-12, w)
for L in ("4", "7"):
    c = R["reach"][L]
    check(f"foreman.k1.next_reach_bounded L={L} c in [1.5, 6.0]", 1.5 <= c["c"] <= 6.0, c)
    f = R["far"][L]
    check(f"foreman.k1.next_far_band L={L} worst <= 0.155 with >= 500 tokens", f["worst"] <= 0.155 and f["tokens"] >= 500, f)
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
