# Foreman K1: 2-dim slots at n = 16384, period 4096. Written 2026-09-23 04:52 IST (clock read 04:52:17), after
# kp_w62a.json's first rows showed the period-2048 premise failing at L = 7 (783,878 addressed positions with
# c + 2048 <= t; mismatch rate 0.105) and BEFORE kp_trap2.py exists or runs. test_kp_trap.py's w62t row (period 2048)
# stays as registered. Never edited after its RED. Reads kp_trap2.json. Exit 1 on any fail.
#
# Same as test_kp_trap.py's w62t row (sc-HPD W62, 63 heads x 2 dims, flat-top multiplexer, twin ALiBi slope
# get_slopes(64)[39] = 2^-5, float32, full chains, Cameron's [22, 16384, k] k = 0, 1 with calibration [23, 16384, k])
# except: rotary period 4096 (1 - cos(2 pi / 4096) = 1.18e-6, ~20 fp32 spacings below 1) and beta = 3e7.
# foreman.k1.w62p4096_real_16384 : at L = 4 and L = 7, 0 pointer mismatches against idealized sc-HPD W62, and the far
#     band of c(W62) (depth > 136 at L = 4, > 1280 at L = 7) scores <= 1/8 + 0.03 with >= 500 band tokens.
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "kp_trap2.json")).read_text())
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


for L in ("4", "7"):
    c = R[L]
    check(f"foreman.k1.w62p4096_real_16384 L={L}", c["mismatches"] == 0 and c["band_tokens"] >= 500 and c["band_acc"] <= 0.155, c)
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
