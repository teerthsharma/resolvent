# Foreman K1: the flat-top multiplexer. Written 2026-09-23 04:42 IST (clock read 04:41:18), after test_kp_w62.py's
# w30_real_long went RED at n = 8192 (diagnosed: the triangle bump turns a 1e-3 leak in x2 into a 64-position error),
# and BEFORE kp_trap.py exists or runs. Never edited after its RED. Reads kp_trap.json. Exit 1 on any fail.
#
# Multiplexer per slot: trapezoid bump = 2 [relu(d + 1) - relu(d + 0.5) - relu(d - 0.5) + relu(d - 1)] (1 on |d| <= 0.5,
# 0 on |d| >= 1; 4 ReLU units) and a gate unit; fallback unit. 5W + 1 units: 151 (W30), 311 (W62) <= 512. Everything
# else as test_kp_w62.py / test_kp_w62a.py; float32; full chains (every token, every layer); Cameron's beds
# [22, n, k] with calibration [23, n, k].
# foreman.k1.w30t_real_8192 : W30, 31 heads x 4 dims, periods (sqrt(n), n), beta = 3n, n = 8192, 4 beds, L in {4, 7}:
#     0 pointer mismatches against idealized sc-HPD W30.
# foreman.k1.w62t_real_16384 : W62, 63 heads x 2 dims, one rotary pair of period 2048, logit = 2e7 (q . k) - m (t - j),
#     m = get_slopes(64)[39] = 2^-5, n = 16384, 2 beds, L in {4, 7}: 0 pointer mismatches against idealized sc-HPD W62,
#     and on those realized pointers the far band of c(W62) (depth > 8.5 * 2^4 = 136 at L = 4, > 10 * 2^7 = 1280 at
#     L = 7) scores <= 1/8 + 0.03 with >= 500 band tokens.
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "kp_trap.json")).read_text())
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


for L in ("4", "7"):
    c = R["w30_8192"][L]
    check(f"foreman.k1.w30t_real_8192 L={L} mismatches == 0", c["mismatches"] == 0, c)
    c = R["w62_16384"][L]
    check(f"foreman.k1.w62t_real_16384 L={L} mismatches == 0 and far band <= 0.155 (>= 500 tokens)",
          c["mismatches"] == 0 and c["band_tokens"] >= 500 and c["band_acc"] <= 0.155, c)
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
