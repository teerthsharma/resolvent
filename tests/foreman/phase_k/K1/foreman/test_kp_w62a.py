# Foreman K1: 2-dim slots at the far-band lengths, using the twin's OWN ALiBi to break rotary aliasing. Written
# 2026-09-23 04:20 IST (clock read 04:19:23), while kp_w62.py runs and BEFORE kp_w62a.py exists or runs. Never edited
# after its RED. Reads kp_w62a.json. Exit 1 on any fail.
#
# Why: test_kp_w62.py predicts that ONE rotary pair of period P = n cannot separate adjacent positions in fp32 at
# n = 16384. A shorter period P = 2048 separates them easily (1 - cos(2 pi / 2048) = 4.7e-6, ~79 fp32 spacings) but
# aliases at c +- 2048. The aliases above c are invisible when c + 2048 > t (causal), and the aliases below c lose to c
# by m * 2048 logits under an ALiBi bias -m (t - j). m = get_slopes(64)[39] = 2^-5 (a slope the 64-head twin HAS):
# 64 logits. Logit = beta (q . k) - m (t - j), beta = 2e7. Same sc-HPD W62 construction and multiplexer as
# test_kp_w62.py, float32, 63 heads x d_head 2 = 126 <= 128. Beds: Cameron's [22, n, k] / calibration [23, n, k].
# Layer-wise check: the realized layer on the idealized input pointers, 2048 query rows per layer, rows
# default_rng([62, n, k, layer]).
# foreman.k1.w62a_real_16384 : n = 16384, 4 beds, L in {4, 7}: mismatch rate <= 1e-3.
# foreman.k1.w62a_real_8192  : n = 8192, 4 beds, L in {4, 7}: mismatch rate <= 1e-3.
# foreman.k1.w62a_no_alias   : on every checked row, every addressed position c satisfies c + 2048 > t (so the
#     construction's premise holds on the bed rather than by luck): violations == 0.
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "kp_w62a.json")).read_text())
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


for n in ("16384", "8192"):
    for L in ("4", "7"):
        c = R[n][L]
        check(f"foreman.k1.w62a_real_{n} L={L} rate <= 1e-3", c["rate"] <= 1e-3, c)
        check(f"foreman.k1.w62a_no_alias n={n} L={L} violations == 0", c["alias_violations"] == 0, c["alias_violations"])
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
