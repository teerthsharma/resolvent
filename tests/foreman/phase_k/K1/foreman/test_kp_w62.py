# Foreman K1 (Amendment K1-a follow-ups 1 and 2). Written 2026-09-23 04:14 IST (clock read 04:13:58), BEFORE
# kp_w62.py exists or runs. Never edited after its RED. Reads kp_w62.json. Exit 1 on any fail.
#
# Construction: kp_attack.py's self-consistent HPD (sc-HPD) window, k3, realized as real causal softmax heads as in
# test_kp_w14.py / test_kp_next.py (1 content head + W window heads; keys = the oracle position as rotary pairs; value =
# one scalar pointer; ReLU bump multiplexer, 4W + 1 units <= 512; float32 everywhere). Beds: bed_kp.make_test,
# Cameron's K1.F' seeds default_rng([22, n, k]) and calibration default_rng([23, n, k]).
#
# (1) 2-dim slots: 63 heads x d_head 2 = 126 <= 128, W = 62, one rotary pair of period P = n, logit scale beta.
# foreman.k1.w62_real_4096 : n = 4096, P = 4096, beta = 3e7, 8 beds, L in {4, 7}, full chains: 0 pointer mismatches
#     against the idealized sc-HPD W62.
# foreman.k1.w62_real_8192 : n = 8192, P = 8192, beta = 1e8, 4 beds, L = 7, layer-wise check (the realized layer applied
#     to the idealized input pointers, on 2048 random query rows per layer, rows default_rng([61, k, layer])):
#     mismatch rate <= 1e-3.
# foreman.k1.w62_unreal_16384 : the same at n = 16384, P = 16384, beta = 4e8: mismatch rate >= 0.01 (one period cannot
#     separate adjacent positions in fp32 there: 1 - cos(2 pi / 16384) = 7.4e-8 ~ the fp32 spacing below 1, 6e-8).
# foreman.k1.w62_reach : idealized sc-HPD W62 reach factor c (test_kp_next.py's definition: first band of w = 2^L/8
#     depths holding >= 50 tokens below 0.5 credited) on bed_kp test beds n = 32768 [57, k] / calibration [58, k],
#     k = 0..3: c(W62) >= c(W30) (3.0 at L=4, 3.5 at L=7) and c(W62) <= 6.0, at L = 4 and L = 7.
# foreman.k1.w62_far_band : with kappa = 2 * c(W62), sc-HPD W10/W14/W30/W62, local W30 and credited doubling all score
#     <= 1/8 + 0.03 on depth > kappa * 2^L on those beds, with >= 500 such tokens, at L = 4 and L = 7.
# (2) The 30-slot build at n = 8192 and 16384: 31 heads x d_head 4 = 124, periods (sqrt(n), n), beta = 3n, full chains.
# foreman.k1.w30_real_long : 0 pointer mismatches against idealized sc-HPD W30 at n = 8192 (4 beds) and n = 16384
#     (2 beds), L in {4, 7}.
# foreman.k1.w30_far_band_long : on those beds, realized W30 credited accuracy on the far band of Amendment K1-a
#     (depth > 6 * 2^4 at L = 4, depth > 7 * 2^7 at L = 7) <= 1/8 + 0.03, where the band holds >= 500 tokens.
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "kp_w62.json")).read_text())
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


for L in ("4", "7"):
    check(f"foreman.k1.w62_real_4096 L={L} mismatches == 0", R["w62_4096"][L]["mismatches"] == 0, R["w62_4096"][L])
check("foreman.k1.w62_real_8192 mismatch rate <= 1e-3", R["w62_8192"]["rate"] <= 1e-3, R["w62_8192"])
check("foreman.k1.w62_unreal_16384 mismatch rate >= 0.01", R["w62_16384"]["rate"] >= 0.01, R["w62_16384"])
base = {"4": 3.0, "7": 3.5}
for L in ("4", "7"):
    c = R["reach"][L]
    check(f"foreman.k1.w62_reach L={L} c(W30) <= c(W62) <= 6", base[L] <= c["c_w62"] <= 6.0, c)
    f = R["far"][L]
    check(f"foreman.k1.w62_far_band L={L} worst <= 0.155, tokens >= 500", f["worst"] <= 0.155 and f["tokens"] >= 500, f)
for n in ("8192", "16384"):
    for L in ("4", "7"):
        c = R["w30_long"][n][L]
        check(f"foreman.k1.w30_real_long n={n} L={L} mismatches == 0", c["mismatches"] == 0, c["mismatches"])
        if c["band_tokens"] >= 500:
            check(f"foreman.k1.w30_far_band_long n={n} L={L} <= 0.155", c["band_acc"] <= 0.155, c)
        else:
            print(f"SKIP foreman.k1.w30_far_band_long n={n} L={L} band holds {c['band_tokens']} < 500 tokens")
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
