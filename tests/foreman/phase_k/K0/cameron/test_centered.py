# K0_centered_resolvent (optimist route: refuse "s must grow with ln n, or sparsemax"). Bars written before
# centered.py existed. Centered head: W' = W - m_i on every causal entry, m_i = (row sum - row max)/(count - 1),
# i.e. the softmax minus its own off-peak background; z = (I - gamma W')^-1 V, gamma 0.999, argmax z_i.
# Beds make_bed(n,16,default_rng([3,n])), n = 1024, 4096; logit noise N(0, sigma^2) iid on causal entries
# (default_rng([4,n])), same draw for both heads.
#  (a) sigma 0: centered scores 1.0 at s = 8 and 12 at both n; plain softmax at s = 8, n = 4096 scores < 0.2.
#  (b) sigma 1: s_50(centered) <= s_50(softmax) - 1.0 at both n.
#  (c) sigma 1: d s_50 / d ln n from 1k to 4k is smaller for centered than for softmax.
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
from centered import sweep

fails = []
def check(name, ok, got):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got)); (None if ok else fails.append(name))

grid = np.arange(2.0, 20.01, 0.5)
R = {n: sweep(n, grid) for n in (1024, 4096)}      # R[n][(sigma, head)] = accuracy array over grid
s50 = {}
for n in R:
    for key, a in R[n].items():
        s50[(n,) + key] = float(np.interp(0.5, np.maximum.accumulate(a), grid))
        print(n, key, "acc@8", round(float(a[grid == 8][0]), 4), "acc@12", round(float(a[grid == 12][0]), 4), "s50", round(s50[(n,) + key], 2))
for n in R:
    a = R[n][(0.0, "centered")]
    check(f"(a) n={n} centered sigma0 = 1.0 at s=8,12", a[grid == 8][0] == 1.0 and a[grid == 12][0] == 1.0, (a[grid == 8][0], a[grid == 12][0]))
check("(a) softmax sigma0 s=8 n=4096 < 0.2", R[4096][(0.0, "softmax")][grid == 8][0] < 0.2, R[4096][(0.0, "softmax")][grid == 8][0])
for n in R:
    check(f"(b) n={n} sigma1 s50 centered <= softmax - 1", s50[(n, 1.0, "centered")] <= s50[(n, 1.0, "softmax")] - 1.0,
          (s50[(n, 1.0, "centered")], s50[(n, 1.0, "softmax")]))
sl = {h: (s50[(4096, 1.0, h)] - s50[(1024, 1.0, h)]) / np.log(4) for h in ("softmax", "centered")}
check("(c) sigma1 slope centered < softmax", sl["centered"] < sl["softmax"], sl)
json.dump({"s50": {str(k): v for k, v in s50.items()}, "slopes": sl,
           "acc": {str(n): {str(k): v.tolist() for k, v in R[n].items()} for n in R}, "grid": grid.tolist()},
          open("centered.json", "w"), indent=1)
sys.exit(1 if fails else 0)
