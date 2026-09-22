# Route check for K0_logit_law_unidentified (measured, no bar): does multi-length training at depth <= 32 pin
# the scale the 16k test needs? s_99 (worst of 8 train beds) at n = 256/512/1024, fit s = a ln n + b, extrapolate.
import sys, json, numpy as np
sys.path.insert(0, '.')
from bed_k import make_train, resolvent_rec
from k1_floors import s_needed, test_beds
s99 = {n: max(s_needed(make_train(np.random.default_rng([5, n, k]), n=n), 0.99) for k in range(8)) for n in (256, 512, 1024)}
a, b = np.polyfit(np.log(list(s99)), list(s99.values()), 1)
res = {"s99_train": s99, "a_fit": a, "b_fit": b}
for n in (4096, 16384):
    s = a * np.log(n) + b
    res[f"acc_{n}_at_extrapolated_s_{s:.2f}"] = float(np.median([resolvent_rec(t["parent"], t["root"], np.array([s]), 0.999)[0] for t in test_beds(n)]))
print(res); json.dump(res, open("route_multilen.json", "w"), indent=1)
