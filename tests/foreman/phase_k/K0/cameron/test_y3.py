# K0_y3_rebuild. Bars written 2026-09-23 01:48, before any Y3 number of mine existed.
# (a) my vectorised recurrence (bed_k.resolvent_rec, all s at once) on yukawa.py's beds (bed(n,16,seed=n))
#     reproduces yukawa.out: s_50 8.92/10.73/13.00, widths 1.81/2.61/3.82, to 2 dp, with yukawa's own
#     definitions (np.interp; s_10 at 0.1+0.9/16, s_90 at 0.9; law ln(n * median depth>0)).
# (b) np.interp validity: acc(s) on those beds never drops by more than 0.005 between grid points.
# (c) 12 fresh seeds per n (default_rng(10000+k)): the law stays struck iff the seed-median offset
#     falls by more than 0.5 from n=1k to n=16k.
# (d) broadening survives seeds: seed-median width(16k) - width(1k) > 1.0.
# (e) the surviving slope: both seed-median slopes d s_50 / d ln n in [1.1, 1.8].
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
from bed_k import make_bed, resolvent_rec, y3_read

grid = np.arange(4.0, 24.01, 0.25)
fails = []
def check(name, ok, got):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got)); (None if ok else fails.append(name))

pinned = {1024: (8.92, 1.81), 4096: (10.73, 2.61), 16384: (13.00, 3.82)}
out = {"pinned_beds": {}, "seeds": {}}
for n, (s50p, wp) in pinned.items():
    p, d, r = make_bed(n, 16, np.random.default_rng(n))
    a = resolvent_rec(p, r, grid, 0.999)
    s50, w, law = y3_read(a, grid, n, d)
    drop = float(np.max(-np.diff(a)))
    out["pinned_beds"][n] = {"s50": s50, "width": w, "offset": s50 - law, "max_drop": drop}
    check(f"n={n} s_50 {s50p} width {wp}", round(s50, 2) == s50p and round(w, 2) == wp, (round(s50, 2), round(w, 2)))
    check(f"n={n} acc(s) monotone within 0.005", drop <= 0.005, drop)
for n in pinned:
    rows = []
    for k in range(12):
        p, d, r = make_bed(n, 16, np.random.default_rng(10000 + k))
        s50, w, law = y3_read(resolvent_rec(p, r, grid, 0.999), grid, n, d)
        rows.append((s50, w, s50 - law))
    out["seeds"][n] = np.array(rows).tolist()
med = {n: np.median(np.array(out["seeds"][n]), 0) for n in pinned}
for n in pinned: print(f"n={n}: seed-median s_50 {med[n][0]:.2f} width {med[n][1]:.2f} offset {med[n][2]:+.2f}; "
                       f"range s_50 [{min(x[0] for x in out['seeds'][n]):.2f},{max(x[0] for x in out['seeds'][n]):.2f}]")
check("law stays struck: offset falls > 0.5 from 1k to 16k", med[16384][2] - med[1024][2] < -0.5, med[16384][2] - med[1024][2])
check("broadening: width(16k) - width(1k) > 1.0", med[16384][1] - med[1024][1] > 1.0, med[16384][1] - med[1024][1])
sl = [(med[4096][0] - med[1024][0]) / np.log(4), (med[16384][0] - med[4096][0]) / np.log(4)]
check("slopes per ln n in [1.1, 1.8]", all(1.1 <= x <= 1.8 for x in sl), sl)
out["median"] = {n: med[n].tolist() for n in pinned}; out["slopes"] = sl
json.dump(out, open("y3_rebuild.json", "w"), indent=1)
sys.exit(1 if fails else 0)
