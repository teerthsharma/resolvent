# K0_wald_rebuild. Bars written 2026-09-23 01:46, before any rebuild number existed.
# (a) independent dense rebuild (logit matrix -> row softmax -> scipy triangular solve), NOT the author's
#     recurrence, reproduces the pinned wald.out numbers on the pinned bed (default_rng(23)) to 4 dp;
# (b) sparsemax arm = exact 1.0; (c) seed band: over seeds 0..199 of the same generator, each pinned
#     seed-dependent value lies inside the central 95% band (the pinned bed is not a lucky draw).
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
from bed_k import pinned_bed, floor_recency, ceiling_exact, resolvent_dense, resolvent_rec, make_bed

fails = []
def check(name, ok, got):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got)); (None if ok else fails.append(name))

parent, depth, root = pinned_bed()
check("max depth 269, median 127", depth.max() == 269 and int(np.median(depth)) == 127, (depth.max(), np.median(depth)))
check("floor 0.0645", round(floor_recency(parent, root)[0], 4) == 0.0645, floor_recency(parent, root)[0])
for L, v in ((4, .0664), (6, .2539), (8, .9856)):
    c = ceiling_exact(depth, L); check(f"ceiling L={L} {v}", round(c, 4) == v, c)
out = {}
for s, v in ((8, .0869), (12, .9954), (16, 1.0), (20, 1.0)):
    a = resolvent_dense(parent, root, s, 0.999); out[s] = a
    check(f"dense resolvent s={s} -> {v}", round(a, 4) == v, a)
a = resolvent_dense(parent, root, None, 0.999, sparse=True); check("sparsemax exact 1.0", a == 1.0, a)

band = {"floor": [], "L8": [], "s8": [], "s12": []}
for sd in range(200):
    p, d, r = make_bed(4096, 16, np.random.default_rng(sd))
    band["floor"].append(floor_recency(p, r)[0]); band["L8"].append(ceiling_exact(d, 8))
    acc = resolvent_rec(p, r, np.array([8.0, 12.0]), 0.999); band["s8"].append(acc[0]); band["s12"].append(acc[1])
pinned = {"floor": .0645, "L8": .9856, "s8": .0869, "s12": .9954}
q = {k: [float(np.quantile(v, .025)), float(np.median(v)), float(np.quantile(v, .975))] for k, v in band.items()}
for k in pinned:
    check(f"seed band {k}: pinned {pinned[k]} in [q2.5, q97.5]", q[k][0] <= pinned[k] <= q[k][2], q[k])
json.dump({"dense": out, "band_q025_med_q975": q}, open("wald_rebuild.json", "w"), indent=1)
sys.exit(1 if fails else 0)
