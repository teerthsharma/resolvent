"""bars_r4 r4_c runs on planning_consequence_v1: draw-major, one sandbox job at a time, each draw's V3 pool trains once.
    python sun/daedalus/results/r4/r4_bed.py
Writes r4_bed.json after every row (a timebox cut keeps what finished). Run seeds are never written."""
import json
import os
import sys
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(D, "engine"))
import verifier as V  # noqa: E402
import hashlib  # noqa: E402

DRIVER_SHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()  # bars_r4 r4_d

BED = "planning_consequence_v1"
PLAN = [(t, rel) for t in V.R4["r4_c_bed"]["draws"] for rel in ["candidates_rjepa/" + a for a in V.R4["r4_c_bed"]["arms"]] + ["POOL"]]
KEEP = ("verdict", "stage", "law", "reason", "hit_planner", "hit_shuffled", "bayes_hit", "floors", "init_hit",
        "equivariance_dev", "v3", "wall_total_s", "provenance")
out_path = os.path.join(HERE, "r4_bed.json")
rows = json.load(open(out_path)) if os.path.exists(out_path) else []
done = {(r["tag"], r["dir"]) for r in rows}
t0 = time.time()
for tag, rel in PLAN:
    if (tag, rel) in done:
        continue
    free = psutil.virtual_memory().available / 2**30
    if free < 1.5:
        print(f"STOP: {free:.2f} GB free < 1.5 before {tag} {rel}", flush=True)
        break
    if rel == "POOL":  # bed validity (r4_c): the pool's hits on this draw, trained here if no arm reached V3
        d = V._rdata(BED, tuple(V.seeds_for(tag)), tag)
        r = {"verdict": "POOL", "stage": "", "reason": "", "v3": {"pool": V.ranker_pool_hits(BED, tag)},
             "bayes_hit": V._hit(d["bayes"], d["evY"]), "provenance": dict(V.PROVENANCE)}
        r["v3"]["headroom"] = r["bayes_hit"] - max(r["v3"]["pool"].values())
    else:
        r = V.verify(os.path.join(D, rel), tag=tag, bed=BED)
    rows.append({"tag": tag, "dir": rel, "driver_sha256": DRIVER_SHA, **{k: r[k] for k in KEEP if k in r}})
    json.dump(rows, open(out_path, "w"), indent=1, default=float)
    print(tag, rel.split("/")[-1], r["verdict"], r["stage"], f"hs={r.get('hit_shuffled')}", f"bayes={r.get('bayes_hit')}",
          r["reason"][:100], json.dumps(r.get("v3"), default=float), f"{time.time() - t0:.0f}s", flush=True)
