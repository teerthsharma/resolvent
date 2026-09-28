"""bars_r3 runs, one process, one sandbox job at a time, draw-major so each draw's V3 pool trains once.
    python sun/daedalus/results/r3/r3_runs.py
Writes r3_runs.json after every row (a timebox cut keeps what finished). Run seeds are never written."""
import json
import os
import sys
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(D, "engine"))
import verifier as V  # noqa: E402

PLAN = [("m0", "cheats/r06_pool_twin_null_sabotage"), ("m0", "candidates_rjepa/fm_djepa_eps4"),
        ("m0", "candidates_rjepa/ch_dj_stoch_noLN")]
for t in ("r3d1", "r3d2"):
    PLAN += [(t, "candidates_rjepa/" + a) for a in ("fm_resolvent", "fm_hop1", "fm_djepa_eps4", "ch_dj_stoch_noLN")]
PLAN += [("m0", "candidates_rjepa/fm_resolvent"), ("m0", "candidates_rjepa/fm_hop1")]  # V3 read on the r2 draw, last
KEEP = ("verdict", "stage", "law", "reason", "hit_planner", "hit_shuffled", "bayes_hit", "floors", "init_hit",
        "equivariance_dev", "bound_dev", "v3", "wall_total_s")
out_path = os.path.join(HERE, "r3_runs.json")
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
    r = V.verify(os.path.join(D, rel), tag=tag)
    row = {"tag": tag, "dir": rel, **{k: r[k] for k in KEEP if k in r}}
    rows.append(row)
    json.dump(rows, open(out_path, "w"), indent=1, default=float)
    print(tag, rel.split("/")[-1], r["verdict"], r["stage"], f"hs={r.get('hit_shuffled')}", r["reason"][:110],
          json.dumps(r.get("v3"), default=float), f"{time.time() - t0:.0f}s", flush=True)
