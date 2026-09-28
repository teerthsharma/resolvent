"""Arms x draws table from r3_runs.json (+ r2 m0 rows where r3 did not rerun them). Replication rule bars_r3 r3_c."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "r3_runs.json")))
r2 = json.load(open(os.path.join(os.path.dirname(HERE), "r2", "rjepa_arms.json")))
cell = {(r["dir"].split("/")[-1], r["tag"]): r for r in rows}
for a in ("fm_resolvent", "fm_hop1"):
    if (a, "m0") not in cell:
        cell[a, "m0"] = dict(r2[a], tag="m0", note="r2 run (no V3)")
out = {}
for a in ("fm_resolvent", "fm_hop1", "fm_djepa_eps4", "ch_dj_stoch_noLN", "r06_pool_twin_null_sabotage"):
    line = []
    for t in ("m0", "r3d1", "r3d2"):
        r = cell.get((a, t))
        if r is None:
            line.append(f"{t}: not run"); continue
        fl = r.get("floors", {})
        v3 = r.get("v3", {})
        line.append(f"{t}: {r['verdict']}@{r['stage']} hs {r.get('hit_shuffled', float('nan')):.4f} "
                    f"floor {max(fl.values()) if fl else float('nan'):.4f} pool {v3.get('pool', {}).get(v3.get('best'), float('nan')):.4f} "
                    f"bayes {r.get('bayes_hit', float('nan')):.4f}")
    v2 = sum(1 for t in ("m0", "r3d1", "r3d2") if cell.get((a, t), {}).get("verdict") in ("PASS_V2", "PASS_V3")
             or cell.get((a, t), {}).get("stage") == "V3")
    v3n = sum(1 for t in ("m0", "r3d1", "r3d2") if cell.get((a, t), {}).get("verdict") == "PASS_V3")
    out[a] = {"cells": line, "passV2_draws": v2, "passV3_draws": v3n,
              "replicated_V2": v2 >= 3, "replicated_V3": v3n >= 3}
    print(a, f"V2 {v2}/3 V3 {v3n}/3", *line, sep="\n  ")
json.dump(out, open(os.path.join(HERE, "r3_table.json"), "w"), indent=1)
