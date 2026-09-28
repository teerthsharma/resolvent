"""RED for bars_r4 r4_a: the r3 verifier (engine unchanged) on plant r07 (v2, elementwise distances) at draw m0. Expected RED: PASS_V3."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "engine"))
import verifier as V
r = V.verify(os.path.join(os.path.dirname(os.path.dirname(HERE)), "cheats", "r07_label_free_null_sabotage"))
print(r["verdict"], r["stage"], r["reason"], json.dumps({k: r.get(k) for k in ("hit_shuffled", "bayes_hit", "floors", "v3", "init_hit")}, default=float), flush=True)
json.dump(r, open(os.path.join(HERE, "red_r07.json"), "w"), indent=2, default=float)
