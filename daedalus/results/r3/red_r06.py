"""RED for bars_r3 r3_d: the r2 verifier (no ranker V3) on plant r06 at draw m0. Expected RED: PASS_V2."""
import sys, json, os
sys.path.insert(0, r"C:\Users\seal\Desktop\New folder (32)\sun\daedalus\engine")
import verifier as V
r = V.verify(r"C:\Users\seal\Desktop\New folder (32)\sun\daedalus\cheats\r06_pool_twin_null_sabotage")
print(r["verdict"], r["stage"], r["reason"], json.dumps({k: r.get(k) for k in ("hit_shuffled", "floors")}, default=float))
json.dump(r, open(os.path.join(os.path.dirname(__file__), "red_r06.json"), "w"), indent=2, default=float)
