import sys, json
sys.path.insert(0, r"C:\Users\seal\Desktop\New folder (32)\sun\daedalus\engine")
import verifier as V
d = r"C:\Users\seal\Desktop\New folder (32)\sun\daedalus\cheats\c21_seed_pick_nonpool"
r = V.verify(d, pool=["alibi_twin"])
print(json.dumps({k: r.get(k) for k in ("verdict", "stage", "reason", "band_acc", "v3", "v5", "wall_total_s")}, default=float))
