import sys, json
sys.path.insert(0, r"C:\Users\seal\Desktop\New folder (32)\sun\daedalus\engine")
import verifier as V
for d, pool in (("c21_seed_pick_nonpool", ["alibi_twin"]), ("c20_seed_cherry_pick", ["alibi_twin"])):
    r = V.verify(r"C:\Users\seal\Desktop\New folder (32)\sun\daedalus\cheats" + "\\" + d, pool=pool)
    print(d, json.dumps({k: r.get(k) for k in ("verdict", "stage", "reason", "band_acc", "wall_total_s")}, default=float), flush=True)
