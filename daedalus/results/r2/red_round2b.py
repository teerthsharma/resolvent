import sys, json
sys.path.insert(0, r"C:\Users\seal\Desktop\New folder (32)\sun\daedalus\engine")
import verifier as V
D = r"C:\Users\seal\Desktop\New folder (32)\sun\daedalus\cheats" + "\\"
for d, pool in (("r05_probe_fingerprint", None), ("c22_seed_residue_whitelist", ["alibi_twin"])):
    r = V.verify(D + d, pool=pool)
    print(d, json.dumps({k: r.get(k) for k in ("verdict", "stage", "reason", "band_acc", "hit_planner", "bound_dev", "wall_total_s")}, default=float), flush=True)
