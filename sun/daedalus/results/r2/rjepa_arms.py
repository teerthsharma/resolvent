import sys, json, os, time
sys.path.insert(0, r"C:\Users\seal\Desktop\New folder (32)\sun\daedalus\engine")
import verifier as V
from r2_rankers import INTEGRITY
D = r"C:\Users\seal\Desktop\New folder (32)\sun\daedalus\candidates_rjepa"
t0, out = time.time(), {}
for a in ("fm_point", "fm_hop1", "fm_resolvent", "fm_djepa", "ch_dj_stoch"):
    r = V.verify(os.path.join(D, a))
    r["admissible"] = r["verdict"] != "ERROR" and not (r["verdict"] == "REJECT" and r["stage"] in INTEGRITY)
    out[a] = r
    print(a, r["verdict"], r["stage"], "admissible" if r["admissible"] else "INADMISSIBLE", r["reason"][:120],
          json.dumps({k: r.get(k) for k in ("hit_planner", "hit_shuffled", "bayes_hit", "floors", "equivariance_dev", "bound_dev")}, default=float), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "rjepa_arms.json"), "w"), indent=2, default=float)
print(f"{sum(r['admissible'] for r in out.values())}/{len(out)} admissible; {sum(r['verdict'] == 'PASS_V2' for r in out.values())} PASS_V2; {time.time() - t0:.0f} s")
