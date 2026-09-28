"""bars_r4 r4_f: fm_resolvent_f64 on planning_consequence_v1 draw r4d1 (V1 precision hypothesis). One job."""
import hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(D, "engine"))
import verifier as V
r = V.verify(os.path.join(D, "candidates_rjepa", "fm_resolvent_f64"), tag="r4d1", bed="planning_consequence_v1")
r["driver_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
print(r["verdict"], r["stage"], r["reason"][:200], json.dumps({k: r.get(k) for k in ("equivariance_dev", "hit_planner", "hit_shuffled", "bayes_hit", "floors")}, default=float), flush=True)
json.dump(r, open(os.path.join(HERE, "r4_f_run.json"), "w"), indent=1, default=float)
