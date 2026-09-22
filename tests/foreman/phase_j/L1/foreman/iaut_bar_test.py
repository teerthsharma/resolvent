"""RED if no finite SU(2) image (2I, 2O, and so 2T and every cyclic subgroup) of the two I-AUT
symbols carries the L1-IAUT row's own registered pass bar 0.8120, even read by a train-fit Bayes
oracle on the exact anchored product. Reads iaut_su2_ceiling.json (produced by iaut_su2_ceiling.py)."""
import json, os
r = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "iaut_su2_ceiling.json")))
per_seed = {s: max(v["best_2I"], v["best_2O"]) for s, v in r["seeds"].items()}
mean = sum(per_seed.values()) / len(per_seed)
print(per_seed, mean)
assert mean >= 0.8120, f"RED: best finite-SU(2) Bayes ceiling per seed {per_seed}, mean {mean:.4f} < L1-IAUT pass bar 0.8120"
