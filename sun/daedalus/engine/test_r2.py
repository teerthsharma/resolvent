"""Round-2 unit checks for the verifier's new instruments (fast, CPU, no sandbox). Run:
python sun/daedalus/engine/test_r2.py"""
import os
import sys

import numpy as np
import torch

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import planning_bed as PB  # noqa: E402
import verifier as V  # noqa: E402

D = os.path.dirname(HERE)


def load(rel):
    import importlib.util
    spec = importlib.util.spec_from_file_location("c_" + os.path.basename(rel), os.path.join(D, rel, "candidate.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.build({"K": 4, "F": 2, "param_budget": 20000})


# 1. r2_c seed rule: 4 run seeds, both parities, every residue mod 3 and mod 4, none of the retired public seeds,
#    and the residue set is not fixed across tags (r2_a's fixed {0,1,2,3} mod 12 was whitelistable: c22)
s = V.seeds_for("m0", 3)
assert len(s) == 4 and {x % 2 for x in s} == {0, 1} and {x % 3 for x in s} == {0, 1, 2} and {x % 4 for x in s} == {0, 1, 2, 3}
assert not set(s) & {53876, 70658, 45346}
sets = {tuple(sorted(x % 12 for x in V.seeds_for(f"probe-tag-{i}", 3))) for i in range(40)}
assert len(sets) >= 10, sets

# 2. ceiling: a ranker that returns the sealed labels breaches the Bayes ceiling; the Bayes rule itself does not
b = PB.make_bed(4000, 1.5, 11)
bp = PB.bayes_pick(b)
hb = float((bp == b["best"]).mean())
assert V.ceiling_breach(1.0, hb, 4000) is not None            # hit of the sealed labels themselves
assert V.ceiling_breach(hb, hb, 4000) is None and V.ceiling_breach(hb + 0.02, hb, 4000) is None

# 3-4. V1 instruments on probes
Z, Zp, perms = V.ranker_probes(PB.make_bed(256, 1.5, 12)["zhat"], np.random.default_rng(0))
with torch.no_grad():
    S = lambda m: (m(torch.tensor(Z)).double().numpy(), np.stack([m(torch.tensor(z)).double().numpy() for z in Zp]))
    for rel, want in (("controls/r_deepsets", False), ("controls/r_latent_distance", False),
                      ("cheats/r03_false_equivariance", True), ("cheats/r04_false_bound", False)):
        dev, tol = V.equivariance_dev(*S(load(rel)), perms)
        assert (dev > tol) == want, (rel, dev, tol)
    m = load("cheats/r04_false_bound")
    assert V.bound_dev(S(m)[0], Z) > 0.2 + 1e-5

    class Tight(torch.nn.Module):  # a correctly bounded D-JEPA-shaped score: base + 0.2 tanh(.)
        def forward(self, z):
            n = z.norm(dim=-1)
            return torch.tensor(PB.base_score(z.numpy()), dtype=z.dtype) + 0.2 * torch.tanh(n)
    assert V.bound_dev(S(Tight())[0], Z) <= 0.2 + 1e-5

# 5. order: a slot reader moves between planner order and a shuffle; latent distance does not
Xe = PB.features(b)
Xs, Ys, P = V.shuffle_rows(Xe, b["best"], np.random.default_rng(3))
assert (np.take_along_axis(Xe, P[..., None], 1) == Xs).all()
pick = lambda X: -np.linalg.norm(X, axis=-1)
assert (Ys == np.array([np.flatnonzero((Xs[i] == Xe[i, b["best"][i]]).all(-1))[0] for i in range(len(Ys))])).all()
assert abs(float((pick(Xe).argmax(1) == b["best"]).mean()) - float((pick(Xs).argmax(1) == Ys).mean())) < 1e-12
slot = np.zeros(len(Xe), dtype=int)
assert float((slot == b["best"]).mean()) - float((slot == Ys).mean()) > 0.03

# 6. the executed view is sealed
try:
    PB.features(b, "executed")
    raise AssertionError("executed view served")
except ValueError:
    pass

# 7. infrastructure deaths are ERROR, never a verdict and never a crash (r2 M0 run died at c16: the runner was
#    killed with empty stderr, infra "" was falsy, so no retry, and pool_accs raised a bare RuntimeError)
import time as _t
_orig_run, _orig_sleep, _orig_te = V._run_sandbox, _t.sleep, V.train_eval
V._run_sandbox = lambda *a: ({"infra": "", "error": "infra", "violations": []}, "h", "h")
_t.sleep = lambda s: None
try:
    V.run_sandbox("x", {}, {})
    raise AssertionError("empty-stderr runner death returned as a candidate result")
except V.InfraError:
    pass
V.train_eval = lambda *a: ({"error": "infra", "violations": []}, None, None, "h", "h")
try:
    V.pool_accs("x", 1, ["gru"], "infra-test")
    raise AssertionError("pool member death returned")
except V.InfraError:
    pass
V._run_sandbox, _t.sleep, V.train_eval = _orig_run, _orig_sleep, _orig_te
# 8. r3 (bars_r3 r3_c): ranker floors are per draw and handed out as copies. r2 cached them by bed only, so a second
#    draw in one process was scored against the first draw's floors, and verify_ranker's null write went into the cache.
import json
bed = "planning_consequence_v0"
_steps = V.R2["beds"][bed]["train_steps"]
V.R2["beds"][bed]["train_steps"] = 5  # the pointwise floor is irrelevant here; latent distance needs no training


def fl(tag):
    return V.ranker_floors(bed, V._rdata(bed, tuple(V.seeds_for(tag)), tag), tag)


fa, fb = fl("floor-test-a"), fl("floor-test-b")
assert fa["latent_distance"] != fb["latent_distance"], "second draw scored against the first draw's floors"
fa["label_shuffle_null"] = 1.0
assert "label_shuffle_null" not in fl("floor-test-a"), "a caller's null leaked into the floor cache"
V.R2["beds"][bed]["train_steps"] = _steps

# 9. r3 (r3_d): every V3 pool member's source is byte-identical to its registered sha256
for name, sha in json.load(open(os.path.join(D, "registry", "bars_r3.json")))["r3_d_ranker_v3"]["pool"].items():
    assert V._sha(open(os.path.join(D, "controls", name, "candidate.py"), "rb").read()) == sha, name
# 10. r4 (bars_r4 r4_b): a bounded claim with 2 * epsilon >= the base range (1) is vacuous and REJECTed before any run;
#     epsilon 0.2 (the pool's D-JEPA) stays admissible
import shutil
import tempfile
V._TRUTH = V._truth_windows() if V._TRUTH is None else V._TRUTH
for eps, want in ((4.0, "V1"), (0.5, "V1"), (0.2, None)):
    with tempfile.TemporaryDirectory() as td:
        shutil.copy(os.path.join(D, "candidates_rjepa", "fm_djepa_eps4", "candidate.py"), td)
        json.dump({"name": "t", "kind": "ranker", "bed": bed, "features": "predicted",
                   "claims": ["permutation_equivariant", "bounded"], "epsilon": eps}, open(os.path.join(td, "manifest.json"), "w"))
        man, rej = V.v0_static(td, {})
        assert (rej["stage"] if rej else None) == want, (eps, rej)

# 11. r4 (r4_a): the label-shuffle null is its own process whose job equals the seed-0 job except train_Y, and every
#     job holds one seed, so build order and job shape say nothing about which model is the null
jobs = []
_orig = V.run_sandbox


def _capture(cdir, man, job, *a, **k):
    jobs.append(job)
    runs = [{"n_params": 1, "n_hidden": 0}]
    return {"violations": [], "error": "" if len(jobs) < 5 else "stop", "probe_shape": torch.tensor([1024, 4]), "runs": runs}, "h", "h"


V.run_sandbox = _capture
r = V.verify_ranker(os.path.join(D, "controls", "r_latent_distance"), {"bed": bed}, {}, tag="null-shape-test")
V.run_sandbox = _orig
assert len(jobs) == 5 and r["stage"] == "V0" and all(len(j["seeds"]) == 1 for j in jobs), (len(jobs), r)
for key in jobs[0]:
    same = all(torch.equal(x, y) for x, y in zip(jobs[0][key], jobs[4][key])) if isinstance(jobs[0][key], list) and         isinstance(jobs[0][key][0], torch.Tensor) else (torch.equal(jobs[0][key], jobs[4][key]) if isinstance(jobs[0][key], torch.Tensor)
                                                        else jobs[0][key] == jobs[4][key])
    assert same == (key != "train_Y"), key
assert sorted(jobs[0]) == sorted(jobs[4]) and "null_Y" not in jobs[0]
print("PASS test_r2: seed coverage (residues not printed: they are secret), ceiling, equivariance, bound, order, sealed view, infra = ERROR, r3 floors per draw, pool shas, r4 vacuous bound, r4 null job = seed-0 job but labels")
