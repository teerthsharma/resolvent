"""bars_r4 r4_c bed-choice claim, bound as a test (Inspector r4): on the shared-error bed the set-level edge (Bayes
minus the best pointwise rule, a grid of argmax +-| |zhat| - c | rules) is < 0.02 at K 63 and >= 0.05 at K 6 sigma 2.
Floors only, no ranker trained; the executed pool headroom on the draws is the bed-validity read, not this.
    python sun/daedalus/engine/test_r4_bed.py"""
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planning_bed as PB  # noqa: E402


def edge(K, sigma, n=4000, seed=101):
    b = PB.make_bed(n, sigma, seed, K=K)
    hb = float((PB.bayes_pick(b) == b["best"]).mean())
    r = np.linalg.norm(b["zhat"], axis=-1)
    pw = max(float(((s * np.abs(r - c)).argmax(1) == b["best"]).mean()) for c in np.linspace(0, 4, 33) for s in (1, -1))
    return hb, pw, hb - pw


out = {(K, s): edge(K, s) for K, s in ((63, 1.5), (63, 3.0), (6, 2.0))}
print({f"K{K} sigma{s}": "bayes %.4f pointwise* %.4f edge %.4f" % v for (K, s), v in out.items()})
assert out[63, 1.5][2] < 0.02 and out[63, 3.0][2] < 0.02, "K 63 has set-level headroom: the claim is dead"
assert out[6, 2.0][2] >= 0.05, "K 6 sigma 2 lacks 0.05 set-level headroom: the claim is dead"
print("PASS test_r4_bed: K 63 edge < 0.02 at sigma 1.5 and 3; K 6 sigma 2 edge >= 0.05")
