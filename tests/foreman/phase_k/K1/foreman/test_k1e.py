# Foreman K1 fifth bar: what a trained L=4 twin could reach on its own training distribution. Written 2026-09-23 03:09
# IST (clock read 03:09:34), BEFORE k1e.py exists or runs and before any twin result exists. Never edited after its RED.
# Reads k1e.json. Exit 1 on any fail.
#
# Beds: the twin's evaluation beds, bed_k.make_train(default_rng([91, k])), k = 0..63 (n = 1024, depth capped at 32).
# Constructions at R0 pricing, credited (most recent root at or before the final pointer), scored on 17 <= depth <= 32.
# foreman.k1.train_dist_voidable : the HPD-oracle C2 (k1b.py) at L = 4 scores > 0.5 there, i.e. the training
#     distribution itself has a hand-set beyond-2^L route above the kill line, so a trained twin's acc(17..32) is a test
#     of whether SGD finds it, not of whether it exists.
# foreman.k1.train_dist_no_oracle : the no-oracle ALiBi-local window (hunt.py), best at R0 pricing, scores <= 0.25 there.
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "k1e.json")).read_text())
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


check("foreman.k1.train_dist_voidable HPD L4 acc(17..32) > 0.5", R["hpd_best"] > 0.5, f"{R['hpd_best']:.4f} [{R['hpd_cfg']}] dbl {R['dbl']:.4f}")
check("foreman.k1.train_dist_no_oracle local L4 acc(17..32) <= 0.25", R["local_best"] <= 0.25, f"{R['local_best']:.4f} [{R['local_cfg']}]")
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
