# K0_gamma_window. Bars written 2026-09-23 01:47, before any gamma other than 0.999 was run.
# Claim: the softmax resolvent's "fixed gamma" has a window on the pinned bed at s = 16:
#   lower edge (range 1/(1-gamma) must cover the depth): gamma = 0.9  -> accuracy < 0.9
#   upper edge (token 0 has W_00 = 1, a pole at gamma = 1, and absorbs all leaked mass): 1-gamma = 1e-10 -> accuracy < 0.2
#   inside: gamma = 0.999 -> accuracy >= 0.999 (the pinned value).
# Dense independent solve (bed_k.resolvent_dense).
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
from bed_k import pinned_bed, resolvent_dense

parent, depth, root = pinned_bed()
g = {0.9: resolvent_dense(parent, root, 16, 0.9), 0.99: resolvent_dense(parent, root, 16, 0.99),
     0.999: resolvent_dense(parent, root, 16, 0.999), 0.99999: resolvent_dense(parent, root, 16, 0.99999),
     1 - 1e-10: resolvent_dense(parent, root, 16, 1 - 1e-10)}
for k, v in g.items(): print(f"gamma {k!r}: accuracy {v:.4f}")
json.dump({repr(k): v for k, v in g.items()}, open("gamma_window.json", "w"), indent=1)
ok = [g[0.9] < 0.9, g[1 - 1e-10] < 0.2, g[0.999] >= 0.999]
print("PASS" if all(ok) else f"FAIL {ok}")
sys.exit(0 if all(ok) else 1)
