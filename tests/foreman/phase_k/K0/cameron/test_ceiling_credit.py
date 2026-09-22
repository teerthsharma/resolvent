# K0_doubling_ceiling_chance_credit. Bar written 2026-09-23 01:46, before the number existed.
# Claim: wald.py's doubling "ceiling" gives unresolved tokens ZERO credit. The same perfect-doubling opponent
# that, on tokens deeper than 2^L, guesses with the author's own position-only rule applied at its 2^L-th
# ancestor scores MORE than the ceiling. Bar: simulated L=6 opponent > 0.2539 + 0.02 on the pinned bed,
# i.e. R-DEPTH's "(a_L) stays under 0.254" fires on the opponent's own best response.
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
from bed_k import pinned_bed, ceiling_exact, ceiling_credit

parent, depth, root = pinned_bed()
res = {L: (ceiling_exact(depth, L), ceiling_credit(parent, depth, root, L)) for L in (4, 6, 8)}
for L, (e, c) in res.items():
    print(f"L={L}: exact-only ceiling {e:.4f}   chance-credited opponent {c:.4f}")
json.dump({str(k): v for k, v in res.items()}, open("ceiling_credit.json", "w"), indent=1)
ok = res[6][1] > 0.2539 + 0.02
print(("PASS" if ok else "FAIL") + f" L=6 credited opponent {res[6][1]:.4f} > 0.2739")
sys.exit(0 if ok else 1)
