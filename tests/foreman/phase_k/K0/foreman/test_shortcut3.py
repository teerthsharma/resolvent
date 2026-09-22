# Foreman K0 round 3 bar. Written 2026-09-23 01:57 IST, BEFORE shortcut3.py existed or ran.
# T7 foreman.k0.softmax_heads_exact: the anchored hybrid (L=6, W=26, k=3) run with REAL causal softmax heads
#    (d_head = 12, +-1 binary id codes, logit scale beta = 4, float64) and an explicit ReLU multiplexer reproduces
#    the idealized pointer arrays at every layer (0 mismatches), so its accuracy equals the idealized 0.3982
#    (all tokens) / 0.1934 (depth > 64), with minimum sign-decode margin >= 0.5. Exit non-zero on failure.
import json, sys
from pathlib import Path
R = json.loads((Path(__file__).parent / "results3.json").read_text())
ok = (R["mismatches_total"] == 0 and abs(R["all"] - 0.3982) < 5e-5 and abs(R["beyond"] - 0.1934) < 5e-5
      and R["min_margin"] >= 0.5)
print(("PASS" if ok else "FAIL") + f" softmax_heads_exact: {R}")
sys.exit(0 if ok else 1)
