# Foreman K0 T9 bar, written 01:58 IST before running. Cameron's K1 bed (bed_k.make_test, imported read-only,
# no bytecode written) gives "opponent best response beyond 2^L = 0.041-0.068". Bar: IF the twin has a positional
# addressing channel, the centered hybrid (k=3, L=6, W=26 = R1 slot budget) beats that band on depth > 64 at
# n = 4096: mean over 8 test beds (seed family default_rng([7, n, k]), disjoint from Cameron's [2, n, k]) > 0.068.
# Also reported (no bar): n = 8192 and 16384, and the n-dilution of the beyond-2^L metric. Exit 1 on failure.
import json, sys
sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K0/cameron")
import numpy as np
import bed_k
from shortcut import hybrid, doubling
out = {}
for n in (4096, 8192, 16384):
    rows = []
    for k in range(8):
        b = bed_k.make_test(np.random.default_rng([7, n, k]), n)
        p, d, r = b["parent"], b["depth"], b["root"]
        deep = d > 64
        rows.append({"hybrid_c_W26": float((hybrid(p, d, 6, 26, 3, "centered") == r)[deep].mean()),
                     "hybrid_a_W26": float((hybrid(p, d, 6, 26, 3, "anchored") == r)[deep].mean()),
                     "hybrid_c_W48": float((hybrid(p, d, 6, 48, 3, "centered") == r)[deep].mean()),
                     "doubling": float((doubling(p, 6) == r)[deep].mean()), "max_depth": int(d.max())})
    out[n] = {key: [float(np.mean([x[key] for x in rows])), float(np.min([x[key] for x in rows])), float(np.max([x[key] for x in rows]))] for key in rows[0]}
print(json.dumps(out, indent=1))
ok = out[4096]["hybrid_c_W26"][0] > 0.068
print(("PASS" if ok else "FAIL") + f" bedk_beyond64_n4096 hybrid_c_W26 mean {out[4096]['hybrid_c_W26'][0]:.4f} > 0.068")
from pathlib import Path
Path(__file__).with_name("results_bedk.json").write_text(json.dumps(out, indent=1))
sys.exit(0 if ok else 1)
