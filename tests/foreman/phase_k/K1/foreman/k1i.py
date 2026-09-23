# Foreman K1 ninth set: L = 4 reach with a band floor a 2-depth band can meet. Bars: test_k1i.py.
import json, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import k1g  # noqa: E402  (beds, families, pooled_ok; its own reach() is not used)
from hunt import doubling  # noqa: E402

L = 4


def reach(ok):
    D = 0
    for d in range(2, k1g.DEPTH.max() + 1, 2):
        m = (k1g.DEPTH > d - 2) & (k1g.DEPTH <= d)
        if m.sum() < 100:
            continue
        if ok[m].mean() < 0.5:
            break
        D = d
    return D / 2 ** L


fam = k1g.families(L, "R0")
dbl = k1g.pooled_ok(lambda b: doubling(b["parent"], L))
ch, cl = max(reach(o) for o in fam["hpd"]), max(reach(o) for o in fam["local"])
kappa = 2 * max(ch, cl)
far = k1g.DEPTH > kappa * 2 ** L
accs = [float(o[far].mean()) for o in fam["hpd"] + fam["local"] + [dbl]]
out = {"doubling": reach(dbl), "hpd": ch, "local": cl,
       "far": {"kappa": kappa, "tokens": int(far.sum()), "worst": max(accs), "doubling_far": accs[-1]}}
(HERE / "k1i.json").write_text(json.dumps(out, indent=1))
print(out)
