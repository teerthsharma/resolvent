# Foreman K1 fifth set: hand-set constructions on the twin's own evaluation beds (train distribution). Bars: test_k1e.py.
import json, sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from hunt import bed_k, doubling, hybrid_space, R0  # noqa: E402
from k1b import hpd_hybrid                          # noqa: E402

L = 4
ok = {}
for k in range(64):
    b = bed_k.make_train(np.random.default_rng([91, k]))
    p, d, r = b["parent"], b["depth"], b["root"]
    lr = bed_k._last_root(p)
    band = (d >= 17) & (d <= 32)
    ok.setdefault("dbl", []).append((lr[doubling(p, L)] == r)[band])
    for kk, W in R0:
        ok.setdefault(f"hpd_k{kk}_W{W}", []).append((lr[hpd_hybrid(p, d, L, W, kk)] == r)[band])
        ok.setdefault(f"local_k{kk}_W{W}", []).append(
            (lr[hybrid_space(p, d, np.arange(len(p)), L, W, kk, "anchored", anchor_self=True)[0]] == r)[band])
mean = {key: float(np.concatenate(v).mean()) for key, v in ok.items()}   # pooled over tokens, like the twin's eval
hk = max((x for x in mean if x.startswith("hpd")), key=mean.get)
lk = max((x for x in mean if x.startswith("local")), key=mean.get)
out = {"dbl": mean["dbl"], "hpd_best": mean[hk], "hpd_cfg": hk, "local_best": mean[lk], "local_cfg": lk, "sweep": mean}
(HERE / "k1e.json").write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k != "sweep"})
