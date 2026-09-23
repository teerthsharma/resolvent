# Foreman K1 fourth set: HPD-oracle C2 at R1 pricing on bed_k at the replacement ratio. Bars: test_k1d.py.
import json, sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from hunt import bed_k, doubling  # noqa: E402
from k1b import hpd_hybrid        # noqa: E402

R1 = [(3, W) for W in range(1, 27)] + [(4, W) for W in range(1, 11)]


def main():
    out = {}
    for L, n in ((6, 2048), (6, 4096), (6, 8192), (6, 16384), (9, 32768)):
        acc = {}
        for k in range(8):
            b = bed_k.make_test(np.random.default_rng([54, n, k]), n)
            p, d, r = b["parent"], b["depth"], b["root"]
            lr = bed_k._last_root(p)
            deep = d > 2 ** L
            acc.setdefault("dbl", []).append(float((lr[doubling(p, L)] == r)[deep].mean()))
            for kk, W in R1:
                acc.setdefault(f"hpd_k{kk}_W{W}", []).append(float((lr[hpd_hybrid(p, d, L, W, kk)] == r)[deep].mean()))
        mean = {key: float(np.mean(v)) for key, v in acc.items()}
        best = max((x for x in mean if x.startswith("hpd")), key=mean.get)
        out[f"{L}_{n}"] = {"best": mean[best], "cfg": best, "dbl": mean["dbl"], "sweep": mean}
    (HERE / "k1d.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    for key, c in main().items():
        print(key, round(c["best"], 4), c["cfg"], "dbl", round(c["dbl"], 4))
