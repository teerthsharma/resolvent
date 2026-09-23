# Foreman K1 third set: floor row of the replacement bed (bed_k at n >= 64 * 2^L) and the harness positional channel.
# Bars: test_k1c.py.
import json, sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from hunt import bed_k, doubling, hybrid_space, R0  # noqa: E402
from k1b import hpd_hybrid                          # noqa: E402


def main():
    out = {"deep": {}}
    for L, n in ((4, 1024), (5, 2048), (6, 4096), (7, 8192), (7, 16384), (7, 32768)):
        acc = {}
        frac = []
        for k in range(8):
            b = bed_k.make_test(np.random.default_rng([53, n, k]), n)
            p, d, r = b["parent"], b["depth"], b["root"]
            lr = bed_k._last_root(p)
            deep = d > 2 ** L
            frac.append(float((d > 4 * 2 ** L).mean()))
            f = lambda A: float((lr[A] == r)[deep].mean())
            acc.setdefault("recency", []).append(float((lr == r)[deep].mean()))
            acc.setdefault("dbl", []).append(f(doubling(p, L)))
            for kk, W in R0:
                acc.setdefault(f"hpd_k{kk}_W{W}", []).append(f(hpd_hybrid(p, d, L, W, kk)))
                acc.setdefault(f"local_k{kk}_W{W}", []).append(
                    f(hybrid_space(p, d, np.arange(n), L, W, kk, "anchored", anchor_self=True)[0]))
        mean = {key: float(np.mean(v)) for key, v in acc.items()}
        hk = max((x for x in mean if x.startswith("hpd")), key=mean.get)
        lk = max((x for x in mean if x.startswith("local")), key=mean.get)
        out["deep"][f"{L}_{n}"] = {"recency": mean["recency"], "dbl_cred": mean["dbl"], "hpd_best": mean[hk], "hpd_cfg": hk,
                                   "local_best": mean[lk], "local_cfg": lk,
                                   "line": max(mean["recency"], mean["dbl"], mean[hk], mean[lk]),
                                   "frac_gt_4x": float(np.mean(frac)), "max_depth_mean": None}
    out["harness_channel"] = json.loads((HERE / "poschan_harness.json").read_text())
    (HERE / "k1c.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
