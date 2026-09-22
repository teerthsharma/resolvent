# Foreman K0 round 2: (T5) the hybrid's critical depth across the ladder's own widths; (T6) a gap-dilated bed.
import json
from pathlib import Path
import numpy as np
from shortcut import fresh_bed, doubling, hybrid

HERE = Path(__file__).parent
RUNGS = [(4, 128), (6, 320), (8, 512)]  # (L, d_model) for R0-R2 per the contract's ladder


def dc(pred, depth, root):
    ok = pred == root
    acc = np.array([ok[depth == d].mean() for d in range(depth.max() + 1)])
    return int((acc >= 0.5).sum()), acc


def main():
    R = {}
    parent, depth, root = fresh_bed(3000, n=16384, m=16)
    rows = []
    for L, d in RUNGS:
        W = d // 12
        dd, _ = dc(doubling(parent, L), depth, root)
        dh, acc = dc(hybrid(parent, depth, L, W, 3, "centered"), depth, root)
        da, _ = dc(hybrid(parent, depth, L, W, 3, "anchored"), depth, root)
        rows.append({"L": L, "d": d, "W": W, "2^L": 2 ** L, "Dc_doubling": dd, "Dc_hybrid_centered": dh, "Dc_hybrid_anchored": da})
    x = np.log([r["2^L"] for r in rows])
    R["rscale"] = {"bed": {"n": 16384, "M": 16, "seed": 3000, "max_depth": int(depth.max())}, "rows": rows,
                   "doubling_slope": float(np.polyfit(x, np.log([r["Dc_doubling"] for r in rows]), 1)[0]),
                   "hybrid_slope": float(np.polyfit(x, np.log([r["Dc_hybrid_centered"] for r in rows]), 1)[0]),
                   "hybrid_anchored_slope": float(np.polyfit(x, np.log([r["Dc_hybrid_anchored"] for r in rows]), 1)[0])}
    # T6: dilated replacement bed, and the pinned-shape bed at the same n for reference
    out = {}
    for m in (16, 64, 128):
        p, dep, r = fresh_bed(3001, n=16384, m=m)
        base = float((doubling(p, 6) == r).mean())
        hyb = float((hybrid(p, dep, 6, 26, 3, "centered", m=m) == r).mean())
        hyb4 = float((hybrid(p, dep, 6, 26, 4, "centered", m=m) == r).mean())
        out[f"M{m}"] = {"max_depth": int(dep.max()), "median_depth": int(np.median(dep)), "doubling_L6": base,
                        "hybrid_L6_W26_k3": hyb, "hybrid_L6_W26_k4": hyb4, "ratio_k3": hyb / base, "ratio_k4": hyb4 / base,
                        "beyond64_hybrid_k3": float((hybrid(p, dep, 6, 26, 3, "centered", m=m) == r)[dep > 64].mean())}
    R["dilated"] = {"beds": out, "ratio": out["M64"]["ratio_k3"]}
    (HERE / "results2.json").write_text(json.dumps(R, indent=1))
    return R


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
