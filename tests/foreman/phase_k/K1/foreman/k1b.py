# Foreman K1 second set: HPD-window C2 (same oracle and pricing as K1.F) and the no-oracle local window, width-priced.
# Bars: test_k1b.py.
import json, sys
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.special import gammaln

sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from hunt import bed_k, pointer0, hybrid_space, R0, M  # noqa: E402

P = 1 / M


@lru_cache(maxsize=None)
def pmf(J):
    """P(sum of J iid Geometric(1/16) gaps = s), s = 0..smax (0 below J)."""
    smax = int(M * J + 60 * np.sqrt(J) * M ** 0.5 + 400)
    s = np.arange(smax + 1)
    out = np.zeros(smax + 1)
    v = s >= J
    sv = s[v]
    out[v] = np.exp(gammaln(sv) - gammaln(J) - gammaln(sv - J + 1) + J * np.log(P) + (sv - J) * np.log1p(-P))
    return out


@lru_cache(maxsize=None)
def hpd_start(J, W):
    if J == 0:
        return 1
    c = np.concatenate([[0.0], np.cumsum(pmf(J))])
    mass = c[W:] - c[:-W]                     # mass[a] = P(a <= s < a + W)
    mass[0] = -1.0                            # gaps are >= 1
    return int(np.argmax(mass))


def hpd_hybrid(parent, depth, L, W, k):
    A = pointer0(parent)
    for _ in range(L):
        v1 = A
        cur = A[v1]
        J = depth - depth[v1]
        a = np.array([hpd_start(int(j), W) for j in range(J.max() + 1)])[J]
        lo, hi = v1 - (a + W - 1), v1 - a + 1
        live = np.ones(len(A), bool)
        for _ in range(k - 2):
            inw = live & (((cur >= lo) & (cur < hi)) | (cur == v1))
            cur = np.where(inw, A[cur], cur)
            live = inw
        A = cur
    return A


def machinery():
    j1 = all(hpd_start(1, W) == 1 for W in range(1, 11))
    bad = 0
    for J in range(1, 401):
        f = pmf(J)
        for W in range(1, 11):
            a = hpd_start(J, W)
            h = f[a:a + W].sum()
            s_lo, s_hi = max(M * J - W // 2 + 1, 1), M * J + W // 2          # centered window in gap coordinates
            cen = f[s_lo:s_hi + 1].sum()
            bad += int(h < cen - 1e-15)
    return {"j1_equals_anchored": bool(j1), "hpd_ge_centered_violations": bad}


def beyond(ok, d, L):
    return float(ok[d > 2 ** L].mean())


def main():
    out = {"hpd_machinery": machinery(), "hpd": {}, "local_wide": {}}
    for n in (4096, 8192, 16384):
        beds = [bed_k.make_test(np.random.default_rng([12, n, k]), n) for k in range(8)]
        out["hpd"][str(n)] = {}
        for L in (4, 7):
            acc = {}
            for b in beds:
                p, d, r = b["parent"], b["depth"], b["root"]
                lr = bed_k._last_root(p)
                for k, W in R0:
                    acc.setdefault(f"hpd_k{k}_W{W}", []).append(beyond(lr[hpd_hybrid(p, d, L, W, k)] == r, d, L))
            mean = {kk: float(np.mean(v)) for kk, v in acc.items()}
            best = max(mean, key=mean.get)
            out["hpd"][str(n)][str(L)] = {"best": mean[best], "best_cfg": best, "sweep": mean}
    beds = [bed_k.make_test(np.random.default_rng([52, 4096, k]), 4096) for k in range(8)]
    curve = {}
    for W in range(11, 129):
        v = []
        for b in beds:
            p, d, r = b["parent"], b["depth"], b["root"]
            A, _ = hybrid_space(p, d, np.arange(4096), 7, W, 3, "anchored", anchor_self=True)
            v.append(beyond(bed_k._last_root(p)[A] == r, d, 7))
        curve[W] = float(np.mean(v))
    cross = [W for W, v in curve.items() if v > 0.5]
    out["local_wide"] = {"W64": curve[64], "crossing_W": min(cross) if cross else None,
                         "curve": {str(W): v for W, v in curve.items()}}
    (HERE / "k1b.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    R = main()
    print(json.dumps(R["hpd_machinery"]))
    for n, row in R["hpd"].items():
        for L, c in row.items():
            print(f"HPD n={n} L={L} best {c['best']:.4f} [{c['best_cfg']}]  k3W10 {c['sweep']['hpd_k3_W10']:.4f} "
                  f"k4W6 {c['sweep']['hpd_k4_W6']:.4f}")
    lw = R["local_wide"]
    print("local W64", lw["W64"], "crossing", lw["crossing_W"],
          {W: round(lw["curve"][str(W)], 4) for W in (11, 16, 24, 32, 40, 48, 56, 64, 96, 128)})
