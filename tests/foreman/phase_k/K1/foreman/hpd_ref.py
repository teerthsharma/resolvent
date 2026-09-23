# Independent slow reference for the HPD-window hybrid (the construction that fired the kill). Bar: test_k1f.py.
# Plain Python per token; bed_k only for the beds; k1b imported only AFTER the reference ran, to compare.
import json, math, sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Users/seal/Desktop/New folder (32)/tests/foreman/phase_k/K0/cameron")
import bed_k  # noqa: E402

HERE = Path(__file__).parent
L, W, P = 7, 10, 1 / 16
_cache = {}


def best_start(J):
    """argmax over a >= 1 of P(a <= S < a + W), S = sum of J Geometric(1/16) gaps on {1, 2, ...}."""
    if J == 0:
        return 1
    if J not in _cache:
        smax = 16 * J + 200 + int(80 * math.sqrt(J))
        pm = [0.0] * (smax + 1)
        for s in range(J, smax + 1):
            pm[s] = math.exp(math.lgamma(s) - math.lgamma(J) - math.lgamma(s - J + 1) + J * math.log(P) + (s - J) * math.log(1 - P))
        best, arg = -1.0, 1
        for a in range(1, smax - W + 2):
            m = sum(pm[a:a + W])
            if m > best:
                best, arg = m, a
        _cache[J] = arg
    return _cache[J]


def reference(parent):
    n = len(parent)
    depth = [0] * n
    for i in range(n):
        depth[i] = 0 if parent[i] < 0 else depth[parent[i]] + 1
    A = [i if parent[i] < 0 else int(parent[i]) for i in range(n)]
    for _ in range(L):
        new = []
        for t in range(n):
            v1 = A[t]
            v2 = A[v1]
            a = best_start(depth[t] - depth[v1])
            lo, hi = v1 - (a + W - 1), v1 - a + 1
            new.append(A[v2] if (v2 == v1 or lo <= v2 < hi) else v2)
        A = new
    return A, depth


def truth_and_credit(parent, A):
    n = len(parent)
    root, pred = [], []
    for t in range(n):
        x = t
        while parent[x] >= 0:
            x = parent[x]
        root.append(x)
        y = A[t]
        while parent[y] >= 0:          # most recent root at or before the final pointer
            y -= 1
        pred.append(y)
    return root, pred


def main():
    accs, ptrs = [], []
    for k in range(4):
        b = bed_k.make_test(np.random.default_rng([12, 4096, k]), 4096)
        p = [int(x) for x in b["parent"]]
        A, depth = reference(p)
        root, pred = truth_and_credit(p, A)
        deep = [t for t in range(4096) if depth[t] > 2 ** L]
        accs.append(sum(pred[t] == root[t] for t in deep) / len(deep))
        ptrs.append(A)
    sys.path.insert(0, str(HERE))
    from k1b import hpd_hybrid
    mism, diff = 0, 0.0
    for k in range(4):
        b = bed_k.make_test(np.random.default_rng([12, 4096, k]), 4096)
        p, d, r = b["parent"], b["depth"], b["root"]
        V = hpd_hybrid(p, d, L, W, 3)
        mism += int((V != np.array(ptrs[k])).sum())
        diff = max(diff, abs(float((bed_k._last_root(p)[V] == r)[d > 2 ** L].mean()) - accs[k]))
    out = {"mismatches": mism, "max_acc_diff": diff, "ref_accs": accs, "ref_mean": float(np.mean(accs))}
    (HERE / "hpd_ref.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(main())
