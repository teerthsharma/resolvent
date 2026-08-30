"""caustic Theorem 2, pointed at attention rows.

T2: a map sending k entities to one value is a pooling equilibrium and bounds ANY
downstream function at 1/k. An attention operator is such a map -- query i produces
row A[i,:]. Two queries with the same row are pooled and no later layer separates
them. T1 then counts it with no labels: err >= n - m, m = number of distinct rows.

The structural claim: a non-negative row is a point on the SIMPLEX; a signed row
with the same L1 budget lives in the strictly larger CROSS-POLYTOPE. More
distinguishable rows available at equal budget, so the orbit partition should be
finer. That is a statement about reachable sets, not about a training run.

Rows are compared after L1 normalization so the comparison is about DIRECTION and
not about the scale rho, which the signed arm was allowed to tune.
"""
from __future__ import annotations
import sys, pathlib, statistics
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ceq import lm


def rows(kind, q, k, rho=1.5, lam=0.10):
    old = (lm.RHO, lm.SGATE_LAM)
    lm.RHO, lm.SGATE_LAM = rho, lam
    try:
        m = lm.TinyLM(kind=kind if kind != "softmax" else "softmax_x")
        a = m.blocks[0].attn.operator(q, k)
    finally:
        lm.RHO, lm.SGATE_LAM = old
    a = a[..., 2:, :]                                    # rows 0,1 are degenerate
    n = a.abs().sum(-1, keepdim=True).clamp_min(1e-30)
    return (a / n).reshape(-1, a.shape[-1])              # direction only


def orbits(r, tol):
    """Greedy single-linkage orbit count at tolerance tol (L1 on directions)."""
    reps, count = [], 0
    for v in r:
        if not any(float((v - p).abs().sum()) < tol for p in reps):
            reps.append(v); count += 1
    return count


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"caustic T1/T2 orbit probe. n rows per draw = 46, tol on L1 of the")
    print(f"L1-normalized row (direction only, so rho cannot flatter the signed arm).\n")
    print(f"{'tol':>6} {'arm':>10} {'orbits m':>9} {'n - m':>7} {'spread':>11}")
    for tol in (0.05, 0.10, 0.25, 0.50):
        for kind in ("softmax", "sgate"):
            ms = []
            for seed in (0, 1, 2):
                g = torch.Generator(device="cpu").manual_seed(seed)
                q = torch.randn(1, 4, 48, 32, generator=g, dtype=torch.float64).to(dev)
                k = torch.randn(1, 4, 48, 32, generator=g, dtype=torch.float64).to(dev)
                r = rows(kind, q, k)
                ms.append(orbits(r, tol))
            n = 4 * 46
            med = statistics.median(ms)
            print(f"{tol:>6.2f} {kind:>10} {med:>9.0f} {n - med:>7.0f} "
                  f"{min(ms):>5}-{max(ms):<5}")
        print()


if __name__ == "__main__":
    main()
