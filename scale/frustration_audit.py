"""Zaslavsky frustration: is a signed arm SIGNED, or signed in name only?

THE CHECK THE ARSENAL HAS CARRIED SINCE ROUND 1 AND NOBODY RAN.
"Frustration ~ 0 = signed-in-name-only; the probe must be switching-gauge
invariant. This item is itself a kill."

WHY GAUGE INVARIANCE IS THE WHOLE POINT. Switching a signed graph by
D = diag(d_i), d_i in {+1,-1}, sends sign(A_ij) -> d_i d_j sign(A_ij). That
changes the COUNT of negative entries while changing nothing structural: a
matrix whose sign pattern is exactly d_i d_j is *balanced*, and its signs can be
gauged away entirely. An operator like that is unsigned wearing a sign.

So counting negatives is not a measurement. The invariant is the TRIANGLE SIGN
PRODUCT: for nodes i, j, k the quantity

    sign(A_ij) * sign(A_ik) * sign(A_jk)

is unchanged by switching, because each d appears exactly twice. A signed graph
is balanced iff every triangle product is +1 (Harary). The frustrated fraction
-- triangles reading -1 -- is therefore a gauge-invariant measure of how much
real sign structure an operator carries.

    frustration ~ 0.0   the signs are a gauge artifact: SIGNED IN NAME ONLY
    frustration ~ 0.5   signs carry structure a switching cannot remove

CALIBRATED BOTH ENDS, per instrument law. A planted balanced pattern must read
~0 and an iid random sign pattern must read ~0.5, or the probe cannot tell the
two apart and a low reading means nothing.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, r"C:\Users\seal\Desktop\New folder (32)")
from ceq import bench

torch.set_num_threads(2)
TRIANGLES = 20000
SIZES = (16, 128, 512)


def frustrated_fraction(sign_mat: torch.Tensor, s: int, gen) -> tuple[float, int]:
    """Fraction of sampled triangles whose sign product is -1. Gauge-invariant.

    Triangles are drawn from the STRICTLY LOWER TRIANGULAR support i > j > k, so
    all three entries A_ij, A_ik, A_jk exist in a causal operator.
    """
    if s < 3:
        return float("nan"), 0
    neg = n = 0
    for _ in range(TRIANGLES):
        idx = torch.randperm(s, generator=gen)[:3]
        i, j, k = sorted(idx.tolist(), reverse=True)      # i > j > k
        a, b, c = sign_mat[i, j], sign_mat[i, k], sign_mat[j, k]
        if a == 0 or b == 0 or c == 0:
            continue
        n += 1
        if a * b * c < 0:
            neg += 1
    return (neg / n if n else float("nan")), n


def op_signs(kind: str, s: int, gen) -> torch.Tensor:
    q = torch.randn(s, 16, generator=gen)
    k = torch.randn(s, 16, generator=gen)
    if kind == "sgate":
        a = bench._causal_sgate_operator(q, k, rho=1.5, lam=0.10)
    elif kind == "sgate_lam1":                # the same operator at lam = 1.0
        a = bench._causal_sgate_operator(q, k, rho=1.5, lam=1.00)
    elif kind == "tgate":
        g = torch.sigmoid(torch.randn(s, generator=gen))
        a = bench._causal_tgate_operator(q, k, g, 1.0)
    elif kind == "softmax":
        a = bench._softmax_operator(q, k)
    elif kind == "random_signed":              # iid signs: the unbalanced end
        a = torch.randn(s, s, generator=gen).tril(-1)
    elif kind == "balanced":                   # planted d_i d_j: the balanced end
        d = torch.where(torch.rand(s, generator=gen) < 0.5, -1.0, 1.0)
        a = (d.view(-1, 1) * d.view(1, -1)) * torch.rand(s, s, generator=gen)
        a = a.tril(-1)
    else:
        raise ValueError(kind)
    return torch.sign(a)


def main() -> int:
    print("ZASLAVSKY FRUSTRATION -- switching-gauge invariant (triangle products)")
    print(f"{TRIANGLES} triangles per cell, i > j > k in the causal support\n")
    print("=== CALIBRATION: both ends must be SEEN, or a low reading means nothing ===")
    for kind in ("balanced", "random_signed"):
        g = torch.Generator().manual_seed(0)
        f, n = frustrated_fraction(op_signs(kind, 128, g), 128, g)
        print(f"  {kind:>15}  frustration = {f:.6f}   ({n} usable triangles)")

    print("\n=== ARMS ===")
    print(f"{'arm':>15} {'s':>6} {'frustration':>13} {'neg entries':>13} {'verdict':>22}")
    for kind in ("softmax", "sgate", "sgate_lam1", "tgate"):
        for s in SIZES:
            g = torch.Generator().manual_seed(0)
            sm = op_signs(kind, s, g)
            f, _ = frustrated_fraction(sm, s, g)
            tril = sm.tril(-1)
            negfrac = float((tril < 0).sum()) / max(1, int((tril != 0).sum()))
            if f != f:
                v = "n/a"
            elif f < 0.02:
                v = "SIGNED IN NAME ONLY"
            elif f < 0.15:
                v = "weak sign structure"
            else:
                v = "real sign structure"
            print(f"{kind:>15} {s:>6} {f:>13.6f} {negfrac:>13.6f} {v:>22}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
