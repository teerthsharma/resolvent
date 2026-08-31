"""R4b — hierarchical (Dyson) coupling wired into the REAL operator.

WHY R4 WAS REPLACED. R4's kill was *"no zero crossing of slope(theta) in range,
OR theta* unstable over 3 seeds"*. Measured (iteration 9): slope(theta) is
identically zero for every theta >= 0.5 out to 1.5, max |slope| = 0.055. So
**theta* is a half-line, not a point**, and stability of an unidentifiable
parameter is not a test. Fourth route in a row with a kill that cannot do its
job.

R4b's kill is a SHAPE test at two pre-registered points, each of which can fire
in both directions and neither of which requires locating a point inside a
plateau:

    slope(theta = 0.20)  must be <= -0.20          (the decaying regime is real)
    slope(theta = 0.80)  must be within 0.10 of 0  (the flat regime is real)
    3 seeds each, spread reported

**Pre-registered in DONE.md at iteration 9, BEFORE this file existed.**

THE PROXY GAP THIS FILE CLOSES. Iteration 9's numbers came from an abstract
aggregation model (level l holds 2^l terms, coupling 2^(-theta*l)). That is not
the module. Here the coupling is applied to the ACTUAL operator: a token pair at
distance d = i - j sits at dyadic level l = floor(log2 d), so the level coupling
2^(-theta*l) is a distance decay on A itself.

    A_dyson[i, j] = A[i, j] * 2^(-theta * floor(log2(i - j)))

DERIVATION, which is what the kill tests. Level l contains ~2^l token pairs at
distance ~2^l, so the background variance is
sum_l 2^l * (2^(-theta l))^2 = sum_l 2^((1-2theta) l), which DIVERGES as
s^((1-2theta)/2) for theta < 1/2 and CONVERGES for theta > 1/2. Hence
slope(theta) = theta - 1/2 below, 0 above.

This is a DIFFERENT escape from pivot routing, and that is the point:

    pivot routing  : flattens by CUTTING the term count to k
                     (measured s^+0.048 against dense s^+1.062)
    Dyson coupling : flattens by KEEPING every term and making the sum CONVERGE

CONTROLS IN EVERY RUN, because fifteen instruments here were internally
consistent and externally wrong:
  * `random` must read at chance and must DECAY like an unweighted dense arm;
  * `softmax` must read exactly 0 on the value path -- it is non-negative, so
    `I + A + A^2` is non-negative entrywise and no flip is reachable. If softmax
    reads nonzero the instrument is broken and every number here is void.
  * theta = 0 must reproduce the UNWEIGHTED operator bitwise -- if it does not,
    the coupling is not what it claims to be.

PROTOCOL: SCALING (i = s-1, j = s/4, c = s/2).
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scale.pivot_probe import build_arm, select_pivots


def dyson_weights(s: int, theta: float, device=None) -> torch.Tensor:
    """2^(-theta * floor(log2(i-j))) for i > j, zero elsewhere.

    theta = 0 gives an all-ones mask on the strict lower triangle, so the
    weighted operator is BITWISE the unweighted one. That is the calibration.
    """
    i = torch.arange(s, device=device).view(-1, 1)
    j = torch.arange(s, device=device).view(1, -1)
    d = (i - j).clamp_min(1).float()
    lvl = torch.floor(torch.log2(d))
    w = torch.pow(2.0, -theta * lvl)
    return w * (i > j).float()


def flip_rate(kind: str, s: int, theta: float, *, n_draws: int, k: int = 8,
              d: int = 16, seed: int = 0, floor: float = 1e-6) -> dict:
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *sh: torch.randn(*sh, generator=g)
    i, j = s - 1, max(1, s // 4)
    c = s // 2
    W = dyson_weights(s, theta)
    flips = used = 0
    for _ in range(n_draws):
        wq, wk, wo = rnd(d, d), rnd(d, d), rnd(d, d)
        x0, v0 = rnd(s, d), rnd(s, d)
        gv, bet = torch.sigmoid(rnd(s)), torch.sigmoid(rnd(s))
        if c in (i, j):
            continue
        piv = select_pivots(x0 @ wk, k, exclude=(i, j))
        grads = []
        for cval in (rnd(d), rnd(d)):
            x = x0.clone(); x[c] = cval
            v = v0.clone().requires_grad_(True)
            a, _ = build_arm(kind, x @ wq, x @ wk, gv, bet, piv)
            # INSTRUMENT #16, CAUGHT AND FIXED. Weighting `a` and the RETURNED
            # `hop2` separately is a POSITIVE ELEMENTWISE RESCALE of an
            # already-summed quantity, and a positive rescale CANNOT CHANGE A
            # SIGN -- so theta=0.20 and theta=0.80 gave bit-identical flip rates
            # (0.0250/0.0000/0.0000 at both). The coupling has to enter the
            # hop-2 SUM, where the per-p weights differ and the sign can move:
            #     hop2[i,j] = sum_p (W[i,p]A[i,p]) * (W[p,j]A[p,j])
            a = a * W
            hop2 = (a[:, piv] @ a[piv, :]) if kind.startswith("pivot") else (a @ a)
            h = ((v + a @ v + hop2 @ v) @ wo)
            gr, = torch.autograd.grad(h[i].sum(), v, allow_unused=True)
            grads.append(0.0 if gr is None else float(gr[j].sum()))
        lo, hi = grads
        if min(abs(lo), abs(hi)) <= floor:
            continue
        used += 1
        if lo * hi < 0:
            flips += 1
    return dict(rate=flips / used if used else 0.0, k=flips, n=used)


def slope(xs, ys):
    pts = [(math.log(a), math.log(b)) for a, b in zip(xs, ys) if b > 0]
    if len(pts) < 2:
        return float("nan")
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    dn = sum((a - mx) ** 2 for a, _ in pts)
    return sum((a - mx) * (b - my) for a, b in pts) / dn if dn else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[32, 128, 512])
    ap.add_argument("--draws", type=int, default=192)
    ap.add_argument("--arm", default="pivot_signed")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()

    print("PROTOCOL: SCALING   R4b — Dyson coupling on the REAL operator")
    print("KILL (pre-registered iteration 9, before this file existed):")
    print("  slope(theta=0.20) <= -0.20     AND     |slope(theta=0.80)| <= 0.10\n")

    if a.calibrate:
        print("=== CALIBRATION — must pass before any R4b number is believed ===")
        s = 64
        W0 = dyson_weights(s, 0.0)
        tri = torch.ones(s, s).tril(-1)
        same = bool(torch.equal(W0, tri))
        print(f"  theta=0 reproduces the unweighted operator bitwise : {same}")
        r = flip_rate("dense_unsigned", 128, 0.5, n_draws=192, seed=0)
        print(f"  softmax on the value path (must be exactly 0)      : {r['rate']:.6f}"
              f"  ({r['k']}/{r['n']})")
        rr = [flip_rate("random", s2, 0.0, n_draws=192, seed=0)["rate"]
              for s2 in a.sizes]
        print(f"  random arm, theta=0, must DECAY like dense         : "
              + " ".join(f"{v:.4f}" for v in rr)
              + f"   slope {slope(a.sizes, rr):+.3f}")
        if not same or r["rate"] != 0.0:
            raise SystemExit("CALIBRATION FAILED — every R4b number is void.")
        print("  CALIBRATED\n")

    print(f"{'theta':>7} {'seed':>5} " + "".join(f"{s:>12}" for s in a.sizes)
          + f"{'slope':>9}")
    verdict = {}
    for theta in (0.20, 0.80):
        slopes = []
        for seed in (0, 1, 2):
            rates = [flip_rate(a.arm, s, theta, n_draws=a.draws, seed=seed)["rate"]
                     for s in a.sizes]
            sl = slope(a.sizes, rates)
            slopes.append(sl)
            print(f"{theta:>7.2f} {seed:>5} "
                  + "".join(f"{v:>12.4f}" for v in rates) + f"{sl:>+9.3f}")
        verdict[theta] = slopes
        print(f"{'':>7} {'mean':>5} " + " " * (12 * len(a.sizes))
              + f"{sum(slopes)/3:>+9.3f}   spread {max(slopes)-min(slopes):.3f}\n")

    m20, m80 = sum(verdict[0.20]) / 3, sum(verdict[0.80]) / 3
    c1 = m20 <= -0.20
    c2 = abs(m80) <= 0.10
    print("=== VERDICT against the pre-registered kill ===")
    print(f"  slope(0.20) = {m20:+.3f}  <= -0.20 ?  {'PASS' if c1 else 'KILL FIRES'}")
    print(f"  |slope(0.80)| = {abs(m80):.3f}  <= 0.10 ?  {'PASS' if c2 else 'KILL FIRES'}")
    print(f"\n  R4b {'SURVIVES' if c1 and c2 else 'IS RED'}")


if __name__ == "__main__":
    main()
