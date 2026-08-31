"""G7 — does the arm change the EVENT, or only the STATISTIC? Both arms, pre-build.

This is the test that struck R5 and R8 at **0/20000** before they cost anything.
Dividing both sides of `|t_c| > |Σ ε_p t_p|` by the same positive scalar leaves
the event set invariant, so a route that only rescales is a provable no-op that
would have read as a clean null after being built. R8 was scheduled FIRST on
grounds of being cheap.

ARM B (discrepancy-steered signs) is the one that matters here, and its answer is
NOT obvious. Choosing `ε` to minimise `|Σ_{p≠c} ε_p t_p|` certainly changes the
background MAGNITUDE. Whether it changes the EVENT is a different question: a
discrepancy pass that shrinks an already-losing background buys nothing, because
the event was already decided.

At k=8 the minimisation is EXACT by exhaustive enumeration over 2^8 = 256 sign
patterns. No Lovett–Meka approximation is needed at this size, so the reference
assignment here is optimal rather than merely constructive, and any later
approximate pass can be measured against it.

ARM A (difference-set schedule) gets its birth gate as a VALUE: a cyclic Singer
(v,k,1)-difference set has every nonzero residue mod v appearing exactly once as
a difference, so `|D−D| = v−1` is an identity to assert, not a property to hope
for.

F17 IS OBSERVED: every reading states its LOGIT SCALE. The same operator read
frustration 0.2779 at unit scale and 0.000000 at harness scale, so a sign
measurement without its scale is not a measurement. Both scales are reported.
"""
from __future__ import annotations

import itertools
import math
import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ceq import bench                                          # noqa: E402
from scale.pivot_probe import select_pivots                    # noqa: E402

K = 8
DRAWS = 3000

#: Cyclic Singer (v, k, 1)-difference sets. Every nonzero residue mod v occurs
#: exactly once as a difference d_i - d_j, so |D-D| = v-1 EXACTLY.
SINGER = {
    7:  [1, 2, 4],
    13: [0, 1, 3, 9],
    21: [0, 1, 4, 14, 16],
    31: [1, 5, 11, 24, 25, 27],
    57: [0, 1, 3, 13, 32, 36, 43, 52],
}


def difference_cover(v: int, D: list[int]) -> int:
    """|D - D| over nonzero residues mod v. Must equal v-1 for a (v,k,1) set."""
    return len({(a - b) % v for a in D for b in D if (a - b) % v != 0})


def best_signs(t: torch.Tensor) -> torch.Tensor:
    """argmin over eps in {-1,+1}^k of |sum eps_p t_p|. EXACT at k=8 (256 patterns)."""
    k = t.numel()
    best, arg = None, None
    for bits in itertools.product((-1.0, 1.0), repeat=k):
        e = torch.tensor(bits, dtype=t.dtype)
        v = abs(float((e * t).sum()))
        if best is None or v < best:
            best, arg = v, e
    return arg


def g7_arm_b(scale: float, seed: int = 0) -> dict:
    """Event-change rate when the background signs are CHOSEN rather than natural."""
    g = torch.Generator().manual_seed(seed)
    changed = n = 0
    nat_bg = cho_bg = 0.0
    for _ in range(DRAWS):
        s = 64
        x = torch.randn(s, 16, generator=g) * scale
        wq = torch.randn(16, 16, generator=g) / 4.0
        wk = torch.randn(16, 16, generator=g) / 4.0
        q, k = x @ wq, x @ wk
        a = bench._causal_sgate_operator(q, k, rho=1.5, lam=1.00)
        i, j = s - 1, s // 4
        piv = select_pivots(k, K, exclude=(i, j))
        w = a[i, piv] * a[piv, j]                    # the hop-2 terms t_p
        if w.numel() < 3 or float(w.abs().sum()) == 0.0:
            continue
        n += 1
        c = 0                                        # c is the first pivot
        t_c = float(w[c])
        rest = torch.cat([w[:c], w[c + 1:]])
        nat = float(rest.sum())                      # natural signs, as the operator gives them
        eps = best_signs(rest.abs() * torch.sign(rest))
        cho = float((eps * rest.abs()).sum())        # chosen signs on the same magnitudes
        nat_bg += abs(nat); cho_bg += abs(cho)
        if (abs(t_c) > abs(nat)) != (abs(t_c) > abs(cho)):
            changed += 1
    return dict(n=n, changed=changed, rate=changed / n if n else float("nan"),
                nat_bg=nat_bg / n if n else float("nan"),
                cho_bg=cho_bg / n if n else float("nan"))


def main() -> int:
    torch.set_num_threads(2)
    print("G7 -- EVENT change, not statistic change. Threshold: >= 1% or the arm is a no-op.\n")

    print("=== ARM A birth gate: |D-D| = v-1 as a VALUE, not a hope ===")
    ok = True
    for v, D in SINGER.items():
        cov = difference_cover(v, D)
        good = cov == v - 1
        ok &= good
        print(f"  v={v:<4} k={len(D)}  D={D}   |D-D|={cov:<4} v-1={v-1:<4} {'OK' if good else 'FAIL'}")
    print(f"  all difference sets cover exactly: {ok}\n")

    print("=== ARM B: G7 event-change from CHOOSING the background signs ===")
    print("    (F17: logit scale stated. lam=1.00, where the operator is genuinely signed.)")
    print(f"  {'scale':>18} {'n':>6} {'changed':>8} {'event-change':>13} "
          f"{'E|bg| natural':>14} {'E|bg| chosen':>13} {'Spencer 6sqrt(k)':>17}")
    spencer = 6.0 * math.sqrt(K)
    for name, sc in (("harness (x0.1)", 0.1), ("unit", 1.0)):
        r = g7_arm_b(sc)
        print(f"  {name:>18} {r['n']:>6} {r['changed']:>8} {r['rate']:>12.4%} "
              f"{r['nat_bg']:>14.6e} {r['cho_bg']:>13.6e} {spencer:>17.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
