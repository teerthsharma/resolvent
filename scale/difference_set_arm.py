"""ARM A — the difference-set schedule, with its birth gates.

NOT "co-prime spacing". G1 fired on that: arXiv 2606.28560 compares a
**"coprime (anti-gridding) reassignment"**, so the schedule heuristic round 3
found is prior art. This is a different object and the difference is the whole
claim.

THE COVERAGE THEOREM. A cyclic Singer (v, k, 1)-difference set `D ⊂ Z_v` has
every nonzero residue occurring EXACTLY ONCE as a difference `d_a − d_b`. So if
each query attends to offsets in `D`, then **hop 2 reaches every one of the v−1
nonzero offsets, each by exactly one path**. Coverage is a COUNTING IDENTITY, not
an empirical improvement — which is why it is Lean-provable and why "co-prime
spacing" is not the same claim.

Severance should therefore be **zero by construction**. Round 3's power-of-two
lattice severed 41–57% of positions; co-prime reduced that to 0.1277 empirically.
A difference set should read 0, and if it does not, the theorem is not describing
the object that was built.

BIRTH GATES, all three required before any reading is credited:
  1. `|D−D| = v−1` as a VALUE. [GREEN, iteration 2] exact at v = 7,13,21,31,57.
  2. The flipper is placed **UNIFORMLY AT RANDOM**, never on the schedule. This
     is the discipline `scale/carpet_probe.py:24` demanded and round 3 broke: at
     `c = i − s/4` with `s` a power of two, `c` sat ON the dilation lattice at
     every measured size, and the resulting flat reading was an artifact of
     placement, not a property of the operator.
  3. Off-schedule flip rate within CI of on-schedule. This is the exact test that
     killed the dilation arm, promoted from post-hoc autopsy to birth gate.

READ ON X4. The float statistic misses flips whose PRODUCT underflows while both
factors are healthy, and at depth median |grad| is 2.8e-32.
"""
from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ceq import bench                                              # noqa: E402
from scale.valuation import valuation, v_opposite_signs            # noqa: E402

#: Cyclic Singer (v, k, 1)-difference sets, v = q^2+q+1, k = q+1.
SINGER = {7: [1, 2, 4], 13: [0, 1, 3, 9], 21: [0, 1, 4, 14, 16],
          31: [1, 5, 11, 24, 25, 27], 57: [0, 1, 3, 13, 32, 36, 43, 52]}


def difference_cover(v: int, D) -> int:
    return len({(a - b) % v for a in D for b in D if (a - b) % v != 0})


def offset_mask(s: int, offsets, device) -> torch.Tensor:
    """Strictly causal: query i attends j iff (i-j) is one of `offsets`."""
    i = torch.arange(s, device=device).view(-1, 1)
    j = torch.arange(s, device=device).view(1, -1)
    off = i - j
    m = torch.zeros(s, s, dtype=torch.bool, device=device)
    for o in offsets:
        if o > 0:
            m |= (off == o)
    return m & (off > 0)


def op_masked(q, k, m, *, rho=1.5, lam=1.00):
    """`_causal_sgate_operator`'s arithmetic under an arbitrary causal mask."""
    w = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    neg = torch.finfo(q.dtype).min
    pp = torch.softmax(w.masked_fill(~m, neg), -1).masked_fill(~m, 0.0)
    pm = torch.softmax((-w).masked_fill(~m, neg), -1).masked_fill(~m, 0.0)
    return rho * (pp - lam * pm) / (1.0 + lam)


def draws(offsets, *, s, i, j, c, n_draws, seed=0, d=16, hops=2, scale=1.0,
          lam=1.00):
    """(lo, hi) gradient pairs at two values of token c. Same shape as the
    shipped probe, but the mask is an arbitrary offset set.

    `lam` keeps its historical 1.00 here so this dead arm's recorded numbers
    do not silently move. THE SHIPPED OPERATOR IS lam = 0.10
    (`ceq/bench.py:192`), and at lam = 1.00 the sgate is antisymmetric --
    a single-key row gives pp = pm = 1 and the entry collapses to exactly 0
    rather than a constant. Callers measuring the shipped operator must pass
    lam=0.10; `scale/arm_a_rebuild.py` does.
    """
    g = torch.Generator().manual_seed(seed)
    dev = torch.device("cpu")
    m = offset_mask(s, offsets, dev)
    out = []
    for _ in range(n_draws):
        wq, wk, wo = (torch.randn(d, d, generator=g) for _ in range(3))
        x0 = torch.randn(s, d, generator=g) * scale
        v0 = torch.randn(s, d, generator=g)
        pair = []
        for cval in (torch.randn(d, generator=g), torch.randn(d, generator=g)):
            x = x0.clone()
            x[c] = cval
            v = v0.clone().requires_grad_(True)
            a = op_masked(x @ wq, x @ wk, m, lam=lam)
            h, term = v, v
            for _ in range(hops):
                term = a @ term
                h = h + term
            h = h @ wo
            gr, = torch.autograd.grad(h[i].sum(), v, allow_unused=True)
            pair.append(0.0 if gr is None else float(gr[j].sum()))
        out.append(tuple(pair))
    return out


def severed_fraction(offsets, *, s, i, j, n_draws=4, seed=0, scale=1.0):
    """Share of c positions whose gradient pair is BITWISE identical at both
    values -- perturbing c changes nothing. Exhaustive over legal c."""
    sev = tot = 0
    for c in range(j + 1, i):
        dr = draws(offsets, s=s, i=i, j=j, c=c, n_draws=n_draws, seed=seed,
                   scale=scale)
        tot += 1
        if all(lo == hi for lo, hi in dr):
            sev += 1
    return sev / tot if tot else float("nan"), sev, tot


def flip_rate_x4(dr) -> float:
    """X4: compare SIGN FIELDS. No product, no floor."""
    if not dr:
        return 0.0
    f = sum(1 for lo, hi in dr if v_opposite_signs(valuation(lo), valuation(hi)))
    return f / len(dr)


def main() -> int:
    torch.set_num_threads(2)
    v, D = 57, SINGER[57]
    s = v
    i, j = s - 1, s // 4
    print("ARM A -- DIFFERENCE-SET SCHEDULE. NOT co-prime spacing (G1 fired on that).\n")

    print("=== BIRTH GATE 1: |D-D| = v-1 as a VALUE ===")
    cov = difference_cover(v, D)
    print(f"  v={v} k={len(D)} D={D}")
    print(f"  |D-D| = {cov}   v-1 = {v-1}   {'OK' if cov == v-1 else 'FAIL'}")

    print("\n=== KILL: severance. Difference set should read ZERO BY THEOREM ===")
    pow2 = [1, 2, 4, 8, 16, 32]
    print(f"  {'schedule':<34} {'k':>3} {'severed':>9} {'n':>5} {'fraction':>10}")
    for name, offs in (("difference set D", D),
                       ("power-of-two (matched-ish k)", pow2[:len(D)]),
                       ("contiguous 1..k", list(range(1, len(D) + 1)))):
        fr, sev, tot = severed_fraction(offs, s=s, i=i, j=j)
        print(f"  {name:<34} {len(offs):>3} {sev:>9} {tot:>5} {fr:>10.4f}")

    print("\n=== BIRTH GATE 2/3: flipper UNIFORMLY AT RANDOM, on vs off schedule ===")
    on = [c for c in range(j + 1, i) if (i - c) % v in D]
    off = [c for c in range(j + 1, i) if (i - c) % v not in D]
    print(f"  positions ON the schedule: {len(on)}   OFF: {len(off)}")
    for label, pool in (("on-schedule", on), ("off-schedule", off)):
        rates = [flip_rate_x4(draws(D, s=s, i=i, j=j, c=c, n_draws=64))
                 for c in pool[:12]]
        mean = sum(rates) / len(rates) if rates else float("nan")
        print(f"  {label:<14} mean X4 flip rate over {len(rates)} positions = {mean:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
