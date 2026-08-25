"""M2 — context-stable signed influence at GLOBAL reach, via pivot routing.

THE DESIGN. Fix the multi-hop PATH COUNT by content-selected pivots rather than
by locality:

    out = v + A v + (A[:,P] A[P,:]) v,     |P| = k fixed, P content-selected

One token's share of the two-hop sum becomes 1/k, INDEPENDENT of s -- the exact
quantity whose 1/s collapse killed every dense arm -- while reach stays global,
which windowing surrendered. Causality is automatic and needs no extra mask:
`(A[:,P] A[P,:])_ij = sum_p A_ip A_pj` is nonzero only when j < p < i, because A
is already strictly lower triangular.

ARSENAL A1 (anti-concentration / Littlewood-Offord) attaches HERE, so its
decomposition is built in rather than bolted on. `sign_flip_rate` IS the
small-ball probability P(|background| < |term_c|), so the exponent should factor:

    rate ~ s^-(a-b)   with   E|term_c| ~ s^-a,   sigma(background) ~ s^-b

The dense prediction is a-b near -1.389 (measured). The pivot prediction is that
a k-term background anti-concentrates as k^-1/2 with NO s dependence, so both a
and b go flat and the rate stops decaying.

PROTOCOL. Every run prints PROTOCOL: SCALING (i=s-1, j=s/4, c=s/2) or PINNED.
These give OPPOSITE answers -- scaling decays about -1.4, pinned is flat over a
64x context growth -- and A1's first attempt was VOID because it ran PINNED
(fixed i=7, j=1, c=4), which holds the 2-hop intermediate count at ~5 regardless
of s. There was no dilution to measure and every exponent read ~0.00 at R^2
0.003. M2 slope claims use SCALING.

THE RIGGING HAZARD, stated because this project has already been bitten by it.
The `c in P` arm needs c to be a pivot. Choosing P so that it CONTAINS c would
be the multizoom artifact again (that harness put c = s//2 into the schedule by
construction, so the result was True at every s tested). Here P is selected by
CONTENT ONLY, never using c; the arm then draws c from whichever tokens landed
in P, and the `c not in P` arm draws from the complement. That makes it a
conditional measurement, not a rigged one -- and the pre-registered kill covers
the remaining hole: if `c not in P` is ALSO flat, the mechanism story is false
even if the pivot number looks good.

KILL NUMBERS (CHECKLIST.md M2, fixed before running, never to be edited):
  * slope(c in P) < -0.3
  * OR `c not in P` is also flat
  * OR any previously published bench.py number moves (G2)
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ceq import bench


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def loglog_slope(xs, ys):
    """Slope over points with a NONZERO rate. Zeros are reported, never clamped:
    clamping turns 'the property is gone' into a finite slope and hides exactly
    the death being looked for."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if y > 0]
    if len(pts) < 2:
        return float("nan"), len(pts)
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    den = sum((a - mx) ** 2 for a, _ in pts)
    num = sum((a - mx) * (b - my) for a, b in pts)
    return (num / den if den else float("nan")), len(pts)


def select_pivots(key: torch.Tensor, k: int, *, exclude=()) -> torch.Tensor:
    """Content-selected pivots. USES ONLY CONTENT -- never `c`, never `i`, `j`.

    Score is the key-norm, a pure function of the token's own representation.
    Deliberately crude: a cleverer selector is a confound at this stage, and the
    point of M2 is whether ROUTING fixes the path count, not whether a selector
    is good. S2/C4 (nucleolus vs top-k salience) is where selection is compared.
    """
    score = key.norm(dim=-1).clone()
    for e in exclude:
        score[e] = float("-inf")
    return torch.topk(score, min(k, int((score > float("-inf")).sum()))).indices


def pivot_hop2(a: torch.Tensor, pivots: torch.Tensor) -> torch.Tensor:
    """A[:,P] @ A[P,:] -- hop 2 restricted to routing through P.

    Causal automatically: nonzero only where j < p < i, since A is strictly
    lower triangular. Rank <= |P| by construction, which IS the mechanism.
    """
    return a[:, pivots] @ a[pivots, :]


def batched_select_pivots(key: torch.Tensor, k: int) -> torch.Tensor:
    """`select_pivots(key[i], k)` for every i, as ONE topk. [n,s,d] -> [n,k'].

    k' is `min(k, s)` -- `select_pivots`'s `min(k, int((score > -inf).sum()))`
    with no exclusions, where the survivor count is exactly `s`. It is NOT
    `min(k, n*s)`: summing the batched score matrix would give that, and would
    part company with the loop whenever k > s.

    NO `exclude`. `select_pivots`'s `exclude` sets entries to -inf and then
    counts survivors, which is per-example bookkeeping this does not replicate;
    the batched form is not offered for that call shape rather than silently
    ignoring the argument.
    """
    score = key.norm(dim=-1)                                   # [n,s]
    return torch.topk(score, min(k, score.shape[-1]), dim=-1).indices


def batched_pivot_hop2(a: torch.Tensor, pivots: torch.Tensor) -> torch.Tensor:
    """`pivot_hop2(a[i], pivots[i])` for every i. a [n,s,s], pivots [n,k].

    Same arithmetic as `pivot_hop2`, with the batch axis carried by tensor ops
    instead of by the interpreter: gather the k pivot COLUMNS along dim 2, the k
    pivot ROWS along dim 1, and replace the n matmuls with one `bmm`.

    WHY IT MATTERS, and it is not the forward. Forward-only the loop is merely
    slow (4-11x). Its BACKWARD is superlinear -- 0.0369 -> 0.6588 -> 24.5587 s
    at n = 128 -> 512 -> 2048 -- because it builds one autograd subgraph PER
    EXAMPLE. The batched form is ~linear (0.0024 -> 0.0156 -> 0.0656), a 374x
    difference at n=2048, and that graph count is the memory cost as well as the
    time cost.

    BITWISE-BOUND against the loop -- `torch.equal`, never `allclose`.
    Gradients are bound to n <= 64; at n = 2048 and 8192 the bind is FORWARD
    ONLY, because the loop's backward there costs ~25 s per call.
    """
    n, s, _ = a.shape
    kp = pivots.shape[-1]
    cols = torch.gather(a, 2, pivots[:, None, :].expand(n, s, kp))   # [n,s,k]
    rows = torch.gather(a, 1, pivots[:, :, None].expand(n, kp, s))   # [n,k,s]
    return torch.bmm(cols, rows)


#: Every arm this probe accepts. Maintained BY HAND, and the distinctness bind
#: in `tests/loop/test_pivot_arms_distinct.py` iterates it: an arm added to the
#: dispatch and forgotten here is exactly the ParaFormer hazard (G3).
ARMS = ("pivot_signed", "dense_signed", "pivot_unsigned", "dense_unsigned",
        "deltanet", "sgate", "random")


def build_arm(kind: str, qq, kk, gvec, bet, pivots, *, gen=None, device=None,
              tau: float = 1.0, rho: float = 1.5, lam: float = 0.10):
    """(A, hop2) for one arm. THE SINGLE SOURCE OF TRUTH for arm construction.

    Extracted so the distinctness bind can exercise the SAME code path the probe
    runs. A test that rebuilds the operators itself proves only that two
    reimplementations agree -- which is how this project published a parity
    number of 89,400.180 against 1.667, and how "alpha=0 is bitwise stock"
    survived until it was checked against the real sdpa path instead of ours.
    """
    s = qq.shape[-2]
    if kind in ("pivot_signed", "dense_signed"):
        a = bench._causal_tgate_operator(qq, kk, gvec, tau)
    elif kind in ("pivot_unsigned", "dense_unsigned"):
        a = bench._softmax_operator(qq, kk)
    elif kind == "deltanet":
        a = bench._causal_deltanet_operator(kk, bet)
    elif kind == "sgate":
        a = bench._causal_sgate_operator(qq, kk, rho=rho, lam=lam)
    elif kind == "random":
        a = (torch.rand(s, s, generator=gen).to(device or qq.device) - 0.5).tril(-1)
    else:
        raise ValueError(kind)
    hop2 = pivot_hop2(a, pivots) if kind.startswith("pivot") else a @ a
    return a, hop2


def run_arm(kind: str, s: int, *, n_draws: int, k: int, d: int = 16,
            placement: str = "in_P", protocol: str = "SCALING",
            seed: int = 0, floor: float = 1e-6, tau: float = 1.0,
            rho: float = 1.5, lam: float = 0.10, device=None,
            wrt: str = "v"):
    """Sign-flip rate plus the A1 small-ball decomposition, for one arm."""
    g = torch.Generator().manual_seed(seed)
    dev = device or torch.device("cpu")
    rnd = lambda *sh: torch.randn(*sh, generator=g).to(dev)

    if protocol == "SCALING":
        i, j = s - 1, max(1, s // 4)
    else:
        i, j = min(7, s - 1), 1

    flips = 0
    terms, bgs, used = [], [], 0
    for _ in range(n_draws):
        wq, wk, wo = rnd(d, d), rnd(d, d), rnd(d, d)
        x0, v0 = rnd(s, d), rnd(s, d)
        gvec = torch.sigmoid(rnd(s))
        bet = torch.sigmoid(rnd(s))

        # ---- pivots and c, both chosen WITHOUT reference to each other -----
        kk0 = x0 @ wk
        pivots = select_pivots(kk0, k, exclude=(i, j))
        pivot_set = set(int(p) for p in pivots)
        cands_in = [p for p in pivot_set if p not in (i, j)]
        cands_out = [t for t in range(1, s - 1)
                     if t not in pivot_set and t not in (i, j)]
        if placement == "in_P":
            pool = cands_in
        elif placement == "not_in_P":
            pool = cands_out
        else:                                   # windowed: adjacent to the query
            pool = [t for t in range(max(1, i - 8), i) if t not in (i, j)]
        if not pool:
            continue
        c = pool[int(torch.randint(0, len(pool), (1,), generator=g))]
        used += 1

        grads, rec = [], None
        for c_val in (rnd(d), rnd(d)):
            x = x0.clone()
            x[c] = c_val
            # WHICH CHANNEL THE GRADIENT IS TAKEN ON.
            #
            # `wrt="v"` is the value path, where a NON-NEGATIVE operator is
            # pinned at exactly zero by theorem: `I + A + A^2` is non-negative
            # entrywise, so no softmax arm can ever flip a sign there. That makes
            # softmax the perfect FLOOR control and a useless ABLATION arm --
            # measured in S2 bucket 1 as k=0 in 4096 draws at four sizes, which
            # is vacuous, not a win.
            #
            # `wrt="x"` is the representation path, where a softmax arm DOES have
            # a sign to lose (the repo's record: 0.025391 / 0.009766 / 0.000000)
            # because `A` itself depends on `x`. It is the only channel on which
            # signed-routed vs unsigned-routed is a real comparison.
            if wrt == "x":
                xx = x.requires_grad_(True)      # assign BEFORE requires_grad_:
                leaf, v = xx, v0                 # in-place on a leaf would raise
            else:
                v = v0.clone().requires_grad_(True)
                leaf = v
            qq, kk = x @ wq, x @ wk
            a, hop2 = build_arm(kind, qq, kk, gvec, bet, pivots, gen=g,
                                device=dev, tau=tau, rho=rho, lam=lam)
            h = v + a @ v + hop2 @ v
            h = h @ wo
            grad, = torch.autograd.grad(h[i].sum(), leaf, allow_unused=True)
            grads.append(0.0 if grad is None else float(grad[j].sum()))

            if rec is None:                     # A1 decomposition, first branch
                if kind.startswith("pivot"):
                    w = a[i, pivots] * a[pivots, j]
                else:
                    w = a[i, :] * a[:, j]
                idx = ((pivots == c).nonzero().flatten()
                       if kind.startswith("pivot")
                       else torch.tensor([c], device=w.device))
                tc = float(w[idx].abs().sum()) if idx.numel() else 0.0
                rec = (tc, float(w.sum()) - float(w[idx].sum()) if idx.numel()
                       else float(w.sum()))
        terms.append(rec[0]); bgs.append(rec[1])
        lo, hi = grads
        if lo * hi < 0 and min(abs(lo), abs(hi)) > floor:
            flips += 1

    if used == 0:
        raise ValueError(
            f"NO DRAWS USED: arm={kind!r} s={s} placement={placement!r} k={k}. "
            f"Every draw hit an empty placement pool, so the reported rate would "
            f"be 0.0 from ZERO measurements. That is not a measurement and must "
            f"not be journalled. (Seen at s=8 with k=8 pivots: after excluding i "
            f"and j there are no non-pivot candidates at all.)"
        )
    n = used
    mean_t = sum(terms) / n if terms else 0.0
    var = (sum(b * b for b in bgs) / n - (sum(bgs) / n) ** 2) if bgs else 0.0
    return dict(rate=flips / n, k=flips, n=used, term=mean_t,
                sigma=math.sqrt(max(var, 0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[32, 128, 512])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--protocol", default="SCALING", choices=["SCALING", "PINNED"])
    ap.add_argument("--arms", nargs="+",
                    default=["pivot_signed", "dense_signed", "pivot_unsigned",
                             "deltanet", "sgate", "random"])
    a = ap.parse_args()

    print(f"PROTOCOL: {a.protocol}"
          + ("  (i=s-1, j=s/4, c drawn from the placement pool)"
             if a.protocol == "SCALING" else "  (fixed offsets)"))
    print(f"M2 pivot probe. k={a.k} pivots, content-selected by key-norm, "
          f"never using c. n={a.n} draws.")
    print("KILL (CHECKLIST M2, pre-registered): slope(c in P) < -0.3, "
          "OR c-not-in-P also flat, OR any published bench number moves.\n")

    rows = {}
    for kind in a.arms:
        placement = "in_P" if kind.startswith("pivot") else "not_in_P"
        cells, rates = [], []
        for s in a.sizes:
            r = run_arm(kind, s, n_draws=a.n, k=a.k, placement=placement,
                        protocol=a.protocol)
            lo, hi = clopper_pearson(r["k"], max(r["n"], 1))
            rates.append(r["rate"])
            cells.append(f"{r['rate']:.4f}[{lo:.3f},{hi:.3f}]")
        sl, npts = loglog_slope(a.sizes, rates)
        rows[kind] = (rates, sl)
        note = "" if npts == len(a.sizes) else f" ({npts}/{len(a.sizes)} nonzero)"
        print(f"{kind:>15} " + "".join(f"{c:>22}" for c in cells)
              + f"  slope {sl:+.3f}{note}")

    print("\nCONTROLS (the run is void if these misread):")
    if "random" in rows:
        print(f"  random  slope {rows['random'][1]:+.3f}  rates {rows['random'][0]}")
    print("  an exact 0.0000 or 1.0000 anywhere means broken until a "
          "known-truth synthetic passes")
    return rows


if __name__ == "__main__":
    main()
