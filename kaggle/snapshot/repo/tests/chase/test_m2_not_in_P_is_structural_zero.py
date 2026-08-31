"""Is the M2 `pivot_signed` / `not_in_P` arm a MEASUREMENT or an IDENTITY?

The arm perturbs row `c` of `x` and asks whether the signed influence
`d(h[i].sum())/d(v[j])` flips sign. With

    h = v + A v + (A[:,P] A[P,:]) v,   then  h @ Wo,

`d(h[i].sum())/d(v[j])` collapses to `(delta_ij + A[i,j] + hop2[i,j]) * Wo.sum()`.
`A` here is `_causal_tgate_operator`, which is UNNORMALIZED: `A[i,j]` reads only
`q[i]` and `k[j]`, and `hop2[i,j] = sum_{p in P} A[i,p] A[p,j]` reads only
`q[i]`, `k[j]`, and `q[p], k[p]` for `p in P`. A draw that moves only row `c`
therefore cannot reach the influence at all unless `c in P u {i, j}` -- and the
`not_in_P` pool is exactly `c not in P` with `i`, `j` already excluded.

This file states the property the arm would have IF it were a measurement:
a `not_in_P` draw CAN move the influence. If that assertion is red at exactly
0.0 delta across every draw, the arm's rate is zero by construction, not by
measurement, and the pre-registered "c not in P is also flat" kill reads a
tautology rather than evidence.
"""
import torch

from scale.pivot_probe import build_arm, run_arm, select_pivots

S, D, K, SEED, N_DRAWS = 32, 16, 8, 0, 8


def _draws(placement, n_draws=N_DRAWS, s=S, d=D, k=K, seed=SEED):
    """Mirror of `run_arm`'s draw loop, keeping the per-branch raw numbers.

    Same generator order, same pivot selection, same `c` pool, same operator
    construction (via the shared `build_arm`), so the `c` values drawn here are
    the `c` values `run_arm` draws.
    """
    g = torch.Generator().manual_seed(seed)
    dev = torch.device("cpu")
    rnd = lambda *sh: torch.randn(*sh, generator=g).to(dev)
    i, j = s - 1, max(1, s // 4)          # PROTOCOL: SCALING

    rows = []
    for _ in range(n_draws):
        wq, wk, wo = rnd(d, d), rnd(d, d), rnd(d, d)
        x0, v0 = rnd(s, d), rnd(s, d)
        gvec = torch.sigmoid(rnd(s))
        bet = torch.sigmoid(rnd(s))

        pivots = select_pivots(x0 @ wk, k, exclude=(i, j))
        pset = set(int(p) for p in pivots)
        if placement == "in_P":
            pool = [p for p in pset if p not in (i, j)]
        else:
            pool = [t for t in range(1, s - 1)
                    if t not in pset and t not in (i, j)]
        if not pool:
            continue
        c = pool[int(torch.randint(0, len(pool), (1,), generator=g))]

        infl, aij, h2ij = [], [], []
        for c_val in (rnd(d), rnd(d)):
            x = x0.clone()
            x[c] = c_val
            v = v0.clone().requires_grad_(True)
            qq, kk = x @ wq, x @ wk
            a, hop2 = build_arm("pivot_signed", qq, kk, gvec, bet, pivots,
                                gen=g, device=dev)
            h = (v + a @ v + hop2 @ v) @ wo
            grad, = torch.autograd.grad(h[i].sum(), v, allow_unused=True)
            infl.append(0.0 if grad is None else float(grad[j].sum()))
            aij.append(float(a[i, j]))
            h2ij.append(float(hop2[i, j]))
        rows.append(dict(c=c, i=i, j=j, in_P=c in pset,
                         infl=tuple(infl), aij=tuple(aij), h2ij=tuple(h2ij),
                         d_infl=abs(infl[0] - infl[1]),
                         d_aij=abs(aij[0] - aij[1]),
                         d_h2ij=abs(h2ij[0] - h2ij[1])))
    return rows


def test_a_not_in_P_draw_can_move_the_signed_influence():
    out = _draws("not_in_P")
    inp = _draws("in_P")

    a_max = max(r["d_infl"] for r in out)                       # (A)
    b_max = max(r["d_aij"] for r in out)                        # (B)
    h_max = max(r["d_h2ij"] for r in out)
    c_max = max(r["d_infl"] for r in inp)                       # (C)
    d_dict = run_arm("pivot_signed", S, n_draws=N_DRAWS, k=K,   # (D)
                     placement="not_in_P", seed=SEED, protocol="SCALING")
    e_dict = run_arm("pivot_signed", S, n_draws=N_DRAWS, k=K,   # (E)
                     placement="in_P", seed=SEED, protocol="SCALING")

    report = [
        f"(A) not_in_P  max|delta influence| = {a_max!r}   "
        f"exact zeros {sum(r['d_infl'] == 0.0 for r in out)}/{len(out)}",
        f"(B) not_in_P  max|delta A[i,j]|    = {b_max!r}   "
        f"exact zeros {sum(r['d_aij'] == 0.0 for r in out)}/{len(out)}",
        f"(B') not_in_P max|delta hop2[i,j]| = {h_max!r}   "
        f"exact zeros {sum(r['d_h2ij'] == 0.0 for r in out)}/{len(out)}",
        f"(C) in_P      max|delta influence| = {c_max!r}   "
        f"exact zeros {sum(r['d_infl'] == 0.0 for r in inp)}/{len(inp)}",
        f"(D) run_arm(placement='not_in_P') = {d_dict!r}",
        f"(E) run_arm(placement='in_P')     = {e_dict!r}",
        "per-draw not_in_P (c, i, j, influence lo/hi, |delta|):",
    ] + [f"    c={r['c']:>3} i={r['i']} j={r['j']} in_P={r['in_P']} "
         f"infl={r['infl']!r} |d|={r['d_infl']!r}" for r in out] + [
        "per-draw in_P (c, i, j, influence lo/hi, |delta|):",
    ] + [f"    c={r['c']:>3} i={r['i']} j={r['j']} in_P={r['in_P']} "
         f"infl={r['infl']!r} |d|={r['d_infl']!r}" for r in inp]
    body = "\n".join(report)
    print(body)

    assert a_max > 0.0, (
        "A `not_in_P` draw did not move the signed influence AT ALL -- the two "
        "branches are BITWISE IDENTICAL, so the arm's 0.0 rate is an identity, "
        "not a measurement.\n" + body
    )
