"""Mechanistic attack probe on the M2 pivot mechanism.

Replays scale.pivot_probe.run_arm's EXACT draw loop and RNG stream (same
torch.Generator().manual_seed(seed), same order of rnd() calls, same
select_pivots, same c-selection) but instead of collapsing to a flip rate,
returns the raw scalars behind it for every draw:

    a_ij   = A[i, j]                                  (one-hop term)
    bg     = sum_{p in P, p != c} A[i,p] * A[p,j]      (two-hop background)
    t1, t2 = A[i,c] * A[c,j] at the first/second draw of token c
    lo, hi = the actual autograd pair run_arm computes

See scale/pivot_probe.py (run_arm, build_arm) and
ceq/bench.py::_causal_tgate_operator for what these quantities mean.
"""
from __future__ import annotations

import pathlib
import statistics
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scale.pivot_probe import select_pivots, build_arm, run_arm, pivot_hop2
from ceq import bench


def draws(kind: str, s: int, *, n_draws: int, k: int, d: int = 16,
          placement: str = "in_P", protocol: str = "SCALING",
          seed: int = 0, tau: float = 1.0, rho: float = 1.5,
          lam: float = 0.10, device=None, fast: bool = False,
          pivots_fn=None) -> list:
    """One dict per draw USED (empty-pool draws are skipped, exactly like
    run_arm). Fields: i, j, c, wo_sum, a_ij, bg, t1, t2, lo, hi, hop2_lo,
    hop2_hi. `hop2_lo`/`hop2_hi` are hop2[i,j] read directly off the two
    branches' own hop2 matrix -- kept independent of the (bg + t1)/(bg + t2)
    reconstruction so CHECK 1 and CHECK 2 test different code paths.

    kind: "pivot_signed" -- P is the k content-selected pivots (hop2 = A[:,P]@A[P,:]).
          "dense_signed"  -- P is the full index range (hop2 = A@A), but pivots
          are still selected and c is still drawn from the pivot set, matching
          the journal's `dense_signed__at_pivots` arm.
          "pivot_sgate"/"dense_sgate" -- same P convention as the two arms
          above, but A = ceq.bench._causal_sgate_operator(qq, kk, rho=rho,
          lam=lam), the DENOMINATOR-CARRYING operator (softmax difference),
          instead of the denominator-free tgate operator build_arm uses for
          the "*_signed" kinds. build_arm (scale/pivot_probe.py) has no
          "pivot_sgate"/"dense_sgate" case, so these two kinds bypass it and
          build (A, hop2) inline; every other kind's call to build_arm, and
          the RNG stream up to that call, is untouched.
          "pivot_tgatex"/"dense_tgatex" -- same P convention as
          "pivot_signed"/"dense_signed", but A =
          ceq.bench._causal_tgate_operator(qq, kk, gvec, tau,
          static_scale=True), the UNMEASURED tgate variant that divides row i
          by sqrt(#visible keys) -- a fixed function of position only, no sum
          over content. build_arm has no "pivot_tgatex"/"dense_tgatex" case
          either, so these two kinds also bypass it and build (A, hop2)
          inline, same as the sgate pair.

    pivots_fn: optional `(kk0, k, exclude) -> LongTensor`, called in place of
    `scale.pivot_probe.select_pivots` so a caller can swap in a different pivot
    chooser (e.g. content-blind random pivots) for an A/B comparison. Default
    `None` reproduces the original select_pivots call exactly, and does not
    touch the shared generator `g` (neither select_pivots nor a well-behaved
    pivots_fn should), so the RNG call order -- and every number this module
    published before pivots_fn existed -- is unchanged.
    """
    if kind not in ("pivot_signed", "dense_signed", "pivot_sgate", "dense_sgate",
                    "pivot_tgatex", "dense_tgatex"):
        raise ValueError(kind)
    g = torch.Generator().manual_seed(seed)
    dev = device or torch.device("cpu")
    rnd = lambda *sh: torch.randn(*sh, generator=g).to(dev)

    if protocol == "SCALING":
        i, j = s - 1, max(1, s // 4)
    else:
        i, j = min(7, s - 1), 1

    out = []
    for _ in range(n_draws):
        wq, wk, wo = rnd(d, d), rnd(d, d), rnd(d, d)
        x0, v0 = rnd(s, d), rnd(s, d)
        gvec = torch.sigmoid(rnd(s))
        bet = torch.sigmoid(rnd(s))

        kk0 = x0 @ wk
        pivots = (select_pivots(kk0, k, exclude=(i, j)) if pivots_fn is None
                  else pivots_fn(kk0, k, exclude=(i, j)))
        pivot_set = set(int(p) for p in pivots)
        cands_in = [p for p in pivot_set if p not in (i, j)]
        cands_out = [t for t in range(1, s - 1)
                     if t not in pivot_set and t not in (i, j)]
        if placement == "in_P":
            pool = cands_in
        elif placement == "not_in_P":
            pool = cands_out
        else:
            pool = [t for t in range(max(1, i - 8), i) if t not in (i, j)]
        if not pool:
            continue
        c = pool[int(torch.randint(0, len(pool), (1,), generator=g))]

        P = (pivots if kind in ("pivot_signed", "pivot_sgate", "pivot_tgatex")
             else torch.arange(s, device=dev))
        wo_sum = float(wo.sum())

        branch = {}
        for name, c_val in zip(("lo", "hi"), (rnd(d), rnd(d))):
            x = x0.clone()
            x[c] = c_val
            v = v0.clone()
            qq, kk = x @ wq, x @ wk
            if kind in ("pivot_sgate", "dense_sgate"):
                a = bench._causal_sgate_operator(qq, kk, rho=rho, lam=lam)
                hop2 = pivot_hop2(a, pivots) if kind == "pivot_sgate" else a @ a
            elif kind in ("pivot_tgatex", "dense_tgatex"):
                a = bench._causal_tgate_operator(qq, kk, gvec, tau, static_scale=True)
                hop2 = pivot_hop2(a, pivots) if kind == "pivot_tgatex" else a @ a
            else:
                a, hop2 = build_arm(kind, qq, kk, gvec, bet, pivots, gen=g,
                                    device=dev, tau=tau, rho=rho, lam=lam)

            a_ij = float(a[i, j])
            in_p = bool((P == c).any())
            t = float(a[i, c] * a[c, j]) if in_p else 0.0
            bg = float((a[i, P] * a[P, j]).sum()) - t

            if fast:
                grad_sum = None
            else:
                v.requires_grad_(True)
                h = v + a @ v + hop2 @ v
                h = h @ wo
                grad, = torch.autograd.grad(h[i].sum(), v, allow_unused=True)
                grad_sum = 0.0 if grad is None else float(grad[j].sum())

            branch[name] = dict(grad=grad_sum, a_ij=a_ij, bg=bg, t=t,
                                hop2_ij=float(hop2[i, j]))

        if fast:
            lo = (branch["lo"]["a_ij"] + branch["lo"]["bg"] + branch["lo"]["t"]) * wo_sum
            hi = (branch["lo"]["a_ij"] + branch["lo"]["bg"] + branch["hi"]["t"]) * wo_sum
        else:
            lo, hi = branch["lo"]["grad"], branch["hi"]["grad"]

        out.append(dict(
            i=i, j=j, c=c, wo_sum=wo_sum,
            a_ij=branch["lo"]["a_ij"], bg=branch["lo"]["bg"],
            t1=branch["lo"]["t"], t2=branch["hi"]["t"],
            lo=lo, hi=hi,
            hop2_lo=branch["lo"]["hop2_ij"], hop2_hi=branch["hi"]["hop2_ij"],
        ))
    return out


def _max_dev(pred, actual):
    diffs = [abs(p - a) for p, a in zip(pred, actual)]
    rels = [d / max(abs(a), 1e-12) for d, a in zip(diffs, actual)]
    return max(diffs), max(rels)


def main():
    t0 = time.time()
    kind, s, k, n_draws, seed = "pivot_signed", 128, 8, 64, 0
    rec = draws(kind, s, n_draws=n_draws, k=k, placement="in_P",
                protocol="SCALING", seed=seed)
    assert len(rec) > 0, "no draws used"
    n = len(rec)

    # CHECK 1: grad[j].sum() == (A[i,j] + hop2[i,j]) * wo.sum(), both branches
    pred1, actual1 = [], []
    for r in rec:
        pred1.append((r["a_ij"] + r["hop2_lo"]) * r["wo_sum"]); actual1.append(r["lo"])
        pred1.append((r["a_ij"] + r["hop2_hi"]) * r["wo_sum"]); actual1.append(r["hi"])
    abs1, rel1 = _max_dev(pred1, actual1)
    ok1 = abs1 < 1e-3 and rel1 < 1e-3
    print(f"CHECK 1 (grad == (A[i,j]+hop2[i,j])*wo.sum()): "
          f"max_abs_dev={abs1:.6e} max_rel_dev={rel1:.6e} "
          f"n_compared={len(pred1)} -> {'PASS' if ok1 else 'FAIL'}")
    assert ok1, f"CHECK 1 FAILED: abs={abs1} rel={rel1}"

    # CHECK 2: lo/hi == (a_ij + bg + t1/t2) * wo.sum()
    pred2, actual2 = [], []
    for r in rec:
        pred2.append((r["a_ij"] + r["bg"] + r["t1"]) * r["wo_sum"]); actual2.append(r["lo"])
        pred2.append((r["a_ij"] + r["bg"] + r["t2"]) * r["wo_sum"]); actual2.append(r["hi"])
    abs2, rel2 = _max_dev(pred2, actual2)
    ok2 = abs2 < 1e-3 and rel2 < 1e-3
    print(f"CHECK 2 (offset reconstruction): "
          f"max_abs_dev={abs2:.6e} max_rel_dev={rel2:.6e} "
          f"n_compared={len(pred2)} -> {'PASS' if ok2 else 'FAIL'}")
    assert ok2, f"CHECK 2 FAILED: abs={abs2} rel={rel2}"

    # CHECK 3: flip-rate parity against run_arm, on the SAME RNG stream
    floor = 1e-6
    flips = sum(
        1 for r in rec
        if (r["a_ij"] + r["bg"] + r["t1"]) * (r["a_ij"] + r["bg"] + r["t2"]) < 0
        and min(abs(r["lo"]), abs(r["hi"])) > floor
    )
    recomputed_rate = flips / n
    ref = run_arm(kind, s, n_draws=n_draws, k=k, placement="in_P",
                  protocol="SCALING", seed=seed)
    ok3 = recomputed_rate == ref["rate"] and n == ref["n"]
    print(f"CHECK 3 (flip-rate parity): recomputed_rate={recomputed_rate!r} "
          f"run_arm_rate={ref['rate']!r} n_draws_used={n} "
          f"run_arm_n={ref['n']} -> {'PASS' if ok3 else 'FAIL'}")
    assert ok3, f"CHECK 3 FAILED: {recomputed_rate} != {ref['rate']}"

    # CHECK 4: fast=True (no-autograd) path matches fast=False EXACTLY
    def flip_rate(recs):
        flips_ = sum(
            1 for r in recs
            if (r["a_ij"] + r["bg"] + r["t1"]) * (r["a_ij"] + r["bg"] + r["t2"]) < 0
            and min(abs(r["lo"]), abs(r["hi"])) > floor
        )
        return flips_ / len(recs)

    rec_slow = draws(kind, s, n_draws=n_draws, k=k, placement="in_P",
                      protocol="SCALING", seed=seed, fast=False)
    rec_fast = draws(kind, s, n_draws=n_draws, k=k, placement="in_P",
                      protocol="SCALING", seed=seed, fast=True)
    assert len(rec_slow) == len(rec_fast), "fast/slow draw counts differ"
    rate_slow, rate_fast = flip_rate(rec_slow), flip_rate(rec_fast)
    max_lo_dev = max(abs(a["lo"] - b["lo"]) for a, b in zip(rec_slow, rec_fast))
    max_hi_dev = max(abs(a["hi"] - b["hi"]) for a, b in zip(rec_slow, rec_fast))
    ok4 = rate_slow == rate_fast
    print(f"CHECK 4 (fast-path parity): rate_slow={rate_slow!r} rate_fast={rate_fast!r} "
          f"max_abs_dev_lo={max_lo_dev:.6e} max_abs_dev_hi={max_hi_dev:.6e} "
          f"n={len(rec_slow)} -> {'PASS' if ok4 else 'FAIL'}")
    assert ok4, f"CHECK 4 FAILED: {rate_slow} != {rate_fast}"

    # CHECK 5: the CHECK-1 identity, grad[j].sum() == (A[i,j]+hop2[i,j])*wo.sum(),
    # on the denominator-carrying pivot_sgate operator (fast=False, s=128, k=8,
    # n_draws=32, seed=0). Reported plainly either way -- PASS or FAIL.
    rec5 = draws("pivot_sgate", 128, n_draws=32, k=8, placement="in_P",
                 protocol="SCALING", seed=0, fast=False)
    n5 = len(rec5)
    pred5, actual5 = [], []
    for r in rec5:
        pred5.append((r["a_ij"] + r["hop2_lo"]) * r["wo_sum"]); actual5.append(r["lo"])
        pred5.append((r["a_ij"] + r["hop2_hi"]) * r["wo_sum"]); actual5.append(r["hi"])
    abs5, rel5 = _max_dev(pred5, actual5)
    ok5 = abs5 < 1e-3 and rel5 < 1e-3
    print(f"CHECK 5 (grad == (A[i,j]+hop2[i,j])*wo.sum(), kind=pivot_sgate, "
          f"s=128, k=8, n_draws=32, seed=0): max_abs_dev={abs5:.6e} "
          f"max_rel_dev={rel5:.6e} n_compared={len(pred5)} n_draws_used={n5} "
          f"-> {'PASS' if ok5 else 'FAIL'}")

    # CHECK 6: the CHECK-1 identity, grad[j].sum() == (A[i,j]+hop2[i,j])*wo.sum(),
    # on kind="pivot_tgatex" (fast=False, s=128, k=8, n_draws=32, seed=0).
    # Reported plainly either way -- PASS or FAIL.
    rec6 = draws("pivot_tgatex", 128, n_draws=32, k=8, placement="in_P",
                 protocol="SCALING", seed=0, fast=False)
    n6 = len(rec6)
    pred6, actual6 = [], []
    for r in rec6:
        pred6.append((r["a_ij"] + r["hop2_lo"]) * r["wo_sum"]); actual6.append(r["lo"])
        pred6.append((r["a_ij"] + r["hop2_hi"]) * r["wo_sum"]); actual6.append(r["hi"])
    abs6, rel6 = _max_dev(pred6, actual6)
    ok6 = abs6 < 1e-3 and rel6 < 1e-3
    print(f"CHECK 6 (grad == (A[i,j]+hop2[i,j])*wo.sum(), kind=pivot_tgatex, "
          f"s=128, k=8, n_draws=32, seed=0): max_abs_dev={abs6:.6e} "
          f"max_rel_dev={rel6:.6e} n_compared={len(pred6)} n_draws_used={n6} "
          f"-> {'PASS' if ok6 else 'FAIL'}")

    # summary statistics over the 64 draws
    a_ij_vals = [r["a_ij"] for r in rec]
    bg_vals = [r["bg"] for r in rec]
    t1_vals = [r["t1"] for r in rec]
    t1_minus_t2 = [r["t1"] - r["t2"] for r in rec]
    sum_vals = [r["a_ij"] + r["bg"] for r in rec]
    abs_a_ij = [abs(x) for x in a_ij_vals]
    abs_bg = [abs(x) for x in bg_vals]

    print(f"\nSummary over {n} draws:")
    print(f"  mean|a_ij|={statistics.mean(abs_a_ij):.6f} "
          f"std(a_ij)={statistics.pstdev(a_ij_vals):.6f} "
          f"median|a_ij|={statistics.median(abs_a_ij):.6f}")
    print(f"  mean|bg|={statistics.mean(abs_bg):.6f} "
          f"std(bg)={statistics.pstdev(bg_vals):.6f} "
          f"median|bg|={statistics.median(abs_bg):.6f}")
    print(f"  mean|t1|={statistics.mean([abs(x) for x in t1_vals]):.6f} "
          f"mean|t1-t2|={statistics.mean([abs(x) for x in t1_minus_t2]):.6f}")
    print(f"  std(a_ij+bg)={statistics.pstdev(sum_vals):.6f} "
          f"std(bg)={statistics.pstdev(bg_vals):.6f}")
    frac = sum(1 for x, y in zip(abs_a_ij, abs_bg) if x > y) / n
    print(f"  frac(|a_ij| > |bg|)={frac:.4f}")

    print(f"\nwall_clock_seconds={time.time() - t0:.3f}")


if __name__ == "__main__":
    main()
