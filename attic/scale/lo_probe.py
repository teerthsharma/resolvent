"""Littlewood-Offord decomposition of the -1.389 exponent, and the KPZ
localization test of the 1/s dilution premise.

THE OPEN ITEM. Measured decay of the content-conditional sign property:
slope -1.389, R^2 0.9938. The naive share argument predicts -1. The extra -0.389
has never been explained.

THE DECOMPOSITION. The mechanism identified was: the first term of J = sum A^k
containing a path through a third token is k=2, which sums over ~s intermediates.
Write that term as

    J_ij ~ (A^2)_ij = A_ic A_cj  +  sum_{m != c} A_im A_mj
                      \_________/   \____________________/
                        term_c              S_notc

A sign flip needs term_c to overcome the background S_notc. That is exactly an
anti-concentration (small-ball) probability for a signed random sum:

    rate ~ P(|S_notc| < |term_c|) ~ E|term_c| / sigma(S_notc)      (for small ball)

so the exponent FACTORS:   rate ~ s^-(a - b)   where
    E|term_c| ~ s^-a        (the share argument's contribution)
    sigma(S_notc) ~ s^-b    (the background's concentration)

Candidate laws: share-only gives -1. Share x CLT small-ball gives -3/2.
Log-correlated / heavy-tail corrections pull toward -4/3 = -1.333. The measured
-1.389 sits BETWEEN -4/3 and -3/2, so this run decides which law it is -- or
shows the exponent does not factor, which is itself the finding.

THE KPZ TEST. The dilution premise is that hop mass spreads over ~s intermediates
so one token's share vanishes like 1/s. Directed-polymer/KPZ theory predicts mass
LOCALIZES on O(s^{2/3}) corridors instead. Measured by the participation ratio

    PR = (sum_m |w_m|)^2 / sum_m w_m^2,     w_m = A_im A_mj

PR ~ s  means uniform spreading (premise holds).
PR ~ s^{2/3} means KPZ localization (premise is WRONG and the decay has a floor).
"""
from __future__ import annotations
import sys, math, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ceq import bench


def loglog_slope(xs, ys):
    lx = [math.log(x) for x in xs]; ly = [math.log(y) for y in ys]
    mx = sum(lx) / len(lx); my = sum(ly) / len(ly)
    den = sum((a - mx) ** 2 for a in lx)
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    r2n = num * num
    r2d = den * sum((b - my) ** 2 for b in ly)
    return num / den, (r2n / r2d if r2d > 0 else float("nan"))


def measure(kind, s, n_draws, d=16, i=7, j=1, c=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *sh: torch.randn(*sh, generator=g)
    tc, bg, hits, prs = [], [], 0, []
    for _ in range(n_draws):
        wq, wk = rnd(d, d), rnd(d, d)
        x = rnd(s, d)
        q, k = x @ wq, x @ wk
        if kind == "sgate":
            A = bench._causal_sgate_operator(q, k, rho=1.5, lam=0.10)
        elif kind == "tgate":
            A = bench._causal_tgate_operator(q, k, torch.sigmoid(rnd(s)), 1.0)
        elif kind == "deltanet":
            A = bench._causal_deltanet_operator(k, torch.sigmoid(rnd(s)))
        else:
            A = bench._softmax_operator(q, k)
        w = A[i, :] * A[:, j]                 # per-intermediate 2-hop weights
        w[i] = 0.0; w[j] = 0.0
        term_c = float(w[c])
        s_notc = float(w.sum() - w[c])
        tc.append(abs(term_c)); bg.append(s_notc)
        if abs(term_c) > abs(s_notc):         # term_c can overcome the background
            hits += 1
        aw = w.abs()
        l1, l2 = float(aw.sum()), float((aw ** 2).sum())
        if l2 > 0:
            prs.append(l1 * l1 / l2)
    n = len(bg)
    mean_tc = sum(tc) / n
    var = sum(v * v for v in bg) / n - (sum(bg) / n) ** 2
    sigma = math.sqrt(max(var, 0.0))
    return dict(term=mean_tc, sigma=sigma, small_ball=hits / n,
                pr=(sum(prs) / len(prs) if prs else float("nan")))


def main():
    SS = [16, 32, 64, 128, 256, 512]
    N = 512
    for kind in ("sgate", "tgate", "deltanet"):
        print(f"\n{'='*78}\n{kind}\n{'='*78}")
        rows = [measure(kind, s, N) for s in SS]
        print(f"{'s':>6} {'E|term_c|':>13} {'sigma(S_notc)':>15} "
              f"{'P(|S|<|t|)':>12} {'particip.ratio':>15}")
        for s, r in zip(SS, rows):
            print(f"{s:>6} {r['term']:>13.5e} {r['sigma']:>15.5e} "
                  f"{r['small_ball']:>12.4f} {r['pr']:>15.2f}")
        a, a2 = loglog_slope(SS, [r["term"] for r in rows])
        b, b2 = loglog_slope(SS, [r["sigma"] for r in rows])
        pr_e, pr2 = loglog_slope(SS, [r["pr"] for r in rows])
        sb = [r["small_ball"] for r in rows]
        print(f"\n  E|term_c|      ~ s^{a:+.3f}   (R2 {a2:.4f})")
        print(f"  sigma(S_notc)  ~ s^{b:+.3f}   (R2 {b2:.4f})")
        print(f"  PREDICTED rate ~ s^{a-b:+.3f}   (= a - b, if the exponent factors)")
        if all(v > 0 for v in sb):
            m, m2 = loglog_slope(SS, sb)
            print(f"  MEASURED  rate ~ s^{m:+.3f}   (R2 {m2:.4f})")
            print(f"  factorization residual: {abs((a-b) - m):.3f}")
        print(f"\n  participation ratio ~ s^{pr_e:+.3f}  (R2 {pr2:.4f})")
        print(f"    s^1.000 = uniform spreading over ~s intermediates (dilution premise)")
        print(f"    s^0.667 = KPZ localization on O(s^2/3) corridors (premise WRONG)")
    print(f"\n{'='*78}")
    print("candidate laws for the -1.389: share only -1.000 | share x CLT -1.500 "
          "| heavy-tail -1.333")


if __name__ == "__main__":
    main()
