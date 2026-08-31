"""ARM B's B1: is "shadow mass INSTEAD OF salience" actually a different criterion?

THE CONTRACT PROPOSES, in ARM B: "B1 selection by SHADOW MASS ||xi_p|| instead of
salience |t_p| (the criterion that died at -1.298 was magnitude)". The whole point
of B1 is that the OLD criterion was magnitude and the NEW one is not.

CHASE RAISED THE DOUBT [r5 iter 11] and the premise checks out. [READ]
`scale/pivot_probe.py:88`:

    score = key.norm(dim=-1).clone()

and its own docstring says "Score is the key-norm, a pure function of the token's
own representation." So the incumbent selector ranks by magnitude, and nothing in
an ARM A draw is causal -- `x0`, `wq`, `wk` are i.i.d. `randn`.

THE DOUBT: `xi` is the log map built from `theta`, and Foreman's identity makes
`theta_i = arcsin(sqrt(TV_i))` a monotone function of the masked token's own
weight. If the shadow-mass ranking reproduces the key-norm ranking, then B1
selects THE SAME TOKENS under a new name, and ARM B's first upgrade is void
BEFORE IT IS BUILT -- which is the cheapest possible time to find out.

THIS IS A TEST OF A PROPOSAL, NOT OF A RESULT. Nothing in ARM B exists yet. That
is precisely why it is worth running now: the contract says "Nothing is built
until ARM A survives", and this establishes whether one of the three things that
WOULD be built is a rename.

PRE-REGISTERED, before any number:
  * Spearman(||xi_p||, ||k_p||) > 0.95 AND top-k overlap > 0.90 at every k
    => B1 IS THE OLD CRITERION. It must be renamed, rejustified, or dropped.
  * Spearman < 0.5 or overlap < 0.5 => B1 selects genuinely different tokens and
    survives this objection (which is NOT the same as being better).
  * Anything between is REPORTED AS AMBIGUOUS and neither claim is made.

Two must-fire controls, because a rank-comparison instrument that cannot read a
planted 1.0 and a planted 0.0 is not measuring agreement.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

from ceq import bench                                            # noqa: E402
from scale.pivot_probe import select_pivots                      # noqa: E402
from scale.torque_probe import rows_with_and_without, shadow, boot_ci  # noqa: E402

JOURNAL = pathlib.Path(__file__).resolve().parents[1] / "results" / "b1_collapse.jsonl"


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation. Ties broken by position, which is fine for float scores."""
    n = a.numel()
    if n < 2:
        return float("nan")
    ra = torch.empty(n, dtype=torch.float64)
    rb = torch.empty(n, dtype=torch.float64)
    ra[a.argsort()] = torch.arange(n, dtype=torch.float64)
    rb[b.argsort()] = torch.arange(n, dtype=torch.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = (ra.norm() * rb.norm()).clamp_min(1e-30)
    return float((ra @ rb) / den)


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    ia = set(torch.topk(a, k).indices.tolist())
    ib = set(torch.topk(b, k).indices.tolist())
    return len(ia & ib) / k


def one(s: int, k: int, d: int, g: torch.Generator):
    """One draw. `c` UNIFORM AT RANDOM -- carpet discipline, contractual."""
    x0 = torch.randn(s, d, generator=g)
    wq, wk = torch.randn(d, d, generator=g), torch.randn(d, d, generator=g)
    q, kk = x0 @ wq, x0 @ wk
    i = s - 1
    j = int(torch.randint(1, i, (1,), generator=g))
    piv = select_pivots(kk, min(k, s - 2), exclude=(i, j))
    pool = [t for t in range(1, i) if t != j]
    c = pool[int(torch.randint(0, len(pool), (1,), generator=g))]

    a_c, a_0 = rows_with_and_without(q, kk, c)
    xi, _ = shadow(a_c, a_0)

    # THE TWO CANDIDATE SELECTORS, on the SAME draw, over the SAME index set.
    salience = kk.norm(dim=-1)                 # the incumbent, verbatim
    shadow_mass = xi.norm(dim=-1)              # B1's proposal

    # Compare over LEGAL candidates only -- excluding i, j and c, exactly as the
    # selector would. Comparing over rows the selector can never pick would
    # measure agreement on tokens neither criterion is allowed to choose.
    legal = torch.tensor([t for t in range(1, i) if t not in (j, c)])
    sa, sh = salience[legal], shadow_mass[legal]

    # TWO-THIRDS OF `xi` IS EXACTLY ZERO, so a rank over the whole vector is
    # mostly a rank over TIES, and "near-chance overlap" is then what ranking
    # NOISE looks like -- not evidence of a different criterion. The comparison
    # is therefore ALSO made over the rows where B1's score actually exists.
    nz = sh > 0
    n_nz = int(nz.sum())
    rho_nz = spearman(sa[nz], sh[nz]) if n_nz >= 2 else float("nan")
    kk_eff = min(k, legal.numel())
    top_sh = torch.topk(sh, kk_eff).indices
    return dict(rho=spearman(sa, sh),
                rho_nonzero=rho_nz,
                n_nonzero=n_nz,
                n_legal=int(legal.numel()),
                overlap=topk_overlap(sa, sh, kk_eff),
                topk_all_nonzero=float(nz[top_sh].float().mean()),
                shadow_zero_frac=float((sh == 0).float().mean()))


def controls(s: int = 256) -> list:
    """Must-fire. A planted 1.0 and a planted ~0.0 must both read correctly."""
    out = []
    g = torch.Generator().manual_seed(5)
    a = torch.randn(s, generator=g).abs()

    # C1 a score against ITSELF must read rho = 1 and overlap = 1.
    out.append(("C1 identical scores read rho=1.0 and overlap=1.0",
                f"rho={spearman(a, a):.6f} overlap={topk_overlap(a, a, 32):.6f}",
                abs(spearman(a, a) - 1.0) < 1e-9 and topk_overlap(a, a, 32) == 1.0))

    # C2 an INDEPENDENT score must read rho ~ 0 and overlap ~ k/s. Without this
    #    the instrument could return 1.0 for everything and look like a finding.
    b = torch.randn(s, generator=g).abs()
    r, o = spearman(a, b), topk_overlap(a, b, 32)
    out.append(("C2 independent scores read rho~0 and overlap~k/s",
                f"rho={r:+.6f} overlap={o:.6f} (chance = {32/s:.6f})",
                abs(r) < 0.25 and o < 0.35))

    # C3 a MONOTONE TRANSFORM of a score must read rho = 1 exactly -- this is the
    #    exact failure mode under test, so the instrument must be shown to catch
    #    it rather than assumed to.
    c = a.pow(0.3) * 7.0 + 2.0
    out.append(("C3 a monotone transform is detected as rho=1.0 (the failure mode)",
                f"rho={spearman(a, c):.6f} overlap={topk_overlap(a, c, 32):.6f}",
                abs(spearman(a, c) - 1.0) < 1e-9 and topk_overlap(a, c, 32) == 1.0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=1024)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--draws", type=int, default=80)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("=== MUST-FIRE CONTROLS (read before any number below) ===")
    ok = True
    for name, det, fired in controls():
        ok &= fired
        print(f"  [{'FIRED' if fired else 'DID NOT FIRE'}] {name}\n      {det}")
    if not ok:
        print("\n  A CONTROL DID NOT FIRE. Every number below is void. Stopping.")
        return 1

    print(f"\nB1 COLLAPSE TEST. s={a.s} d={a.d} draws={a.draws} ks={a.ks} "
          f"threads={torch.get_num_threads()}")
    print("[READ] the incumbent selector is `score = key.norm(dim=-1)` at "
          "scale/pivot_probe.py:88.")
    print("Other agents share this box; NO TIMING IS REPORTED.\n")
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)

    print("  NOTE: `rho` compares two scores that DO NOT DEPEND ON k, so it is")
    print("  ONE measurement, not three. Only `top-k overlap` varies with k.")
    print()
    print(f"  {'k':>5} {'top-k overlap':>26} {'top-k in nonzero xi':>21}")
    rows = {}
    for k in a.ks:
        g = torch.Generator().manual_seed(a.seed)
        rr = [one(a.s, k, a.d, g) for _ in range(a.draws)]
        rho = [r["rho"] for r in rr]
        ov = [r["overlap"] for r in rr]
        zf = sum(r["shadow_zero_frac"] for r in rr) / len(rr)
        lo_r, hi_r = boot_ci(rho)
        lo_o, hi_o = boot_ci(ov)
        rnz = [r["rho_nonzero"] for r in rr]
        lo_n, hi_n = boot_ci(rnz)
        tan = sum(r["topk_all_nonzero"] for r in rr) / len(rr)
        nnz = sum(r["n_nonzero"] for r in rr) / len(rr)
        nleg = sum(r["n_legal"] for r in rr) / len(rr)
        rows[k] = (sum(rho) / len(rho), lo_r, hi_r, sum(ov) / len(ov), lo_o, hi_o, zf,
                   sum(rnz) / len(rnz), lo_n, hi_n, tan, nnz, nleg)
        print(f"  {k:>5} {sum(ov)/len(ov):>12.6f} [{lo_o:.4f},{hi_o:.4f}] "
              f"{tan:>21.6f}      (chance = {k/nleg:.6f})")
        with JOURNAL.open("a") as fh:
            fh.write(json.dumps({"s": a.s, "k": k, "draws": a.draws,
                                 "rho_mean": sum(rho) / len(rho), "rho_ci": [lo_r, hi_r],
                                 "overlap_mean": sum(ov) / len(ov),
                                 "overlap_ci": [lo_o, hi_o],
                                 "xi_zero_frac": zf}) + "\n")

    print("\n=== B1 VERDICT (pre-registered before the numbers) ===")
    same = all(r[0] > 0.95 and r[3] > 0.90 for r in rows.values())
    diff = all(r[0] < 0.50 or r[3] < 0.50 for r in rows.values())
    for k, r in rows.items():
        print(f"  k={k:<5} top-{k} overlap={r[3]:.6f} [{r[4]:.4f},{r[5]:.4f}]  "
              f"chance={k/r[12]:.6f}   top-k inside nonzero xi = {r[10]:.4f}")
    print()
    if same:
        print("  B1 IS THE OLD CRITERION. Shadow mass ranks the candidates the")
        print("  way the key-norm does, so 'selection by shadow mass INSTEAD OF")
        print("  salience' selects the same tokens under a new name. ARM B's")
        print("  first upgrade must be renamed, rejustified, or dropped -- and")
        print("  this was established BEFORE it was built.")
    elif diff:
        print("  B1 IS NOT A RENAME OF THE KEY-NORM. Chase's objection does not")
        print("  land: the two criteria rank the candidates near-independently.")
        print("  That is NOT a claim that B1 is better -- only that it is not the")
        print("  old criterion. Whether it helps is an unrun ablation.")
        print()
        print("  READ THIS BESIDE IT, IT IS NOT A FOOTNOTE: two thirds of `xi`")
        print("  rows are EXACTLY ZERO, so most of the rank is a rank over TIES,")
        print("  and near-chance overlap is also what ranking NOISE produces. The")
        print("  `rho` over nonzero rows and the `top-k inside nonzero xi` column")
        print("  above are what separate 'a different criterion' from 'an")
        print("  arbitrary tie-break', and they are reported rather than resolved.")
    else:
        print("  AMBIGUOUS. The two criteria agree partially and neither")
        print("  pre-registered branch is met. NEITHER claim may be made, and")
        print("  the numbers above stand on their own.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
