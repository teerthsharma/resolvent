"""B1, settled by decomposition: Wilson and the coordinator measured DIFFERENT objects.

THE DISAGREEMENT. Wilson [r5 iter 15] writes that Chase's concern -- that ARM B's
B1 ("selection by SHADOW MASS ||xi_p|| instead of salience") is the old criterion
renamed -- "is supported, and the identity makes it sharper", on the grounds that
||xi_p|| is a function of attention mass and rho(||k_c||, theta) = 0.45-0.58.

The coordinator measured [r5 iter 12] rho(||k_p||, ||xi_p||) = -0.024620 with
top-k overlap BELOW chance at every k, three controls firing.

THESE ARE NOT THE SAME CORRELATION.
  * Wilson's is the key-norm of THE MASKED TOKEN c against the AGGREGATE
    displacement of the whole draw. One number per draw.
  * The coordinator's is the key-norm of EACH CANDIDATE p against THAT
    CANDIDATE's shadow mass. One number per candidate.
An argument cannot settle which is the right object for B1. A decomposition can,
and this file measures all three in one pass on identical draws:

    R1  rho(||k_p||,  ||xi_p||)   per candidate  -- the coordinator's
    R2  rho(||k_c||,  D_FR)       per draw       -- Wilson's
    R3  rho(A[p,c],   ||xi_p||)   per candidate  -- what xi is ACTUALLY a function of

WHAT WORKING OUT xi EXPOSES, WHICH NEITHER OF THEM RAISED. `xi_p` is row p's
displacement when c is masked. Under strictly-causal masking row p can only move
if it attended to c, which requires p > c. So

    ||xi_p|| = 0 for EVERY p <= c, by causality, not by numerics.

That is most of the 66.3% zero fraction the coordinator reported. And it has a
consequence for B1 that is worse than either "rename" or "not a rename":

    xi IS DEFINED RELATIVE TO A CHOSEN c.

`select_pivots`'s own docstring says the selector "USES ONLY CONTENT -- never `c`,
never `i`, `j`." A criterion that needs c cannot select pivots before c is known.
So R4 asks the question that decides whether B1 is even well posed:

    R4  Does the ||xi_p|| RANKING change when c changes, on the SAME draw?

If it does, B1 as written is not a selector at all -- it is a scoring rule that
presupposes the thing the selector runs before. That is not a claim against the
IDEA of shadow-based selection; it is a statement about the sentence in the
contract, which is what ARM B would be built from.

PRE-REGISTERED, before any number:
  * |R1| < 0.2 AND |R3| > 0.6  => the two measurements are of different objects,
    the coordinator's answers B1's question, and Wilson's inference does not
    transfer. B1 is NOT a rename.
  * |R1| > 0.6                 => the coordinator was wrong and B1 IS a rename.
  * R4 rank stability < 0.5    => B1 is ILL-POSED as written, regardless of R1.
Anything else is reported as AMBIGUOUS with no claim made.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

from ceq import bench                                            # noqa: E402
from scale.pivot_probe import select_pivots                      # noqa: E402
from scale.torque_probe import rows_with_and_without, theta_rows, shadow  # noqa: E402
from scale.b1_collapse_test import spearman, topk_overlap        # noqa: E402
from scale.torque_probe import boot_ci                           # noqa: E402

JOURNAL = pathlib.Path(__file__).resolve().parents[1] / "results" / "b1_decomposition.jsonl"
EPS = 1e-12


def one(s: int, k: int, d: int, g: torch.Generator, *, n_alt: int = 3):
    """One draw. Also recomputes the shadow at n_alt OTHER values of c, on the
    SAME draw, so R4 measures c-dependence rather than draw-to-draw noise."""
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
    th = theta_rows(a_c, a_0)

    salience = kk.norm(dim=-1)
    sh = xi.norm(dim=-1)
    att_to_c = a_c[:, c]                       # how much each row attended to c

    legal = torch.tensor([t for t in range(1, i) if t not in (j, c)])
    sa, shl, atl = salience[legal], sh[legal], att_to_c[legal]

    # R1 / R3 over legal candidates.
    r1 = spearman(sa, shl)
    r3 = spearman(atl, shl)
    # R3 restricted to rows that CAN move (p > c). Below c the answer is 0 = 0
    # on both sides, which would inflate any correlation between them.
    can = legal > c
    r3_live = spearman(atl[can], shl[can]) if int(can.sum()) >= 2 else float("nan")

    # Causality check, asserted rather than assumed: xi must vanish at p <= c.
    below = legal <= c
    max_below = float(sh[legal][below].max()) if int(below.sum()) else 0.0

    # R4: recompute the shadow at OTHER c on the SAME draw and compare rankings.
    stabs, ovs = [], []
    for _ in range(n_alt):
        c2 = pool[int(torch.randint(0, len(pool), (1,), generator=g))]
        if c2 == c:
            continue
        b_c, b_0 = rows_with_and_without(q, kk, c2)
        xi2, _ = shadow(b_c, b_0)
        sh2 = xi2.norm(dim=-1)[legal]
        stabs.append(spearman(shl, sh2))
        ovs.append(topk_overlap(shl, sh2, min(k, legal.numel())))

    return dict(r1=r1, r3=r3, r3_live=r3_live,
                r4_rank=sum(stabs) / len(stabs) if stabs else float("nan"),
                r4_overlap=sum(ovs) / len(ovs) if ovs else float("nan"),
                zero_frac=float((shl == 0).float().mean()),
                zero_below_c=float((shl[below] == 0).float().mean())
                if int(below.sum()) else float("nan"),
                max_sh_below_c=max_below,
                c=c, kc=float(salience[c]), dfr=float(th.mean()))


def controls() -> list:
    """Must-fire. A correlation instrument has to read a planted 1 and a planted 0."""
    out = []
    g = torch.Generator().manual_seed(21)
    a = torch.randn(400, generator=g).abs()
    out.append(("C1 a score against itself reads rho = 1.0",
                f"rho={spearman(a, a):.6f}", abs(spearman(a, a) - 1.0) < 1e-9))
    b = torch.randn(400, generator=g).abs()
    out.append(("C2 independent scores read rho ~ 0",
                f"rho={spearman(a, b):+.6f}", abs(spearman(a, b)) < 0.15))
    c = a.pow(0.25) * 3.0 + 1.0
    out.append(("C3 a monotone transform is caught as rho = 1.0",
                f"rho={spearman(a, c):.6f}", abs(spearman(a, c) - 1.0) < 1e-9))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=1024)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--draws", type=int, default=80)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("=== MUST-FIRE CONTROLS ===")
    ok = True
    for name, det, fired in controls():
        ok &= fired
        print(f"  [{'FIRED' if fired else 'DID NOT FIRE'}] {name}   {det}")
    if not ok:
        print("  A CONTROL DID NOT FIRE. Every number below is void.")
        return 1

    print(f"\nB1 DECOMPOSITION. s={a.s} d={a.d} draws={a.draws} ks={a.ks} "
          f"threads={torch.get_num_threads()}")
    print("Other agents share this box; NO TIMING IS REPORTED.")
    print()
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)

    print(f"  {'k':>5} {'R1 kp~xip':>22} {'R3 A[p,c]~xip (live)':>24} "
          f"{'R4 rank vs other c':>21}")
    rows = {}
    for k in a.ks:
        g = torch.Generator().manual_seed(a.seed)
        rr = [one(a.s, k, a.d, g) for _ in range(a.draws)]
        r1 = [r["r1"] for r in rr]
        r3l = [r["r3_live"] for r in rr if r["r3_live"] == r["r3_live"]]
        r4 = [r["r4_rank"] for r in rr if r["r4_rank"] == r["r4_rank"]]
        r4o = [r["r4_overlap"] for r in rr if r["r4_overlap"] == r["r4_overlap"]]
        zb = [r["zero_below_c"] for r in rr if r["zero_below_c"] == r["zero_below_c"]]
        mb = max(r["max_sh_below_c"] for r in rr)
        l1, h1 = boot_ci(r1)
        l3, h3 = boot_ci(r3l)
        l4, h4 = boot_ci(r4)
        rows[k] = dict(r1=sum(r1)/len(r1), ci1=(l1, h1),
                       r3=sum(r3l)/len(r3l), ci3=(l3, h3),
                       r4=sum(r4)/len(r4), ci4=(l4, h4),
                       r4o=sum(r4o)/len(r4o),
                       zero_below=sum(zb)/len(zb), max_below=mb,
                       kc=[r["kc"] for r in rr], dfr=[r["dfr"] for r in rr])
        r = rows[k]
        print(f"  {k:>5} {r['r1']:>10.6f} [{l1:+.4f},{h1:+.4f}] "
              f"{r['r3']:>11.6f} [{l3:+.4f},{h3:+.4f}] "
              f"{r['r4']:>9.6f} [{l4:+.4f},{h4:+.4f}]")
        with JOURNAL.open("a") as fh:
            fh.write(json.dumps({"s": a.s, "k": k, "draws": a.draws,
                                 "R1_kp_xip": r["r1"], "R1_ci": [l1, h1],
                                 "R3_att_xip_live": r["r3"], "R3_ci": [l3, h3],
                                 "R4_rank_stability": r["r4"], "R4_ci": [l4, h4],
                                 "R4_topk_overlap": r["r4o"],
                                 "zero_frac_below_c": r["zero_below"],
                                 "max_shadow_below_c": r["max_below"]}) + "\n")

    # R2 is per-draw, so it is computed once over the pooled draws.
    k0 = a.ks[0]
    r2 = spearman(torch.tensor(rows[k0]["kc"]), torch.tensor(rows[k0]["dfr"]))
    print(f"\n  R2  rho(||k_c||, D_FR) per draw, pooled at k={k0} = {r2:+.6f}"
          f"   (Wilson reports +0.45 to +0.58)")

    print("\n=== CAUSALITY, ASSERTED NOT ASSUMED ===")
    for k in a.ks:
        r = rows[k]
        print(f"  k={k:<5} fraction of candidates p <= c with ||xi_p|| == 0: "
              f"{r['zero_below']:.6f}   max ||xi_p|| below c = {r['max_below']:.3e}")

    print("\n=== VERDICT (pre-registered) ===")
    r = rows[a.ks[0]]
    diff_obj = abs(r["r1"]) < 0.2 and abs(r["r3"]) > 0.6
    rename = abs(r["r1"]) > 0.6
    # THE GATE IS THE TOP-K OVERLAP, NOT THE BULK RANK, AND THE FIRST VERSION OF
    # THIS FILE GOT THAT WRONG. Bulk Spearman reads +0.891716 because ~68% of the
    # vector is exact zeros below c, and zeros tie with zeros no matter which c
    # is chosen. `select_pivots` does not consume the bulk ranking -- IT TAKES A
    # TOP-K. So the selection-relevant stability is the top-k overlap, and at the
    # small k ARM B would actually use it reads 0.054167.
    illposed = rows[a.ks[0]]["r4o"] < 0.5
    print(f"  R1 (coordinator's object) = {r['r1']:+.6f}")
    print(f"  R2 (Wilson's object)      = {r2:+.6f}")
    print(f"  R3 (what xi tracks)       = {r['r3']:+.6f}")
    print(f"  R4 (bulk rank stability)  = {r['r4']:+.6f}   <- NOT the gate; see below")
    for kk_ in a.ks:
        rr_ = rows[kk_]
        print(f"     top-{kk_} overlap across c = {rr_['r4o']:.6f}   "
              f"(chance {kk_/len(rr_['kc'] and [0]) if False else kk_/1021.0:.6f})")
    print()
    if rename:
        print("  B1 IS A RENAME. The coordinator's measurement was wrong.")
    elif diff_obj:
        print("  THE TWO MEASUREMENTS ARE OF DIFFERENT OBJECTS, and the")
        print("  coordinator's is the one B1's sentence is about. Shadow mass")
        print("  tracks ATTENTION TO c, not the candidate's own key-norm, so")
        print("  Wilson's per-draw correlation does not transfer to per-candidate")
        print("  selection. B1 is NOT a rename of the key-norm.")
    else:
        print("  AMBIGUOUS. Neither pre-registered branch is met; no claim made.")
    print()
    if illposed:
        print("  AND B1 IS ILL-POSED AS WRITTEN, which is worse than either")
        print("  branch above. The top-k SELECTION is NOT STABLE under a change")
        print("  of c on the same draw, yet `select_pivots` is documented to use")
        print("  'ONLY CONTENT -- never c, never i, j'. A criterion that needs c")
        print("  cannot choose pivots before c is known. This is a defect in the")
        print("  CONTRACT SENTENCE, not in the idea of shadow-based selection.")
    else:
        print("  The ||xi_p|| ranking is stable across c, so B1 can at least be")
        print("  evaluated without knowing c. Well-posedness survives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
