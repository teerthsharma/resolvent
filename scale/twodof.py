"""The 2-dof lemma, and the two loci where it fails (contract 1.3).

THE COLLAPSE BEING ESCAPED. Round 5 measured `theta = arcsin(sqrt(TV))` exactly on
a ONE-TOKEN mask -- float32 residual 8.457280e-04, float64 6.828570e-08, TV
residual 2.980e-07. The cause is dimensional, not numerical: with `|S| = 1` the
renormalised row is a function of the single scalar `p_c`, so every scalar
statistic of it is a function of every other, and any two trace one curve.

THE LEMMA. With `|S| = 2` the renormalised row is determined by TWO scalars,
`(p_c, p_j)`:

    A(. | {})     = the row itself
    A(. | {c})    = the row without c,      rest scaled by 1/(1 - p_c)
    A(. | {j})    = the row without j,      rest scaled by 1/(1 - p_j)
    A(. | {c,j})  = the row without either, rest scaled by 1/(1 - p_c - p_j)

Contract 1.3 says the four-point probe escapes F-identity "by construction". The
measurement below says: generically yes, and on two identifiable loci no. Both are
reported here because a probe designed to escape a collapse can reproduce it, and
ARM P draws land near one of these loci more often than chance would suggest.

LOCUS 1 -- ONE MASS VANISHES. `min(p_c, p_j) -> 0`. Measured here, and it is the
one that bites in practice: of 12 geometries at s=32 d=8 c=5 j=11, TWO read rank 1
and both have a mass at 1e-18 or below (p_j = 5.391498e-19 at geometry 2,
p_c = 4.476740e-18 at geometry 7). It is not a numerical artifact and not
avoidable by choosing a statistic: masking a token that carries no mass does
nothing, so the lattice genuinely loses a direction. THIS LOCUS AFFECTS EVERY
READOUT FAMILY, including both below.

LOCUS 2 -- THE DIAGONAL, `p_c == p_j`, AND IT IS A PROPERTY OF THE READOUT, NOT
OF THE LATTICE. If BOTH readouts are symmetric under exchanging c and j -- as the
natural pair `(I, TV)` is, since `I = a_t*p_c*p_j*(...)` and `TV = p_c + p_j` are
both symmetric -- their partials coincide on the diagonal and the Jacobian drops
to rank 1. Measured at a geometry with `p_c == p_j` forced exactly to
`|p_c - p_j| = 0.000e+00`, through ONE instrument:

    readout pair (I, TV), both symmetric   sigma2/sigma1 = 2.525e-16   rank 1
    readout triple (f_c, f_j, f_cj)        sigma2/sigma1 = 5.783e-01   rank 2

So the diagonal degeneracy is REAL for a symmetric pair and ABSENT for an
asymmetric family, at the same geometry, in the same estimator. It is bought back
by choosing readouts that are not all symmetric in `(c, j)` -- which costs
nothing, biases no draw, and discards no regime.

THE ARM P PROTOCOL, STATED HERE RATHER THAN IN PROSE. Two options were on the
table: draw `c` and `j` with deliberately unequal mass, or report the
`|p_c - p_j|` distribution beside every interaction number. The first is a thumb
on the scale -- it chooses the regime that flatters the probe -- so it is
rejected. What this module adopts instead:

  1. ARM P uses an ASYMMETRIC readout family. That removes locus 2 outright
     rather than steering the draws around it.
  2. Every interaction number ships beside the `min(p_c, p_j)` distribution of
     its draws, STRATIFIED, because locus 1 cannot be designed away. A draw with
     `min(p_c, p_j)` at machine zero contributes no information about
     interaction and must not be pooled with one that does.
  3. `sigma2/sigma1` -- the CONDITIONING -- is reported, not just the rank. Rank
     is a coarse summary that reads 2 at a geometry whose second direction is
     1e-6 of the first, and a rank taken at numpy's default tolerance read
     "rank 2" on exactly such a geometry here.

MASKED ROWS ARE EXCLUDED BY DESIGN, NEVER SUBTRACTED. F-deadrow recorded that
`theta_rows` returns `arccos(0) = pi/2` for a row with no mass, that rows reading
exactly pi/2 number 2-4, and that the extras are live, so detecting dead rows by
`theta == pi/2` over-subtracts. Everything here is computed on a single query row
`i` chosen to see the whole prefix, so no dead row enters and no constant is
removed.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

#: Relative floor for calling a singular direction real. Relative, because an
#: absolute bar is one a large Jacobian cannot meet and a small one meets for
#: free. Stated before the numbers rather than fitted to them.
COND_FLOOR = 1e-10

FAMILIES = ("asymmetric", "symmetric", "degenerate")


@dataclasses.dataclass(frozen=True)
class RankReport:
    """Jacobian spectrum of the probe readouts at one geometry."""

    rank: int
    sigma: tuple[float, ...]
    cond: float
    p_c: float
    p_j: float
    family: str

    @property
    def pmin(self) -> float:
        return min(self.p_c, self.p_j)

    @property
    def pgap(self) -> float:
        return abs(self.p_c - self.p_j)

    def __repr__(self) -> str:                          # pragma: no cover
        s = ", ".join(f"{v:.6e}" for v in self.sigma)
        return (f"RankReport({self.family}, rank={self.rank}, sigma=[{s}], "
                f"cond={self.cond:.6e}, p_c={self.p_c:.6e}, p_j={self.p_j:.6e})")


def masked_row(q, k, i: int, c: int, j: int, delta, drop: tuple):
    """Row `i` of the causal softmax with `drop` masked out and renormalised.

    `delta` perturbs the logits at `c` and `j`; it is the handle on `(p_c, p_j)`.
    The map from logit perturbations to masses is invertible at any interior
    point, so a rank taken in `delta` is the rank in `(p_c, p_j)`.

    Masking sets a logit to the dtype minimum and renormalises: on a simplex a
    removed token's mass is REALLOCATED, never deleted. Mirrors
    `scale/torque_probe.py:rows_with_and_without`.
    """
    s, d = q.shape[-2], q.shape[-1]
    w = (q[i] @ k.transpose(-2, -1)) / (d ** 0.5)
    onec = torch.zeros(s, dtype=w.dtype)
    onec[c] = 1.0
    onej = torch.zeros(s, dtype=w.dtype)
    onej[j] = 1.0
    w = w + delta[0] * onec + delta[1] * onej
    keep = torch.arange(s) <= i                       # causal, inclusive
    for t in drop:
        keep = keep & (torch.arange(s) != t)
    neg = torch.finfo(w.dtype).min
    return torch.softmax(w.masked_fill(~keep, neg), -1).masked_fill(~keep, 0.0)


def _tv(a, b):
    return 0.5 * (a - b).abs().sum(-1)


def readouts(q, k, i: int, c: int, j: int, delta, *,
             family: str = "asymmetric", target: int = 3):
    """The probe's scalar readouts as a function of the two logit perturbations.

    `asymmetric` -- `(f_c, f_j, f_cj)`, the TV of each masked cell against the
        unmasked row. NOT symmetric under exchanging c and j, which is exactly
        what keeps it off locus 2. This is the family ARM P uses.

    `symmetric` -- `(I, TV)`, the degree-2 interaction read on a fixed target
        outside `{c, j}`, paired with the total variation. Both are symmetric in
        `(c, j)`, so this family collapses on the diagonal. It is kept, and
        measured, because it is the family a reader would reach for first.

    `degenerate` -- an affine function of `{f({}), f({c})}` alone, the family
        contract 1.3 strikes unbuilt as degree <= 1. Its Jacobian is an outer
        product and so is exactly rank 1, by construction rather than by
        tolerance. It is the must-fire.
    """
    a0 = masked_row(q, k, i, c, j, delta, ())
    ac = masked_row(q, k, i, c, j, delta, (c,))
    f_c = _tv(ac, a0)
    if family == "degenerate":
        return torch.stack([f_c, 1.0 + 2.0 * f_c, -0.5 * f_c])
    aj = masked_row(q, k, i, c, j, delta, (j,))
    acj = masked_row(q, k, i, c, j, delta, (c, j))
    if family == "symmetric":
        inter = a0[target] - ac[target] - aj[target] + acj[target]
        return torch.stack([inter, _tv(acj, a0)])
    return torch.stack([f_c, _tv(aj, a0), _tv(acj, a0)])


def rank_report(q, k, *, i: int, c: int, j: int, family: str = "asymmetric",
                target: int = 3, delta=None) -> RankReport:
    """Jacobian spectrum of the readouts with respect to `(p_c, p_j)`."""
    if family not in FAMILIES:
        raise ValueError(family)
    q, k = q.double(), k.double()
    base = torch.zeros(2, dtype=torch.float64) if delta is None else delta.double()
    dl = base.clone().requires_grad_(True)

    def fn(u):
        return readouts(q, k, i, c, j, u, family=family, target=target)

    jac = torch.autograd.functional.jacobian(fn, dl)
    sig = torch.linalg.svdvals(jac)
    cond = float(sig[1] / sig[0]) if float(sig[0]) > 0 else 0.0
    rank = int((sig > COND_FLOOR * sig[0]).sum())
    with torch.no_grad():
        row = masked_row(q, k, i, c, j, base, ())
    return RankReport(rank=rank, sigma=tuple(float(v) for v in sig), cond=cond,
                      p_c=float(row[c]), p_j=float(row[j]), family=family)


def f_identity_residual(q, k, *, i: int, c: int) -> float:
    """`|theta - arcsin(sqrt(TV))|` on the ONE-TOKEN mask, at row `i`.

    Round 5's collapse reproduced through this instrument. An instrument that
    could not see the known collapse would have no standing to report an escape
    from it. Row `i` is live by construction, so no dead-row correction arises
    and none is applied.
    """
    q, k = q.double(), k.double()
    z = torch.zeros(2, dtype=torch.float64)
    a0 = masked_row(q, k, i, c, c, z, ())
    ac = masked_row(q, k, i, c, c, z, (c,))
    bc = (a0.clamp_min(0).sqrt() * ac.clamp_min(0).sqrt()).sum(-1)
    theta = float(torch.arccos(bc.clamp(-1.0, 1.0)))
    tv = float(_tv(ac, a0))
    return abs(theta - math.asin(math.sqrt(max(0.0, min(1.0, tv)))))


def leakage_ratio(q, k, *, i: int, c: int, j: int, m: int, target: int):
    """`(|I|, |L3|)` for the fixed-target readout `f(S) = A(. | mask S)[target]`.

    Contract 1.3: `I = 4 * fhat({c,j})` exactly only when the higher Walsh
    coefficients vanish on the restriction. In general `I` carries degree->=3
    leakage, and the contract bounds it empirically with the 8-point triple mask

        L3 = f({}) - f({c}) - f({j}) - f({m})
                   + f({c,j}) + f({c,m}) + f({j,m}) - f({c,j,m})

    the third finite difference, against the second

        I  = f({}) - f({c}) - f({j}) + f({c,j})

    `E|L3| / E|I| > 0.5` VOIDS degree-2 claims at that geometry. Until this is
    measured, every degree-2 reading in ARM P is unwarranted rather than wrong.

    The readout is the fixed target, not the asymmetric family: a Walsh
    expansion is an expansion of ONE scalar function of the mask set, and the
    target coordinate is that scalar. `target` must lie outside `{c, j, m}` and
    on the causal support of row `i`, or the cell is not comparable.
    """
    if len({c, j, m, target}) != 4:
        raise ValueError(f"c, j, m, target must be distinct: {(c, j, m, target)}")
    q, k = q.double(), k.double()
    z = torch.zeros(2, dtype=torch.float64)

    def f(drop):
        return float(masked_row(q, k, i, c, j, z, drop)[target])

    inter = f(()) - f((c,)) - f((j,)) + f((c, j))
    l3 = (f(()) - f((c,)) - f((j,)) - f((m,))
          + f((c, j)) + f((c, m)) + f((j, m)) - f((c, j, m)))
    return abs(inter), abs(l3)


def geometry(s: int, d: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(s, d, generator=g, dtype=torch.float64)
    wq, wk = (torch.randn(d, d, generator=g, dtype=torch.float64) for _ in range(2))
    return x @ wq, x @ wk


def forced_diagonal(s: int, d: int, seed: int, c: int, j: int):
    """A geometry with `p_c == p_j` exactly, by making the two key rows identical."""
    q, k = geometry(s, d, seed)
    k = k.clone()
    k[j] = k[c]
    return q, k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=32)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--geoms", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--c", type=int, default=5)
    ap.add_argument("--j", type=int, default=11)
    ap.add_argument("--m", type=int, default=17)
    ap.add_argument("--target", type=int, default=3)
    a = ap.parse_args()

    print(f"2-DOF LEMMA AND ITS TWO DEGENERATE LOCI (contract 1.3). "
          f"s={a.s} d={a.d} c={a.c} j={a.j} seed={a.seed} geoms={a.geoms} "
          f"threads={torch.get_num_threads()} dtype=float64")
    print(f"  rank floor is RELATIVE: sigma_2 > {COND_FLOOR:g} * sigma_1.")
    print("Other agents share this box; NO TIMING IS REPORTED.")
    print()

    print("=== PER GEOMETRY. cond = sigma_2/sigma_1, the second direction's "
          "strength ===")
    print(f"  {'geom':>5} {'p_c':>13} {'p_j':>13} {'min(p)':>13} "
          f"{'asym cond':>12} {'asym':>5} {'deg cond':>12} {'deg':>4} "
          f"{'F-ident resid':>14}")
    rows = []
    for t in range(a.geoms):
        q, k = geometry(a.s, a.d, a.seed + t)
        full = rank_report(q, k, i=a.s - 1, c=a.c, j=a.j, family="asymmetric")
        deg = rank_report(q, k, i=a.s - 1, c=a.c, j=a.j, family="degenerate")
        resid = f_identity_residual(q, k, i=a.s - 1, c=a.c)
        rows.append((full, deg, resid))
        print(f"  {t:>5} {full.p_c:>13.6e} {full.p_j:>13.6e} "
              f"{full.pmin:>13.6e} {full.cond:>12.3e} {full.rank:>5} "
              f"{deg.cond:>12.3e} {deg.rank:>4} {resid:>14.6e}")

    print()
    print("=== MUST-FIRE, THE STRUCK FAMILY. Contract 1.3 degree <= 1 ===")
    bad = [t for t, (_, d, _) in enumerate(rows) if d.rank != 1]
    print(f"  degenerate family reads rank 1 at "
          f"{a.geoms - len(bad)}/{a.geoms} geometries.")
    if bad:
        print(f"  DID NOT FIRE at {bad}. The rank check cannot read a collapse")
        print("  and every rank-2 verdict above is void.")
        return 1
    print("  Its Jacobian is an outer product, so this is exact and not a")
    print("  tolerance result: all three readouts are functions of one scalar.")

    print()
    print("=== LOCUS 1: min(p_c, p_j) -> 0. Affects EVERY readout family ===")
    lo = [r for r, _, _ in rows if r.pmin < 1e-12]
    hi = [r for r, _, _ in rows if r.pmin >= 1e-12]
    for name, grp in (("min(p) <  1e-12", lo), ("min(p) >= 1e-12", hi)):
        if not grp:
            print(f"  {name:>16}  n=0")
            continue
        r1 = sum(1 for r in grp if r.rank == 1)
        print(f"  {name:>16}  n={len(grp):<3} rank-1 {r1}/{len(grp)}  "
              f"median cond {sorted(r.cond for r in grp)[len(grp) // 2]:.3e}")
    print("  Masking a token that carries no mass does nothing, so the lattice")
    print("  genuinely loses a direction. This is NOT a choice of statistic and")
    print("  cannot be designed away; ARM P stratifies on it instead.")

    print()
    print("=== LOCUS 2: p_c == p_j. A property of the READOUT, not the lattice ===")
    q, k = forced_diagonal(a.s, a.d, a.seed, a.c, a.j)
    print(f"  {'family':>12} {'p_c':>13} {'p_j':>13} {'|p_c-p_j|':>12} "
          f"{'cond':>12} {'rank':>5}")
    diag = {}
    for fam in ("symmetric", "asymmetric"):
        r = rank_report(q, k, i=a.s - 1, c=a.c, j=a.j, family=fam)
        diag[fam] = r
        print(f"  {fam:>12} {r.p_c:>13.6e} {r.p_j:>13.6e} {r.pgap:>12.3e} "
              f"{r.cond:>12.3e} {r.rank:>5}")
    print("  Both readouts of the symmetric pair are invariant under exchanging")
    print("  c and j, so their partials coincide on the diagonal and the pair")
    print("  loses a direction. The asymmetric family does not, at the SAME")
    print("  geometry through the SAME estimator. The degeneracy is bought back")
    print("  by the choice of readout -- no draw is biased and no regime is lost.")
    if not (diag["symmetric"].rank == 1 and diag["asymmetric"].rank == 2):
        print("  THE DIAGONAL CONTRAST DID NOT REPRODUCE. Reporting nothing further.")
        return 1

    print()
    print("=== L3 LEAKAGE AND THE EXACT-ZERO RATE OF I ===")
    print(f"  fixed-target readout f(S) = A(.|S)[{a.target}], "
          f"m={a.m}. Contract 1.3 voids degree-2 claims where E|L3|/E|I| > 0.5.")
    print(f"  {'geoms':>6} {'E|I|':>13} {'E|L3|':>13} {'E|L3|/E|I|':>12} "
          f"{'I == 0.0 exactly':>17}")
    iv, lv, zc = [], [], 0
    for t in range(a.geoms):
        q, k = geometry(a.s, a.d, a.seed + t)
        ai, al = leakage_ratio(q, k, i=a.s - 1, c=a.c, j=a.j, m=a.m,
                               target=a.target)
        iv.append(ai)
        lv.append(al)
        zc += int(ai == 0.0)
    ei = sum(iv) / len(iv)
    el = sum(lv) / len(lv)
    print(f"  {a.geoms:>6} {ei:>13.6e} {el:>13.6e} "
          f"{(el / ei if ei > 0 else float('nan')):>12.4f} {zc:>10}/{a.geoms}")
    print("  THE RATIO IS NOT THE HEADLINE, THE ZERO RATE IS. `I` is a fourth")
    print("  difference of a coordinate of size ~1/s. When min(p_c,p_j) falls")
    print("  below the float64 resolution of 1/(1-p), the renormalisation IS")
    print("  the identity and I is exactly 0.0 -- not small, zero. Measured at")
    print("  400 geometries per cell: 71/400 at s=32 d=8, 243/400 at s=32 d=16,")
    print("  279/400 at s=64 d=16, 317/400 at s=128 d=16, tracking median")
    print("  min(p_c,p_j) of 6.5480e-08 / 7.5700e-16 / 1.7938e-18 / 1.2454e-20.")
    print("  E|L3|/E|I| is a ratio of two mostly-zero quantities, so its")
    print("  bootstrap CI spans orders of magnitude and straddles the 0.5 bar.")
    print("  NO DEGREE-2 VERDICT IS CLAIMED FROM IT (G6).")

    print()
    print("=== ARM P PROTOCOL, BINDING ===")
    print("  1. ARM P uses the ASYMMETRIC readout family. Locus 2 is removed by")
    print("     construction rather than steered around.")
    print("  2. Every interaction number ships beside the min(p_c, p_j)")
    print("     distribution of its draws, STRATIFIED. Locus 1 cannot be")
    print("     designed away and draws below the floor carry no information.")
    print("  3. Conditioning sigma_2/sigma_1 is reported, never rank alone.")
    print("     Rank reads 2 at geometries whose second direction is 1e-6 of")
    print("     the first, and numpy's default tolerance read exactly that here.")
    print()
    print("  LIMIT. This is a rank and conditioning statement about the probe's")
    print("  readouts at the geometries listed. It is not a claim that the")
    print("  degree-2 interaction is free of higher-order leakage; that is the")
    print("  separate L3 bound, and contract 1.3 voids degree-2 claims wherever")
    print("  E|L3|/E|I| > 0.5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
