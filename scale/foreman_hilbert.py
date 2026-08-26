"""Round 6, iteration 0, Foreman. The Hilbert metric: positivity, and the size
of the projective diameter that the contraction certificate rests on.

Three questions, in the order the contract asks them.

WHAT d_H IS HERE. On the open simplex, for p, q > 0,

    d_H(p, q) = max_j log(p_j / q_j) - min_j log(p_j / q_j)

an oscillation of log-ratios. It is invariant under positive rescaling of either
argument, so it is a metric on rays, not on vectors. It is +infinity whenever the
two supports differ, and that is not a numerical accident: a coordinate where one
argument is zero and the other is not makes the ratio unbounded. The whole
positivity requirement in the contract's 1.1 exists to keep this from happening.

THE IDENTITY THIS FILE LEANS ON, and binds as a control. For two softmax rows
over the SAME support, with logits u and v,

    log(softmax(u)_j / softmax(v)_j) = (u_j - v_j) - (logZ_u - logZ_v)

so the constant cancels in the oscillation and

    d_H(softmax(u), softmax(v)) = max_j (u_j - v_j) - min_j (u_j - v_j)

The Hilbert distance between two softmax rows IS the oscillation of their logit
difference. Nothing about the softmax survives except the logits. This is why
d_H over softmax rows scales linearly with the logit scale, and it is the reason
the numbers below are as large as they are: it is arithmetic, not a defect.

POSITIVITY IS A SUPPORT STATEMENT, NOT A SAMPLING STATEMENT. `ceq/bench.py`
builds its causal mask as `tril(-1)`, strictly lower triangular, so row i is
supported on {0, ..., i-1} and row 0 is supported on nothing at all. Two
structural consequences, both measured below rather than asserted:

  1. Two attention rows with different indices have DIFFERENT SUPPORTS. Any
     d_H taken between them over the full index range is +infinity by
     construction, for every draw, at every geometry, forever. A settling map
     built naively out of such rows triggers kill K-A on its first reading and
     the trigger carries no information about the operator.
  2. Row 0 is the all-zero row -- softmax over an all-masked row, then
     `masked_fill(nm, 0.0)`. It cannot be normalised and cannot be a pivot.

Both are repaired by DESIGN in the T fixed below, not by subtracting anything --
see F-deadrow in the contract's STATE section.

THE THIRD FAILURE MODE IS FLOAT UNDERFLOW, and it is the one that has to be
measured rather than reasoned about. `torch.softmax` in float32 computes
exp(u_j - max u). float32's smallest positive subnormal is 1.4012985e-45, so the
result is EXACTLY ZERO once u_j - max u drops below about -103.28. If a live
attention row has a within-row logit spread wider than that, a coordinate on the
causal support reads exactly 0 and positivity fails for a reason that has
nothing to do with the mask. float64's corresponding cliff is about -745.13.
This file prints the measured spread beside both cliffs.

THREADS ARE PINNED HERE, not by the launcher. Four agents share this box, so no
wall-clock timing is produced or reported; every number below is arithmetic on
fixed inputs and is unaffected by contention.

G8: no multiplication appears inside any sign or comparison decision in this
file. Every branch is a comparison of two already-computed reals.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from decimal import Decimal, getcontext

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

from ceq import bench                                            # noqa: E402
from scale.pivot_probe import select_pivots                      # noqa: E402
from scale import hilbert as coord                               # noqa: E402

JOURNAL = pathlib.Path(__file__).resolve().parents[1] / "results" / "hilbert.jsonl"

# The float64 threshold above which tanh(Delta/4) is indistinguishable from 1.0.
# Carried as a published constant so the audit below compares against it rather
# than rediscovering it.
TANH_SATURATION_DELTA = 76.246190

# ln of the smallest positive subnormal of each dtype: below this, exp() is 0.
UNDERFLOW_F32 = math.log(float(torch.finfo(torch.float32).tiny) * 2.0 ** -23)
UNDERFLOW_F64 = math.log(float(torch.finfo(torch.float64).tiny) * 2.0 ** -52)


# ---------------------------------------------------------------- the metric

def d_hilbert(p: torch.Tensor, q: torch.Tensor) -> float:
    """d_H(p, q) over the common support, +inf when the supports differ.

    Returning +inf on a support mismatch is the honest answer, not a guard: the
    ratio really is unbounded there. Callers that want a finite number must
    restrict both arguments to a shared support first, and doing so is a design
    decision about the map, not about the metric.
    """
    sp, sq = p > 0, q > 0
    if not torch.equal(sp, sq):
        return float("inf")
    if not bool(sp.any()):
        return float("nan")                     # both empty: undefined, not zero
    r = torch.log(p[sp].double()) - torch.log(q[sp].double())
    return float(r.max() - r.min())


def one_minus_kappa(delta: float) -> float:
    """1 - tanh(Delta/4) = 2/(exp(Delta/2) + 1), in closed form.

    Never computed as `1.0 - math.tanh(delta / 4.0)`: that subtraction has lost
    all its significant digits by Delta = 75 and reads exactly 0 from Delta = 100.
    """
    if delta == float("inf"):
        return 0.0
    return 2.0 / (math.exp(delta / 2.0) + 1.0)


# ------------------------------------------------------------------ the map

def pivot_readings(a: torch.Tensor, pivots: torch.Tensor):
    """The pivot rows of A, and the support the mixture of them lives on.

    A is strictly causal, so pivot p's row is supported on {0, ..., p-1}. A
    convex combination of the pivot rows with strictly positive weights is
    therefore supported on the UNION of those supports, which is
    {0, ..., max(P) - 1} -- the support of the highest-indexed pivot alone.

    This is the whole positivity argument in one sentence and it is why the
    union, not the intersection, is the right domain: the intersection would
    throw away every coordinate above min(P), and would still be the support of
    exactly one row (the lowest pivot), so it buys nothing and costs reach.
    """
    p_max = int(pivots.max())
    support = slice(0, p_max)                   # {0, ..., p_max - 1}
    rows = a[pivots][:, support].double()
    rows = rows / rows.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
    return rows, p_max


def settle(rows: torch.Tensor, gate: torch.Tensor, m: torch.Tensor,
           beta: float) -> torch.Tensor:
    """One application of T.

        w_p(m) = gate_p * <a_p, m> ** beta,      T(m) = sum_p w_p(m) a_p / sum_p w_p(m)

    `gate_p` is row i's own softmax weight on pivot p -- the glance's opinion of
    that pivot, which is positive because p < i puts it on row i's causal
    support. `<a_p, m>` is the agreement between pivot p's reading and the
    current reading; it is strictly positive because coordinate 0 is on every
    non-empty causal row's support, so the two vectors always overlap there.

    beta is the only free parameter. beta = 0 makes T constant, which is a
    contraction with ratio 0 and is carried below as a degenerate control.
    """
    agree = rows @ m
    w = gate * agree.pow(beta)
    w = w / w.sum()
    out = w @ rows
    return out / out.sum()


def vertex_diameter(rows: torch.Tensor, gate: torch.Tensor, beta: float,
                    chunk: int = 16) -> float:
    """The EXACT projective diameter of T's image, evaluated at the extreme rays.

    The supremum of d_H(Tm, Tm') is not attained in the interior of the simplex,
    so estimating it by interior draws returns a LOWER bound on the diameter and
    therefore an OPTIMISTIC upper bound on kappa. For a positive LINEAR map the
    standard repair is to read the columns, since L e_j is column j. T is not
    linear, but it is still continuous on the closed simplex, and at a vertex

        w_p(e_j) proportional to gate_p * a_p[j] ** beta

    which is well defined -- it vanishes for every pivot p <= j and stays
    strictly positive for p = max(P), whose row covers all of S. So T(e_j) is a
    strictly positive vector and the vertices can be evaluated directly.

    This is an exact maximum over all |S| vertices, not a sample: no CI is
    reported because none is needed.
    """
    n = rows.shape[1]
    wv = gate.unsqueeze(0) * rows.transpose(0, 1).pow(beta)      # [|S|, k]
    wv = wv / wv.sum(-1, keepdim=True)
    tv = wv @ rows                                               # [|S|, |S|]
    tv = tv / tv.sum(-1, keepdim=True)
    if not bool((tv > 0).all()):
        return float("inf")
    lg = tv.log()
    best = 0.0
    for lo in range(0, n, chunk):
        blk = lg[lo:lo + chunk].unsqueeze(1) - lg.unsqueeze(0)   # [c, |S|, |S|]
        best = max(best, float((blk.amax(-1) - blk.amin(-1)).max()))
    return best


# ------------------------------------------------------- must-fire controls

def controls() -> list:
    """Every control must be SEEN to fire, and the ones that matter fire in both
    directions: a probe that can only report zero is not a probe.
    """
    out = []
    g = torch.Generator().manual_seed(11)
    p = torch.rand(32, generator=g).double() + 0.5
    p = p / p.sum()

    out.append(("C1 d_H(p, p) == 0 exactly",
                f"d_H={d_hilbert(p, p):.17g}", d_hilbert(p, p) == 0.0))

    scaled = 3.7 * p
    out.append(("C2 d_H is projective: d_H(p, 3.7 p) == 0",
                f"d_H={d_hilbert(p, scaled):.3e}", abs(d_hilbert(p, scaled)) < 1e-14))

    q = p.clone()
    q[7] = 0.0
    out.append(("C3 K-A CAN fire: one planted exact zero gives d_H = +inf",
                f"d_H={d_hilbert(p, q)}", d_hilbert(p, q) == float("inf")))

    u = torch.randn(64, generator=g).double() * 4.0
    v = torch.randn(64, generator=g).double() * 4.0
    lhs = d_hilbert(torch.softmax(u, -1), torch.softmax(v, -1))
    rhs = float((u - v).max() - (u - v).min())
    out.append(("C4 d_H(softmax u, softmax v) == osc(u - v)",
                f"d_H={lhs:.15f} osc={rhs:.15f} |diff|={abs(lhs - rhs):.3e}",
                abs(lhs - rhs) < 1e-12))

    sat = math.tanh(TANH_SATURATION_DELTA / 4.0)
    just_below = math.tanh((TANH_SATURATION_DELTA - 1e-5) / 4.0)
    out.append((f"C5 tanh(Delta/4) saturates to 1.0 at Delta={TANH_SATURATION_DELTA}",
                f"tanh={sat:.17g} at Delta-1e-5 tanh={just_below:.17g} "
                f"1-tanh(below)={1.0 - just_below:.6e}",
                sat == 1.0 and just_below < 1.0))

    getcontext().prec = 50
    worst = 0.0
    for delta in (10.0, 40.0, 75.0, 76.0, 100.0, 500.0, 1400.0):
        exact = Decimal(2) / ((Decimal(delta) / 2).exp() + 1)
        got = one_minus_kappa(delta)
        worst = max(worst, abs(float(exact) - got) / max(float(exact), 1e-300))
    out.append(("C6 closed form 1-kappa = 2/(e^(D/2)+1) matches 50-digit Decimal",
                f"worst relative error over Delta in "
                f"{{10,40,75,76,100,500,1400}} = {worst:.3e}", worst < 1e-13))

    # C7 must fire in the OTHER direction: the float subtraction is wrong at 75
    # and dead at 100, so the closed form is not decoration.
    naive75 = 1.0 - math.tanh(75.0 / 4.0)
    true75 = one_minus_kappa(75.0)
    naive100 = 1.0 - math.tanh(100.0 / 4.0)
    out.append(("C7 the naive 1-tanh path IS wrong where the closed form is not",
                f"Delta=75 naive={naive75:.6e} closed={true75:.6e} "
                f"rel={abs(naive75 - true75) / true75:.4f}  |  "
                f"Delta=100 naive={naive100:.17g} closed={one_minus_kappa(100.0):.6e}",
                naive75 != true75 and naive100 == 0.0))

    # C8 SECOND PATH. `scale/hilbert.py` was written independently against the
    # same formulas. Two implementations agreeing is a second path; two
    # diverging silently is a bug farm, so the divergence is measured here
    # rather than reconciled by picking one.
    worst_d, worst_g = 0.0, 0.0
    for _ in range(64):
        u = torch.rand(24, generator=g).double() + 1e-3
        v = torch.rand(24, generator=g).double() + 1e-3
        worst_d = max(worst_d, abs(d_hilbert(u, v) - coord.d_H(u, v)))
    for delta in (0.5, 1.0, 5.0, 20.0, 60.0, 75.0, 76.5, 100.0, 500.0, 1400.0):
        worst_g = max(worst_g, abs(one_minus_kappa(delta) - coord.one_minus_kappa(delta))
                      / coord.one_minus_kappa(delta))
    out.append(("C8 second path: scale/hilbert.py agrees on the open cone",
                f"max |d_H diff| over 64 interior pairs = {worst_d:.3e}   "
                f"max relative |1-kappa| diff over 10 Delta = {worst_g:.3e}",
                worst_d < 1e-12 and worst_g < 1e-14))

    # C9 THE ONE PLACE THE TWO PATHS DIVERGE, and it is not a bug in either:
    # two vectors sharing the SAME zero coordinate lie in the same part (face)
    # of the cone. Lemmens-Nussbaum's Theorem 2.9 takes its sup over pairs
    # restricted by the comparability relation x ~_C y, which is exactly
    # "same part". This file follows that reading and measures the distance
    # inside the shared face; `scale/hilbert.py` follows the strict-interior
    # reading and calls any boundary point infinitely far from anything.
    z1 = p.clone()
    z2 = (torch.rand(32, generator=g).double() + 0.5)
    z2 = z2 / z2.sum()
    z1[3] = 0.0
    z2[3] = 0.0
    # UPDATED at round 6 iteration 22. This control was written to assert that
    # the two paths DIVERGED on a shared-zero coordinate -- 1.503823 here against
    # +inf there. `scale/hilbert.py` has since adopted the same-part reading, so
    # the divergence is resolved and the control now asserts AGREEMENT. It is
    # kept rather than deleted because it is the only place the shared-zero case
    # is exercised at all, and a silent regression to the strict-interior
    # reading would make K-A fire on every causal draw.
    out.append(("C9 shared-zero coordinate: both paths now agree (same-part)",
                f"this file = {d_hilbert(z1, z2):.6f}   "
                f"scale/hilbert.py = {coord.d_H(z1, z2):.6f}   "
                f"|diff| = {abs(d_hilbert(z1, z2) - coord.d_H(z1, z2)):.3e}",
                math.isfinite(d_hilbert(z1, z2))
                and abs(d_hilbert(z1, z2) - coord.d_H(z1, z2)) < 1e-12))
    return out


# --------------------------------------------------------------- the audit

def positivity_audit(s: int, d: int, seed: int, dtype: torch.dtype):
    """Exact zeros on the causal support, and the within-row logit spread that
    would create them. Structural, in the sense that it reports the SPREAD --
    the quantity that decides the question for every draw -- and not merely
    whether this particular draw happened to underflow.
    """
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(s, d, generator=g)
    wq, wk = torch.randn(d, d, generator=g), torch.randn(d, d, generator=g)
    q, kk = (x0 @ wq).to(dtype), (x0 @ wk).to(dtype)
    a = bench._softmax_operator(q, kk)
    mask = torch.ones(s, s, dtype=torch.bool).tril(-1)

    logits = (q.double() @ kk.double().T) / math.sqrt(d)
    spreads = []
    for i in range(1, s):
        row = logits[i, :i]
        spreads.append(float(row.max() - row.min()))

    allowed = mask
    zeros_on_support = int(((a == 0) & allowed).sum())
    live = a[allowed]
    return dict(
        s=s, d=d, seed=seed, dtype=str(dtype),
        zeros_on_support=zeros_on_support,
        support_entries=int(allowed.sum()),
        row0_all_zero=bool((a[0] == 0).all()),
        min_positive=float(live[live > 0].min()),
        spread_max=max(spreads), spread_median=sorted(spreads)[len(spreads) // 2],
    )


def domain_draws(rows: torch.Tensor, size: int, gen: torch.Generator,
                 per_conc: int) -> list:
    """Draws spanning T's DOMAIN, not merely the neighbourhood of its image.

    Estimating a supremum from draws clustered in the middle of the simplex
    understates it, and the whole question here is how large a supremum can get.
    So the sampler is deliberately adversarial: Dirichlet concentrations from
    0.05 (nearly a vertex) to 5.0 (nearly uniform), plus the pivot readings
    themselves, which are the extreme points of T's image.
    """
    out = [r / r.sum() for r in rows]
    for conc in (0.05, 0.3, 1.0, 5.0):
        for _ in range(per_conc):
            e = -torch.rand(size, generator=gen).double().clamp_min(1e-300).log()
            m = e.pow(1.0 / conc)
            m = m / m.sum()
            out.append(m.clamp_min(torch.finfo(torch.float64).tiny))
    return out


def delta_audit(s: int, d: int, k: int, seed: int, beta: float, per_conc: int,
                dtype: torch.dtype = torch.float64):
    """Delta-hat at the ARM A geometry, three ways, all printed.

    naive   -- d_H between two whole pivot rows over the full index range. This
               is what a reading of the contract's 1.1 that ignores supports
               produces, and it is +inf for a structural reason: strictly-causal
               rows at different indices have different supports.
    hull    -- max over pivot pairs of d_H restricted to the shared support.
               This is the projective diameter of the LINEAR part of T, the map
               alpha |-> sum_p alpha_p a_p, and it is exactly the quantity
               Birkhoff's theorem takes as its hypothesis, Delta(L).
    image   -- max over sampled pairs of d_H(T m, T m'), with m ranging over the
               domain. This is the contract's own definition of Delta-hat and
               the one kill K-A reads.

    The three are printed together because the gap between `hull` and `image` is
    the finding: Delta(L) is the diameter of the image of the linear part over
    the WHOLE weight simplex, whereas T only ever visits the weights its kernel
    produces.
    """
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(s, d, generator=g)
    wq, wk = torch.randn(d, d, generator=g), torch.randn(d, d, generator=g)
    q, kk = (x0 @ wq).to(dtype), (x0 @ wk).to(dtype)
    i = s - 1
    piv = select_pivots(kk, min(k, s - 2), exclude=(i,))
    piv = piv[piv > 0]                      # pivot 0 has an empty causal row
    a = bench._softmax_operator(q, kk).double()

    naive = 0.0
    hull = 0.0
    pl = piv.tolist()
    for x in range(len(pl)):
        for y in range(x + 1, len(pl)):
            px, py = pl[x], pl[y]
            naive = max(naive, d_hilbert(a[px], a[py]))
            lo = min(px, py)
            hull = max(hull, d_hilbert(a[px, :lo], a[py, :lo]))

    rows, p_max = pivot_readings(a, piv)
    gate = a[i, piv].clone()
    gate = gate / gate.sum()
    vertex = vertex_diameter(rows, gate, beta)

    gg = torch.Generator().manual_seed(seed + 10_000)
    ms = domain_draws(rows, p_max, gg, per_conc)
    ts = [settle(rows, gate, m, beta) for m in ms]
    # The INVARIANT set. Delta(L) in Birkhoff is a sup over the whole cone, but
    # the iteration only ever sees T's image, and after two steps only T(T(.)).
    # Measuring the diameter on the set the iteration actually visits is the
    # difference between a Neumann series of 1e+10 terms and one of ~20, so the
    # second iterate is measured rather than argued about.
    t2 = [settle(rows, gate, m, beta) for m in ts]

    image = 0.0
    image2 = 0.0
    din_max = 0.0
    ratios = []
    for x in range(len(ms)):
        for y in range(x + 1, len(ms)):
            din = d_hilbert(ms[x], ms[y])
            dout = d_hilbert(ts[x], ts[y])
            d2 = d_hilbert(t2[x], t2[y])
            if math.isfinite(dout):
                image = max(image, dout)
            if math.isfinite(d2):
                image2 = max(image2, d2)
            if math.isfinite(din):
                din_max = max(din_max, din)
            if din > 1e-9 and math.isfinite(din) and math.isfinite(dout):
                ratios.append(dout / din)
    ratios.sort()
    kappa_emp = ratios[len(ratios) // 2] if ratios else float("nan")

    return dict(
        s=s, d=d, k=k, seed=seed, beta=beta, dtype=str(dtype),
        n_pivots=len(pl), p_max=p_max, support_size=p_max, n_draws=len(ms),
        delta_naive=naive, delta_hull=hull, delta_image=image,
        delta_image2=image2, delta_domain=din_max, delta_vertex=vertex,
        kappa_cert_image=math.tanh(image / 4.0),
        one_minus_kappa_image=one_minus_kappa(image),
        kappa_cert_hull=math.tanh(hull / 4.0),
        one_minus_kappa_hull=one_minus_kappa(hull),
        kappa_emp=kappa_emp, kappa_emp_max=ratios[-1] if ratios else float("nan"),
        structural_bound=beta,
        bound_respected=bool(ratios) and ratios[-1] <= beta + 1e-12,
        # Contract 1.2's Neumann truncation, priced at each of the three
        # diameters. This is the number that decides whether the implicit
        # gradient is computable, and it is not the same question as kappa < 1.
        neumann_vertex=coord.neumann_terms(vertex),
        neumann_image=coord.neumann_terms(image),
        neumann_image2=coord.neumann_terms(image2),
        neumann_hull=coord.neumann_terms(hull),
        # kappa = beta corresponds to Delta = 4 atanh(beta) = 2 log((1+b)/(1-b)).
        neumann_structural=(coord.neumann_terms(2.0 * math.log((1.0 + beta) / (1.0 - beta)))
                            if 0.0 <= beta < 1.0 else -1),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ss", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--per-conc", dest="per_conc", type=int, default=3)
    a = ap.parse_args()

    print("=== MUST-FIRE CONTROLS (read these before any number below) ===")
    ok = True
    for name, detail, fired in controls():
        ok &= fired
        print(f"  [{'FIRED' if fired else 'DID NOT FIRE'}] {name}\n      {detail}")
    if not ok:
        print("\n  A CONTROL DID NOT FIRE. Every number below would be decoration.")
        return 1

    print(f"\n=== POSITIVITY AUDIT. threads={torch.get_num_threads()} ===")
    print(f"  exp() underflows to exactly 0 below  float32 {UNDERFLOW_F32:.6f} nats"
          f"   float64 {UNDERFLOW_F64:.6f} nats")
    print(f"  {'s':>6} {'dtype':>16} {'zeros/support':>22} {'row0 dead':>10} "
          f"{'min positive':>14} {'spread med':>12} {'spread max':>12}")
    fail_pos = False
    for s in a.ss:
        for dt in (torch.float32, torch.float64):
            r = positivity_audit(s, a.d, 0, dt)
            fail_pos |= r["zeros_on_support"] > 0
            print(f"  {r['s']:>6} {r['dtype']:>16} "
                  f"{r['zeros_on_support']:>10}/{r['support_entries']:<11} "
                  f"{str(r['row0_all_zero']):>10} {r['min_positive']:>14.6e} "
                  f"{r['spread_median']:>12.4f} {r['spread_max']:>12.4f}")

    print(f"\n=== DELTA-HAT AT THE ARM A GEOMETRY. beta={a.beta} "
          f"per_conc={a.per_conc} ===")
    print(f"  {'s':>6} {'k':>5} {'seed':>5} {'|P|':>5} {'Dnaive':>8} {'Dhull':>9} "
          f"{'Dvertex':>9} {'Dimg':>9} {'Dimg2':>8} "
          f"{'kcert(vtx)':>12} {'k_emp':>9} {'<=b':>5} {'N(vtx)':>9}")
    rows = []
    for s in a.ss:
        for k in a.ks:
            for seed in a.seeds:
                r = delta_audit(s, a.d, k, seed, a.beta, a.per_conc)
                rows.append(r)
                print(f"  {r['s']:>6} {r['k']:>5} {r['seed']:>5} "
                      f"{r['n_pivots']:>5} {r['delta_naive']:>8.3g} "
                      f"{r['delta_hull']:>9.4f} {r['delta_vertex']:>9.4f} "
                      f"{r['delta_image']:>9.4f} {r['delta_image2']:>8.4f} "
                      f"{math.tanh(r['delta_vertex'] / 4.0):>12.10f} "
                      f"{r['kappa_emp']:>9.6f} "
                      f"{str(r['bound_respected']):>5} "
                      f"{r['neumann_vertex']:>9}")

    fin = [r for r in rows if math.isfinite(r["delta_image"])]
    above = [r for r in fin if r["delta_image"] >= TANH_SATURATION_DELTA]
    sat = [r for r in fin if r["kappa_cert_image"] == 1.0]
    print(f"\n  cells: {len(rows)}   Delta_image finite: {len(fin)}   "
          f"Delta_image >= {TANH_SATURATION_DELTA}: {len(above)}   "
          f"kappa_cert reads EXACTLY 1.0: {len(sat)}")
    if fin:
        print(f"  Delta_image range: {min(r['delta_image'] for r in fin):.4f} .. "
              f"{max(r['delta_image'] for r in fin):.4f} nats")
        print(f"  Delta_hull  range: {min(r['delta_hull'] for r in fin):.4f} .. "
              f"{max(r['delta_hull'] for r in fin):.4f} nats")
    print(f"  Delta_vertex (EXACT, extreme rays, no sampling) range: "
          f"{min(r['delta_vertex'] for r in rows):.4f} .. "
          f"{max(r['delta_vertex'] for r in rows):.4f} nats;  "
          f">= {TANH_SATURATION_DELTA} in "
          f"{sum(1 for r in rows if r['delta_vertex'] >= TANH_SATURATION_DELTA)}"
          f"/{len(rows)} cells")
    print(f"  interior-sampled Delta_image understates the exact vertex diameter "
          f"by a factor {min(r['delta_vertex'] / r['delta_image'] for r in rows):.3f}"
          f" .. {max(r['delta_vertex'] / r['delta_image'] for r in rows):.3f}")
    inf_naive = sum(1 for r in rows if r["delta_naive"] == float("inf"))
    print(f"  Delta_naive == +inf in {inf_naive}/{len(rows)} cells "
          f"(unrestricted supports)")
    broke = [r for r in rows if not r["bound_respected"]]
    print(f"  structural bound kappa <= beta = {a.beta}: "
          f"{len(rows) - len(broke)}/{len(rows)} cells respect it "
          f"(max observed ratio {max(r['kappa_emp_max'] for r in rows):.6f})")
    print(f"  Neumann terms for tol=1e-6 at the STRUCTURAL kappa = beta = "
          f"{a.beta}: N = {rows[0]['neumann_structural']}")
    print(f"  Neumann terms at Delta_image:  min {min(r['neumann_image'] for r in rows)}"
          f"   max {max(r['neumann_image'] for r in rows)}")
    print(f"  Neumann terms at Delta_image2 (one further settling step): "
          f"min {min(r['neumann_image2'] for r in rows)}"
          f"   max {max(r['neumann_image2'] for r in rows)}")
    print(f"  Delta_image2 range: {min(r['delta_image2'] for r in rows):.4f} .. "
          f"{max(r['delta_image2'] for r in rows):.4f} nats;  "
          f">= {TANH_SATURATION_DELTA} in "
          f"{sum(1 for r in rows if r['delta_image2'] >= TANH_SATURATION_DELTA)}/{len(rows)} cells")
    print(f"  Neumann terms at Delta_hull:   "
          f"{sorted(set(r['neumann_hull'] for r in rows))}")

    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    import json
    with JOURNAL.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\n  journal: {JOURNAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
