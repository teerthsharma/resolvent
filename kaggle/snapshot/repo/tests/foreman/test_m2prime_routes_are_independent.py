"""M2' claims FOUR routes, any ONE sufficing. This file asserts that premise.

Every assertion below is M2's own text turned into a test. They are written to
PASS if the four routes are four independent shots with kills that can fire, and
they are RED against the code as it stands. RED first, then the finding.

Each test states the M2' clause it encodes in its docstring. Nothing here is a
new mechanism; the operator, the geometry (`i=s-1, j=s/4`), the selector and the
arm construction are `scale/pivot_probe.py`'s, reached through
`scale/route_dependency.py` so the SAME code path runs.

DECLARED CHEATS:
  * draw counts are small (n=256 at s=128) so the file runs in seconds. Every
    assertion below fails by a margin far larger than the interval at that n,
    and the larger runs are in DONE.md with their Clopper-Pearson intervals.
  * the flip indicator is algebraic, not autograd; `route_dependency.py check`
    verifies the identity at 40/40 with max deviation 1.06e-06.
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scale import route_dependency as rd
from scale import s2_probe
from scale.pivot_probe import build_arm, select_pivots

S, N, K = 128, 256, 8


def _det_fraction(recs):
    return sum(rd.r2_closed_form(r["A0"], r["w0"]) for r in recs) / len(recs)


def test_r2_determinacy_is_not_also_maximal_on_a_sign_free_operator():
    """R2: 'sign-determinacy — magnitude-randomization invariance on the pivot
    block. Kill: determined fraction ~0.'

    For R2 to be evidence about SIGNED influence, an operator with no signs must
    not score better on it than the signed one."""
    with s2_probe.absmag():
        twin = _det_fraction(rd.draws(S, K, N, fast=True))
    assert twin < 0.99, (
        f"|A| -- the SAME matrix with signs stripped, magnitudes bit-identical "
        f"-- has R2 determined fraction {twin:.4f}. R2's kill is 'determined "
        f"fraction ~0'; a non-negative operator maxes the criterion out, so a "
        f"GREEN on R2 is not evidence of signedness."
    )


def test_r2_and_the_m2_flip_rate_are_independent_events():
    """M2' offers R2 as a route ADDITIONAL to M2's flip statistic. If they are
    two routes, the flip rate must not be determined by R2's own predicate."""
    recs = rd.draws(S, K, N, fast=True)
    det = [r for r in recs if rd.r2_closed_form(r["A0"], r["w0"])]
    und = [r for r in recs if not rd.r2_closed_form(r["A0"], r["w0"])]
    assert det and und, "degenerate split"
    fd = sum(r["flip"] for r in det) / len(det)
    fu = sum(r["flip"] for r in und) / len(und)
    ld, hd = rd.clopper_pearson(sum(r["flip"] for r in det), len(det))
    lu, hu = rd.clopper_pearson(sum(r["flip"] for r in und), len(und))
    assert not (hd < lu or hu < ld), (
        f"flip|R2-determined = {fd:.4f} CP[{ld:.4f},{hd:.4f}] on {len(det)} "
        f"draws vs flip|R2-undetermined = {fu:.4f} CP[{lu:.4f},{hu:.4f}] on "
        f"{len(und)} draws. Disjoint intervals: R2's predicate PARTITIONS M2's "
        f"flip statistic. They are one measurement read two ways, and R2 GREEN "
        f"is M2's flip rate going to zero."
    )


def test_r1_decoder_recall_can_fail():
    """R1: 'certified selection — group-testing decoder recall >= 1-eps.'
    A recall that cannot fall below 1 is not a measurement. The decoder's items
    are tokens; a defective item is one whose perturbation moves the readout."""
    g = torch.Generator().manual_seed(0)
    rnd = lambda *sh: torch.randn(*sh, generator=g)
    i, j = rd._geom(S)
    worst, zeros, tot = 0.0, 0, 0
    for _ in range(48):
        wq, wk = rnd(rd.D, rd.D), rnd(rd.D, rd.D)
        x0 = rnd(S, rd.D)
        gv, bt = torch.sigmoid(rnd(S)), torch.sigmoid(rnd(S))
        piv = select_pivots(x0 @ wk, K, exclude=(i, j))
        pset = {int(p) for p in piv}
        a0, h0 = build_arm("pivot_signed", x0 @ wq, x0 @ wk, gv, bt, piv)
        base = float(a0[i, j] + h0[i, j])
        off = [t for t in range(1, S - 1) if t not in pset and t not in (i, j)]
        t = off[int(torch.randint(0, len(off), (1,), generator=g))]
        x = x0.clone()
        x[t] = rnd(rd.D)
        a1, h1 = build_arm("pivot_signed", x @ wq, x @ wk, gv, bt, piv)
        d = abs(float(a1[i, j] + h1[i, j]) - base)
        worst = max(worst, d)
        zeros += int(d == 0.0)
        tot += 1
    assert worst > 0.0, (
        f"perturbing an OFF-support token moved the readout by {worst:.3e} "
        f"({zeros}/{tot} draws bitwise identical). hop2[i,j] = sum_{{p in P}} "
        f"A[i,p]A[p,j] has no term with index t when t not in P, so the "
        f"decoder's non-members are inert by construction and recall is 1.0 "
        f"whatever the decoder does. This is M2 clause 2 again, wearing R1."
    )


def test_r3_balanced_ambiguity_can_fire_on_a_non_negative_operator():
    """R3: 'non-Archimedean routing — S_max signed-tropical arm trains to <=1.10
    and balanced-ambiguity <10%.' Balanced ambiguity is the symmetrized tropical
    semiring's own failure mode: the top two paths near-tie with OPPOSITE signs.
    If it cannot be nonzero without signs, passing it says nothing about them."""
    with s2_probe.absmag():
        recs = rd.draws(S, K, N, fast=True)
    amb = 0
    for r in recs:
        w = r["w0"]
        mag = w.abs()
        o = torch.argsort(mag, descending=True)
        if w.numel() > 1:
            amb += int(mag[o[1]] > 0.9 * mag[o[0]]
                       and torch.sign(w[o[1]]) != torch.sign(w[o[0]]))
    frac = amb / len(recs)
    assert frac > 0.0, (
        f"balanced-ambiguity on |A| is exactly {frac:.4f}. A non-negative "
        f"operator has no opposite-signed pair to balance, so R3's '<10%' bar "
        f"is met perfectly by an operator with no signs. The criterion is "
        f"satisfied hardest by the thing it is supposed to rule out."
    )


@pytest.mark.parametrize("s", [32, 128, 512])
def test_r4_theta_is_injective_over_its_own_range(s):
    """R4: 'slope(theta) crosses 0 at theta* IN RANGE.' A theta* in range needs
    distinct thetas to give distinct operators."""
    a, b = rd.kbudget(s, 0.7, K), rd.kbudget(s, 1.0, K)
    assert a != b, (
        f"at s={s} the pivot budget is k={a} at theta=0.7 and k={b} at "
        f"theta=1.0 -- identical, because the budget saturates at s-3. Every "
        f"theta above ~0.65 is the SAME arm, so 'theta* in range' is not "
        f"well-posed on the upper third of the range."
    )


def test_r4_slope_theta_is_defined_everywhere_on_the_grid():
    """A zero crossing cannot be located where the function is undefined."""
    undefined = []
    for th in (0.0, 0.5, 1.0):
        rates = []
        for s in (32, 128):
            k = rd.kbudget(s, th, K)
            recs = rd.draws(s, k, N, fast=True)
            rates.append(sum(r["flip"] for r in recs) / len(recs))
        sl, npts = rd.loglog_slope([32, 128], rates)
        if math.isnan(sl):
            undefined.append((th, rates))
    assert not undefined, (
        f"slope(theta) is NaN at {undefined}: the rate hits exactly 0 at the "
        f"dense end, and `loglog_slope` returns NaN rather than clamping. "
        f"Instrument #15 died of a NaN slope read as a GREEN; R4 puts one back "
        f"in the region where its own kill has to be read."
    )


def test_r2_gives_the_same_verdict_under_both_magnitude_priors():
    """R2 cites Brualdi-Shader, whose qualitative class is magnitudes over
    (0, inf), and then specifies '10^4 resamples' without naming a prior. If the
    route is well-posed, the verdict must not depend on that unstated choice."""
    recs = rd.draws(S, K, N, fast=True)
    bounded = sum(rd.r2_closed_form(r["A0"], r["w0"]) for r in recs) / len(recs)
    unbounded = sum(rd.r2_closed_form_unbounded(r["A0"], r["w0"])
                    for r in recs) / len(recs)
    fires_b, fires_u = bounded < 0.2, unbounded < 0.2
    assert fires_b == fires_u, (
        f"R2's determined fraction is {bounded:.4f} under m~U(0,1) and "
        f"{unbounded:.4f} under m~(0,inf) on the SAME {len(recs)} draws. The "
        f"kill 'determined fraction ~0' "
        f"{'FIRES' if fires_b else 'CANNOT fire'} under the first prior and "
        f"{'FIRES' if fires_u else 'CANNOT fire'} under the second. R2 never "
        f"states which, so the route returns RED or GREEN by a choice nobody "
        f"made."
    )
