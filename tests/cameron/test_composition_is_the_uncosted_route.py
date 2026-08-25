"""The stated tradeoff, encoded as an assertion, on the shipped operator.

`CHECKLIST.md:50`, the M2 baseline-impossibility line, in its own words:

    every dense signed arm decays s^-1.1..-1.7; windowed arms buy flatness
    only by SURRENDERING REACH.

and `CHECKLIST.md:198` and `:271`, in the vision paragraph:

    dense signed operators decay, and windowed operators surrender reach.

That is a claim about VALUES -- a bounded-row-width arm's reach at a distance
that grows with `s` -- so it is written here as a comparison of values, and the
arms that are supposed to occupy the two horns are measured side by side on one
instrument at one geometry.

WHAT THE PUBLISHED WINDOWED NUMBERS MEASURED. `scale/window_sweep.py` pins
`j = i - w/2` and `c = i - w/4`. At `w = 8` that is `j = i-4`, `c = i-2`, and
`RESEARCH.md:158` already records the consequence: `out_i` reads only
`[i-16, i]`, so *"the measured quantity cannot vary with s."* The flat row
0.124023 / 0.133301 / 0.108398 / 0.120605 is therefore not evidence that a
bounded receptive field arrests the decay. THE GEOMETRY HERE IS DIFFERENT ON
PURPOSE: `j = i - s//2`, so the span the operator must cross GROWS with `s`, and
the quantity is free to vary with `s`. `test_the_geometry_is_s_sensitive` is the
must-fire control for exactly that, and it is checked before any verdict.

Every number is CPU-only and off `ceq.bench`'s own `flip_rate`, at `floor = 0.0`
-- the scale-free cut. `flip_rate`'s own docstring says an absolute floor is
"an ARM-DEPENDENT SAMPLE FILTER rather than a dust rule", and depth moves the
gradient scale by 30 orders of magnitude here (median |grad| 2.8e-32 at depth 4
against 1.3e+03 at depth 4 on a shorter span), so `floor = 1e-6` would not be
comparing arms at all. `flipabs` is reported beside it, never asserted on.
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import dilated as D  # noqa: E402
import ladder as L  # noqa: E402
from ceq import bench  # noqa: E402

#: `CHECKLIST.md:52` -- "Kill: slope(c in P) < -0.3".
M2_BAR = -0.3

#: Kept small enough that the whole file is a test rather than a job. The
#: precision runs behind the reported numbers are `n = 1024` (`s <= 512`) and
#: `n = 256 x 4 seeds` (`s = 2048`); this file re-derives the SHAPE at `n = 256`
#: and every assertion below is about a value that is 0.0 or 1.0, not marginal.
N = 256
SIZES = (32, 128, 512)
DEV = torch.device("cpu")


def _cell(arm: str, s: int, n: int = N):
    draws, sch, dist = L.run(arm, s, n, seed=0, device=DEV)
    return dict(reach=D.reach_fraction(draws),
                flip0=bench.flip_rate(draws, floor=0.0),
                flipabs=bench.flip_rate(draws, floor=1e-6),
                depth=len(sch), dist=dist, sch=sch, n=n)


def _slope(sizes, rates):
    """Least-squares log-log slope, on the cells whose rate is strictly positive.

    A zero rate has no logarithm. Fitting through it is what produced this
    project's withdrawn -1.389, so zeros are DROPPED and their count is reported
    by the caller rather than being silently coerced.
    """
    pts = [(math.log10(s), math.log10(r)) for s, r in zip(sizes, rates) if r > 0]
    if len(pts) < 2:
        return float("nan")
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return num / den if den else float("nan")


# ---------------------------------------------------------------------------
# BINDS. Instrument #17 was a correct measurement of an operator that ships
# nowhere. `dilated.py` is a reimplementation, so it is pinned to the shipped
# code BITWISE before any number off it is believed.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("s", [16, 32, 64, 128])
@pytest.mark.parametrize("w", [4, 8, 16])
def test_the_dilated_mask_is_the_shipped_mask_at_dilation_one(s, w):
    """`dilated_mask(..., dilation=1)` IS `bench._causal_mask(..., window=w)`."""
    assert torch.equal(D.dilated_mask(s, DEV, w, 1),
                       bench._causal_mask(s, DEV, w))


def test_the_dilated_mask_is_not_the_shipped_mask_at_dilation_two():
    """MUST-FIRE CONTROL for the bind above: `torch.equal` can return False.

    A bind that only ever sees `True` is a check that cannot fail. This is the
    same object at `dilation = 2`, and it must differ -- otherwise the bind
    above is comparing something to itself.
    """
    assert not torch.equal(D.dilated_mask(32, DEV, 8, 2),
                           bench._causal_mask(32, DEV, 8))


def test_the_composed_probe_is_the_shipped_probe_at_depth_one():
    """`composed_draws([1])` reproduces `bench.sign_flip_draws(depth=1, window=w)`.

    Bitwise on the float pairs, not approximately: same RNG stream, same path
    sum, same readout. If this fails, every number in this file is off a
    different operator than the one the repository ships.
    """
    mine = D.composed_draws(n_draws=32, s=32, d=16, i=31, j=27, c=29,
                            dilations=[1], window=8, hops=2, seed=0, device=DEV)
    shipped = bench.sign_flip_draws("sgate", n_draws=32, s=32, d=16, i=31, j=27,
                                    c=29, hops=2, window=8, depth=1, seed=0,
                                    device=DEV)
    assert mine == shipped


# ---------------------------------------------------------------------------
# MUST-FIRE CONTROL for the verdict tests.
# ---------------------------------------------------------------------------

def test_the_geometry_is_s_sensitive():
    """The defect that voided the published windowed row must not repeat here.

    `RESEARCH.md:158`: at `j = i-4` the measured quantity CANNOT vary with `s`,
    so a flat row there is a tautology. This control demands that at THIS
    geometry the instrument is seen to move with `s`: the contiguous `w = 8`
    band, composed to the same depth the dilated arm uses, must reach `j` at
    `s = 32` and must FAIL to reach it at `s = 512`.

    That is the same statistic, on the same operator, at the same depth, moving
    from 1.0 to 0.0 as `s` grows -- so a value read here is a reading and not a
    constant. If this control is silent, no verdict below may be taken.
    """
    lo = _cell("contig", 32)
    hi = _cell("contig", 512)
    assert lo["reach"] == 1.0, f"CONTROL SILENT: contig s=32 reach {lo['reach']}"
    assert hi["reach"] == 0.0, f"CONTROL SILENT: contig s=512 reach {hi['reach']}"


def test_the_instrument_still_reads_the_m2_decay():
    """MUST-FIRE CONTROL: the arm M2 killed must still be seen to die here.

    The whole verdict below is "an arm did not decay". That is worth nothing
    unless this instrument, at THIS geometry, still reads the decay it was built
    to read. So the shipped `sgate` at `window = 0`, `depth = 1` -- the arm
    `PROGNOSIS.md:40` records at slope -1.298 -- is run through the same
    `_cell`/`_slope` path and must come in under the same -0.3 bar.

    If this goes silent, `test_the_arm_that_reaches_decays_at_the_m2_rate` is
    measuring an instrument that cannot return a decay and its RED means nothing.
    """
    got = {s: _cell("glob_1", s) for s in SIZES}
    rates = [got[s]["flip0"] for s in SIZES]
    sl = _slope(SIZES, rates)
    row = "  ".join(f"s={s} flip0={got[s]['flip0']:.6f}" for s in SIZES)
    assert sl <= M2_BAR, f"CONTROL SILENT: glob_1 slope={sl:.6f}  {row}"


@pytest.mark.parametrize("s", [32, 128, 512, 2048])
def test_the_row_width_is_constant_and_the_support_is_the_whole_context(s):
    """The two numbers the tradeoff says cannot both hold. Pinned, not argued.

    `workdone2.md` section 7 states the mechanism: the background `B_k(s)`
    inherits the scale of tokens promoted by a selection ranging over `s`, so
    what is needed is an aggregation whose surviving background does not. A row
    that sums over a FIXED `w` entries at every `s` has no such selection --
    there is nothing ranging over `s` to promote.

    So this pins both halves at once, as exact integers rather than bounds:

      * `max_j nnz(A[i, :])` over every layer -- must be exactly `window`, at
        every `s`. This is the quantity that decays the global arm.
      * `|{ j : d(out_i)/d(v_j) != 0 }|` -- must be exactly `s`. This is REACH,
        and `s` is not "most of the context", it is all of it.

    `RESEARCH.md:158` voided the published windowed row because `out_i` read
    only `[i-16, i]` and so could not vary with `s`. The second number here is
    the direct refutation of that failure mode for this arm: the support is the
    whole context and it GROWS with `s`, 32 -> 2048.
    """
    sch = D.log_schedule(s)
    masks = [D.dilated_mask(s, DEV, 8, dl) for dl in sch]
    assert max(int(m.sum(-1).max()) for m in masks) == 8

    g = torch.Generator().manual_seed(0)
    d, i, depth = 16, s - 1, len(sch)
    rnd = lambda *sh: torch.randn(*sh, generator=g)
    wq = [rnd(d, d) for _ in range(depth)]
    wk = [rnd(d, d) for _ in range(depth)]
    wo = [rnd(d, d) for _ in range(depth)]
    x, v = rnd(s, d), rnd(s, d).requires_grad_(True)
    h = v
    for layer in range(depth):
        a = D.sgate_masked(x @ wq[layer], x @ wk[layer], masks[layer])
        acc, term = h, h
        for _ in range(2):
            term = a @ term
            acc = acc + term
        h = acc @ wo[layer]
    grad, = torch.autograd.grad(h[i].sum(), v)
    assert int((grad.abs().sum(-1) > 0).sum()) == s


# ---------------------------------------------------------------------------
# THE VERDICT TESTS. These assert the repository's stated position.
# ---------------------------------------------------------------------------

def test_windowed_operators_surrender_reach():
    """`CHECKLIST.md:50` -- "windowed arms buy flatness only by surrendering reach".

    A dilated band holds the row width at exactly `window` entries, so it is a
    windowed operator by the only definition that matters to the decay mechanism
    -- its row normalizer sums over `w` terms and never over `s`. The claim says
    such an arm cannot reach `j` at a distance that grows with `s`.

    The control in `test_the_geometry_is_s_sensitive` shows this exact statistic
    reading 0.0 for the contiguous band at `s = 512`, so 0.0 is reachable here
    and a non-zero reading is a fact rather than an instrument artifact.
    """
    got = {s: _cell("dil", s) for s in SIZES}
    row = "  ".join(f"s={s} reach={c['reach']:.6f} depth={c['depth']} "
                    f"dist={c['dist']} rowwidth={8}" for s, c in got.items())
    for s, c in got.items():
        assert c["reach"] == 0.0, (
            f"windowed operators do NOT surrender reach: {row}")


def test_the_arm_that_reaches_decays_at_the_m2_rate():
    """`CHECKLIST.md:52` -- the M2 kill: an arm at global reach decays below -0.3.

    Applied to whichever arm actually reaches `j` at every `s`. If a bounded
    -row-width arm reaches, the claim is that it must pay the dense arm's decay;
    this pins that as a slope against the pre-registered bar.

    The zeros are dropped from the fit rather than fitted through -- the
    withdrawn -1.389 was produced by fitting through a floor artifact -- and the
    number of dropped cells is carried in the failure message.
    """
    got = {s: _cell("dil", s) for s in SIZES}
    if min(c["reach"] for c in got.values()) < 1.0:
        pytest.skip("no bounded-row-width arm reaches at every s; nothing to fit")
    rates = [got[s]["flip0"] for s in SIZES]
    sl = _slope(SIZES, rates)
    dropped = sum(1 for r in rates if r == 0.0)
    row = "  ".join(f"s={s} flip0={got[s]['flip0']:.6f} "
                    f"flipabs={got[s]['flipabs']:.6f} depth={got[s]['depth']}"
                    for s in SIZES)
    assert sl <= M2_BAR, (
        f"an arm at global reach did NOT decay: slope={sl:.6f} against bar "
        f"{M2_BAR}, {dropped} zero cells dropped from the fit.  {row}")
