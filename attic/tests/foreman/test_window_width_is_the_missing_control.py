"""S1's missing control, pinned as a test so it cannot go missing again.

THE CLAIM THAT WAS STRUCK. README.md:469-471 reads: "The windowed rate is flat
across a 64x growth in context ... That is a real non-vanishing regime, and it
is the standard bounded field of Mistral, Gemma and gpt-oss."

WHY THE FLATNESS PROVES NOTHING. Every windowed number in that table is at
w = 8, with the probe positions pinned at `j = i - w/2` and `c = i - w/4`. With
w = 8 those are `i-4` and `i-2`, so `out_i` depends only on tokens `[i-16, i]`
and the measured quantity is a function of a fixed 20-token tail. It CANNOT
vary with `s`. The sweep varied `s`, which by construction the answer does not
depend on, while holding fixed the variable the answer does depend on. A GREEN
here was never evidence; it was arithmetic.

THE CONTROL, AND IT IS THE SAME DEFECT AS THE FLOOR. `w` is the real scale
parameter and it was never swept. Sweeping it (below) gives a log-log slope in
WINDOW WIDTH of about -1.5, statistically indistinguishable from the -1.4 decay
in context that the windowed regime was supposed to escape. The property does
not survive a bounded field; it survives w = 8. gpt-oss's sliding window is 128;
Mistral's and Gemma's are 4096 -- at those widths the measured rate is at or
below the resolution of the draw budget.

Both this file and `test_absolute_floor_is_an_arm_filter.py` are one failure
wearing two costumes: an UNCONTROLLED SCALE PARAMETER. There, the parameter was
the discard threshold, held absolute while the arms' gradient scale moved. Here
it is the window width, held at 8 while the context moved. In both cases the
sweep varies the labelled variable and a second, unlabelled variable sets the
answer.

CPU ONLY, ON PURPOSE -- see the sibling file's docstring. Whole file is ~20 s.
"""
from __future__ import annotations

import functools
import math

import pytest
import torch

from ceq import bench

#: Draws per cell. 384 keeps the whole file under ~20 s on CPU while leaving the
#: smallest cell (w=64) resolvable at 2/384. It is NOT enough to distinguish
#: -1.5 from -1.4; the test asserts the SIGN and rough size of the slope, which
#: is what the struck claim turns on, and nothing finer.
N_DRAWS = 384
HOPS = 2

#: Fixed context for the window sweep, and fixed window for the context sweep.
S_FIXED, W_FIXED = 256, 8
WIDTHS = (8, 16, 32, 64)
SIZES = (32, 64, 128, 256)


@functools.lru_cache(maxsize=None)
def rate(s: int, w: int) -> float:
    """`window_sweep.py`'s geometry exactly: i = s-1, j = i - w/2, c = i - w/4."""
    i = s - 1
    return bench.sign_flip_rate("sgate", n_draws=N_DRAWS, s=s, i=i,
                                j=i - w // 2, c=i - w // 4, hops=HOPS,
                                window=w, seed=0, device=torch.device("cpu"))


def loglog(xs, ys) -> tuple[float, float]:
    """OLS slope and R^2 on log10 points.

    A zero rate is clamped to half the resolution `1/(2N)` rather than dropped.
    Dropping it -- which `ceq/diagnose.py::_loglog_fit` does, defensibly, since
    log10(0) is undefined -- would silently delete the steepest evidence for the
    very decay being fitted, so here the clamp is stated and the floor value is
    printed with the row.
    """
    lx = [math.log10(x) for x in xs]
    ly = [math.log10(max(y, 1.0 / (2 * N_DRAWS))) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    slope = (sum((a - mx) * (b - my) for a, b in zip(lx, ly))
             / sum((a - mx) ** 2 for a in lx))
    ss_res = sum((b - my - slope * (a - mx)) ** 2 for a, b in zip(lx, ly))
    ss_tot = sum((b - my) ** 2 for b in ly)
    return slope, (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def _both_slopes():
    w_rates = [rate(S_FIXED, w) for w in WIDTHS]
    s_rates = [rate(s, W_FIXED) for s in SIZES]
    return (WIDTHS, w_rates, loglog(WIDTHS, w_rates),
            SIZES, s_rates, loglog(SIZES, s_rates))


def _report() -> tuple[float, float]:
    ws, wr, (sw, rw), ss, sr, (ss_, rs) = _both_slopes()
    print(f"\n  sgate, hops={HOPS}, {N_DRAWS} draws/cell, cpu, seed 0")
    print(f"  window width swept at s={S_FIXED}:")
    for w, r in zip(ws, wr):
        print(f"      w={w:5d}  rate {r:.6f}  k={round(r * N_DRAWS):3d}/{N_DRAWS}")
    print(f"  context swept at w={W_FIXED}:")
    for s, r in zip(ss, sr):
        print(f"      s={s:5d}  rate {r:.6f}  k={round(r * N_DRAWS):3d}/{N_DRAWS}")
    print(f"\n  SLOPE IN WINDOW WIDTH  {sw:+.3f}  (R^2 {rw:.4f})")
    print(f"  SLOPE IN CONTEXT       {ss_:+.3f}  (R^2 {rs:.4f})")
    print("  the published table reports only the second and calls it a regime.")
    return sw, ss_


def test_both_slopes_are_printed_side_by_side():
    """The two numbers the published table needed and carried only one of.

    Printing is the deliverable here as much as asserting: any future reader of
    a windowed rate sees the window-width slope beside the context slope, which
    is the comparison that decides whether "flat in s" means anything.
    """
    slope_w, slope_s = _report()
    assert slope_w < -0.5, (
        f"window-width slope {slope_w:+.3f} is not steeply negative; the "
        f"measured decay in w is the finding this file exists to hold.")
    assert slope_s > -0.3, (
        f"context slope {slope_s:+.3f}; the windowed rate is supposed to be "
        f"flat in s, and it is -- that is the point, it is flat BY "
        f"CONSTRUCTION and therefore uninformative.")


@pytest.mark.xfail(
    strict=True,
    reason="S1. This is the README's claim -- that the non-vanishing regime is "
           "a property of a BOUNDED FIELD rather than of w=8 specifically. If "
           "it were, the rate would be flat in window width. It is not. Kept "
           "as a strict xfail so the control can never go missing again.")
def test_the_property_is_a_bounded_field_property_and_not_a_w_equals_8_property():
    """README.md:471 -- "the standard bounded field of Mistral, Gemma and gpt-oss".

    Those windows are 4096, 4096 and 128. If the property belonged to bounded
    attention as such, widening the field from 8 to 64 would leave it alone.
    """
    slope_w, _ = _report()
    assert abs(slope_w) < 0.3, (
        f"the rate falls with window width at slope {slope_w:+.3f}. The "
        f"'non-vanishing regime' is a property of w=8, not of a bounded field. "
        f"Mistral and Gemma window at 4096, gpt-oss at 128.")


def test_the_w8_geometry_cannot_depend_on_context_at_all():
    """Ground (ii) of the strike, made structural rather than statistical.

    At w=8 with j=i-4 and c=i-2, `out_i` reads tokens `[i-16, i]` and no others,
    so two runs at different `s` are the same computation on the same tail. The
    windowed row is compared directly rather than the rate, which removes the
    draw budget from the argument entirely: this is not "flat within noise", it
    is the same numbers.

    NOT BITWISE, AND THE REASON IS NOT CONTEXT. Measured residual is 2.384e-07
    against a row whose entries are order 1e-1 -- about 1e-6 relative, which is
    float32 epsilon. It is nonzero because `x @ wq` is a different SHAPE at
    s=32 and s=128, so BLAS picks a different reduction order; it is not a
    dependence on the prefix, whose tokens never enter this row. The bound is
    written RELATIVE to the row's own magnitude rather than as a bare absolute
    tolerance, which is the whole subject of the sibling test file.
    """
    torch.manual_seed(0)
    d, w = 16, 8
    rows = []
    for s in (32, 128):
        g = torch.Generator().manual_seed(7)
        x = torch.randn(s, d, generator=g)
        wq = torch.randn(d, d, generator=torch.Generator().manual_seed(11))
        wk = torch.randn(d, d, generator=torch.Generator().manual_seed(13))
        # the SAME 32-token tail in both runs, so only the prefix length differs
        tail = torch.randn(32, d, generator=torch.Generator().manual_seed(17))
        x[-32:] = tail
        a = bench._causal_sgate_operator(x @ wq, x @ wk, window=w)
        rows.append(a[s - 1, s - 1 - w:s - 1])
    delta = float((rows[0] - rows[1]).abs().max())
    scale = float(rows[0].abs().max())
    print(f"\n  max |A[i, i-8:i]| difference between s=32 and s=128: {delta:.3e}"
          f"   row scale {scale:.3e}   relative {delta / scale:.3e}")
    assert delta / scale < 1e-5, (delta, scale)
