"""Structural-zero guard: one check for the class that outnumbers every other
failure mode in this project's record.

CLASS: a control or kill quantity is scored as a MEASUREMENT (a rate, a delta,
a slope) while the code path that produces it is wired so it can return
nothing but a constant -- because the input being "perturbed" is excluded, by
construction, from every term the output reads.

FOUND IN THIS REPO, same shape, each caught by hand, after the fact, in a
separate investigation:

  - `scale/pivot_probe.py` M2 `not_in_P` arm: `hop2[i,j] = sum_{p in P}
    A[i,p]A[p,j]` sums only over P; a `c not in P` draw cannot reach it.
    max|delta| = 0.0 over 8/8 draws. (tests/chase/test_m2_not_in_P_is_structural_zero.py)
  - `ceq/eviction.py` M4 control: `settle_evicted` reads `x[keep]` only, and
    the perturbed token is chosen via `exclude=keep` -- it can never land
    inside `keep`. `tests/w3/test_w3_eviction.py:96` asserted the resulting
    `0.0` as if it were a passing measurement.
  - `scale/dfloor_probe.py` v1: kept `min(4, nblk)` blocks; at two of three
    sizes tested, `nblk <= 4`, so every block was kept and recall was 1.0 by
    construction. DONE.md calls it "the eighth instrument in this project
    that was internally consistent and externally wrong".
  - the Lean-refcount schedule selector: in a single sequence every
    block-aligned prefix occurs exactly once, so every score is `refcount -
    1 = 0` -- "the selector is a coin flip by construction" (DONE.md Round 12).
  - the multizoom R5 harness: its schedule placed `c = s // 2` inside the
    schedule by construction, so the claimed result was true at every size
    tested and false as a measurement (DONE.md R5, deleted).

Five incidents, five separate hand-run investigations, iterations apart, all
answering the same one-line question that was never asked in advance: CAN the
input you are about to perturb reach the output you are about to read?

THE CHECK. Perturb the input a control claims to vary; run the REAL, SHIPPED
computation (never reimplemented -- see the "two implementations" note below)
across several independent draws; require at least one draw to move the
output past a floor. If every draw lands on the same value, the quantity is
an identity wearing a measurement's clothes, and this raises before anyone
gets to read a rate or a slope off of it.

WHY NOT REIMPLEMENT THE OPERATOR. A checker that rebuilds the computation
under test can drift from what ships and pass against its own copy while the
shipped path is broken -- this is the exact shape of the 89,400.180-vs-1.667
defect (a parity test that compared the gated path only against this file's
own `stock_attention`, both wrong the same way). Callers of this guard must
import the function under test, not restate its logic.

CALIBRATION. A guard that has never been observed passing is not trustworthy
either -- see `run_calib.py`'s own `--self-test`. Every caller of this file
should exercise it once against the arm's own positive-control counterpart
(the `in_P` / "before anything reads it" side) and require that call GREEN,
beside the RED call on the structural-zero side. Both example callers in
`tests/chase/test_structural_zero_guard.py` do this.
"""
from __future__ import annotations

from collections.abc import Iterable


def assert_perturbation_moves_output(
    deltas: Iterable[float], *, what: str, floor: float = 0.0
) -> None:
    """`deltas`: one `|before - after|` per independent draw, already computed
    by perturbing the input and re-running the real, shipped computation.

    Raises with every value shown if none exceeds `floor` -- the exact shape
    of all five incidents above: not a small effect, EXACTLY constant, every
    draw, not one outlier.
    """
    vals = list(deltas)
    assert vals, f"{what}: zero draws were usable -- nothing was tested"
    moved = [v for v in vals if v > floor]
    assert moved, (
        f"STRUCTURAL ZERO: {what} -- {len(vals)} draws, every |delta| <= "
        f"{floor}: {vals!r}. The perturbed input cannot reach this output; "
        f"the quantity is an identity dressed as a measurement, not a "
        f"control."
    )
