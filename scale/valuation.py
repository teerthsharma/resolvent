"""X4 — the sign-flip statistic on VALUATIONS, not floats. The floor is DELETED.

WHAT THE FLOAT INSTRUMENT ACTUALLY DOES WRONG, and it is sharper than "the floor
is set too high".

The shipped statistic is

    if lo * hi < 0 and min(abs(lo), abs(hi)) > floor:  flips += 1

Two independent defects, and the second is the one nobody had named:

  1. THE FLOOR discards true flips whose magnitude is small. Measured at depth:
     a composed arm reads 0.386719 at floor=0 and 0.000000 at floor=1e-6 --
     **100% of its flips discarded** -- because depth moves gradient scale ~30
     orders (median |grad| 2.8e-32 at depth 4). Deleting the floor fixes this.

  2. `lo * hi` MULTIPLIES, and the PRODUCT can underflow to zero when neither
     FACTOR does. float32's smallest normal is ~1.18e-38. At lo = 1e-30 and
     hi = -1e-30 -- both perfectly representable, both far from underflow --
     the product is -1e-60, which flushes to **-0.0**, and `-0.0 < 0` is
     **False**. A genuine, unambiguous sign flip is counted as NO FLIP.
     **Deleting the floor does NOT fix this. Only not multiplying does.**

Defect 2 is why this is a VALUATION instrument rather than "the same test with
floor=0". A valuation carries (sign, exponent, mantissa) separately, so the sign
of a product is the product of the signs -- an exact operation on {-1,+1} that
has no dynamic range and cannot underflow, at any depth, ever.

CALIBRATED AT BOTH ENDS, as the contract requires:
  * it must REPRODUCE every published floor=0 number EXACTLY -- a new instrument
    that moves an old number is a new arm, not an instrument;
  * a planted 30-order-spread case must be read correctly HERE where the float
    path is PROVABLY wrong -- not "differently", provably: the float path
    returns a value whose own arithmetic is demonstrably a flushed zero.

All previously published FLOORED numbers stay quoted as historical readings of a
float instrument. They were correct readings of the wrong instrument.
"""
from __future__ import annotations

import math

#: A valuation of a real number: (sign, exponent, mantissa) with
#: `x = sign * mantissa * 2**exponent`, `mantissa in [0.5, 1)` for x != 0.
#: Zero is (0, 0, 0.0) -- a distinct state, not a very small number.
Valuation = tuple[int, int, float]


def valuation(x: float) -> Valuation:
    """(sign, exponent, mantissa). Exact for any finite float, no floor."""
    if x == 0.0:
        return (0, 0, 0.0)
    m, e = math.frexp(abs(x))
    return (1 if x > 0.0 else -1, e, m)


def v_is_zero(v: Valuation) -> bool:
    return v[0] == 0


def v_opposite_signs(a: Valuation, b: Valuation) -> bool:
    """True iff a and b are both nonzero with OPPOSITE signs.

    THIS IS THE WHOLE POINT. It compares the sign FIELDS -- an operation on
    {-1, 0, +1} -- and never forms the product `a * b`. The float path forms
    that product and can flush it to zero while both operands are healthy, which
    turns a real flip into a missed one. Here there is no product to flush.
    """
    return a[0] != 0 and b[0] != 0 and a[0] != b[0]


def v_compare_magnitude(a: Valuation, b: Valuation) -> int:
    """-1 / 0 / +1 for |a| vs |b|, exact across ANY dynamic range.

    Compares exponents first and mantissas only to break a tie, so |a| and |b|
    may differ by hundreds of orders without either being formed as a ratio or a
    difference -- both of which lose the comparison to underflow or cancellation.
    """
    if v_is_zero(a) and v_is_zero(b):
        return 0
    if v_is_zero(a):
        return -1
    if v_is_zero(b):
        return 1
    if a[1] != b[1]:
        return -1 if a[1] < b[1] else 1
    if a[2] == b[2]:
        return 0
    return -1 if a[2] < b[2] else 1


def flip_rate(draws, *, floor: float = 0.0) -> float:
    """Sign-flip rate over (lo, hi) gradient pairs. NO FLOOR by default.

    `floor` exists ONLY so this can reproduce a historical floored reading for
    comparison. It defaults to 0.0 -- deleted, not defaulted to a small number --
    and the docstring of every caller that passes a nonzero floor must say which
    historical number it is reproducing and why.
    """
    if not draws:
        return 0.0
    n = flips = 0
    for lo, hi in draws:
        n += 1
        a, b = valuation(lo), valuation(hi)
        if not v_opposite_signs(a, b):
            continue
        if floor > 0.0:
            fv = valuation(floor)
            smaller = a if v_compare_magnitude(a, b) < 0 else b
            if v_compare_magnitude(smaller, fv) <= 0:
                continue
        flips += 1
    return flips / n if n else 0.0


def float_flip_rate(draws, *, floor: float = 0.0) -> float:
    """The SHIPPED float statistic, verbatim, kept for calibration and contrast."""
    if not draws:
        return 0.0
    n = flips = 0
    for lo, hi in draws:
        n += 1
        if lo * hi < 0 and min(abs(lo), abs(hi)) > floor:
            flips += 1
    return flips / n if n else 0.0
