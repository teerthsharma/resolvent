"""Bind item (c) -- ceq/arm_pl.py:315's dtype fix on `brute_force_path_sums`.

THE REPAIR THIS FILE OWNS. `brute_force_path_sums` built its accumulator as
`r = np.eye(n)`, a REAL float64 array, then wrote a possibly-complex `total`
into it entry by entry. NumPy silently truncates the imaginary part on that
assignment (and raises `ComplexWarning`); the shipped line reads
`r = np.eye(n, dtype=m.dtype)` so the accumulator carries the INPUT's dtype
instead. Lines 313-329 are the whole body of the function past its docstring,
and `test_arm_pl.py`'s two `test_bind3_*` parametrizations only ever draw
REAL `A` (`arm_pl.draw_dag` is real-valued), so the complex branch of that
assignment has never executed anywhere in the repository before this file.

WHAT MAKES EACH ASSERTION RED.
  test_complex_input_needs_complex_accumulator
      Reverting ceq/arm_pl.py:315 to `r = np.eye(n)` turns this red: the
      50-seed worst gap between the LU resolvent and the brute-force sum
      jumps from float64 noise (2.8e-16, seed 39, measured below) to O(1)
      (0.55, seed 7, measured below) because every off-diagonal `total`'s
      imaginary part is truncated to zero on assignment. The threshold sits
      five orders of magnitude below the failure and five above the noise.
  test_pre_fix_accumulator_reproduces_the_measured_o1_failure
      The control for the assertion above: it runs the REJECTED line
      directly (copied, since the shipped module no longer contains it
      anywhere) against the SAME `dag_resolvent` LU solve, on the worst seed
      from the sweep, and demands the O(1) gap actually appears. If this one
      is not red, the first test's margin is not measuring what it claims.
  test_real_input_is_a_bitwise_no_op
      `m.dtype` is `float64` for real input, so `np.eye(n, dtype=m.dtype)`
      and `np.eye(n)` are the same array; this demands BITWISE equality
      against the shipped function on real `A`, so any future change to the
      real-valued path -- not just this dtype fix -- trips it too.

RULE 2 -- GUARDED ENTRIES ACTUALLY EVALUATED, counted by wrapping
`arm_pl.dag_resolvent`, `arm_pl.brute_force_path_sums` and this file's
`_brute_force_path_sums_pre_fix` and running each test alone, not reasoned
about from the grid:
  test_complex_input_needs_complex_accumulator
      50 complex128 matrices, 50 `dag_resolvent` calls, 50
      `brute_force_path_sums` calls, 1,101 nonzero complex128 edge entries
      summed over the sweep. THIS IS THE BIND.
  test_pre_fix_accumulator_reproduces_the_measured_o1_failure
      1 complex128 matrix, 1 `dag_resolvent` call, 17 nonzero entries, and
      ZERO `brute_force_path_sums` calls -- the function under repair
      executes no statement here at all. A CONTROL on the bind's margin,
      never the bind itself.
  test_real_input_is_a_bitwise_no_op
      2 real float64 matrices, 2 `brute_force_path_sums` calls, 0
      `dag_resolvent` calls.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from ceq import arm_pl

N = 9
DENSITY = 0.6
LO, HI = 0.05, 0.45
SEEDS = range(50)


def _draw_complex_dag(seed: int, n: int = N, density: float = DENSITY,
                       lo: float = LO, hi: float = HI) -> torch.Tensor:
    """A strictly-lower-triangular complex128 edge-gate matrix.

    Magnitude drawn the same way `arm_pl.draw_dag` draws its real one; phase
    uniform on the circle. `np.tril(..., -1)` keeps it nilpotent so the
    resolvent terminates, matching the real DAG's topological-order
    convention.
    """
    rng = np.random.default_rng(seed)
    mag = rng.uniform(lo, hi, (n, n))
    phase = rng.uniform(0.0, 2 * np.pi, (n, n))
    mask = rng.random((n, n)) < density
    a = mag * np.exp(1j * phase) * mask
    return torch.from_numpy(np.tril(a, -1))


def _brute_force_path_sums_pre_fix(a: torch.Tensor) -> torch.Tensor:
    """`brute_force_path_sums`, reverted to the planted negative `r = np.eye(n)`.

    Copied rather than imported: the whole point is to execute the REJECTED
    line, which the shipped module no longer contains anywhere. Everything
    else is identical to `ceq.arm_pl.brute_force_path_sums`.
    """
    m = a.detach().cpu().numpy()
    n = m.shape[0]
    r = np.eye(n)  # the planted negative: real float64, no dtype=m.dtype
    for i in range(n):
        for j in range(i):
            total = 0.0
            middle = range(j + 1, i)
            for size in range(len(middle) + 1):
                for pick in itertools.combinations(middle, size):
                    path = (j,) + pick + (i,)
                    w = 1.0
                    for u, vtx in zip(path[:-1], path[1:]):
                        w *= m[vtx, u]
                    total += w
            r[i, j] = total
    return torch.from_numpy(r).to(a.dtype)


def test_complex_input_needs_complex_accumulator():
    """RED without ceq/arm_pl.py:315's `dtype=m.dtype`. See module docstring.

    THE BIND. Instrumented: 50 guarded complex128 matrices, 50
    `dag_resolvent` calls, 50 `brute_force_path_sums` calls, 1,101 nonzero
    complex128 edge entries over the sweep.
    """
    worst = 0.0
    worst_seed = None
    for seed in SEEDS:
        a = _draw_complex_dag(seed)
        assert a.dtype == torch.complex128
        r = arm_pl.dag_resolvent(a)
        brute = arm_pl.brute_force_path_sums(a)  # executes lines 313-329
        gap = (r - brute).abs().max().item()
        if gap > worst:
            worst, worst_seed = gap, seed
    # Measured: worst post-fix gap 2.789e-16 at seed 39 (float64 noise); the
    # pre-fix accumulator's worst gap over the same sweep is 0.553 (seed 7),
    # an O(1) failure -- see test_pre_fix_accumulator_reproduces_the_measured_o1_failure.
    assert worst < 1e-9, (
        f"worst |LU - brute_force| over {len(SEEDS)} seeds = {worst:.6e} "
        f"at seed {worst_seed}; expected float64 noise (~1e-15), not the "
        f"O(1) gap a truncated-imaginary accumulator produces")


def test_pre_fix_accumulator_reproduces_the_measured_o1_failure():
    """Control: the rejected line, run directly, must actually fail at O(1).

    Not a construction-property check (rule 4): both sides call the SAME
    `arm_pl.dag_resolvent` LU solve from the shipped module; only the
    brute-force accumulator differs, by exactly the one line under repair.

    A CONTROL, NOT THE BIND. Instrumented: 1 guarded complex128 matrix, 1
    `dag_resolvent` call, 17 nonzero entries, and ZERO calls into
    `arm_pl.brute_force_path_sums` -- this test executes no statement of the
    function under repair. It measures only that the rejected line's failure
    is real, so the bind (`test_complex_input_needs_complex_accumulator`) has
    a margin worth stating.
    """
    seed = 7  # the sweep's worst seed for the pre-fix accumulator
    a = _draw_complex_dag(seed)
    r = arm_pl.dag_resolvent(a)
    broken = _brute_force_path_sums_pre_fix(a)
    gap = (r - broken).abs().max().item()
    assert gap > 0.1, f"expected an O(1) pre-fix gap, measured {gap:.6e}"


def test_real_input_is_a_bitwise_no_op():
    """The dtype fix must not move the real-valued path at all (bind item c).

    Instrumented: 2 guarded real float64 matrices, 2 `brute_force_path_sums`
    calls, 0 `dag_resolvent` calls.
    """
    for lo, hi, density in ((0.05, 0.45, 0.6), (0.30, 0.90, 1.0)):
        a = arm_pl.draw_dag(n=N, seed=15, density=density, lo=lo, hi=hi)
        assert a.dtype == torch.float64
        shipped = arm_pl.brute_force_path_sums(a)
        pre_fix = _brute_force_path_sums_pre_fix(a)
        assert torch.equal(shipped, pre_fix)
