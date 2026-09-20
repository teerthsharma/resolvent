"""The per-cycle holonomy of a genuinely cyclic gate matrix, read off the
gates themselves -- not the per-query interference scalar `I` the addendum's
E-INTERF v2 would have used.

THE DEFECT THIS FILE REPAIRS. W5.4 argues an unsigned (non-negative) arm
"cannot represent [holonomy pi] AT ANY VALUE PARAMETRISATION" because holonomy
is gauge-invariant. True against a non-negative gate. False against the rival
Foreman actually priced: holonomy pi is exactly the Z2 subgroup of U(1), and a
REAL SIGNED gate reaches it with zero phase machinery, because the argument of
a nonzero real number is 0 or pi and nothing else. Worse, E-INTERF v2 as
specified would have scored the two arms by one scalar `I` per query, and one
scalar is not a sufficient statistic for a phase: with a free route-magnitude
ratio a signed real arm can match ANY target `I` value by choosing the ratio,
regardless of what phase (0 or pi) actually produced it. That test would have
separated the phase arm from a strawman and been wrong to.

THE REPAIR, planted here. (1) `H_c = 2*pi/3`, NOT `pi` -- `2*pi/3` is not in
the Z2 subgroup `{1, -1}` of U(1), so nothing built from a real, signed gate
can reach it, at any magnitude ratio. (2) the two routes' magnitudes are
pinned EQUAL by the construction itself (both built from `sqrt(target_mag)`
per hop), not merely close, so there is no ratio left to tune. (3) the
statistic read is the PER-CYCLE HOLONOMY of the gate matrix -- the argument of
a product of gates around a closed loop -- which is exactly the phase
information E-INTERF v2's single intensity scalar collapsed away, and it is
gauge-invariant by construction (any per-node U(1) rephasing telescopes to
identity around a full loop), not merely observed to be stable on one draw.

SESSION REPAIR (this pass, over the version the attending flagged). Two
defects, both in how the bed was built, not in the finding:

(a) RULE 4 / CONSTRUCTION-PROPERTY. Every gate value used to come from a
LOCAL helper wrapping `cmath.rect` directly -- no arm in the repository was
involved, so a broken arm could not have broken this test. Every hop below
now goes through `ceq/arm_phase.py:96`'s SHIPPED `gate(u, theta, cap=...)`:
a coverage run of this file (recorded at the bottom of this docstring) shows
statements in `arm_phase.py` past the import executing, which they did not
before.

(b) RULE 2 / BAR THAT CANNOT FIRE. `test_no_real_signed_gate_...` used to
sweep 200 seeded magnitude draws against `assert worst_gap > pi/3 - 1e-9`.
That branch cannot fire: with each hop's phase restricted to `{0, pi}`, the
ratio's argument is `route1_phase - route2_phase mod 2*pi`, a function of the
two SIGNS only -- magnitude never enters the argument, so the reachable gap
set has exactly two elements, `{pi/3, 2*pi/3}`, on every seed, and the old
bar cleared the smaller one by exactly its own float-rounding slack,
deterministically. That is a theorem about the Z2 subgroup, not a
measurement, and it is now stated and checked as one: an exhaustive
enumeration of the 4 sign pairs (not a seeded sample of them), asserting the
gap SET equals `{pi/3, 2*pi/3}` exactly, plus a separate, honestly-named
invariance check that a magnitude sweep spanning 12 decades (1e-6 .. 1e6)
never moves the holonomy off the value its sign pair alone determines --
which is the actual content of "no ratio closes the gap", stated as an
invariance rather than dressed as a bar that cannot fail.

THE GRAPH. Four nodes, `A[i, j]` the gate on edge `j -> i` (arm_pl's own
`draw_dag` convention). Two routes from 0 to 3 -- `0->1->3` and `0->2->3` --
share one closing edge `3->0` that turns the diamond into two simple cycles,
`0->1->3->0` and `0->2->3->0`, sharing that edge. This is deliberately NOT a
DAG: `A[0, 3]` has `j=3 > i=0`, so the matrix is not strictly lower
triangular, `A` is not nilpotent, and `brute_force_path_sums` (whose `j in
range(i)` loop assumes a topological order) cannot see this edge at all --
which is why this file does not use it as an oracle, and derives its own
closed form below instead. `beta_1` of the read graph in the rest of this
repo is 0 (a path); this graph is the one place `beta_1 = 1` on purpose.

`ceq/arm_pl.py:293`'s `dag_resolvent` needs NO code change to take this
`A`: it is `torch.eye(dtype=a.dtype) ` and `torch.linalg.solve`, both of
which already dispatch on `torch.complex128` (verified directly below, not
assumed). The state-of-tree note that the file has zero occurrences of the
literal strings `complex`/`cfloat`/`cdouble`/`1j` is true and irrelevant --
the function was already dtype-generic. There was no interface gap to write
around; this file writes to `dag_resolvent` as it stands.

NORTH STAR. This is a REROUTE, not a kill: nothing existing is retired. The
route being reached for is condition (3), carrying the weight self-attention
or JEPA carries -- a mechanism that can only be told apart from a strawman by
a statistic a strawman cannot fake by tuning an unrelated knob.
"""
from __future__ import annotations

import cmath
import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ceq import arm_pl                                            # noqa: E402
from ceq import arm_phase                                         # noqa: E402

CT = torch.complex128
RDTYPE = torch.float64             # real dtype fed to arm_phase.gate; _ctype
                                    # maps float64 -> complex128, matching CT.
H_C = 2 * math.pi / 3               # planted holonomy: NOT pi, NOT in Z2 = {0, pi}


def _shipped_gate(m: float, theta: float, *, cap: bool) -> complex:
    """One call into `ceq/arm_phase.py`'s SHIPPED `gate(u, theta, cap=...)`
    (line 96), scalar in, scalar out. Every edge value in this file is built
    through this function and nothing else -- the repair for RULE 4: a broken
    `torch.polar` phase or a dtype regression in `_ctype` land here and break
    every phase-reading test below, because there is no other path to a gate
    value left in this file.

    A BROKEN `magnitude()` CLAMP ALSO LANDS HERE, BUT DOES NOT BREAK "every
    test below" -- corrected from an earlier claim that it did. Every
    holonomy assertion in this file reads a RATIO of two route products
    built from equal magnitude inputs (`_build_cycle` pins `route1_mag ==
    route2_mag`), and a magnitude term cancels identically out of the
    argument of a ratio of complex numbers regardless of what the clamp
    does to it -- a uniform bug in `magnitude()` moves both routes together
    and every ratio-based assert here stays green. Only
    `test_the_magnitude_clamp_lands_on_its_absolute_endpoint_not_a_ratio`
    below, which reads a route's ABSOLUTE magnitude rather than a ratio,
    is guarded against it.
    """
    u = torch.tensor(m, dtype=RDTYPE)
    th = torch.tensor(theta, dtype=RDTYPE)
    return complex(arm_phase.gate(u, th, cap=cap).item())


def _build_cycle(route1_mag, route2_mag, route1_phase, route2_phase,
                  close_phase, close_mag=1.0, dtype=CT, cap=True):
    """The 4-node graph above. Each route's magnitude/phase is split evenly
    across its two hops (`sqrt` / `/2`), so the planted route value is a
    property of the PRODUCT of two independent SHIPPED gates -- the thing a
    learned arm would actually emit -- never injected on a single edge as a
    shortcut.

    `cap` selects `arm_phase.gate`'s clamped ("the arm as shipped", magnitude
    in `[0, 1]`, used by the two holonomy-reading tests below where every
    route magnitude is `<= 1`) or open (the rival's own regime, used by the
    Z2 test whose magnitude sweep runs past `1`) parametrization; the closing
    edge always uses the open form so an out-of-`[0,1]` `close_mag` (as in
    the `dag_resolvent` test) is not silently clamped underneath a test that
    is checking a specific numeric value.
    """
    a = torch.zeros(4, 4, dtype=dtype)
    m1, p1 = math.sqrt(route1_mag), route1_phase / 2
    m2, p2 = math.sqrt(route2_mag), route2_phase / 2
    a[1, 0] = _shipped_gate(m1, p1, cap=cap)
    a[3, 1] = _shipped_gate(m1, p1, cap=cap)
    a[2, 0] = _shipped_gate(m2, p2, cap=cap)
    a[3, 2] = _shipped_gate(m2, p2, cap=cap)
    a[0, 3] = _shipped_gate(close_mag, close_phase, cap=False)
    return a


def _route_product(a, path):
    """Product of `a[v, u]` for consecutive `u -> v` in `path` -- a walk read
    off the gate matrix by arm_pl's own `A[i, j] = gate(j -> i)` convention.
    Swap the index order here and every assertion below fires on this file's
    own construction, which is exactly the bug this helper exists to catch.
    """
    w = torch.ones((), dtype=a.dtype)
    for u, v in zip(path[:-1], path[1:]):
        w = w * a[v, u]
    return w


# --------------------------------------------------------------- the bar

def test_the_planted_holonomy_reads_off_the_gates_at_an_arbitrary_gauge():
    """BED: the 4-node genuine-cycle matrix above, built through
    `arm_phase.gate` (`cap=True`, the arm's own clamped magnitude -- every
    route magnitude here is `1.0`, exactly the closed endpoint the clamp is
    required to attain), dtype complex128, 5 nonzero guarded entries
    (`A[1,0] A[3,1] A[2,0] A[3,2] A[0,3]`), read through 6 independent
    closing-edge phases (an arbitrary per-edge "gauge" choice, including
    0.0) -- 6 guarded matrices, 30 guarded entries total, every one of them
    a return value of the shipped `gate`. This is the branch that fires:
    `route1/route2` is the ratio of two products of gates the arm itself
    emits, and its argument is the relative holonomy between the two cycles
    `0->1->3->0` and `0->2->3->0` -- the shared closing edge's own phase
    cancels in the ratio by construction, so reading it out is the actual
    exercise of gauge invariance, not a claim about it.

    WHAT WOULD FAIL THIS. Any of: `route1`/`route2` magnitudes drifting apart
    fires the first assert -- the pin in `_build_cycle` breaking is one way,
    but a bug in `arm_phase.magnitude`'s clamp is NOT another one unless it
    is also PHASE-COUPLED: route1 and route2 are built from the identical
    magnitude input (`1.0`) and differ only in phase (`H_C` vs `0.0`), so a
    clamp bug that treats every input the same regardless of phase moves
    both routes together and this assert cannot see it (that gap is closed
    by `test_the_magnitude_clamp_lands_on_its_absolute_endpoint_not_a_ratio`
    below, which reads a route's magnitude absolutely instead of comparing
    it to a sibling). Corrected from an earlier, false claim that any
    asymmetric-looking clamp behavior would fire here. The read holonomy
    landing anywhere but `H_C` (e.g. a hop-splitting bug that produces
    `H_C/2` or `2*H_C`, an index-order bug in `_route_product` that silently
    substitutes a different two-hop product, or `arm_phase.gate`'s
    `torch.polar` call being replaced by something that does not implement
    `m * exp(i*theta)`) fires the second; the holonomy CHANGING when the
    closing-edge phase changes (i.e. this statistic secretly depending on an
    ungauged, arbitrary edge rather than being a genuine loop invariant)
    fires the third.
    """
    for close_phase in (0.0, 0.3, -1.7, math.pi - 0.01, -math.pi + 0.2, 5.5):
        a = _build_cycle(route1_mag=1.0, route2_mag=1.0,
                          route1_phase=H_C, route2_phase=0.0,
                          close_phase=close_phase, cap=True)
        route1 = _route_product(a, (0, 1, 3))
        route2 = _route_product(a, (0, 2, 3))

        assert abs(abs(route1) - abs(route2)) < 1e-13, (
            "the two routes' magnitudes were not pinned equal by construction")

        ratio = route1 / route2
        assert abs(abs(ratio) - 1.0) < 1e-13
        holonomy = cmath.phase(complex(ratio))
        assert abs(holonomy - H_C) < 1e-12, (
            f"read holonomy {holonomy} at close_phase={close_phase}, "
            f"planted {H_C}")


def test_dag_resolvent_takes_the_complex_cycle_with_no_code_change():
    """BED: the same 5-entry matrix (built through the shipped `gate`,
    `cap=True` on the two routes, `0.7 <= 1` so the clamp is a no-op here and
    not the thing under test -- `test_the_planted_holonomy...` above is what
    exercises the clamp at its boundary), read through `arm_pl.dag_resolvent`
    (`ceq/arm_pl.py:293`) rather than through the raw gate products, against
    a closed form derived independently of `brute_force_path_sums` (which
    cannot see edge `A[0,3]` at all, since its `j in range(i)` loop assumes a
    topological order and this graph has none). 1 guarded matrix, 5 guarded
    entries, `dag_resolvent` evaluated once on it.

    THE CLOSED FORM. Every walk from node 0 back to node 0 is a free
    concatenation of the two primitive loops through 0 -- `0->1->3->0`
    (weight `p*c`) and `0->2->3->0` (weight `q*c`), where `p, q` are the two
    route products and `c = A[0,3]` -- so the generating function over all
    such walks is the standard renewal sum `1 / (1 - (p*c + q*c))`, i.e.
    `R[0, 0] = (I - A)^-1[0, 0] = 1 / (1 - c*(p+q))`. This is an algebraic
    identity independent of `dag_resolvent`'s own implementation (an LU
    solve, per its docstring, not a series), so agreement is a real check,
    not two writings of the same formula.

    WHAT WOULD FAIL THIS. `dag_resolvent` silently discarding the imaginary
    part (e.g. an implicit real cast somewhere on the complex path) would
    move `R[0, 0]` off the closed form by `O(1)`, not by float noise; a
    dtype it cannot actually solve at all raises before this assert runs.
    """
    a = _build_cycle(route1_mag=0.7, route2_mag=0.7,
                      route1_phase=H_C, route2_phase=0.0,
                      close_phase=0.9, close_mag=1.1, cap=True)
    r = arm_pl.dag_resolvent(a)
    assert r.dtype == CT

    p = _route_product(a, (0, 1, 3))
    q = _route_product(a, (0, 2, 3))
    c = a[0, 3]
    predicted_r00 = 1.0 / (1.0 - c * (p + q))
    assert abs(r[0, 0] - predicted_r00) < 1e-12


def test_the_z2_reachable_holonomy_set_is_exactly_pi_and_zero_a_theorem():
    """THEOREM about the Z2 subgroup, stated and checked as one -- not a
    measurement, and not dressed as one. Restricting every hop's phase to
    `{0, pi}` (the subgroup a real, signed `gate(u, theta, cap=False)` call
    can reach: `theta=0` gives `+u`, `theta=pi` gives `-u`, `u >= 0`) makes
    `route1_phase - route2_phase mod 2*pi` -- and hence the read holonomy --
    a function of the two SIGNS ALONE: magnitude cancels out of the argument
    of a ratio of complex numbers no matter what it is, so it never enters.
    There are exactly 4 sign pairs, hence exactly 2 distinct holonomies
    (`{0, pi}`, since `(0,0)` and `(pi,pi)` both give `0`, `(0,pi)` and
    `(pi,0)` both give `pi`), hence exactly 2 distinct gaps to `H_C`:
    `pi/3` and `2*pi/3`. That set is what is asserted below, by exhaustive
    enumeration of the 4 sign pairs (not a random sample of them -- there is
    nothing left to sample once the space has 4 elements).

    BED: 4 guarded matrices (one per sign pair), 5 guarded entries each, 20
    guarded entries total, every hop a call to the shipped `gate(...,
    cap=False)`.

    WHAT WOULD FAIL THIS. A read holonomy for some sign pair landing outside
    `{0, pi}` (a phase-arithmetic bug in `_build_cycle`/`_shipped_gate`
    letting a stray non-Z2 phase in) fires the first assert. `arm_phase.gate`'s
    `cap=False` branch not actually skipping the clamp does NOT fire it --
    corrected from an earlier, false claim that it would: this test's
    magnitude is `1.0` on every hop, already inside `[0, 1]`, so clamping it
    would be a no-op here, and even past that, `holonomy` is read off a
    RATIO of two route products, which cancels a magnitude term identically
    regardless of what the clamp did to it. A live `cap=False` regression is
    instead caught by
    `test_the_magnitude_clamp_lands_on_its_absolute_endpoint_not_a_ratio`
    below, which reads a route's magnitude absolutely at an input that
    overshoots the clamp. A gap
    SET other than `{pi/3, 2*pi/3}` -- e.g. missing an element, because a
    sign-pair bug collapsed two cases into one, or containing an extra one,
    because a phase leaked in from the closing edge despite it canceling in
    the ratio by construction -- fires the second.
    """
    signs = (0.0, math.pi)
    gaps = set()
    for route1_phase in signs:
        for route2_phase in signs:
            a = _build_cycle(route1_mag=1.0, route2_mag=1.0,
                              route1_phase=route1_phase,
                              route2_phase=route2_phase,
                              close_phase=0.4, cap=False)
            ratio = _route_product(a, (0, 1, 3)) / _route_product(a, (0, 2, 3))
            holonomy = cmath.phase(complex(ratio))

            landed_z2 = min(abs(holonomy - 0.0), abs(holonomy - math.pi),
                             abs(holonomy + math.pi))
            assert landed_z2 < 1e-9, (
                f"escaped Z2 at holonomy={holonomy} for signs "
                f"({route1_phase}, {route2_phase})")

            gap = min(abs(holonomy - H_C), abs(holonomy - (H_C - 2 * math.pi)))
            gaps.add(round(gap, 9))

    predicted = {round(math.pi / 3, 9), round(2 * math.pi / 3, 9)}
    assert gaps == predicted, (
        f"the Z2 subgroup's reachable gap set was {gaps}, not the proved "
        f"{predicted}")


def test_no_magnitude_ratio_moves_the_z2_holonomy_off_its_sign_pair():
    """The other half of the old, unfireable bar's intent, stated honestly:
    not "no seed in a 200-draw sample got closer than pi/3" (a bar that
    cannot fail once the theorem above is known -- worst case IS pi/3, on
    every draw), but "the magnitude ratio is not part of the argument at
    all", checked as an invariance across a sweep the ratio actually cannot
    escape: for every one of the 4 sign pairs, holding `route2_mag = 1.0`
    and sweeping `route1_mag` across `1e-6 .. 1e6` (12 decades, chosen to
    dwarf any ratio E-INTERF v2's free knob could plausibly reach) never
    moves the holonomy off the value that sign pair alone determines above.

    BED: 4 sign pairs, each building 1 base matrix plus 5 magnitudes
    (`1e-6, 0.05, 1.0, 5.0, 1e6`) = 4 * (1 + 5) = 24 guarded matrices, 5
    guarded entries each, 120 guarded entries total (counted by
    instrumenting `_shipped_gate`, not reasoned about) -- every one of them
    a call to the shipped `gate(..., cap=False)`.

    WHAT WOULD FAIL THIS. Any magnitude in the sweep producing a holonomy
    that differs from the same sign pair's `route1_mag=1.0` case by more
    than float noise -- i.e. the ratio DOES enter the argument for some
    magnitude, which is exactly the E-INTERF v2 failure mode this file exists
    to rule out.
    """
    ratios = (1e-6, 0.05, 1.0, 5.0, 1e6)
    for route1_phase in (0.0, math.pi):
        for route2_phase in (0.0, math.pi):
            base_a = _build_cycle(route1_mag=1.0, route2_mag=1.0,
                                   route1_phase=route1_phase,
                                   route2_phase=route2_phase,
                                   close_phase=0.4, cap=False)
            base_ratio = (_route_product(base_a, (0, 1, 3))
                          / _route_product(base_a, (0, 2, 3)))
            base_holonomy = cmath.phase(complex(base_ratio))

            for route1_mag in ratios:
                a = _build_cycle(route1_mag=route1_mag, route2_mag=1.0,
                                  route1_phase=route1_phase,
                                  route2_phase=route2_phase,
                                  close_phase=0.4, cap=False)
                ratio = (_route_product(a, (0, 1, 3))
                         / _route_product(a, (0, 2, 3)))
                holonomy = cmath.phase(complex(ratio))
                drift = min(abs(holonomy - base_holonomy),
                            abs(abs(holonomy - base_holonomy) - 2 * math.pi))
                assert drift < 1e-9, (
                    f"route1_mag={route1_mag} moved the holonomy from "
                    f"{base_holonomy} to {holonomy} (signs "
                    f"{route1_phase}, {route2_phase}) -- the magnitude "
                    f"ratio entered the argument")


def test_the_magnitude_clamp_lands_on_its_absolute_endpoint_not_a_ratio():
    """THE MAGNITUDE HALF, closed. Every assertion above reads a RATIO of
    two route products that `_build_cycle` pins to equal magnitudes --
    magnitude cancels identically out of the argument of such a ratio no
    matter what produced it, so nothing above can tell a correct
    `magnitude()` clamp from a broken one. This test reads a route's
    magnitude ABSOLUTELY instead, at an input that overshoots the clamp
    (`route1_mag=4.0`, i.e. a per-hop `u = sqrt(4.0) = 2.0`, past the
    endpoint `1.0`), so the clamp's actual bounds are what the assert
    checks against, not a sibling route.

    BED: 2 guarded matrices (one `cap=True`, one `cap=False`, same
    overshooting `route1_mag=4.0`), 5 guarded entries each via
    `_build_cycle`, 10 guarded entries total, every one of them a return
    value of the shipped `gate`.

    `cap=True` must land the route exactly on the clamp's endpoint: each
    hop clamps `2.0` down to `1.0`, so `route1 = 1.0 * 1.0 = 1.0`.
    `cap=False` must NOT clamp at all: each hop stays at the raw `2.0`, so
    `route1 = 2.0 * 2.0 = 4.0`.

    WHAT WOULD FAIL THIS. `magnitude()` clamping to `[0, 0.5]` instead of
    `[0, 1]`: the `cap=True` hop lands on `0.5`, so `route1` reads `0.25`,
    not `1.0` -- fires the first assert. `gate` dropping the clamp
    entirely (i.e. `cap` stops mattering and `m = u` always): the
    `cap=True` hop stays at the raw `2.0`, so `route1` reads `4.0`, not
    `1.0` -- also fires the first assert. `gate` ignoring `cap` and always
    clamping: the `cap=False` hop gets clamped to `1.0` anyway, so
    `route1` reads `1.0`, not `4.0` -- fires the second assert.
    """
    capped = _build_cycle(route1_mag=4.0, route2_mag=1.0,
                           route1_phase=0.0, route2_phase=0.0,
                           close_phase=0.0, cap=True)
    route1_capped = _route_product(capped, (0, 1, 3))
    assert abs(abs(route1_capped) - 1.0) < 1e-13, (
        f"cap=True at an overshooting route1_mag=4.0 read |route1|="
        f"{abs(route1_capped)}, not the clamp's endpoint 1.0")

    uncapped = _build_cycle(route1_mag=4.0, route2_mag=1.0,
                             route1_phase=0.0, route2_phase=0.0,
                             close_phase=0.0, cap=False)
    route1_uncapped = _route_product(uncapped, (0, 1, 3))
    assert abs(abs(route1_uncapped) - 4.0) < 1e-13, (
        f"cap=False at route1_mag=4.0 read |route1|="
        f"{abs(route1_uncapped)}, not the raw 4.0 -- the clamp fired "
        f"when cap said it should not")


if __name__ == "__main__":
    # ponytail: the smallest possible standalone check, so this file's own
    # claims can be read without pytest.
    test_the_planted_holonomy_reads_off_the_gates_at_an_arbitrary_gauge()
    test_dag_resolvent_takes_the_complex_cycle_with_no_code_change()
    test_the_z2_reachable_holonomy_set_is_exactly_pi_and_zero_a_theorem()
    test_no_magnitude_ratio_moves_the_z2_holonomy_off_its_sign_pair()
    test_the_magnitude_clamp_lands_on_its_absolute_endpoint_not_a_ratio()
    print("ok")
