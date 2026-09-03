"""V20 R15 it.4 -- JUPITER. The merge verdict, bound at last, on MARS's reroute.

WHY THIS FILE EXISTS. `V20_R15_IT2_INSPECTOR.md` C9 struck JUPITER's
`N = 1 primitive` merge verdict as UNBOUND: a grep for `primitive|isomorph|
cumprod|merge` over `tests/jupiter/` returned zero files. The it.2 collection
RED binds the SUITE, and the suite asserted the L-DOM census and `crude ==
exact` and nothing else. The only node in the tree asserting the ALGEBRA is
MARS's control (`tests/mars_v20/test_jupiter_it2_merge_hides_the_upper_cap.py`),
and an adversary's control is not the author's bind.

WHAT IS AND IS NOT ASSERTED HERE. The algebra -- `exp . cumsum . log == cumprod`
on `m in (0,1]` -- is CONCEDED, by MARS's control, probed by the Inspector at
`1 + 1e-6` and failing at `8.999999999703689e-07` against a `1e-12` bar. This
file asserts the VERDICT: whether the merge is EXERCISED by the trained record.

MARS STRIKE 7's reroute, restated as the proposition under test:
  "one primitive on the common image `g <= 0`, and the record contains no
   trained W3 cell inside that image."

TOLERANCES.
  * The image partition uses `1.0` EXACTLY. It is not a tolerance: it is the
    upper clamp endpoint of `ceq/arm_smprime.py:113` `torch.clamp(u, 0, 1)`,
    which is where `g = log m` changes sign. No cell sits within 1e-1 of it
    from above (nearest is seed 6 at `1.102945`), so no float bar is load-bearing.
  * The atom mass equality `n_zero_gates / 8192 == frac_gate_annihilated` is
    asserted BITWISE, not within tolerance. SATURN's it.3 A.4 reports it exact
    on all eight cells; this file re-derives it rather than citing it.
  * Definedness (`isfinite` / `isnan`) is a predicate, not a measurement. No
    tolerance applies and none is used. A `nan` is never widened to `inf`.
"""
from __future__ import annotations

import json
import math
import pathlib

import pytest
import torch

from ceq import arm_pl, arm_smprime

ROOT = pathlib.Path(__file__).resolve().parents[2]
RETAKE = ROOT / "results" / "v17k_r4_retake.jsonl"

#: The clamp endpoint of `ceq/arm_smprime.py:113`. `g = log m` changes sign here.
CLAMP_HI = 1.0

#: The gate vector length `n_zero_gates` counts over. Asserted, not assumed.
GATE_N = 8192

#: MARS STRIKE 4/7's partitioning threshold on `a_hat_max`. NOT the cap: the cap
#: is 1.0 and every W3 cell exceeds it, so 1.0 partitions nothing. The Inspector
#: recorded this distinction as a precision note on MARS's prose.
PARTITION_AT = 2.0


def cells(kind: str):
    """The eight trained cells of one arm, from the journal. `[READ]` only."""
    out = []
    for line in RETAKE.read_text(encoding="utf-8").splitlines():
        if '"a_hat_min"' not in line:
            continue
        d = json.loads(line)
        if d.get("kind") == kind:
            out.append(d)
    return sorted(out, key=lambda d: d["seed"])


def in_common_image(rows):
    """Cells whose whole trained magnitude range lies in W1's image, `g <= 0`.

    `g = log m` and W1's `m` is `clamp(u, 0, 1)`, so W1's image is `g <= 0`,
    i.e. `a_hat_max <= 1.0`. This is the ONE predicate the verdict turns on and
    it is a single function, used by the RED node, the GREEN node and the
    planted negative alike -- see `test_planted_negative_...`.
    """
    return [d for d in rows if d["a_hat_max"] <= CLAMP_HI]


# ---------------------------------------------------------------------------
# The it.2 verdict, run against the record. RED BY DESIGN, KEPT RED.
# ---------------------------------------------------------------------------
def test_the_it2_merge_verdict_covers_the_trained_record():
    """RED BY DESIGN -- the record of the strike, against its own author.

    `V20_R15_IT2_JUPITER.md` A.1/A.4 merged W1 and W3 into one primitive and
    named the isomorphism's domain exclusion as exactly one point, `0 not in
    R_{>0}` -- the LOWER endpoint. That verdict claims the merge covers the
    trained record, which requires at least one trained W3 cell inside W1's
    image. This node asserts exactly that and nothing more. It fires.
    """
    w3 = cells("arm_pl")
    inside = in_common_image(w3)
    assert len(inside) >= 1, (
        "the it.2 merge verdict claims a coordinate change between W1 and W3, "
        "but W1's image under g=log m is g<=0 (clamp(u,0,1), ceq/arm_smprime.py:113) "
        "and 0 of %d trained W3 cells have a_hat_max <= %.1f; "
        "g = log a_hat_max per seed: %s"
        % (len(w3), CLAMP_HI,
           {d["seed"]: round(math.log(d["a_hat_max"]), 4) for d in w3}))


# ---------------------------------------------------------------------------
# The restated verdict. GREEN.
# ---------------------------------------------------------------------------
def test_the_restated_verdict_the_common_image_holds_no_trained_w3_cell():
    """MARS STRIKE 7's reroute, asserted as a count and not as prose.

    Both halves are measured off the SAME journal field, `a_hat_max`, so the
    two wings are compared on one scale and not on two.
    """
    w1, w3 = cells("arm_smprime"), cells("arm_pl")
    assert len(w1) == 8 and len(w3) == 8

    # W1: every trained cell lies in the common image, g <= 0.
    assert len(in_common_image(w1)) == 8, {d["seed"]: d["a_hat_max"] for d in w1}
    assert max(d["a_hat_max"] for d in w1) == CLAMP_HI      # attained, 6 of 8

    # W3: no trained cell lies in it, g > 0 on all eight.
    assert len(in_common_image(w3)) == 0, {d["seed"]: d["a_hat_max"] for d in w3}
    assert min(d["a_hat_max"] for d in w3) > CLAMP_HI
    assert round(max(math.log(d["a_hat_max"]) for d in w3), 4) == 4.7536

    # The verdict: the merge's exercised set on the trained record is EMPTY.
    assert len({d["seed"] for d in in_common_image(w1)}
               & {d["seed"] for d in in_common_image(w3)}) == 0


def test_the_unnamed_boundary_sorts_the_scoreboard_with_zero_overlap():
    """A boundary that partitions `eval_nrmse` is not a coordinate change.

    This is MARS STRIKE 4/7's partition, re-derived here because it is the
    reason the merge cannot be granted: erasing the distinction would erase a
    split the scoreboard already sorts on.
    """
    w3 = cells("arm_pl")
    lo = [d["eval_nrmse"] for d in w3 if d["a_hat_max"] <= PARTITION_AT]
    hi = [d["eval_nrmse"] for d in w3 if d["a_hat_max"] > PARTITION_AT]
    assert (len(lo), len(hi)) == (5, 3)
    assert max(lo) == pytest.approx(0.662128, abs=5e-7)
    assert min(hi) == pytest.approx(1.113339, abs=5e-7)
    assert max(lo) < min(hi)                                 # zero overlap


# ---------------------------------------------------------------------------
# The atom -- the merge is not a coordinate change where one side has no value
# ---------------------------------------------------------------------------
def _draw_with_atom(s=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(s, 8, dtype=torch.float64, generator=gen)
    k = torch.randn(s, 8, dtype=torch.float64, generator=gen)
    u = torch.rand(s, dtype=torch.float64, generator=gen)
    u[0] = 0.0                                               # the atom, exactly
    th = torch.zeros(s, dtype=torch.float64)
    return q, k, u, th


def test_at_the_atom_w1_is_defined_and_w3_is_not():
    """`cumprod` on a zero gate is a true `0`; `log 0 = -inf` into a softmax
    logit is `nan`. SATURN it.3 A.5 measured this on all 8 cells; re-derived
    here on one draw because the verdict, not the number, is what this file owns.
    """
    q, k, u, th = _draw_with_atom()
    c = cells("arm_smprime")[0]["manifest"]["smp_values"]
    w1 = arm_smprime.operator(q, k, u, th, beta=c["beta"], qk=c["qk"],
                              g=c["g"], route=c["route"])
    w3 = arm_pl.operator(q, k, torch.log(torch.clamp(u, 0.0, 1.0)))
    assert bool(torch.isfinite(w1.to(torch.complex128)).all())
    assert bool(torch.isnan(w3.to(torch.complex128)).any())


def test_the_atom_is_not_a_measure_zero_corner_it_carries_half_the_gate_mass():
    """The exclusion `0 not in R_{>0}` is the it.2 verdict's ONLY named
    exclusion. It is not a corner: on 7 of 8 trained W1 cells the gate is
    annihilated on ~half its entries. Mass read from the record, bitwise.
    """
    w1 = cells("arm_smprime")
    nz = [d for d in w1 if d["manifest"]["smp_values"]["n_zero_gates"] > 0]
    assert len(nz) == 7                              # seed 2 is the escape cell
    for d in w1:                                     # the denominator, asserted
        v = d["manifest"]["smp_values"]
        assert v["n_zero_gates"] / GATE_N == d["frac_gate_annihilated"]
    assert min(d["frac_gate_annihilated"] for d in nz) > 0.49
    assert sum(d["frac_gate_annihilated"] for d in nz) / 7 == pytest.approx(
        0.5023542131696429, abs=1e-15)


# ---------------------------------------------------------------------------
# V-16: a check that cannot fail is not a check
# ---------------------------------------------------------------------------
def test_planted_negative_the_image_predicate_returns_nonzero_when_a_cell_moves():
    """The zero in the GREEN node is a measurement, not a constant.

    The mutation is NAMED AND EXACT: take the W3 cell nearest the boundary
    (seed 6, `a_hat_max = 1.102945`) and set it to `0.9` -- inside W1's image.
    The SAME `in_common_image` the verdict uses then returns 1, not 0.
    """
    w3 = cells("arm_pl")
    assert len(in_common_image(w3)) == 0
    near = min(w3, key=lambda d: d["a_hat_max"])
    assert near["seed"] == 6 and near["a_hat_max"] == pytest.approx(1.102945, abs=1e-6)
    mutated = [dict(d, a_hat_max=0.9) if d is near else d for d in w3]
    assert len(in_common_image(mutated)) == 1
