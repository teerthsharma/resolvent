"""WINGS-DISTINCT, RE-MEASURED ON THE PER-CELL TRAINED GATE.

WHY THIS EXISTS. `V20_R15_IT2_SATURN.md` A.6 reported every wing pair separating
by `>= 0.30` at both shapes "at the trained settings". MARS STRIKE 5
(`V20_R15_IT2_MARS.md`) showed the gate that measurement ran on is one W1 cannot
occupy: `tests/saturn/test_v20_r15_wings_distinct.py::trained_gate_range` reads
`a_hat_min`/`a_hat_max` off the **`arm_pl`** cells, `[0.0318116, 116.006073]`,
and `ceq/arm_smprime.py:109` is `clamp(u, 0, 1)`. Over 99% of that draw clamps
to exactly `1.0` -- the identity gate -- and `lo > 0` means the draw never
reaches `m = 0`, the annihilating endpoint 7 of 8 trained W1 cells DO reach.

THE REROUTE, which is MARS's and needs no new training. Every input is
journalled per W1 cell in `results/v17k_r4_retake.jsonl`:
  * `a_hat_min` / `a_hat_max`  -- top level of the `arm_smprime` cell row;
  * `n_zero_gates`             -- `manifest.smp_values`, and
    `n_zero_gates / 8192 == frac_gate_annihilated` exactly on all eight cells.
`u` is drawn per cell on that cell's own `[a_hat_min, a_hat_max]`, with an atom
at exactly `0.0` of mass `n_zero_gates / 8192`.

THE ATOM IS REPORTED SEPARATELY AND NOT AS A NUMBER. At `m = 0` W1's `cumprod`
returns a true `0` while W2/W3 both route `log m = -inf` into a softmax logit
(`ceq/arm_phase.py:126`, `ceq/arm_pl.py:97`), which is `nan`. A `nan` read as
`inf` separation would be an artifact promoted to a measurement, so the atom leg
reports the DEFINEDNESS structure and the continuous leg carries the number.

TOLERANCE. `1e-12`, unchanged, and its provenance is unchanged: 105x
`9.547918011776346e-15`, the worst float64 identity residual this tree records
for itself. Restated where it is used.
"""
from __future__ import annotations

import json
import math
import pathlib

import pytest
import torch

from ceq import arm_phase, arm_pl, arm_smprime

ROOT = pathlib.Path(__file__).resolve().parents[2]
RETAKE = ROOT / "results" / "v17k_r4_retake.jsonl"

TOL = 1e-12
SHAPES = (8, 64)

#: The gate vector length the journal's `n_zero_gates` counts over. Asserted
#: against `frac_gate_annihilated` rather than assumed -- see the control.
GATE_N = 8192

#: `V20_R15_IT2_SATURN.md` A.6's published floor, the figure under strike.
IT2_PUBLISHED_FLOOR = 0.30

#: The void draw, `arm_pl`'s range, kept so the saturation control has a
#: positive it is required to detect (`MISTAKES.md` V-7).
VOID_LO, VOID_HI = 0.0318116, 116.006073


def per_cell():
    """Per W1 cell: the trained switches AND that cell's own gate support."""
    out = []
    for line in RETAKE.read_text(encoding="utf-8").splitlines():
        if '"a_hat_min"' not in line:
            continue
        d = json.loads(line)
        if d.get("kind") != "arm_smprime":
            continue
        v = d["manifest"]["smp_values"]
        out.append({"seed": d["seed"], "beta": v["beta"], "qk": v["qk"],
                    "g": v["g"], "route": v["route"],
                    "lo": d["a_hat_min"], "hi": d["a_hat_max"],
                    "n_zero": v["n_zero_gates"],
                    "frac_annih": d["frac_gate_annihilated"]})
    return out


def draw(s, seed, lo, hi, n_zero, *, atom):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(s, 8, dtype=torch.float64, generator=gen)
    k = torch.randn(s, 8, dtype=torch.float64, generator=gen)
    u = torch.rand(s, dtype=torch.float64, generator=gen) * (hi - lo) + lo
    if atom and n_zero:
        hit = torch.rand(s, dtype=torch.float64, generator=gen) < n_zero / GATE_N
        u = torch.where(hit, torch.zeros_like(u), u)
    th = (torch.rand(s, dtype=torch.float64, generator=gen) * 2 - 1) * math.pi
    return q, k, u, th


def w3(q, k, u, th):
    """W3 on W2's gate content: `g = log(clamp(u, 0, 1))`, `ceq/compat.py:393`."""
    return arm_pl.operator(q, k, torch.log(torch.clamp(u, 0.0, 1.0))).to(
        torch.complex128)


def wings(q, k, u, th, c):
    return (arm_smprime.operator(q, k, u, th, beta=c["beta"], qk=c["qk"],
                                 g=c["g"], route=c["route"]),
            arm_phase.operator(q, k, u, th),
            w3(q, k, u, th))


def delta(a, b) -> float:
    """max |A - B|. `nan` is returned AS `nan`, never widened to `inf`."""
    d = (a.to(torch.complex128) - b.to(torch.complex128)).abs()
    return float("nan") if torch.isnan(d).any() else float(d.max())


PAIRS = (("W1/W2", 0, 1), ("W1/W3", 0, 2), ("W2/W3", 1, 2))


def sweep(s, *, atom):
    """min over the eight trained cells of max|delta|, plus the nan census."""
    worst, undef = {}, {}
    for c in per_cell():
        ops = wings(*draw(s, c["seed"], c["lo"], c["hi"], c["n_zero"],
                          atom=atom), c)
        for tag, i, j in PAIRS:
            d = delta(ops[i], ops[j])
            if math.isnan(d):
                undef.setdefault(tag, []).append(c["seed"])
            else:
                worst[tag] = min(worst.get(tag, float("inf")), d)
    return worst, undef


# ==========================================================================
# CONTROLS -- the saturation the strike names must be DETECTABLE by this file
# ==========================================================================

def test_the_void_draw_saturates_the_cap_and_the_per_cell_draw_does_not():
    """The control MARS's reroute needs: it must fire on the gate under strike.

    Planted positive -- the `arm_pl` range, clamped by `ceq/arm_smprime.py:109`,
    must read as saturated. If this instrument could not see 99% saturation it
    could not certify the corrected draw is unsaturated either."""
    gen = torch.Generator().manual_seed(0)
    u = torch.rand(GATE_N, dtype=torch.float64, generator=gen)
    void = float((torch.clamp(u * (VOID_HI - VOID_LO) + VOID_LO, 0, 1)
                  == 1.0).double().mean())
    assert void > 0.98, (
        f"the planted positive did not fire: the void draw reads {void:.4%} of "
        f"the gate at exactly 1.0, but STRIKE 5 measures ~99%. The saturation "
        f"detector is broken, so its 'not saturated' verdict below is worthless")
    for c in per_cell():
        gen = torch.Generator().manual_seed(c["seed"])
        u = torch.rand(GATE_N, dtype=torch.float64, generator=gen)
        got = float((torch.clamp(u * (c["hi"] - c["lo"]) + c["lo"], 0, 1)
                     == 1.0).double().mean())
        assert got < 1e-3, (
            f"seed {c['seed']}: the corrected per-cell draw is STILL saturated "
            f"({got:.4%} of the gate at exactly 1.0) -- the reroute did not "
            f"remove the defect it was written to remove")


def test_the_atom_mass_is_the_journals_own_annihilated_fraction():
    """`n_zero_gates / 8192` is not a guessed denominator: it reproduces the
    journalled `frac_gate_annihilated` bitwise on all eight cells."""
    for c in per_cell():
        assert c["n_zero"] / GATE_N == c["frac_annih"], (
            f"seed {c['seed']}: n_zero_gates/{GATE_N} = "
            f"{c['n_zero'] / GATE_N!r} != frac_gate_annihilated "
            f"{c['frac_annih']!r}; the atom's mass is not read from the record")


def test_the_per_cell_support_reaches_the_annihilating_endpoint():
    """7 of 8 cells reach `m = 0` exactly; the void draw reached it on none."""
    lows = {c["seed"]: c["lo"] for c in per_cell()}
    zero = sorted(s for s, lo in lows.items() if lo == 0.0)
    assert zero == [0, 1, 3, 4, 5, 6, 7], f"a_hat_min == 0.0 on {zero}"
    assert VOID_LO > 0.0


def test_the_measure_calls_an_arm_against_itself_the_same():
    """Planted positive, carried forward: same object twice reads exact 0.0."""
    c = per_cell()[0]
    args = draw(16, 0, c["lo"], c["hi"], c["n_zero"], atom=False)
    for name, op in zip(("W1", "W2", "W3"), (0, 1, 2)):
        d = delta(wings(*args, c)[op], wings(*args, c)[op])
        assert d == 0.0, f"{name} against itself read {d!r}, expected 0.0"


# ==========================================================================
# THE BIND -- Task A: does `>= 0.30` survive the corrected gate?
# ==========================================================================

@pytest.mark.parametrize("s", SHAPES)
def test_the_wings_still_separate_on_the_per_cell_trained_gate(s):
    worst, undef = sweep(s, atom=False)
    print(f"\n  PER-CELL CONTINUOUS s={s}: " +
          str({t: f"{v:.6e}" for t, v in worst.items()}))
    assert not undef, undef
    same = {t: v for t, v in worst.items() if v < TOL}
    assert not same, f"at s={s} these pairs are one operator to {TOL}: {same}"


@pytest.mark.parametrize("s", SHAPES)
def test_the_it2_published_floor_of_030_survives_the_corrected_gate(s):
    """THE STRIKE, bound. `V20_R15_IT2_SATURN.md` A.6 published `>= 0.30` at
    BOTH shapes. On the gate W1 can actually occupy, s=8 does not clear it."""
    worst, _ = sweep(s, atom=False)
    below = {t: f"{v:.6e}" for t, v in worst.items()
             if v < IT2_PUBLISHED_FLOOR}
    assert not below, (
        f"at s={s} the it.2 published floor {IT2_PUBLISHED_FLOOR} is not met on "
        f"the per-cell trained gate: {below}. The separation SHRINKS; it does "
        f"not vanish -- every reading is still >= 1e11 x the tolerance {TOL}")


# ==========================================================================
# Task B -- the W2/W3 leg, and it is a measurement of `theta` alone
# ==========================================================================

@pytest.mark.parametrize("s", SHAPES)
def test_the_w2_w3_leg_is_carried_entirely_by_theta(s):
    """`ceq/arm_phase.py:158` is `arm_pl`'s matrix times `phase_factor(theta)`
    under `g = log m`, so `|W2| = W3` identically. Setting `theta = 0` on the
    SAME per-cell gate must collapse the leg to bitwise zero; anything the leg
    reports at a non-zero `theta` is therefore `theta` and nothing else."""
    for c in per_cell():
        q, k, u, th = draw(s, c["seed"], c["lo"], c["hi"], c["n_zero"],
                           atom=False)
        z = torch.zeros_like(th)
        assert delta(arm_phase.operator(q, k, u, z), w3(q, k, u, z)) == 0.0, (
            f"seed {c['seed']}: W2 and W3 are not bitwise equal at theta = 0, "
            f"so the leg is not carried by theta alone and the report is wrong")
        assert delta(arm_phase.operator(q, k, u, th), w3(q, k, u, th)) > TOL


def test_the_w2_theta_is_drawn_and_not_trained_anywhere_in_results():
    """The leg's `theta` is `U(-pi, pi]`. It cannot be read: no `arm_phase`
    cell exists. The moment one does, this RED is the signal to re-measure."""
    hits = [p.name for p in (ROOT / "results").glob("*")
            if p.is_file() and '"kind": "arm_phase"' in
            p.read_text(encoding="utf-8", errors="ignore")]
    assert not hits, f"arm_phase now has journalled cells in {hits}: the W2/W3 leg must be re-measured at the TRAINED theta"


# ==========================================================================
# THE ATOM -- definedness, not a number
# ==========================================================================

@pytest.mark.parametrize("s", SHAPES)
def test_at_the_annihilating_atom_w2_and_w3_are_undefined_where_w1_is_not(s):
    """The endpoint the void draw excluded. With the atom in, W2 and W3 go
    `nan` on exactly the 7 cells whose `a_hat_min == 0.0`; W1 returns a finite
    matrix on all eight. This is a domain difference, not a magnitude, and it
    is the same object as JUPITER's STRIKE-7 exclusion reached from the gate."""
    bad = []
    for c in per_cell():
        q, k, u, th = draw(s, c["seed"], c["lo"], c["hi"], c["n_zero"],
                           atom=True)
        a, b, cc = wings(q, k, u, th, c)
        has0 = bool((torch.clamp(u, 0.0, 1.0) == 0.0).any())
        fin = (bool(torch.isfinite(a).all()), bool(torch.isnan(b).any()),
               bool(torch.isnan(cc).any()))
        if fin != (True, has0, has0):
            bad.append((c["seed"], has0, fin))
    assert not bad, (
        f"s={s}: (W1 finite, W2 nan, W3 nan) did not track the presence of an "
        f"exact zero gate on {bad}")
