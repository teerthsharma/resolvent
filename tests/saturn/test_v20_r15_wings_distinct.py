"""WINGS-DISTINCT: W1/W2/W3 must be different OPERATORS at their TRAINED settings.

WHY THIS EXISTS. `V20_R15_IT1_SATURN.md` filed N = 3 wings and then flagged,
against its own list, that the three had never been run through
`tests/loop/test_arms_distinct.py::assert_arms_distinct` -- that bind covers the
seven `ceq/bench.py` arms only. W1/W2/W3 distinctness was argued from SOURCE
(`cumsum` vs `cumprod`, `ceq/compat.py:393-394`) and never measured. An argued
distinctness is the ParaFormer shape one level up: three names, and nobody had
checked they were three objects.

MARS measured the three COINCIDE at the corner `beta = qk = g = 1`, `u = 1`,
`theta = 0`, to 1.11e-16..8.11e-16. That corner is not where they run: the
trained `beta` journalled in `results/v17k_r4_retake.jsonl` reads
0.588/0.733/0.782/0.835/0.897/0.901/1.344/1.509 over the eight seeds and is
never 1. So the only question that settles N is whether they are distinct AT
THE TRAINED SETTINGS, and that is what this file measures.

TRAINED SETTINGS ARE READ, NOT CHOSEN. `(beta, qk, g, route)` come out of the
journal's own `manifest.smp_values` cell, per seed. The gate magnitude range
comes out of the same journal's `a_hat_min`/`a_hat_max` on the W3 cells.

TOLERANCE. `1e-12`, which is 105x the worst float64 identity residual this tree
records for itself (`results/v17k_r4_retake.jsonl`, arm_pl bind at s=64,
`residual = 9.547918011776346e-15`). This is MARS's tolerance, reused rather
than re-invented. It is NOT `9.522e-03`: that is a cross-device `eval_nrmse`
delta, wrong units for an operator comparison.

CALIBRATED BEFORE USE (MISTAKES.md:804, V-15). A distinctness test that cannot
detect SAMENESS is unfalsifiable. Two planted positives:
  * an arm against ITSELF, which must read exactly 0.0;
  * MARS's corner, where the three really are one operator, which must be
    called SAME at this tolerance.
Only then does a "distinct" reading mean anything.
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

#: 105x the tree's own worst float64 identity residual (see module docstring).
TOL = 1e-12
WORST_IDENTITY_RESIDUAL = 9.547918011776346e-15

#: The two shapes the journal's own binds are recorded at.
SHAPES = (8, 64)


def trained_smp() -> list:
    """`(seed, beta, qk, g, route)` per seed, READ from the retake journal."""
    out = []
    for line in RETAKE.read_text(encoding="utf-8").splitlines():
        if '"smp_values"' not in line:
            continue
        d = json.loads(line)
        v = d["manifest"]["smp_values"]
        out.append({"seed": d["seed"], "beta": v["beta"], "qk": v["qk"],
                    "g": v["g"], "route": v["route"]})
    return out


def trained_gate_range():
    """Widest journalled trained gate magnitude range over the W3 cells."""
    los, his = [], []
    for line in RETAKE.read_text(encoding="utf-8").splitlines():
        if '"a_hat_min"' not in line:
            continue
        d = json.loads(line)
        if d.get("kind") != "arm_pl":
            continue
        los.append(d["a_hat_min"])
        his.append(d["a_hat_max"])
    return min(los), max(his)


def draw(s, seed, lo, hi):
    """One common `(q, k, u, theta)` all three wings are evaluated on."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(s, 8, dtype=torch.float64, generator=gen)
    k = torch.randn(s, 8, dtype=torch.float64, generator=gen)
    u = torch.rand(s, dtype=torch.float64, generator=gen) * (hi - lo) + lo
    theta = (torch.rand(s, dtype=torch.float64, generator=gen) * 2 - 1) * math.pi
    return q, k, u, theta


# --- the three wings, each through its OWN shipped entry point ---------------

def w1(q, k, u, theta, beta=1.0, qk=1.0, g=1.0, route="product"):
    """W1 `ceq/arm_smprime.py::operator` -- cumprod path product / Z^beta."""
    return arm_smprime.operator(q, k, u, theta, beta=beta, qk=qk, g=g,
                                route=route)


def w2(q, k, u, theta):
    """W2 `ceq/arm_phase.py::operator` -- cumsum(log m) key bias x phase twist."""
    return arm_phase.operator(q, k, u, theta)


def w3(q, k, u, theta):
    """W3 `ceq/arm_pl.py::operator` -- cumsum(g) key bias, REAL.

    `g = log(clamp(u, 0, 1))` is the mapping that puts W3 on the same gate
    content as W2: `arm_phase.key_bias` is `s - cumsum(log m)` and
    `arm_pl.key_bias` is `s - cumsum(g)`. It is the `cumsum` reduction
    `ceq/compat.py:393-394` names for both.
    """
    g = torch.log(torch.clamp(u, 0.0, 1.0))
    return arm_pl.operator(q, k, g).to(torch.complex128)


def delta(a, b) -> float:
    """max |A - B| over the operator matrix. NaN-safe: NaN reads as inf."""
    d = (a.to(torch.complex128) - b.to(torch.complex128)).abs()
    if torch.isnan(d).any():
        return float("inf")
    return float(d.max())


# ==========================================================================
# CALIBRATION -- the test must be SEEN to call two same things SAME
# ==========================================================================

def test_the_measure_calls_an_arm_against_itself_the_same():
    """Planted positive 1. Same object twice: exactly 0.0, not merely small."""
    q, k, u, th = draw(16, 0, 0.05, 1.4)
    cases = (("W1", lambda: w1(q, k, u, th, beta=0.73, qk=1.27, g=1.32)),
             ("W2", lambda: w2(q, k, u, th)),
             ("W3", lambda: w3(q, k, u, th)))
    for name, fn in cases:
        d = delta(fn(), fn())
        assert d == 0.0, f"{name} against itself read {d!r}, expected exact 0.0"


def test_the_measure_calls_marss_corner_the_same():
    """Planted positive 2. At `beta = qk = g = 1`, `u = 1`, `theta = 0` the
    three ARE one operator. A test that reported them distinct here would be
    reporting float noise as a wing."""
    for s in SHAPES:
        q, k, _, _ = draw(s, 0, 0.05, 1.4)
        u = torch.ones(s, dtype=torch.float64)
        th = torch.zeros(s, dtype=torch.float64)
        a, b, c = (w1(q, k, u, th, beta=1.0, qk=1.0, g=1.0),
                   w2(q, k, u, th), w3(q, k, u, th))
        for x, y, tag in ((a, b, "W1/W2"), (a, c, "W1/W3"), (b, c, "W2/W3")):
            d = delta(x, y)
            assert d < TOL, (
                f"s={s} {tag} read {d:.6e} at MARS's corner, where the three "
                f"are the SAME operator. The tolerance {TOL} is too tight or "
                f"the mapping is wrong; either way no 'distinct' reading from "
                f"this file can be trusted.")


def test_the_tolerance_is_above_the_trees_own_identity_residual():
    """The tolerance must not be tighter than the tree's own float64 noise."""
    assert TOL > WORST_IDENTITY_RESIDUAL
    assert str(WORST_IDENTITY_RESIDUAL) in RETAKE.read_text(encoding="utf-8"), (
        "the residual the tolerance is scaled from is not in the journal")


# ==========================================================================
# THE BIND ITSELF
# ==========================================================================

@pytest.mark.parametrize("s", SHAPES)
def test_the_three_wings_are_distinct_operators_at_their_trained_settings(s):
    """SUPERSEDED BY `test_v20_r15_wings_distinct_percell.py`, MARS STRIKE 5.

    `trained_gate_range()` below reads `a_hat_min`/`a_hat_max` off the **arm_pl**
    cells, `[0.0318116, 116.006073]`. `ceq/arm_smprime.py:109` clamps to `[0,1]`,
    so 99.1% of that draw is exactly `1.0` (measured) and it never reaches
    `m = 0`. The per-cell node re-measures this on the gate W1 can occupy and
    its numbers, not these, are the round's. Kept RUNNING and not deleted so the
    void gate stays visible next to its replacement."""
    lo, hi = trained_gate_range()
    rows = trained_smp()
    assert len(rows) == 8, f"expected 8 trained W1 cells, read {len(rows)}"
    worst = {}
    for r in rows:
        q, k, u, th = draw(s, r["seed"], lo, hi)
        a = w1(q, k, u, th, beta=r["beta"], qk=r["qk"], g=r["g"],
               route=r["route"])
        b, c = w2(q, k, u, th), w3(q, k, u, th)
        for x, y, tag in ((a, b, "W1/W2"), (a, c, "W1/W3"), (b, c, "W2/W3")):
            d = delta(x, y)
            worst[tag] = min(worst.get(tag, float("inf")), d)
    print("\n  WINGS-DISTINCT s={} gate in [{:.6g},{:.6g}] over 8 trained "
          "(beta,qk,g): MIN over seeds of max|delta|".format(s, lo, hi))
    for tag in ("W1/W2", "W1/W3", "W2/W3"):
        print("    {}: {:.6e}".format(tag, worst[tag]))
    same = {t: v for t, v in worst.items() if v < TOL}
    assert not same, (
        f"at s={s} these wing pairs are the SAME operator to {TOL} at the "
        f"TRAINED settings: {same}. N is smaller than the filed count.")
