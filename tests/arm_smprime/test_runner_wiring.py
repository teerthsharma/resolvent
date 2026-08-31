"""ARM S-M' AS THE RUNNER SEES IT -- `scripts/v15_r1.py`, the file that scores
R1'.

Two nodes reported the same gap: `make_arm` knew `arm_pl` and
`scale/m3_capability.py::Arm` only, so the arm `CEQ_V16_CONTRACT.md` PART IV
names for R1' was unreachable from the runner and R1' could not run. This suite
covers the wiring, and its centre is ONE trap.

THE TRAP. `V15_ARM_PHASE.md` section 7 item 4: the phase family's identity point
is `m = 1`, not `m = 0`, because the magnitude head IS the magnitude under a
CLOSED `[0,1]` cap. A harness that "resets the heads to zero" therefore puts
such an arm at `m = 0` -- the ANNIHILATING gate, where `V16Domain.lean` clause 4
makes the path product exactly `0` on every off-diagonal window. The cell that
comes out is a whole-corpus zero, and it reads as a CATASTROPHIC ARM rather than
as a broken HARNESS. `ceq/arm_smprime.py` has the same trap: it ships
`identity_heads()` and NO `zero_heads()`.

So the invariant asserted here is the one that separates the two points with no
tolerance in it:

    at the identity gate      the modulus row is the ALL-ONES CAUSAL MASK
    at the annihilating gate  the modulus row is the IDENTITY MATRIX
                              (the diagonal survives: the empty product is 1)

`torch.equal` both ways, and the rejection region is occupied by a test that
substitutes a zeroing reset into the runner and requires the runner's own check
to fire (`MISTAKES.md` V-24 -- a guard that has never been seen refusing is not
a guard).

NOTHING HERE TRAINS (L-LEAN). Every test runs at zero gradient steps.
"""
from __future__ import annotations

import importlib.util
import math
import os
import pathlib

import pytest
import torch

from ceq import arm_pl
from ceq import arm_smprime as smp

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: The runner is a SCRIPT, not a package module, so it is loaded by path -- the
#: same mechanism `scripts/v15_r1.py` itself uses to import MARS's power
#: instrument from the test file that filed it. Importing the shipped file is
#: the point: a copy of `identity_point` here would be a check on the copy.
_spec = importlib.util.spec_from_file_location(
    "_v15_r1_under_test", ROOT / "scripts" / "v15_r1.py")
r1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r1)

DT = torch.float64
#: same convention as the rest of `tests/arm_smprime`:
#: `ARM_SMPRIME_DEVICE=cuda python -m pytest tests/arm_smprime` re-runs the
#: identical suite with the identical ids.
DEV = torch.device(os.environ.get("ARM_SMPRIME_DEVICE", "cpu"))
S = 16                      # small: none of these readings depends on the shape
#: the runner's own width. `make_arm` builds at `D_MODEL` and a local copy of
#: the number would make this suite pass against a width the runner stopped
#: using.
DM = r1.D_MODEL


def _x(n: int = 3, s: int = S, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, s, DM, generator=g, dtype=DT).to(DEV)


def _zero_the_heads(model):
    """What a GENERIC harness reset does, written out. This is the wrong point
    for this arm and the suite uses it as the negative, never as a fixture."""
    with torch.no_grad():
        for h in (model.m_head, model.theta_head):
            h.weight.zero_()
            h.bias.zero_()
    return model


# ============================================ the arm is reachable at all

def test_make_arm_constructs_arm_smprime():
    """The gap both nodes reported: the factory did not know the name."""
    arm = r1.make_arm("arm_smprime", S)
    assert isinstance(arm, smp.ArmSMPrime)
    assert arm.kind == smp.NAME
    #: the drop-in shape the runner's loop assumes: `forward(x) -> [n]`.
    assert arm.to(DT).to(DEV)(_x()).shape == (3,)


def test_arm_smprime_is_declared_gated_and_resolves_to_its_own_module():
    assert "arm_smprime" in r1.GATED_ARMS
    assert r1.ARM_MODULES["arm_smprime"] is smp
    assert r1.ARM_MODULES["arm_pl"] is arm_pl


# ================================================== THE IDENTITY-POINT TRAP

def test_the_runners_identity_point_is_m_equals_one_and_not_zeroed_heads():
    """`identity_heads()`, and the two points are exhibited side by side so the
    difference is a measurement rather than a claim about method names."""
    arm, method = r1.identity_point(r1.make_arm("arm_smprime", S).to(DT).to(DEV))
    assert method == "identity_heads"
    x = _x()
    with torch.no_grad():
        row = smp.hop(*arm.heads(x), g=arm.g)[1]
        m, theta = smp.blend(*arm.heads(x), arm.g)
    #: the identity gate: `m = 1`, `theta = 0`, modulus row = all-ones causal.
    assert torch.equal(m, torch.ones_like(m))
    assert torch.equal(theta, torch.zeros_like(theta))
    assert torch.equal(row, torch.tril(torch.ones_like(row)))

    #: THE WRONG POINT, run rather than described. Zeroing the heads sends the
    #: same arm to `m = 0` and the modulus row collapses to the identity matrix:
    #: every off-diagonal hop is exactly zero, which is the whole corpus.
    zrow = _annihilating_row(x)
    eye = torch.eye(S, dtype=zrow.dtype, device=zrow.device).expand_as(zrow)
    assert torch.equal(zrow, eye)
    assert not torch.equal(zrow, torch.tril(torch.ones_like(zrow)))
    #: `S(S+1)/2` surviving hops against `S`. The gap is the reading a harness
    #: bug would publish as the arm's number.
    assert float(row.sum()) == 3 * S * (S + 1) / 2
    assert float(zrow.sum()) == 3 * S


def _annihilating_row(x):
    with torch.no_grad():
        z = _zero_the_heads(r1.make_arm("arm_smprime", S).to(DT).to(DEV))
        return smp.hop(*z.heads(x), g=z.g)[1]


def test_the_runners_identity_bind_reads_the_identity_gate():
    rec = r1.identity_bind("arm_smprime", S, device=DEV)
    assert rec["method"] == "identity_heads"
    assert rec["all_ones_causal_mask"] is True
    assert rec["annihilating"] is False
    #: LIVE, off the tensor -- not the `--device` argument re-stamped.
    assert rec["device"] == DEV.type


def test_the_identity_check_fires_when_a_harness_zeroes_the_heads(monkeypatch):
    """THE REJECTION REGION (`MISTAKES.md` V-24). Substitute the generic reset
    the trap describes and the runner's own check must refuse it. Without this
    test, `identity_bind` could return `annihilating=False` unconditionally and
    every assertion above would still pass."""
    monkeypatch.setattr(r1, "identity_point",
                        lambda model: (_zero_the_heads(model), "zero_heads"))
    rec = r1.identity_bind("arm_smprime", S, device=DEV)
    assert rec["method"] == "zero_heads"
    assert rec["annihilating"] is True
    assert rec["all_ones_causal_mask"] is False


def test_identity_point_prefers_identity_heads_over_zero_heads():
    """Attribute ORDER decides this, and the order is not alphabetical. An arm
    carrying both methods must go to the identity, never to the annihilating
    gate, so `identity_heads` is tried first and this pins that."""
    arm = r1.make_arm("arm_smprime", S).to(DT).to(DEV)
    arm.zero_heads = lambda: _zero_the_heads(arm)     # a harness that offers both
    _, method = r1.identity_point(arm)
    assert method == "identity_heads"
    with torch.no_grad():
        m, _ = smp.blend(*arm.heads(_x()), arm.g)
    assert torch.equal(m, torch.ones_like(m))


def test_arm_pl_still_goes_to_its_own_identity():
    """The fix must not move the OTHER arm. ARM PL's identity is zeroed heads
    and it has no `identity_heads`, so the same helper must land it there."""
    arm, method = r1.identity_point(arm_pl.ArmPL(S, d_model=DM).to(DT).to(DEV))
    assert method == "zero_heads"
    with torch.no_grad():
        g, s = arm.heads(_x())
    assert torch.equal(g, torch.zeros_like(g))


# ============================================== the PART IV column wiring

def test_the_gate_feature_is_the_arms_own_and_is_not_exponentiated():
    """`recovered_gate` is the one place the two arms differ. ARM PL's head is
    `log a_hat` and exponentiates; ARM S-M''s feature already IS `Re(a_hat)`, so
    exponentiating it would score a different quantity under one column name."""
    arm = r1.make_arm("arm_smprime", S).to(DT).to(DEV)
    x, live = _x(), [S - 2, S - 1]
    feat = r1.gate_features(arm, x, live)
    assert torch.equal(feat, arm.gate_feature(x, live))
    assert feat.shape == (3 * len(live), 1)
    assert torch.equal(r1.recovered_gate("arm_smprime", feat), feat[:, :1])
    pl = arm_pl.ArmPL(S, d_model=DM).to(DT).to(DEV)
    pf = r1.gate_features(pl, x, live)
    assert torch.equal(r1.recovered_gate("arm_pl", pf), pf[:, :1].exp())


def test_lambda_hat_reports_the_annihilating_gate_as_minus_infinity():
    """`log 0 = -inf` is the TRUE reading for a path-product arm, not an
    instrument failure, and it must not be quietly dropped into a mean that
    reads finite. The surviving positions are reported beside it."""
    arm = _zero_the_heads(r1.make_arm("arm_smprime", S).to(DT).to(DEV))
    col = r1.gate_columns("arm_smprime", arm, _x(), [S - 2, S - 1])
    assert col["lambda_hat"] == -math.inf
    assert col["frac_gate_annihilated"] == 1.0
    assert col["a_hat_max"] == 0.0
    assert col["unit_root"] is False


def test_the_unit_root_flag_and_the_dynamic_range_bound_cannot_disagree():
    """Both turn on `a_hat_max >= 1`. Two predicates for one crossing is how a
    scoreboard pays twice for one event."""
    x, live = _x(), [S - 2, S - 1]
    for kind, arm in (("arm_smprime", r1.make_arm("arm_smprime", S)),
                      ("arm_pl", arm_pl.ArmPL(S, d_model=DM)),
                      ("softmax", r1.make_arm("softmax", S))):
        col = r1.gate_columns(kind, arm.to(DT).to(DEV), x, live)
        assert col["unit_root"] == (col["dyn_range_bound"] == math.inf)
        assert col["unit_root"] == (col["a_hat_max"] >= 1.0)


def test_the_softmax_control_states_its_gate_columns_rather_than_measuring_them():
    """`g == 0` is the parity point: `a_hat == 1` everywhere BY CONSTRUCTION.
    The columns are a statement about the arm, and the `Z` winding -- which a
    real gate has no phase to carry -- is absent rather than filled with a `0`
    that would read as a measurement."""
    col = r1.gate_columns("softmax", r1.make_arm("softmax", S).to(DT).to(DEV),
                          _x(), [S - 2, S - 1])
    assert col["a_hat_max"] == 1.0 and col["lambda_hat"] == 0.0
    assert col["unit_root"] is True
    assert col["z_winding_max"] is None and col["z_winding_residual"] is None


def test_only_the_phase_arm_reports_a_z_winding():
    """ARM PL's gate is real: no phase, so no winding. Reported ABSENT, because
    a `0` there would be a reading ARM PL never took in a column ARM S-M' fills
    with one it did."""
    x, live = _x(), [S - 2, S - 1]
    pl = r1.gate_columns("arm_pl", arm_pl.ArmPL(S, d_model=DM).to(DT).to(DEV),
                         x, live)
    assert pl["z_winding_max"] is None and pl["z_winding_residual"] is None
    sm = r1.gate_columns("arm_smprime",
                         r1.make_arm("arm_smprime", S).to(DT).to(DEV), x, live)
    assert isinstance(sm["z_winding_max"], int)
    assert 0.0 <= sm["z_winding_residual"] <= 0.5


def test_the_distance_to_skyline_column_is_absent_and_says_why():
    """PART IV mandates the column; the v15/v16 scan-skyline module does not
    exist (`tests/mars_v15/test_skyline_gate_containment.py::_find_v15_skyline`
    searches four import paths and finds none). A number here would be a
    distance to an object that was never built, so the runner emits `None` with
    the reason beside it."""
    src = (ROOT / "scripts" / "v15_r1.py").read_text(encoding="utf-8")
    assert 'r["dist_to_skyline"] = None' in src
    assert "dist_to_skyline_why" in src


# ==================================================== the planted-negative rule

def test_a_nan_planted_negative_counts_as_fired():
    """ARM S-M''s `exp_scan` negative reads `nan` -- `-inf - (-inf)` at an
    annihilating gate. `nan > bar` is `False`, so a runner written that way
    would read the most decisive negative in the round as SILENT and abort a
    run whose rejection region was in fact occupied."""
    assert r1.fires(float("nan")) is True
    assert r1.fires(1.9) is True
    assert r1.fires(1e-16) is False
    assert r1.fires(0.0) is False


def test_the_shipped_negatives_all_fire_at_the_runners_own_bind():
    """The rejection region, at the runner's call site rather than at the
    module's. `bedm_draw` + `label_cell` is what `bind_check` runs."""
    a, b = smp.bedm_draw(seed=15, s=8, device=DEV)
    res = {m: smp.label_cell(a, b, mutation=m, seed=15)["residual"]
           for m in smp.MUTATIONS[1:]}
    assert smp.label_cell(a, b, seed=15)["residual"] <= 1e-6
    assert all(r1.fires(v) for v in res.values()), res
    assert math.isnan(res["exp_scan"])


# ======================================================= the device, live

@pytest.mark.parametrize("kind", ["arm_smprime", "arm_pl"])
def test_the_bind_rows_carry_the_device_they_were_measured_on(kind):
    """`V16_R1_DEVICE_READY.md`: this file had `device="cpu"` as a STRING
    LITERAL at four sites while calling the pooling guard. Every row a new code
    path emits must read the device off a tensor instead."""
    rows = []
    got = r1.bind_check(rows.append, [kind, "softmax"], device=DEV)
    assert rows == got and got, "bind_check must emit every row it returns"
    assert {r["device"] for r in got} == {DEV.type}
    assert {r["kind"] for r in got} == {kind}
    assert [r["t"] for r in got].count("identity") == 1
    assert [r["t"] for r in got].count("bind") == 2
