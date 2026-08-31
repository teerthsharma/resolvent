"""G0.6 K-CERT: the certificate script's own invariants, each with a planted negative.

This repo has struck 14 vacuous controls. A test that only ever sees the passing
case cannot tell "the check holds" from "the check cannot fire", so every PASS
half below is paired with a FAIL half built from a fabricated reading:

  * a fabricated SLOW DEVICE -- same exponent, 10x the intercept -- must move the
    chunk table by exactly 10x. If it does not, the chunk table is a constant
    wearing a fit's clothes;
  * a fabricated bar row at delta/tol = 60% must HALT. If it does not, the 50%
    line is decoration;
  * a fabricated non-deterministic reduction must read NOT BITWISE. If it does
    not, the determinism column cannot distinguish the two regimes it reports;
  * a fabricated non-power-law series must be REFUSED on R^2. A fit with no R^2
    is not a law, and an R^2 that never rejects anything is not a test.

The last block reads `results/k_cert_local.json` -- the certificate the script
emitted when it was run once locally -- and asserts the invariants HELD on the
real measurement, not merely that the predicates exist.
"""
from __future__ import annotations

import json
import math
import pathlib

import pytest

from scripts import k_cert

ROOT = pathlib.Path(__file__).resolve().parents[2]
CERT = ROOT / "results" / "k_cert_local.json"


# ------------------------------------------------------------------ the fit

def test_an_exact_power_law_is_recovered_with_r2_one():
    ns = [2048, 4096, 8192, 16384, 32768]
    ys = [math.exp(-11.9670) * n ** 0.9734 for n in ns]
    f = k_cert.fit_power_law(ns, ys)
    assert f["r2"] == pytest.approx(1.0, abs=1e-12)
    assert f["b"] == pytest.approx(0.9734, abs=1e-9)
    assert f["log_a"] == pytest.approx(-11.9670, abs=1e-9)


def test_a_series_with_no_power_law_structure_is_refused_on_r2():
    """The must-fire half of the R^2 gate. Without this, MIN_R2 could be 0 and
    every table above would still read GREEN."""
    ns = [2048, 4096, 8192, 16384, 32768]
    ys = [1.0, 5.0, 1.2, 9.0, 1.1]          # no monotone power-law structure
    f = k_cert.fit_power_law(ns, ys)
    assert f["r2"] < k_cert.MIN_R2
    with pytest.raises(k_cert.NotALaw):
        k_cert.refuse_weak_fit(f, "fabricated non-law")


def test_a_fabricated_slow_device_moves_the_chunk_table_by_exactly_its_factor():
    """The chunk table must be a FUNCTION of the refitted law. A device ten times
    slower at every `n` must buy exactly one tenth of the steps."""
    ns = [2048, 4096, 8192, 16384, 32768]
    fast = [math.exp(-11.9670) * n ** 0.9734 for n in ns]
    slow = [10.0 * y for y in fast]
    ff, fs = k_cert.fit_power_law(ns, fast), k_cert.fit_power_law(ns, slow)

    assert fs["b"] == pytest.approx(ff["b"], abs=1e-9)      # same exponent
    assert fs["log_a"] - ff["log_a"] == pytest.approx(math.log(10.0), abs=1e-9)
    for n in ns:
        a = k_cert.steps_in_hours(ff, n, 11.0)
        b = k_cert.steps_in_hours(fs, n, 11.0)
        assert a > 0 and b > 0
        # `rel=1e-4` and not exact: `steps_in_hours` FLOORS to whole steps, so
        # the ratio carries at most one step of rounding out of >=10^5.
        assert a / b == pytest.approx(10.0, rel=1e-4)


# -------------------------------------------------------------- the bar line

def test_a_bar_row_at_sixty_percent_of_tolerance_halts():
    """The 50% line, fired. `delta` is chosen so `delta/tol` is exactly 0.60."""
    rows = [dict(task="fabricated", clause="predict_the_mean",
                 cpu=1.0, cuda=1.0 + 6e-7, tol=1e-6)]
    for r in rows:
        r["delta"], r["delta_over_tol"] = k_cert.delta_over_tol(r["cpu"], r["cuda"], r["tol"])
    assert rows[0]["delta_over_tol"] == pytest.approx(0.60, rel=1e-9)
    assert k_cert.halt_on_bar(rows) is not None
    assert "HALT" in k_cert.halt_on_bar(rows)


def test_a_bar_row_inside_the_line_does_not_halt():
    rows = [dict(task="control", clause="predict_the_mean",
                 cpu=1.0, cuda=1.0 + 8.58e-8, tol=1e-6)]
    for r in rows:
        r["delta"], r["delta_over_tol"] = k_cert.delta_over_tol(r["cpu"], r["cuda"], r["tol"])
    assert rows[0]["delta_over_tol"] < k_cert.HALT_FRACTION
    assert k_cert.halt_on_bar(rows) is None


def test_the_halt_line_is_the_contract_line():
    assert k_cert.HALT_FRACTION == 0.50


# --------------------------------------------------------------- 0-step gate

def test_the_zero_step_gate_admits_the_measured_null_tail_and_rejects_a_real_beat():
    # the reading that aborted an eight-seed run; see tests/test_zero_step_gate.py
    assert k_cert.zero_step_gate(1.0007840721511323, 0.9999703932724174)
    # the must-fire half: an arm that genuinely starts ahead of predict-the-mean
    assert not k_cert.zero_step_gate(1.0, 0.95)
    assert not k_cert.zero_step_gate(0.95, 0.95)
    # NaN first: float('nan') >= 1.0 is False, but say so rather than rely on it
    assert not k_cert.zero_step_gate(float("nan"), 1.0)


# --------------------------------------------------------------- determinism

def test_a_fabricated_non_deterministic_reduction_reads_not_bitwise():
    assert k_cert.determinism_verdict([0.0, 0.0, 0.0])["bitwise"] is True
    v = k_cert.determinism_verdict([0.0, 1e-7, 0.0])
    assert v["bitwise"] is False
    assert v["max_abs"] == pytest.approx(1e-7)


# ------------------------------------------- the certificate this script emitted

def _cert():
    if not CERT.exists():
        pytest.fail("results/k_cert_local.json missing -- run `python scripts/k_cert.py` once")
    return json.loads(CERT.read_text())


def test_certificate_carries_both_refits_and_prints_an_r2_for_each():
    c = _cert()
    laws = c["throughput"]["laws"]
    assert laws, "no throughput law fitted"
    for name, f in laws.items():
        assert "r2" in f, "{} has no R^2 -- a fit with no R^2 is not a law".format(name)
        assert f["r2"] >= k_cert.MIN_R2, (name, f["r2"])
        assert f["n_points"] >= 3
    m = c["memory"]["law"]
    assert "r2" in m and m["r2"] >= k_cert.MIN_R2


def test_certificate_worst_delta_over_tol_clears_the_fifty_percent_line():
    c = _cert()
    w = c["bar"]["worst"]
    assert w["delta_over_tol"] < k_cert.HALT_FRACTION, w
    assert c["bar"]["halt"] is None
    assert len(c["bar"]["rows"]) >= 20, "fewer rows than 4 rungs x 5 clauses"


def test_certificate_zero_step_gate_fired_at_every_cell_shape_it_could_reach():
    c = _cert()
    cells = c["zero_step"]["cells"]
    assert cells
    ran = [x for x in cells if x["ok"] is not None]
    assert ran, "no cell shape was reachable"
    assert all(x["ok"] for x in ran), [x for x in ran if not x["ok"]]
    # non-degeneracy: the gate was evaluated on real readings, not on a constant
    assert len({round(x["nrmse0_eval"], 9) for x in ran}) > 1


def test_certificate_reports_determinism_in_both_regimes():
    c = _cert()
    d = c["determinism"]
    assert d["reduction_length"] == 64
    for regime in ("flag_off", "flag_on"):
        assert regime in d
        r = d[regime]
        assert r["executable"] in (True, False)
        # per QUANTITY, because forward and backward do not stand or fall
        # together under the flag -- that split is the finding, not a detail
        for q in ("hop", "forward", "gradient"):
            assert r[q]["executable"] in (True, False)
            if r[q]["executable"]:
                assert "bitwise" in r[q] and "max_abs" in r[q]
            else:
                assert r[q]["error"]


def test_certificate_names_the_box_and_the_commit_so_kaggle_can_append():
    c = _cert()
    assert c["box"]["device_name"]
    assert c["box"]["torch"]
    assert c["git"]["head"]
    assert c["schema"] == k_cert.SCHEMA
