"""MARS, CEQ v20 R15 it.2 -- ATTACK ON V20_R15_IT2_SATURN.md, as it landed.

Two nodes, both against measurements SATURN files as closing the it.4 freeze.

A. His A.6 distinctness draw builds W1's gate from W3's journalled range.
   `tests/saturn/test_v20_r15_wings_distinct.py:20` says so and `:179` does it:
   `lo`/`hi` are ARM PL's `a_hat_min`/`a_hat_max`, and `u ~ U(lo, hi)`.
   ARM S-M''s magnitude is `torch.clamp(u, 0, 1)` (`ceq/arm_smprime.py:109`).
   The trained ARM S-M' cells reach `a_hat_min == 0.0` exactly on 7 of 8 -- the
   annihilating endpoint -- and `a_hat_max <= 1.0` on all 8. A uniform draw on
   ARM PL's range reaches NEITHER endpoint of W1's own cap.

B. His B seconds-to-floor sums `secs` in JOURNAL FILE ORDER to the first
   crossing cell. Per-cell cost is near constant within an arm, so the figure
   is (index of the first crossing seed) x (that arm's per-cell cost) -- a
   statistic about row order, not about either arm.

Journal-only arithmetic plus one source read. Nothing trains. No git write.
Nothing touches Kaggle.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "results" / "v17k_r4_retake.jsonl"
FLOOR_1 = 0.7071067811865476


def _cells(arm):
    out = []
    for line in JOURNAL.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        c = str(r.get("cell") or r.get("id") or "")
        if c.startswith(arm + ":"):
            out.append(r)
    return out


SMP, PL = _cells("arm_smprime"), _cells("arm_pl")


def test_control_the_two_arms_have_eight_cells_each_with_secs_and_nrmse():
    assert len(SMP) == 8 and len(PL) == 8
    assert all(isinstance(r.get("secs"), (int, float)) for r in SMP + PL)


def test_control_saturns_draw_range_is_read_from_the_w3_cells():
    """The range he draws from is ARM PL's, verified against his own file."""
    src = (ROOT / "tests" / "saturn" / "test_v20_r15_wings_distinct.py").read_text()
    assert "a_hat_min" in src and "a_hat_max" in src
    assert 'r["seed"], lo, hi' in src or "lo, hi" in src
    assert min(r["a_hat_min"] for r in PL) < 0.05
    assert max(r["a_hat_max"] for r in PL) > 100.0


def test_the_distinctness_draw_can_reach_w1s_own_gate_endpoints():
    """RED: the gate W1 is measured on excludes the endpoint W1 trains to.

    ARM S-M's magnitude is clamp(u, 0, 1). Drawing u uniformly on ARM PL's
    range puts P(u <= 1) at under 1%, so ~99% of gate entries clamp to exactly
    1.0 -- the identity gate, MARS's own corner in the m dial -- and NO entry
    can be 0.0, which is where 7 of 8 trained W1 cells sit.
    """
    lo = min(r["a_hat_min"] for r in PL)
    hi = max(r["a_hat_max"] for r in PL)
    p_below_cap = (min(1.0, hi) - lo) / (hi - lo)
    trained_lo = min(r["a_hat_min"] for r in SMP)
    trained_hi = max(r["a_hat_max"] for r in SMP)
    at_zero = [
        str(r.get("cell") or r.get("id")) for r in SMP if r["a_hat_min"] == 0.0
    ]
    assert lo <= trained_lo and p_below_cap > 0.5, (
        f"W1 distinctness is measured with u ~ U({lo:.7f}, {hi:.6f}) then "
        f"clamped to [0,1]: only {p_below_cap * 100:.2f}% of draws land below "
        f"the cap, so ~{100 - p_below_cap * 100:.2f}% of the gate is exactly "
        f"1.0. The trained W1 gate spans [{trained_lo}, {trained_hi}] and "
        f"reaches the annihilating endpoint 0.0 on {len(at_zero)}/8 cells "
        f"({at_zero}), which u never attains since lo={lo:.7f} > 0."
    )


def test_seconds_to_floor_does_not_depend_on_journal_row_order():
    """RED: criterion (3)'s 25.0x is a row-order artifact.

    Recompute the same statistic with the seeds visited best-first instead of
    file order. If the ranking's magnitude moves, the figure is about where the
    crossing seed sits in the file.
    """
    def s2f(cells, order):
        tot = 0.0
        for r in order(cells):
            tot += r["secs"]
            if r["eval_nrmse"] < FLOOR_1:
                return tot
        return None

    file_order = (lambda c: c)
    best_first = (lambda c: sorted(c, key=lambda r: r["eval_nrmse"]))
    a = {k: s2f(c, file_order) for k, c in (("W3", PL), ("W1", SMP))}
    b = {k: s2f(c, best_first) for k, c in (("W3", PL), ("W1", SMP))}
    ra, rb = a["W1"] / a["W3"], b["W1"] / b["W3"]
    spans = {
        "W3 secs": (min(r["secs"] for r in PL), max(r["secs"] for r in PL)),
        "W1 secs": (min(r["secs"] for r in SMP), max(r["secs"] for r in SMP)),
    }
    assert abs(ra - rb) < 0.5 * ra, (
        f"seconds-to-floor in FILE order {a} ranks W3 ahead of W1 by "
        f"{ra:.1f}x; the identical statistic with seeds visited best-first is "
        f"{b}, a ratio of {rb:.1f}x. Per-cell cost barely varies ({spans}), so "
        f"the figure is (index of the first crossing seed) x (per-cell cost). "
        f"W3 crosses on its FIRST journal row and W1 on its THIRD."
    )
