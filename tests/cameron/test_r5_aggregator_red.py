"""CAMERON round 5 -- the RED tests behind the aggregator finding.

Every assertion here states what ARM A's K3 comparison NEEDS to be true for
"theta beats raw TV by 2.3%" to be a verdict about the SPHERE. Each one is run
against the code exactly as it stands. A test that goes green is a bind, not a
finding, and is labelled as such.

The conftest in this directory writes every outcome to house-events.jsonl.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)

from scale import arm_a_run                                        # noqa: E402
from scale import cameron_aggregator_probe as cap                  # noqa: E402
from scale.torque_probe import (rows_with_and_without, theta_rows,  # noqa: E402
                                tv_rows)
from scale.pivot_probe import select_pivots                        # noqa: E402

S, D, K, SEED = 128, 8, 4, 12345
PS, PD, PK = 1024, 16, 8            # the published ARM A cell
ARM_A_JOURNAL = ROOT / "results" / "arm_a.jsonl"


def _published_draw(seed: int = 0):
    """The FIRST draw of the published k=8 causal cell (arm_a_run seed 0).

    `test_bind_probe_reproduces_arm_a_draws` and the exact reproduction of
    `theta_c`/`d_theta` in results/cameron_aggregators.jsonl are what make
    "published" true here rather than "similar"."""
    g = torch.Generator().manual_seed(seed)
    r = cap.draw(PS, PK, PD, g, filler=False, heavy=False)
    assert r is not None
    return r, slice(r["c"] + 1, None)          # rows that CAN move: i > c


def _one_draw(dtype=torch.float32):
    """A draw of arm_a_run's protocol, rows returned at the requested dtype."""
    g = torch.Generator().manual_seed(SEED)
    x0 = torch.randn(S, D, generator=g)
    wq, wk, wo = (torch.randn(D, D, generator=g) for _ in range(3))
    v0 = torch.randn(S, D, generator=g)
    q, kk = (x0 @ wq).to(dtype), (x0 @ wk).to(dtype)
    i = S - 1
    j = int(torch.randint(1, i, (1,), generator=g))
    piv = select_pivots(kk, min(K, S - 2), exclude=(i, j))
    c = sorted(int(p) for p in piv)[0]
    a_c, a_0 = rows_with_and_without(q, kk, c)
    return c, theta_rows(a_c, a_0), tv_rows(a_c, a_0)


# ------------------------------------------------------------------ the bind
def test_bind_probe_reproduces_arm_a_draws():
    """BIND, expected GREEN: cameron's probe consumes the generator exactly as
    `arm_a_run.one` does, so both see the same (q, k, c, i, j) and the same
    reported theta. Without this every number below is on different draws."""
    g1 = torch.Generator().manual_seed(SEED)
    ref = arm_a_run.one(S, K, D, g1, filler=False)
    g2 = torch.Generator().manual_seed(SEED)
    mine = cap.draw(S, K, D, g2, filler=False, heavy=True)
    assert ref is not None and mine is not None
    got = cap.aggs(mine["th"], mine["c"], 0.0)["mean_all"]
    assert got == ref["theta"], f"probe theta {got!r} != arm_a_run {ref['theta']!r}"
    assert mine["tau"] == ref["tau"]
    # and the streams are still aligned after the draw
    assert float(torch.randn(1, generator=g1)) == float(torch.randn(1, generator=g2))


# ------------------------------------------------------------------ RED tests
def test_theta_rows_is_zero_when_a_row_has_no_mass_either_side():
    """`theta_rows`'s docstring: "Rows with no mass either side give 0."

    Row 0 of every causal draw is exactly that row -- `tril(-1)` leaves it with
    no allowed j, and `_softmax_operator` zeroes it -- so this is not a
    hypothetical input."""
    z = torch.zeros(1, 8)
    got = float(theta_rows(z, z)[0])
    assert got == 0.0, (f"theta_rows on a no-mass row returned {got!r} "
                        f"(= pi/2 = {math.pi / 2!r}); tv_rows returns "
                        f"{float(tv_rows(z, z)[0])!r} on the same input")


def test_row_zero_is_not_a_constant_offset_on_every_draw():
    """If row 0 contributes pi/2 to every draw, D_FR carries a constant
    pi/(2s) that is neither causal nor filler signal. At s=1024 that is
    0.0015340, against a published filler D_FR of 0.0033169."""
    c, th, tv = _one_draw()
    assert float(th[0]) != pytest.approx(math.pi / 2), (
        f"row 0 theta = {float(th[0])!r} = pi/2 on this draw; its share of a "
        f"s=1024 mean is {math.pi / 2 / 1024:.7f}, i.e. "
        f"{100 * (math.pi / 2 / 1024) / 0.0033169065058852236:.1f}% of the "
        f"published filler D_FR 0.0033169065058852236 and "
        f"{100 * (math.pi / 2 / 1024) / 0.030850132878792163:.1f}% of the "
        f"published causal D_FR 0.030850132878792163. tv row 0 = {float(tv[0])!r}")


def test_theta_is_not_a_pointwise_relabelling_of_tv():
    """K3 asks whether the SPHERE beats raw TV. That question only exists if
    theta carries information TV does not. For single-token ablation with
    renormalisation, A0_j = Ac_j/(1-p) for j != c, so BC = sqrt(1-p) and
    TV = p exactly -- making theta_i = arcsin(sqrt(TV_i)) row by row. Read in
    float64 on rows whose displacement exceeds 1e-6, where `arccos` near 1
    still has digits left; below that the deviation is the cancellation
    measured by the conditioning test, not a second geometry."""
    r, a = _published_draw()
    th, tv = r["th64"][a], r["tv64"][a]
    m = tv > 1e-6
    dev = float(((th[m] - torch.arcsin(torch.sqrt(tv[m]))).abs() / th[m]).max())
    assert dev > 1e-6, (
        f"max relative |theta_i - arcsin(sqrt(TV_i))| = {dev:.3e} over "
        f"{int(m.sum())} moved rows of the published k=8 seed-0 draw: theta is "
        f"a FIXED SCALAR FUNCTION of TV on this ablation, so K3 compares two "
        f"AGGREGATIONS OF ONE NUMBER, not two geometries")


def test_theta_rows_is_as_well_conditioned_as_tv_rows_in_float32():
    """Both statistics are read off the same float32 attention rows. `arccos`
    near 1 is the classic cancellation: d(theta)/d(BC) = -1/sin(theta), so an
    O(eps) error in BC becomes O(eps/theta) in theta -- unbounded as the
    displacement shrinks, which is exactly the regime this round measures. TV
    has no such term. The bar is 1%, set against K3's decision margin of 2.29%."""
    r, a = _published_draw()
    e_th = abs(float(r["th"][a].double().mean() - r["th64"][a].mean())
               ) / float(r["th64"][a].mean())
    e_tv = abs(float(r["tv"][a].double().mean() - r["tv64"][a].mean())
               ) / float(r["tv64"][a].mean())
    assert e_th < 0.01, (
        f"float32-vs-float64 relative error of the ACTIVE-ROW MEAN on the "
        f"published k=8 seed-0 draw: theta {100 * e_th:.2f}%, TV "
        f"{100 * e_tv:.6f}% -- same rows, same draw, same arithmetic change")


def test_theta_rows_does_not_silently_zero_rows_that_moved():
    """A row where TV reports displacement and theta reports exactly 0.0 is a
    row the sphere cannot see. `theta_rows` clamps BC to <= 1 and float32 BC
    saturates at 1 for small displacements, so those rows land on arccos(1)."""
    r, a = _published_draw()
    th, tv = r["th"][a], r["tv"][a]
    lost = int(((th == 0) & (tv > 0)).sum())
    assert lost == 0, (
        f"{lost} of {th.numel()} rows that moved read exactly 0.0 displacement "
        f"through theta while TV reads them nonzero "
        f"(largest such TV = {float(tv[(th == 0) & (tv > 0)].max()):.3e})")


def test_the_row_mean_is_the_best_available_aggregator():
    """The incumbent statistic is `theta_rows(...).mean()`. If some other
    aggregator of the SAME row vector separates causal from filler much better,
    K3's 2.29% was a fact about `.mean()`. Read off cameron's journal, whose
    cells reproduce the published theta_c/theta_f/d_theta/d_tv exactly."""
    j = ROOT / "results" / "cameron_aggregators.jsonl"
    lines = [json.loads(l) for l in j.read_text().splitlines() if l.strip()]
    lines = [l for l in lines if l.get("tag") == "primary"]
    assert lines, f"NOT FOUND: no tag=primary cells in {j}"
    from scale.torque_probe import cohen_d
    worst = None
    for L in lines:
        ds = {s: abs(cohen_d(L["causal"][s], L["filler"][s])) for s in cap.STATS}
        best = max(ds, key=ds.get)
        ratio = ds[cap.INCUMBENT] / ds[best]
        if worst is None or ratio < worst[0]:
            worst = (ratio, L["k"], best, ds[best], ds[cap.INCUMBENT],
                     ds[cap.RAW_TV])
    ratio, k, best, dbest, dinc, dtv = worst
    assert ratio > 0.9, (
        f"k={k}: incumbent {cap.INCUMBENT} |d|={dinc:.4f}, raw TV |d|={dtv:.4f} "
        f"(the whole K3 margin), but {best} |d|={dbest:.4f} on the IDENTICAL "
        f"draws -- the incumbent is {100 * ratio:.1f}% of the best available")


def test_k3_margin_is_decisive_on_the_published_cells():
    """The contract: theta must beat raw TV or the sphere is notation. A margin
    is decisive at 20%; the bar is stated here and read off the published
    journal, which is immutable (G2) and is only READ."""
    rows = [json.loads(l) for l in ARM_A_JOURNAL.read_text().splitlines() if l.strip()]
    assert rows, f"NOT FOUND: no rows in {ARM_A_JOURNAL}"
    margins = {r["k"]: 100 * (abs(r["d_theta"]) / abs(r["d_tv"]) - 1) for r in rows}
    assert all(m > 20.0 for m in margins.values()), (
        f"K3 margins by k (percent by which |d_theta| exceeds |d_TV|): "
        f"{ {k: round(v, 2) for k, v in margins.items()} }")


def test_arm_a_journal_reports_tau_and_gamma_as_separators():
    """`arm_a_run.one` computes tau and gamma_1 on EVERY draw of BOTH arms and
    journals only the causal-arm mean. Neither has ever been given a
    causal-vs-filler effect size, so the round has no idea whether the free
    statistics separate better than the one under test."""
    rows = [json.loads(l) for l in ARM_A_JOURNAL.read_text().splitlines() if l.strip()]
    assert rows, f"NOT FOUND: no rows in {ARM_A_JOURNAL}"
    missing = sorted({k for k in ("d_tau", "d_gamma") for r in rows if k not in r})
    assert not missing, (
        f"{missing} absent from every one of the {len(rows)} published ARM A "
        f"rows; keys present: {sorted(rows[0])}")
