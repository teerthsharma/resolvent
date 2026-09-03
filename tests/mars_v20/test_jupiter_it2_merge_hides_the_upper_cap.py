"""MARS, CEQ v20 R15 it.2 -- ATTACK ON V20_R15_IT2_JUPITER.md A.1/A.4.

JUPITER merges W1 and W3 into ONE primitive on the algebraic fact
`cumprod = exp . cumsum . log` with `g := log m`, and names the isomorphism's
domain exclusion as exactly one point: "The isomorphism has a domain:
`0 not in R_{>0}`" (A.1). That is the LOWER endpoint.

There is a SECOND domain difference and it is the one the record separates on.
W1's magnitude is `clamp(u, 0, 1)` (`ceq/arm_smprime.py:109`), so under the
isomorphism W1 occupies only `g = log m <= 0`. W3 ships `dyn_range_bound = inf`
and its trained gate exceeds 1.0 on EVERY cell, i.e. `g > 0` on 8 of 8.

A merge is entitled to the isomorphism only where both wings are in its image.
Read on the trained cells they are not in the same image, and the boundary that
separates them is the same one that partitions W3's own eval_nrmse with zero
overlap (MARS it.2 STRIKE 4).

Journal-only arithmetic plus one source read. No git write. Nothing touches
Kaggle.
"""
from __future__ import annotations

import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "results" / "v17k_r4_retake.jsonl"


def _cells(arm):
    out = []
    for line in JOURNAL.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if str(r.get("cell") or r.get("id") or "").startswith(arm + ":"):
                out.append(r)
    return out


SMP, PL = _cells("arm_smprime"), _cells("arm_pl")


def test_control_w1_is_actually_capped_in_source_and_in_the_journal():
    """PLANTED POSITIVE: the cap is real, in the code and in the record."""
    src = (ROOT / "ceq" / "arm_smprime.py").read_text()
    assert "clamp(u, 0.0, 1.0)" in src or "clamp(u, 0, 1)" in src, "no clamp"
    assert max(r["a_hat_max"] for r in SMP) <= 1.0 + 1e-12, [
        r["a_hat_max"] for r in SMP
    ]


def test_control_the_isomorphism_is_exact_where_both_wings_are_defined():
    """PLANTED POSITIVE: JUPITER's algebra is not what is being struck.

    On the shared image `m in (0, 1]`, `exp(cumsum(log m)) == cumprod(m)` to
    float64. The merge's ALGEBRA is correct; its DOMAIN claim is not.
    """
    m = [0.9, 0.5, 0.25, 1.0, 0.03]
    prod, acc, run = 1.0, 0.0, []
    for x in m:
        prod *= x
        acc += math.log(x)
        run.append(abs(prod - math.exp(acc)))
    assert max(run) < 1e-12, run


def test_the_merge_names_every_domain_exclusion_between_w1_and_w3():
    """RED: the merge names the lower endpoint and not the upper cap.

    W1 lives at `g <= 0`; W3's trained gate is above 1.0 on every cell, so
    `g > 0` on 8 of 8. The wings are not in a common image on any trained cell.
    """
    over = {
        str(r.get("cell") or r.get("id")): r["a_hat_max"]
        for r in PL
        if r["a_hat_max"] > 1.0
    }
    txt = (ROOT / "V20_R15_IT2_JUPITER.md").read_text(
        encoding="utf-8", errors="replace"
    )
    names_cap = ("clamp" in txt and "a_hat_max" in txt) or "116.006" in txt
    assert not over or names_cap, (
        f"W1 is clamp(u,0,1) so its image under g=log m is g<=0. W3 exceeds "
        f"the cap on {len(over)}/8 trained cells (g=log a_hat_max > 0): "
        f"{ {k: round(math.log(v), 4) for k, v in over.items()} }. "
        "V20_R15_IT2_JUPITER.md names only the lower exclusion '0 not in "
        "R_{>0}' and does not name the upper cap, which is the boundary the "
        "trained record actually separates on."
    )


def test_the_one_primitive_verdict_survives_the_boundary_it_did_not_name():
    """RED: the unnamed boundary partitions W3's own tournament.

    W3 cells inside W1's cap-image band (a_hat_max <= 1.5) and outside it
    (>= 12.7) split eval_nrmse with no overlap. A boundary that sorts the
    scoreboard is not a coordinate change.
    """
    inside = [r for r in PL if r["a_hat_max"] <= 2.0]
    outside = [r for r in PL if r["a_hat_max"] > 2.0]
    assert not (inside and outside) or min(
        r["eval_nrmse"] for r in outside
    ) <= max(r["eval_nrmse"] for r in inside), (
        f"the cap boundary JUPITER's merge does not name partitions W3's own "
        f"eval_nrmse with zero overlap: {len(inside)} cells at a_hat_max<=2.0 "
        f"score <= {max(r['eval_nrmse'] for r in inside):.6f}; "
        f"{len(outside)} cells above score >= "
        f"{min(r['eval_nrmse'] for r in outside):.6f}."
    )
