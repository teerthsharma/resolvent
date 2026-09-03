"""MARS it.5 — STRIKE 8. The seed-2 cell that decides VENUS's ranking is
separated from its seven siblings BEFORE the first gradient step, and the
separator is a column already in the journal.

W1 `arm_smprime` seed 2 carries the tournament's best `eval_nrmse` 0.203920
against W3's best 0.633739, a factor of 3.1. it.2 said the cause "needs weights
not in the tree". That was wrong in a useful way: `scripts/v15_r1.py` saves no
weights at all (0 of 342 weight files in the tree name `arm_smprime`), but it
journals the UNTRAINED gate state beside the trained one in the `*_0step`
columns. The question is answerable at zero GPU-seconds.

READ  results/v17k_r4_retake.jsonl, t=="cell" records, columns
      `frac_gate_annihilated_0step` (init) and `frac_gate_annihilated`(trained).

The node below asserts the premise VENUS's ranking needs — that W1's best cell
is a property of the primitive rather than of its initialisation draw — and it
is RED.
"""
import json
import pathlib

import pytest

JOURNAL = pathlib.Path(__file__).resolve().parents[2] / "results" / "v17k_r4_retake.jsonl"


def cells(kind: str) -> dict[int, dict]:
    """The eight terminal cells for one arm, by seed. NOT the 128 `trace` polls."""
    out = {}
    for line in JOURNAL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("t") == "cell" and r.get("kind") == kind and r.get("seed") is not None:
            out[r["seed"]] = r
    return out


def separation(col: str, kind: str = "arm_smprime", winner: int = 2) -> float:
    """`min(col over the other seeds) - col[winner]`.

    Positive means the winner is strictly below every sibling on this column and
    the two sets do not overlap; <= 0 means the column does not separate it.
    """
    c = cells(kind)
    others = [c[s][col] for s in c if s != winner]
    return min(others) - c[winner][col]


def test_the_eight_cells_are_eight_and_the_journal_is_mostly_not_cells():
    """Calibration. The freeze reads 8 cells per arm; the file holds ~142
    records per arm carrying `kind`. A search that returned 8 is reading the
    cells, not the polls."""
    rows = [json.loads(l) for l in JOURNAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(cells("arm_smprime")) == 8
    assert len(cells("arm_pl")) == 8
    kinded = [r for r in rows if r.get("kind") == "arm_pl"]
    non_cell = [r for r in kinded if r.get("t") != "cell"]
    #: 134 of 142 `arm_pl`-kinded records in this file are not produced cells.
    assert len(non_cell) > 100, len(non_cell)
    #: and at least one of them runs no training at all.
    assert any(r.get("t") == "identity" for r in non_cell)


def test_w1_best_cell_is_not_decided_before_training():
    """RED BY DESIGN. The strike.

    If seed 2's 3.1x win were a property of the path-product primitive, the
    untrained gate state would not already single it out. It does: seed 2 starts
    with `frac_gate_annihilated_0step = 0.124756`, and the nearest sibling starts
    at 0.200562 — a gap of 0.0758 with zero overlap across all eight seeds.
    """
    gap = separation("frac_gate_annihilated_0step")
    c = cells("arm_smprime")
    init = {s: c[s]["frac_gate_annihilated_0step"] for s in sorted(c)}
    assert gap <= 0.0, (
        "the winning cell is separated from all seven siblings at step 0, before "
        f"any gradient: gap={gap!r}, frac_gate_annihilated_0step per seed={init!r}, "
        f"trained frac_gate_annihilated={ {s: c[s]['frac_gate_annihilated'] for s in sorted(c)} !r}"
    )


def test_planted_negative_the_same_predicate_reports_no_separation_on_a_flat_column():
    """The zero is a measurement. `a_hat_min_0step` is 0.0 on all eight W1 cells,
    so the identical `separation()` call returns exactly 0.0 — not positive.
    A separator that separated everything would be a broken instrument."""
    assert separation("a_hat_min_0step") == 0.0
    #: and it is genuinely flat, not missing.
    c = cells("arm_smprime")
    assert all(c[s]["a_hat_min_0step"] == 0.0 for s in c)


def test_seed2_pins_the_same_a_hat_min_in_two_structurally_different_arms():
    """The corroboration, and it is not an init effect.

    On seed 2 alone, W1 and W3 converge to the same `a_hat_min` to 2.4e-5
    relative, from different starting points (W1 starts at 0.0, W3 at 0.4297).
    On every other seed the two disagree by 85-93%. Whatever pins 0.34076 on
    seed 2 is upstream of the arm.
    """
    smp, pl = cells("arm_smprime"), cells("arm_pl")
    rel = {s: abs(1.0 - pl[s]["a_hat_min"] / smp[s]["a_hat_min"])
           for s in smp if smp[s]["a_hat_min"] > 0.0}
    assert list(rel) == [2], rel
    assert rel[2] < 1e-4, rel
    #: the other seven are not comparable on this column at all: W1's clamp
    #: drives `a_hat_min` to an exact 0.0, which is itself the seed-2 anomaly.
    assert sum(1 for s in smp if smp[s]["a_hat_min"] == 0.0) == 7


def test_seed2_is_w3s_worst_cell_while_it_is_w1s_best():
    """The anti-correlation. A seed that made the primitive better would not
    also make the other primitive diverge."""
    smp, pl = cells("arm_smprime"), cells("arm_pl")
    assert min(smp, key=lambda s: smp[s]["eval_nrmse"]) == 2
    assert max(pl, key=lambda s: pl[s]["eval_nrmse"]) == 2
    assert pl[2]["a_hat_max"] > 12.0 and smp[2]["eval_nrmse"] < 0.21


def test_no_weights_exist_so_this_is_the_only_available_route():
    """`scripts/v15_r1.py` writes no checkpoint. The `*_0step` columns are the
    whole of the untrained record."""
    src = (JOURNAL.parents[1] / "scripts" / "v15_r1.py").read_text(encoding="utf-8")
    assert "torch.save" not in src and "save_file" not in src
    assert "_0step" in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
