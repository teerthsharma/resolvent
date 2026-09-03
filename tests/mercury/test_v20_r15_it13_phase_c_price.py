"""it.13 MERCURY — what Phase C costs and what it can decide.

Every number in V20_R15_IT13_MERCURY.md that is arithmetic is bound here.
RED first: this node was written before `tests/mercury/phase_c_price.py` existed.
"""
import pathlib
import pytest

from tests.mercury.phase_c_price import (
    BANKED_40,
    HASH,
    banked_cells,
    hash_reader_census,
    jupiter_it8_formula,
    s_exponent_band,
    chess_producer_census,
    chess_oracle_cost,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_the_40_banked_cells_are_found_under_one_instrument_hash():
    cells = banked_cells(ROOT)
    assert len(cells) == 40, f"expected the 40 JUPITER prices, got {len(cells)}"
    assert {c["instrument_hash"] for c in cells} == {HASH}


def test_the_retake_of_all_40_costs_317_not_684():
    """JUPITER it.9:239 charges 40 x 17.1 s. 17.1 is arm_smprime's mean; 20 of the
    40 cells are arm_pl or softmax at ~1.7 s. The cells' own secs say 317.092."""
    cells = banked_cells(ROOT)
    total = sum(c["secs"] for c in cells)
    assert total == pytest.approx(317.092, abs=0.01)
    assert 40 * 17.1 == pytest.approx(684.0)
    assert total < 0.47 * 684.0


def test_saturns_156_975_is_the_retake_24_subset():
    cells = [c for c in banked_cells(ROOT) if c["_file"].endswith("v17k_r4_retake.jsonl")]
    assert len(cells) == 24
    assert sum(c["secs"] for c in cells) == pytest.approx(156.975, abs=0.01)


def test_the_point_on_the_band_is_withdrawn_from_every_assertion_in_this_office():
    """it.33 WITHDRAWAL, with the address list grep-produced and asserted empty --
    the route MARS-31-C named after the `62` shipped ONE address and left six live.

    MARS-31-C named TWO addresses, `:48` and `:55` of this file, by scanning this
    file alone. A scan of the whole of `tests/` finds FIVE assertion lines
    carrying the point and THREE of them in this office; the third,
    `tests/mercury/test_v20_r15_it30_admission.py:107`, a file-scoped scan cannot
    see. All three are withdrawn at it.33.

    The remaining two are named, not repaired, and the partition is exact at five:
      `tests/jupiter/...it31_c37_and_table_edits.py:195` -- a `not in` assertion,
        so it ENFORCES the withdrawal rather than quoting the point.
      `tests/mars_v20/...it31_the_repairs_of_it29_it30.py:179` -- MARS's own
        unmutated-instrument control. MARS's address to withdraw, not this
        office's; it is listed so the count is closed, not so it is claimed.
    """
    point = "309." + "047"          # split, so this node is not its own hit
    hits = []
    for p in sorted((ROOT / "tests").rglob("*.py")):
        for i, ln in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if point in ln and ln.lstrip().startswith("assert "):
                hits.append((p.relative_to(ROOT).as_posix(), i, "not in" in ln))
    assert hits, "the scanner found nothing anywhere -- it is broken"
    mine = sorted("%s:%d" % (a, i) for a, i, _ in hits if a.startswith("tests/mercury/"))
    assert mine == [], "the point survives in this office at: %s" % (mine,)
    assert sorted((a, enforces) for a, _, enforces in hits) == [
        ("tests/jupiter/test_v20_r15_it31_c37_and_table_edits.py", True),
        ("tests/mars_v20/test_v20_r15_it31_the_repairs_of_it29_it30.py", False),
    ], hits


def test_the_s_sweep_price_is_a_band_because_the_exponent_is_what_is_unknown():
    """BAND ONLY, NO POINT -- the it.32 ruling, which is MARS's. `3 x (0.25+1+4) x
    (16.16+1.78+1.68)` is written at V20_R15_IT8_JUPITER.md:216 and labelled 275;
    it.13 priced that label with a point on this band. Band and point are ONE
    MONOTONE FAMILY in the exponent `e`, so containment of any point is a
    consequence of the ordering and is never a measurement. What the withdrawal
    costs this office: it.13's `+12.4% against its own label` does not survive it,
    because 275 lies INSIDE the band and the band cannot price the label at all.
    """
    lo, mid, hi = s_exponent_band()
    assert lo < mid < hi, "the family is not monotone in e -- containment is not free"
    assert (round(lo, 3), round(hi, 3)) == (206.031, 537.152)   # S^1 and S^3, the finding
    assert lo < 275.0 < hi, "275 is inside the band; the band does not price the label"
    assert jupiter_it8_formula() == mid, "the withdrawn point was the band midpoint"


def test_instrument_hash_is_read_by_zero_files_under_ceq_and_scale():
    """SATURN's split: the hash MOVES on any edit, but nothing under ceq/ or scale/
    reads it, so the edit forks the pool and invalidates nothing journalled."""
    census = hash_reader_census(ROOT)
    assert census["ceq"] == [], census["ceq"]
    assert census["scale"] == [], census["scale"]
    # control: the searcher is capable of returning non-zero
    assert census["scripts"], "the searcher found nothing anywhere - it is broken"


def test_the_chess_witness_is_two_thirds_built_and_registers_as_no_bed():
    """Clause (4)'s bed, as it actually stands. The tree HAS a chess producer and
    the legality / next-FEN oracles; it has NO registered chess bed, NO eval-delta
    producer, and NO arm that consumes a chess corpus."""
    found = chess_producer_census(ROOT)
    assert "ceq/kdata.py" in found["code"], found["code"]
    assert found["registered_beds"] == ["bed_m", "bed_k", "bed_1"], found["registered_beds"]
    assert "chess" not in " ".join(found["registered_beds"])
    assert found["eval_delta_producers"] == [], found["eval_delta_producers"]
    assert found["engines"] == [], found["engines"]
    assert found["arm_consumers"] == [], found["arm_consumers"]


def test_the_two_oracles_that_do_exist_cost_zero_gpu_seconds():
    n_rows, secs = chess_oracle_cost(ROOT)
    assert n_rows == 512, n_rows
    assert secs < 5.0, f"{secs} s of CPU on the 36-game fixture"
