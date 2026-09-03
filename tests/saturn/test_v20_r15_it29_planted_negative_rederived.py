"""it.29 SATURN REPAIR 1 -- the it.27 planted negative, re-derived against its own code.

The INSPECTOR struck the PROSE beside a working instrument, and the strike holds:

  it.27 said the planted negative fires because `ceq/arm_pl.py:163-172` "is a scalar
  recurrence loop" while the row describes a masked reverse-cumprod. It does not.
  Driving `resolved_arms` directly, EVERY range resolves to `[]` under the planted
  module swap -- `163-172`, `162-168`, `1-2`, `358-360`, `400-410`, `144-174`.
  What fires is `anchored`: no symbol NAMED BY THE ROW is defined in `arm_pl`, so no
  range in that module can be anchored. The range is decorative in that negative.

  it.27 also said "the row becomes UNDERIVABLE". It does not. `ledger_wing_arm` on the
  planted text returns `{'W1': 'arm_smprime', 'W3': 'arm_pl'}` -- byte-identical to
  control -- because L-10's SYMBOL citations still resolve. The line citation dies;
  the wing binding does not.

RED 1 and RED 2 record those two, against the unmutated it.27 module. The remaining
nodes re-derive what the mechanism actually is and make the range load-bearing where
it can be: with an anchor present, the range discriminates (L-13 sweep).

Run:  python -m pytest tests/saturn/test_v20_r15_it29_planted_negative_rederived.py -q
"""
from __future__ import annotations

import pathlib

from tests.saturn.test_v20_r15_it27_wing_arm_citation import (
    LEDGER, ledger_wing_arm, resolved_arms, rows, span_of,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
TEXT = LEDGER.read_text(encoding="utf-8", errors="replace")
ROW = {lid: row for lid, _s, row in rows(TEXT)}
SWAP = ("ceq/arm_smprime.py:163-172", "ceq/arm_pl.py:163-172")


def planted(rng: str) -> list:
    """L-10 with its module swapped to `arm_pl` and its range set to `rng`."""
    row = ROW["L-10"].replace(SWAP[0], f"ceq/arm_pl.py:{rng}")
    return resolved_arms(row).get(f"line ceq/arm_pl.py:{rng}")


# ==========================================================================
#  RED -- two claims in the it.27 filing, measured against the it.27 code
# ==========================================================================

def test_red1_the_planted_negatives_range_is_load_bearing():
    """RED 1. The it.27 prose names the RANGE as what fails. Measure it.

    If the range were load-bearing, at least one range in `arm_pl` would resolve
    under the planted swap and one would not. Every range dies identically.
    """
    sweep = {r: planted(r) for r in
             ("163-172", "162-168", "1-2", "358-360", "400-410", "144-174")}
    assert any(v for v in sweep.values()), (
        "the it.27 planted negative is INSENSITIVE to the range it names: every range "
        f"in arm_pl resolves to [] -- {sweep}. `ArmPL` spans {span_of('arm_pl','ArmPL')} "
        "and `ArmPL.forward` spans "
        f"{span_of('arm_pl','ArmPL.forward')}, so ranges inside real arm_pl bodies die "
        "too. What fires is that no symbol L-10 NAMES is defined in arm_pl -- "
        "`path_product` and `zero_hop_mask` are arm_smprime-only -- so `anchored` is "
        "empty for any range. The '163-172 is a scalar recurrence loop' prose describes "
        "a discrimination the node never performs.")


def test_red2_the_founding_swap_makes_the_row_underivable():
    """RED 2. The it.27 prose says the row becomes underivable. Measure the binding."""
    control = ledger_wing_arm(TEXT)
    after = ledger_wing_arm(TEXT.replace(ROW["L-10"], ROW["L-10"].replace(*SWAP)))
    assert after != control, (
        f"the founding swap changes nothing: derived {after} == control {control}. "
        "L-10's symbol citations `path_product` / `zero_hop_mask` still resolve into "
        "arm_smprime, so W1 keeps its binding. The line citation became unresolvable; "
        "the ROW did not become underivable. it.27 claimed the stronger fact.")


# ==========================================================================
#  The mechanism as it actually is
# ==========================================================================

def test_what_actually_fires_is_the_missing_anchor_not_the_range():
    """The planted negative's real mechanism, stated as a measurement."""
    sweep = {r: planted(r) for r in ("163-172", "162-168", "358-360", "400-410")}
    assert all(v == [] for v in sweep.values()), sweep
    assert span_of("arm_pl", "path_product") is None
    assert span_of("arm_pl", "zero_hop_mask") is None
    assert span_of("arm_smprime", "path_product") == (144, 172)


def test_the_range_is_load_bearing_where_an_anchor_exists():
    """L-13 names `ArmSMPrime.forward` (572-577) AND cites arm_smprime, so the
    range is the only free variable. It discriminates -- which is the claim it.27
    made for L-10, transplanted to the row where it is true."""
    def l13(rng):
        row = ROW["L-13"].replace("ceq/arm_smprime.py:572-577",
                                  f"ceq/arm_smprime.py:{rng}")
        return resolved_arms(row).get(f"line ceq/arm_smprime.py:{rng}")

    assert span_of("arm_smprime", "ArmSMPrime.forward") == (572, 577)
    assert l13("572-577") == ["arm_smprime"]
    for outside in ("560-565", "1-5", "497-500", "572-600"):
        assert l13(outside) == [], f"{outside} resolved but lies outside the anchor"


def test_the_founding_swap_kills_the_line_citation_and_nothing_more():
    """The corrected claim for planted negative 2, asserted at its true strength."""
    l10 = ROW["L-10"]
    assert resolved_arms(l10)["line ceq/arm_smprime.py:163-172"] == ["arm_smprime"]
    after = resolved_arms(l10.replace(*SWAP))
    assert after["line ceq/arm_pl.py:163-172"] == []
    assert after["symbol path_product"] == ["arm_smprime"]
    assert ledger_wing_arm(TEXT.replace(l10, l10.replace(*SWAP))) == ledger_wing_arm(TEXT)


# ==========================================================================
#  Boundary correction -- the node is STRONGER than it.27 claimed
# ==========================================================================

def _swap_names(s: str) -> str:
    s = (s.replace("arm_smprime", "@@A@@").replace("arm_pl", "arm_smprime")
          .replace("@@A@@", "arm_pl"))
    return (s.replace("ArmSMPrime", "@@B@@").replace("ArmPL", "ArmSMPrime")
             .replace("@@B@@", "ArmPL"))


def test_a_prose_only_founding_rewrite_refuses_rather_than_reading_green():
    """it.27 boundary 1 said an internally consistent founding rewrite reads GREEN.
    A rewrite of the LEDGER PROSE alone does not: `path_product` carries no arm name,
    so it stays put and contradicts the swapped ones. Reading GREEN needs the `ceq/`
    SYMBOLS moved too -- a source edit, not a prose edit."""
    try:
        derived = ledger_wing_arm(_swap_names(TEXT))
    except AssertionError as e:
        assert "W1 resolves into ['arm_pl', 'arm_smprime']" in str(e), str(e)
    else:
        raise AssertionError(f"prose-only rewrite derived {derived} instead of refusing")


def test_a_path_only_rewrite_is_the_weaker_case_and_does_read_green():
    """The honest half of the correction: swapping only the `ceq/*.py` PATHS leaves
    every symbol resolving, so the binding survives unchanged. The node's strength
    comes from the symbols, and that is where its blind spot is too."""
    paths = (TEXT.replace("ceq/arm_smprime.py", "@@A@@")
                 .replace("ceq/arm_pl.py", "ceq/arm_smprime.py")
                 .replace("@@A@@", "ceq/arm_pl.py"))
    assert paths != TEXT
    assert ledger_wing_arm(paths) == ledger_wing_arm(TEXT)
