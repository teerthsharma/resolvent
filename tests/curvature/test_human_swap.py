"""The consequence-swap claim, carried off the synthetic bed onto human positions.

WHAT THIS TEST BINDS. ceqjepa.intent_do reports that a swap changing the set of
admitted futures moves the read while a meaning-preserving swap of the same
Hamming distance moves it EXACTLY 0.000000. That was measured on a generated bed
whose decor slots are planted consequence-free. This binds the same claim on the
13,479 human-played K+Q-vs-K positions in kqk_slice.tsv, where the admitted
futures come from ceqjepa.chess_steps' enumeration and nothing is planted.

THE TEST IS WRITTEN TO FAIL IF THE CLAIM SURVIVES UNQUALIFIED. The surface-only
zero is asserted only on the reads that are FUNCTIONS of the admitted-futures set
-- where it is an algebraic identity and not a measurement -- and the committor,
the one exact read of the same enumeration that is not such a function, is
asserted to MOVE. The learned pi-JEPA read is asserted NOT to separate, and the
random-init encoder beside it is asserted to do the same thing, because without
that arm the trained arm's number says nothing.

THE TWO CONTROLS THAT KILLED THE LAST ATTEMPT are both here and both are
required: the move-permutation ablation (ceqjepa/hf/README.md reports -0.0044 +-
0.0138 on DCM-1, so randomising which move was forced changed nothing) and the
oracle headroom (+0.0935 with the wrong sign, so no model could have won). A run
that drops either is not a run.
"""

import numpy as np
import pytest

from ceqjepa import human_swap as hs


# ---------------------------------------------------------------------------
# PROVENANCE. A check that cannot run is RED, never skipped.
# ---------------------------------------------------------------------------

def test_the_slice_is_the_one_this_file_was_written_against():
    prov = hs.slice_provenance()
    assert prov["sha256"] == hs.SLICE_SHA256
    assert prov["n_bytes"] == 512656
    assert prov["n_lines"] == 13479
    assert prov["n_black_queen_mirrored"] == 6630
    assert prov["n_mapped"] == 13479, (
        "half the slice is K vs K+q and only reaches the White-queen "
        "enumeration through a colour mirror; without it 6,630 lines vanish "
        "and the 'real data' is half of what it says")
    assert prov["n_distinct_positions"] == 9360


def test_every_base_position_carries_an_exact_label_or_an_explicit_refusal():
    b = hs.human_bases()
    assert b["n_distinct"] == 9360
    assert b["n_terminal_refused"] == 0
    assert b["n_win_basin"] == 9270
    assert b["n_draw_refused"] == 90
    assert b["n_win_basin"] + b["n_draw_refused"] == b["n_distinct"]


# ---------------------------------------------------------------------------
# SUPPORT. The bed PLANTED the surface-only cell. Real chess has to supply it.
# ---------------------------------------------------------------------------

def test_the_surface_only_cell_has_support_on_real_positions():
    c = hs.relocation_census()
    assert c["n_relocations"] == 1389988
    assert c["n_surface_only"] == 153215
    assert c["n_consequence"] == 1236773
    assert c["n_surface_only"] + c["n_consequence"] == c["n_relocations"]
    assert c["n_dropped_terminal"] == 5539
    assert c["surface_only_fraction"] > 0.10, (
        "a meaning-preserving intervention has to EXIST on a real board before "
        "anything can be measured about one")


def test_every_swap_in_both_families_moves_the_same_surface_hamming():
    c = hs.relocation_census()
    assert c["hamming_one_hot_consequence"] == (2, 2)
    assert c["hamming_one_hot_surface_only"] == (2, 2)


# ---------------------------------------------------------------------------
# THE CONFOUND. On the bed both partners sat at Hamming 3 and nothing else
# varied. On the board a second thing varies and it has to be pinned away.
# ---------------------------------------------------------------------------

def test_the_two_families_are_displacement_mismatched_before_the_pin():
    loose = hs.displacement_audit()["base-piece"]
    gap = loose["mean_consequence"] - loose["mean_surface_only"]
    assert gap > 1.0, (
        "untightened, surface-only relocations sit over a square closer to where "
        "they came from; if this ever reaches zero the tightened pin below stops "
        "being load-bearing and the claim has to be re-argued, not re-run")


def test_the_tightened_pin_removes_the_displacement_difference_exactly():
    tight = hs.displacement_audit()["base-piece-cheb"]
    assert tight["n_cells"] == 71016
    assert tight["mean_consequence"] == tight["mean_surface_only"]
    assert tight["max_abs_difference"] == 0.0


# ---------------------------------------------------------------------------
# THE READS THAT ARE FUNCTIONS OF THE ADMITTED SET. Zero here is an IDENTITY.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("read", ["a_set", "dtm"])
def test_reads_derived_from_the_admitted_set_read_exactly_zero_and_say_why(read):
    t = hs.swap_table(pin="base-piece-cheb")[read]
    assert hs.READ_IS_A_FUNCTION_OF_A[read] is True, (
        "%r must declare itself a function of the admitted-futures set, so its "
        "zero is never reported as a measurement" % read)
    assert t["surface_only"]["mean"] == 0.0
    assert t["surface_only"]["worst"] == 0.0
    assert t["surface_only"]["fraction_nonzero"] == 0.0
    assert t["consequence"]["mean"] > 0.0


# ---------------------------------------------------------------------------
# THE EXACT READ THAT IS NOT. This is the measurement.
# ---------------------------------------------------------------------------

def test_the_committor_is_not_a_function_of_the_admitted_set():
    assert hs.READ_IS_A_FUNCTION_OF_A["committor"] is False


def test_the_claim_does_not_survive_on_the_committor():
    t = hs.swap_table(pin="base-piece-cheb")["committor"]
    s, c = t["surface_only"], t["consequence"]
    assert s["fraction_nonzero"] > 0.99, (
        "a meaning-preserving swap is supposed to move the read 0.000000; it "
        "moves the exact committor on %.6f of pinned cells" % s["fraction_nonzero"])
    assert s["mean"] > 0.0
    assert s["worst"] > 0.15
    ratio = c["mean"] / s["mean"]
    assert 2.0 < ratio < 3.0, (
        "the bed reports an exact surface-only floor of 0.0 and an estimated "
        "ratio of 13.8; the exact non-identity read manages %.6f, and if that "
        "ever clears 3.0 the kill has to be re-derived rather than kept" % ratio)


def test_the_committor_movement_is_not_the_fixpoint_residual():
    t = hs.swap_table(pin="base-piece-cheb")["committor"]
    res = hs.committor_residual()
    assert res < 1e-11
    assert t["surface_only"]["mean"] / res > 1e8


def test_the_mechanism_is_multiplicity_and_not_membership():
    m = hs.multiplicity_audit()["surface_only"]
    assert m["fraction_set_preserved"] == 1.0
    assert m["fraction_multiset_preserved"] < 0.25, (
        "a surface-only swap preserves the SET of admitted futures and drops the "
        "multiset on most swaps; that is the mechanism behind the committor "
        "moving where a set read cannot")
    assert m["fraction_degree_preserved"] < 0.50


# ---------------------------------------------------------------------------
# THE PLANTED NEGATIVE. A similarity read must fail on both families.
# ---------------------------------------------------------------------------

def test_the_alikeness_read_cannot_separate_consequence_from_surface():
    t = hs.swap_table(pin="base-piece-cheb")["alikeness"]
    ratio = t["consequence"]["mean"] / t["surface_only"]["mean"]
    assert 0.5 < ratio < 2.0, (
        "the planted negative must fire on both families; ratio %.6f" % ratio)


# ---------------------------------------------------------------------------
# THE LEARNED READ, WITH THE ARM THAT MAKES IT READABLE
# ---------------------------------------------------------------------------

def test_the_learned_encoder_knows_this_endgame_before_it_is_asked_to_isolate():
    p = hs.jepa_probe()
    assert p["jepa_trained"]["rmse"] < p["jepa_trained"]["rmse_mean_baseline"], (
        "if the trained encoder carried no information at all about exact "
        "distance-to-mate, its failure to separate the two families below would "
        "be the boring kind and could not be reported as anything else")
    assert p["jepa_trained"]["r2"] > 0.10
    assert p["jepa_trained"]["r2"] > p["jepa_random"]["r2"], (
        "training has to buy something over a random-init encoder of the same "
        "architecture or there is nothing here to call learned")
    assert p["jepa_trained"]["r2"] < 2.0 * p["jepa_random"]["r2"], (
        "and it buys less than a doubling: a random projection of the board "
        "one-hot already reaches most of the trained encoder's R2, which bounds "
        "how much of the failure below can be blamed on the isolation task "
        "rather than on the encoder")


@pytest.mark.parametrize("read", ["jepa_trained", "jepa_random"])
def test_the_learned_read_does_not_separate_consequence_from_surface(read):
    t = hs.swap_table(pin="base-piece-cheb")[read]
    assert hs.READ_IS_A_FUNCTION_OF_A[read] is False
    ratio = t["consequence"]["mean"] / t["surface_only"]["mean"]
    assert 0.99 < ratio < 1.01, (
        "%r moves the same amount whether the intervention changed the admitted "
        "futures or not; ratio %.6f" % (read, ratio))
    assert t["surface_only"]["fraction_nonzero"] == 1.0


def test_training_bought_nothing_the_random_encoder_did_not_already_have():
    tab = hs.swap_table(pin="base-piece-cheb")
    def r(n):
        return tab[n]["consequence"]["mean"] / tab[n]["surface_only"]["mean"]
    assert abs(r("jepa_trained") - r("jepa_random")) < 0.01, (
        "the trained encoder and a never-trained one sit in the same cell of "
        "this test, which is what the random arm exists to detect")


# ---------------------------------------------------------------------------
# L-NULL. Two nulls, each naming what it VARIES and what it PINS.
# ---------------------------------------------------------------------------

def test_the_label_permutation_null_collapses_every_separation():
    nul = hs.label_permutation_null()
    assert nul["varies"] == ("family label",)
    assert set(nul["pins"]) == {
        "base position", "piece moved", "chebyshev displacement",
        "the relocation set itself", "the read", "the seed"}
    for read in hs.READS:
        assert abs(nul[read]["ratio"] - 1.0) < 0.05, (
            "under a shuffled family label read %r must separate nothing; "
            "ratio %.6f" % (read, nul[read]["ratio"]))


def test_the_move_permutation_ablation_names_both_sides():
    abl = hs.intervention_permutation_null()
    assert abl["varies"] == ("base-to-intervention pairing",)
    assert set(abl["pins"]) == {
        "piece moved", "chebyshev displacement", "family label",
        "the interventions in each bucket",
        "the base positions in each bucket", "the read", "the metric",
        "the seed"}
    for read in hs.READS:
        assert abs(abl[read]["permuted"]["ratio"] - 1.0) < 0.15, (
            "re-dealing which position each intervention lands on must leave "
            "nothing to separate; read %r permuted ratio %.6f"
            % (read, abl[read]["permuted"]["ratio"]))


def test_the_exact_read_carries_move_specific_information_and_the_learned_one_does_not():
    abl = hs.intervention_permutation_null()
    assert abl["committor"]["ratio_lost_to_permutation"] > 1.0, (
        "the exact committor's separation must depend on WHICH intervention was "
        "applied to WHICH position, or this bed has the same defect DCM-1's did")
    assert abl["jepa_random"]["ratio_lost_to_permutation"] < 0.05
    assert abl["jepa_trained"]["ratio_lost_to_permutation"] < \
        abl["alikeness"]["ratio_lost_to_permutation"], (
        "the trained encoder keeps less move-specific signal than the planted "
        "negative does; if that ever inverts, re-derive the verdict")


# ---------------------------------------------------------------------------
# THE HEADROOM, measured rather than hoped for.
# ---------------------------------------------------------------------------

def test_the_bed_headroom_is_measured_and_splits_in_two():
    h = hs.oracle_headroom()
    assert h["identity_reads"] == ("a_set", "dtm")
    assert h["exact_non_identity_read"] == "committor"
    assert 2.0 < h["ceiling_ratio"] < 3.0, (
        "the ceiling is the EXACT committor: no model can beat an exact "
        "quantity at being that quantity, so this is the headroom, full stop")
    assert h["ceiling_surface_only_mean"] > 0.0
    assert h["ceiling_surface_only_fraction_nonzero"] > 0.99


def test_the_module_self_check_passes():
    assert hs.demo() == 0
