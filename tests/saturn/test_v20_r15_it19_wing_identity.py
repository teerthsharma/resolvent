"""SATURN it.19 REPAIR 2 -- the clause-(b) repair certified the manifest against itself.

MARS's it.18 STRIKE 2, and it lands. The it.14 repair replaced a hardcoded
wing->arm map with `_arm_of`, which DERIVES the arm from the wing's own clause-(a)
module path. Removing the map removed the only place the manifest could be caught
being wrong: clause (b) is now checked against clause (a), and clause (a) against
nothing. A closed loop inside one file cannot contradict itself about a wing's
identity -- it can only be uniformly wrong about it.

The falsifying mutation is MARS's, reused rather than reinvented: exchange W1's and
W3's clause-(a) AND clause-(b) citations together. Every per-collection statistic is
preserved (V-26), and W1 -- whose clauses (c) and (d) still cite V16_ARM_SMPRIME.md
and the 15.970 smprime price -- is certified `arm_pl`.

  RED 1  the swap must be caught by a SEMANTIC node, not merely by the digest.
         Before the repair, zero of seven fire.

  RED 2  `test_every_frozen_wing_is_found_and_not_merely_named` is parametrized on
         ARM NAMES and takes no wing argument, so it cannot witness a wing/arm
         mismatch in either direction. It must be parametrized on WINGS.

Run:  python -m pytest tests/saturn/test_v20_r15_it19_wing_identity.py -x -q
"""
from __future__ import annotations

import inspect

import pytest

from tests.saturn import test_v20_r15_freeze_manifest as FM
from tests.mars_v20.test_it18_wing_identity_is_self_certified import _swapped


def _declared_params(fn):
    """The node's OWN parametrize list -- so the harness never invents its args.

    Calling the found-wing node with wing ids while it is still parametrized on arm
    names makes it fire for the wrong reason and hands the strike a false GREEN.
    """
    for m in getattr(fn, "pytestmark", []):
        if m.name == "parametrize":
            return [(v,) for v in m.args[1]]
    return [()]


def _semantic_nodes():
    """Every node in the manifest file that reads the ROWS, with its call args."""
    return [
        ("every_clause_b_line_journals_that_wings_own_arm",
         FM.test_every_clause_b_line_journals_that_wings_own_arm, [()]),
        ("every_frozen_wing_is_found_and_not_merely_named",
         FM.test_every_frozen_wing_is_found_and_not_merely_named,
         _declared_params(FM.test_every_frozen_wing_is_found_and_not_merely_named)),
        ("every_frozen_citation_resolves_at_head",
         FM.test_every_frozen_citation_resolves_at_head, [(c,) for c in FM.CLAUSES]),
        ("the_frozen_list_names_only_found_wings",
         FM.test_the_frozen_list_names_only_found_wings, [()]),
        ("every_frozen_wing_cites_all_four_clauses",
         FM.test_every_frozen_wing_cites_all_four_clauses, [()]),
        ("no_wing_cites_the_same_line_for_two_different_clauses",
         FM.test_no_wing_cites_the_same_line_for_two_different_clauses, [()]),
        ("the_declared_N_equals_the_number_of_wing_rows",
         FM.test_the_declared_N_equals_the_number_of_wing_rows, [()]),
    ] + [
        # Discovered, not named, so this file is byte-identical across the RED
        # measurement and the GREEN one -- the repair adds the node, not the test.
        ("every_wings_arm_is_corroborated_outside_clause_a", fn, [()])
        for fn in [getattr(FM, "test_every_wings_arm_is_corroborated_outside_clause_a", None)]
        if fn is not None
    ]


def _fire(nodes):
    out = []
    for name, fn, args in nodes:
        for a in args:
            try:
                fn(*a)
            except Exception as exc:
                out.append(f"{name}{a}: {type(exc).__name__}: {exc}")
    return out


def test_the_true_rows_pass_every_semantic_node():
    """CALIBRATION, the other side. A node that always fires is not evidence."""
    assert _fire(_semantic_nodes()) == [], (
        "the unmutated manifest already fails a node; the repair below would be "
        "unfalsifiable")


def test_the_mutation_is_statistic_preserving():
    """The V-26 precondition, asserted before the falsification is claimed."""
    orig = FM.rows()
    mut = _swapped(orig)
    assert len(mut) == len(orig)
    assert sorted((w, c) for w, c, _p, _a in mut) == sorted((w, c) for w, c, _p, _a in orig)
    assert sorted(p for _w, _c, p, _a in mut) == sorted(p for _w, _c, p, _a in orig)
    assert mut != orig, "the mutation changed nothing; W1/W3 rows missing"


def test_the_identity_swap_is_caught_by_a_semantic_node(monkeypatch):
    """RED 1: clause (a) must be corroborated by something outside clause (a)."""
    mut = _swapped(FM.rows())
    monkeypatch.setattr(FM, "rows", lambda: mut)
    assert FM._arm_of("W1") == "arm_pl", (
        "premise gone: the swap did not move W1's derived arm; re-derive the strike")
    fired = _fire(_semantic_nodes())
    assert fired, (
        "W1 is certified arm_pl by its own clause (a) and its own clause (b) while "
        "its clauses (c) and (d) still cite V16_ARM_SMPRIME.md and the 15.970 "
        "smprime price, and not one semantic node fires. Clause (b) is checked "
        "against (a); (a) is checked against nothing.")


def test_the_found_wing_node_takes_a_wing_and_not_an_arm():
    """RED 2: parametrized on arm names, it cannot witness a mismatch either way."""
    fn = FM.test_every_frozen_wing_is_found_and_not_merely_named
    params = [a for (a,) in _declared_params(fn)]
    assert params, "the node lost its parametrize; re-derive the strike"
    wings = {r[0] for r in FM.rows()}
    assert set(params) <= wings, (
        f"the node is parametrized on {params}, which are ARM names, not "
        f"wing ids {sorted(wings)}. It takes no wing, so a wing whose clause (a) "
        "names the wrong arm still reads GREEN: it asks 'does this arm exist?', "
        "never 'is this arm THIS wing's?'")
    assert "wing" in inspect.signature(fn).parameters
