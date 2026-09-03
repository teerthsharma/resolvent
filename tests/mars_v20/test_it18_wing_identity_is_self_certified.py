"""MARS it.18 STRIKE 2 -- the clause-(b) repair binds the manifest to itself.

SATURN's own strike: the `kind` exchange left all four it.4 nodes 8/8 GREEN while
the manifest certified W1 by an `arm_pl` record. His repair (tests/saturn/
test_v20_r15_freeze_manifest.py::_arm_of) derives the arm name from the wing's
OWN clause-(a) module path rather than a hardcoded map, "because a hardcoded map
would be a second place for the manifest to be wrong".

A hardcoded map is also the only place the manifest can be CAUGHT being wrong.
After the repair, clause (b) is checked against clause (a) and clause (a) is
checked against nothing. The wing->arm binding is now a closed loop inside one
file, so the manifest cannot contradict itself about a wing's identity -- it can
only be uniformly wrong about it.

V-26's rule: the falsifying mutation must preserve every per-collection
statistic. This one does. Exchanging W1's and W3's clause-(a) AND clause-(b)
citations leaves:
  * the same 2 wings and the same 8 rows            (FROZEN-N unchanged)
  * 4 clauses per wing, no clause duplicated        (unchanged)
  * every anchor still on its exact cited line      (resolve unchanged)
  * every clause-(b) `kind` equal to its wing's derived arm  (the repair's joint)
  * two distinct arms, one per wing                 (distinctness unchanged)
and yet W1 -- the smprime wing, whose clauses (c) and (d) still cite
V16_ARM_SMPRIME.md and the 15.970 price -- is now certified as `arm_pl`.

RED against unmutated code: the mutation is applied to the manifest ROWS in
memory, the shipped test functions are called on them unchanged, and none of
them fires.

Run:  python -m pytest tests/mars_v20/test_it18_wing_identity_is_self_certified.py -x -q
"""
from __future__ import annotations

import pytest

from tests.saturn import test_v20_r15_freeze_manifest as FM


def _swapped(rows):
    """Exchange W1's and W3's clause (a) and (b) citations. Nothing else moves."""
    by = {(w, c): (cite, anch) for w, c, cite, anch in rows}
    out = []
    for w, c, cite, anch in rows:
        if c in ("a", "b"):
            other = "W3" if w == "W1" else ("W1" if w == "W3" else w)
            if (other, c) in by:
                cite, anch = by[(other, c)]
        out.append((w, c, cite, anch))
    return out


def test_the_mutation_is_statistic_preserving():
    """The V-26 precondition, asserted before the falsification is claimed."""
    orig = FM.rows()
    mut = _swapped(orig)
    assert len(mut) == len(orig), "row count moved"
    assert sorted(w for w, _c, _p, _a in mut) == sorted(w for w, _c, _p, _a in orig)
    assert sorted((w, c) for w, c, _p, _a in mut) == sorted((w, c) for w, c, _p, _a in orig)
    # same multiset of citations -- only their wing changed
    assert sorted(p for _w, _c, p, _a in mut) == sorted(p for _w, _c, p, _a in orig)
    assert mut != orig, "the mutation did not change anything; W1/W3 rows missing"


def _declared_params(fn):
    """The node's OWN parametrize list, so this file never invents its arguments.

    REPAIRED at it.20 by SATURN, whose own it.19 repair broke this test: he
    re-parametrized `test_every_frozen_wing_is_found_and_not_merely_named` from
    ARM NAMES to WING IDS, the hardcoded `("arm_smprime",), ("arm_pl",)` below
    went stale, and the node then raised identically with and without the
    mutation (`AssertionError` on both sides, an arm name being no wing at all). This test went GREEN on that raise while the defect it
    guards was untouched -- and a test that passes for a reason unrelated to the
    bug is worse than no test. Duplicated from
    `tests/saturn/test_v20_r15_it19_wing_identity.py` rather than imported: that
    file imports `_swapped` from this one, and a cycle is a worse dependency than
    six lines.
    """
    for m in getattr(fn, "pytestmark", []):
        if m.name == "parametrize":
            return [(v,) for v in m.args[1]]
    return [()]


def _nodes():
    return (
        ("clause_b_journals_that_wings_own_arm",
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
    ) + tuple(
        # Discovered, not named: the corroboration node SATURN's repair adds.
        ("every_wings_arm_is_corroborated_outside_clause_a", fn, [()])
        for fn in [getattr(FM, "test_every_wings_arm_is_corroborated_outside_clause_a", None)]
        if fn is not None
    )


def _fired(nodes):
    out = {}
    for name, fn, args in nodes:
        for a in args:
            try:
                fn(*a)
            except Exception as exc:
                out[(name, a)] = f"{type(exc).__name__}: {exc}"
    return out


def test_the_clause_b_repair_does_not_catch_the_identity_swap(monkeypatch):
    """RED at it.18: SATURN's repaired joint passed on a manifest that swaps two
    wings. GREEN only once a node fires on the swap AND is silent on the true
    rows -- the differential, added at it.20. A node that raises either way is
    not evidence, and that is exactly how this test was passing."""
    on_truth = _fired(_nodes())
    mut = _swapped(FM.rows())
    monkeypatch.setattr(FM, "rows", lambda: mut)

    # sanity: the repair really is deriving from the swapped clause (a)
    assert FM._arm_of("W1") == "arm_pl", (
        "premise gone: the swap did not move W1's derived arm; re-derive the strike"
    )

    fired = [f"{k}: {v}" for k, v in _fired(_nodes()).items() if k not in on_truth]

    assert fired, (
        "W1 is now certified as arm_pl by its own clause (a) and its own clause "
        "(b), while its clauses (c) and (d) still cite V16_ARM_SMPRIME.md and the "
        "15.970 smprime price -- and not one semantic node fires. The repair moved "
        "the wing->arm binding INSIDE the manifest, so the manifest now agrees "
        "with itself by construction. The `kind` exchange SATURN struck is still "
        "reachable: exchange clause (a) with it."
    )
