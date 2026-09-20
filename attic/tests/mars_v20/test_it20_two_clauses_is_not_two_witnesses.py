"""MARS it.20 STRIKE 1 -- clause (c) is not a witness outside clause (a).

SATURN's it.19 REPAIR 2 added `test_every_wings_arm_is_corroborated_outside_clause_a`
and priced it himself: "Two clauses is a second witness, not an independent one. It
would not catch a manifest in which clause (a) and clause (c) were moved together.
The third would be clause (d)'s price, and it is not asserted here."

The price is understated by one clause and overstated in kind.

* (a)+(c) alone does NOT survive -- clause (b) is derived from clause (a) by
  `_arm_of`, so moving (a) without (b) breaks the joint node. The manifest looks
  defended against the two-clause move for a reason that is not the new witness.
* (a)+(b)+(c) moved together survives EVERYTHING the freeze file ships, the new
  witness included. W1 is certified `arm_pl` while its clause (d) still cites
  `V17_R4_RETAKE_PRICE.md:194`, the 15.970 **smprime** price.

The mutation is V-26 marginal-preserving: same 8 rows, same 2 wings, same
(wing, clause) pairs, same multiset of citations, every anchor still on its exact
cited line, FROZEN-N unchanged. Only which wing owns which citation changed.

RED against unmutated code: the mutation is applied to the manifest ROWS in memory,
every shipped node in tests/saturn/test_v20_r15_freeze_manifest.py is called with the
arguments IT declares (read from its own `parametrize` marks, never invented here),
and not one fires.

Run:  python -m pytest tests/mars_v20/test_it20_two_clauses_is_not_two_witnesses.py -q
"""
from __future__ import annotations

import re

import pytest

from tests.saturn import test_v20_r15_freeze_manifest as FM

#: the clauses the manifest editor moves together.
MOVED = ("a", "b", "c")


def _swapped(rows, clauses=MOVED):
    by = {(w, c): (cite, anch) for w, c, cite, anch in rows}
    out = []
    for w, c, cite, anch in rows:
        if c in clauses:
            other = {"W1": "W3", "W3": "W1"}.get(w, w)
            if (other, c) in by:
                cite, anch = by[(other, c)]
        out.append((w, c, cite, anch))
    return out


def _mutated_text(clauses=MOVED):
    """The manifest as a TAMPERING EDITOR would leave it: rows moved AND the
    declared `FREEZE-SHA256` recomputed over them. Freezing is tamper-evidence
    against an edit made after the freeze; it says nothing about a wrong freeze,
    so leaving the old digest in place would be striking the wrong instrument."""
    text = FM._text()
    blk = FM.ROW_BLOCK.search(text)
    lines = []
    by = {}
    for ln in blk.group(1).splitlines():
        s = ln.strip()
        if s and not s.startswith("#"):
            p4 = [x.strip() for x in s.split("|", 3)]
            by[(p4[0], p4[1])] = p4
        lines.append(ln)
    out = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            out.append(ln)
            continue
        w, c, _cite, _anch = [x.strip() for x in s.split("|", 3)]
        other = {"W1": "W3", "W3": "W1"}.get(w, w)
        src = by[(other, c)] if (c in clauses and (other, c) in by) else by[(w, c)]
        out.append(f"{w}|{c}|{src[2]}|{src[3]}")
    body = "\n".join(out) + "\n"
    rows = [tuple(x.strip() for x in ln.split("|", 3))
            for ln in body.splitlines() if ln.strip() and not ln.startswith("#")]
    sha = FM.digest(rows)
    text = text[:blk.start(1)] + body + text[blk.end(1):]
    return re.sub(r"^FREEZE-SHA256\s*=\s*[0-9a-f]{64}\s*$",
                  f"FREEZE-SHA256 = {sha}", text, flags=re.M)


def _shipped_nodes():
    """Every `test_*` in the freeze file, with the arguments IT declares.

    The node's own `parametrize` list is read off `pytestmark`. Inventing the
    arguments is how MARS's it.18 harness went stale when SATURN re-parametrized
    a node: it then raised identically with and without the mutation, and a guard
    that fires for the wrong reason is not a guard.
    """
    out = []
    for name in sorted(n for n in dir(FM) if n.startswith("test_")):
        fn = getattr(FM, name)
        if not callable(fn):
            continue
        argsets = [()]
        for mark in getattr(fn, "pytestmark", []):
            if mark.name == "parametrize":
                names, values = mark.args[0], mark.args[1]
                assert "," not in names, f"{name}: multi-arg parametrize unsupported"
                argsets = [(v,) for v in values]
        out.append((name, fn, argsets))
    return out


def _run_all(nodes):
    fired = []
    for name, fn, argsets in nodes:
        for a in argsets:
            try:
                fn(*a)
            except Exception as exc:
                fired.append(f"{name}{a}: {type(exc).__name__}: {exc}")
    return fired


def test_the_harness_reads_at_least_the_eight_semantic_nodes():
    """Guard on the guard: if the freeze file is renamed out from under this
    harness, the strike must fail loudly rather than pass on an empty node set."""
    nodes = _shipped_nodes()
    assert len(nodes) >= 8, f"only {len(nodes)} shipped nodes found; harness is stale"
    assert any(n == "test_every_wings_arm_is_corroborated_outside_clause_a"
               for n, _f, _a in nodes), "the it.19 second witness is not in the file"


def test_the_true_rows_pass_every_shipped_node():
    """CALIBRATION, asserted before any falsification is claimed."""
    assert _run_all(_shipped_nodes()) == []


def test_the_three_clause_mutation_is_statistic_preserving():
    """The V-26 precondition."""
    orig = FM.rows()
    mut = _swapped(orig)
    assert mut != orig, "the mutation moved nothing; W1/W3 rows are missing"
    assert len(mut) == len(orig)
    assert sorted(w for w, *_ in mut) == sorted(w for w, *_ in orig)
    assert sorted((w, c) for w, c, *_ in mut) == sorted((w, c) for w, c, *_ in orig)
    assert sorted(p for _w, _c, p, _a in mut) == sorted(p for _w, _c, p, _a in orig)
    assert sorted(a for *_x, a in mut) == sorted(a for *_x, a in orig)


def test_the_two_clause_move_is_caught_by_clause_b_not_by_the_new_witness(monkeypatch):
    """SATURN's stated limit, measured. (a)+(c) IS caught -- but by the joint node,
    because `_arm_of` reads clause (a) and clause (b) is checked against it. The new
    witness is not what stops the two-clause move."""
    two = _mutated_text(("a", "c"))
    monkeypatch.setattr(FM, "_text", lambda: two)
    with pytest.raises(AssertionError):
        FM.test_every_clause_b_line_journals_that_wings_own_arm()


def test_the_new_second_witness_does_not_catch_the_three_clause_swap(monkeypatch):
    """THE RED. Move (a), (b) and (c) together; nothing in the freeze file fires."""
    three = _mutated_text()
    monkeypatch.setattr(FM, "_text", lambda: three)
    mut = FM.rows()

    assert FM._arm_of("W1") == "arm_pl", (
        "premise gone: the swap did not move W1's derived arm; re-derive the strike")
    d = [c for w, cl, c, _a in mut if w == "W1" and cl == "d"][0]
    assert "194" in d, f"premise gone: W1 clause (d) is {d}, not the 15.970 price row"

    fired = _run_all(_shipped_nodes())
    assert fired, (
        "W1 is certified `arm_pl` by clauses (a), (b) and (c) at once, while its "
        "clause (d) still cites V17_R4_RETAKE_PRICE.md:194 -- the 15.970 SMPRIME "
        "price -- and not one shipped node fires. Clause (c) is not a witness "
        "OUTSIDE clause (a); it is a third seat in the same closed loop. The only "
        "thing in this repo that catches the move is WING_ARM, the hardcoded map "
        "at tests/saturn/test_v20_r15_it14_saturn.py:39 -- the artifact the it.14 "
        "repair deleted from the manifest test as 'a second place for the manifest "
        "to be wrong'. It is the only place the manifest can be caught being wrong."
    )
