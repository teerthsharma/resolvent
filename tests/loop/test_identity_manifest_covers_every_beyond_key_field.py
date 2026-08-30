"""Every config field the arm-key cannot see must have a must-fire behind it.

WHY THIS FILE EXISTS RATHER THAN AN EDIT. `tests/cameron/test_identity_manifest.py`
is a PRE-SEEDED KEEP: the round-10 contract carries it as "proven this branch;
carried, not re-audited", on the strength of "refusal fires on five config fields
naming the moved field". That claim is true and the five are well chosen -- they
are the fields `_key` is blind to, which is where a collision can hide. This file
does not touch that instrument. It measures whether the five are ALL of them.

THE GAP, MEASURED. `scale/identity_manifest.py:66-70` defines 16 CONFIG_FIELDS,
and its own comment states why the list is wider than the arm key's:

    "is a field two different arms can collide on, which is the whole defect, so
     this list is deliberately wider than `_key`'s: it adds `beta`, `n_neumann`,
     `d_model` and `kind`, and it keeps `torch_version` because a number taken
     under a different torch IS a number taken under a different object."

So the module names five fields as the collision-bearing ones. The must-fire at
`tests/cameron/test_identity_manifest.py:121-126` parametrises five cases, but
they are not the same five:

    must-fire covers   : beta, cell, d_model, n_neumann, task
    named beyond _key  : beta, d_model, kind, n_neumann, torch_version
    BEYOND-KEY, UNTESTED: kind, torch_version

`cell` and `task` are in the key and are covered anyway -- correctly, they are the
headline collision cases. But `kind` -- the field the comment singles out with
"two different arms can collide on" -- has no must-fire, and neither does
`torch_version`, whose whole purpose is to fire on an interpreter upgrade.

"5 of 16" is the wrong way to say this and was the first framing tried: eleven of
the sixteen are in `_key`, so mutating them changes the key and the collision
cannot occur. The real gap is exactly two, and both are fields the module itself
flags.

WHAT THIS TEST DOES NOT CLAIM. It does not assert the refusal is broken. It
asserts the coverage is incomplete, which is a different and weaker thing -- an
untested guard may work perfectly. The repair is two entries in the existing
parametrize list, and it belongs to whoever owns that file. This test goes green
the moment they land, and stays as the guard against the list drifting apart from
CONFIG_FIELDS again.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE = ROOT / "scale" / "identity_manifest.py"
MUSTFIRE = ROOT / "tests" / "cameron" / "test_identity_manifest.py"

#: Named in scale/identity_manifest.py's own comment as the fields added because
#: the arm key cannot see them. Hard-coded rather than parsed out of prose: a
#: regex over a comment would silently return the empty set if the wording moved,
#: and an empty expectation is a test that cannot fail.
BEYOND_KEY = ("beta", "n_neumann", "d_model", "kind", "torch_version")


def config_fields() -> tuple[str, ...]:
    """CONFIG_FIELDS read from the AST, not imported -- import must stay free."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "CONFIG_FIELDS":
                    return tuple(
                        e.value for e in node.value.elts if isinstance(e, ast.Constant)
                    )
    raise AssertionError("CONFIG_FIELDS not found in scale/identity_manifest.py")


def mustfire_fields() -> set[str]:
    """Field names in the parametrize list of the must-fire test."""
    tree = ast.parse(MUSTFIRE.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "test_mutating_one_config_field_makes_the_refusal_fire":
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call) or not dec.args:
                continue
            names = dec.args[0]
            if not isinstance(names, ast.Constant) or "field" not in str(names.value):
                continue
            cases = dec.args[1]
            if isinstance(cases, (ast.List, ast.Tuple)):
                for case in cases.elts:
                    if isinstance(case, ast.Tuple) and case.elts:
                        first = case.elts[0]
                        if isinstance(first, ast.Constant):
                            found.add(first.value)
    return found


def test_the_reader_finds_the_parametrized_cases_at_all():
    """Must-fire for this file's own instrument. An empty read would pass every
    assertion below by vacuity, which is the failure mode MISTAKES.md V-7 records
    for a scan that could not descend to where its values lived."""
    got = mustfire_fields()
    assert got, "read zero parametrized fields; the reader is broken, not the coverage"
    assert got == {"beta", "n_neumann", "d_model", "cell", "task"}, (
        f"the must-fire list changed to {sorted(got)}; update this file's premise "
        "before trusting its verdict"
    )


def test_every_named_beyond_key_field_is_in_config_fields():
    """Premise. If the module dropped one, this file's expectation is stale."""
    fields = config_fields()
    assert len(fields) == 16, f"CONFIG_FIELDS is now {len(fields)}, not 16: {fields}"
    missing = [f for f in BEYOND_KEY if f not in fields]
    assert not missing, f"{missing} no longer in CONFIG_FIELDS; this file is stale"


@pytest.mark.parametrize("field", BEYOND_KEY)
def test_a_collision_bearing_field_has_a_must_fire(field: str):
    """THE GAP, one case per field so a partial repair reads as a partial pass."""
    assert field in mustfire_fields(), (
        f"scale/identity_manifest.py names {field!r} as a field the arm key cannot "
        "see -- 'a field two different arms can collide on, which is the whole "
        "defect' -- and no case in "
        "tests/cameron/test_identity_manifest.py::"
        "test_mutating_one_config_field_makes_the_refusal_fire mutates it. "
        "The guard may well work; it is untested. Repair is one entry in that "
        "parametrize list."
    )
