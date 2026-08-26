"""A sealed journal that legitimately grows must be distinguishable from one edited.

`verify_journal` answers same-or-different, and that is not the question an
append-only record raises. Every honest append changes the root, so a tool that
only compares roots reports every append as a failure -- and a check that fires on
routine behaviour stops being read, which costs more than not having it.

The distinguishing property is that an APPEND leaves the sealed prefix intact:
re-rooting the first `n` lines, where `n` is the count recorded at sealing time,
must reproduce the sealed root exactly. An EDIT to any of those lines cannot. That
is why the seal records the line count alongside the root -- the count is not
metadata, it is the second half of the check.

This was found by running the seal against the tree four iterations after taking
it. Twenty-two journals were unchanged and one had grown by three lines, and the
shipped tool called the grown one False with no way to ask what kind of change it
was.
"""
from __future__ import annotations

import json

import pytest

from scale.merkle import (GENESIS_LABEL, merkle_root, journal_root,
                          verify_journal, verify_append_only)


@pytest.fixture()
def sealed(tmp_path):
    c = tmp_path / "contract.md"
    c.write_text("rules v1\n", encoding="utf-8")
    j = tmp_path / "j.jsonl"
    body = [json.dumps({"i": i, "v": i * 0.5}) for i in range(16)]
    j.write_text("\n".join(body) + "\n", encoding="utf-8")
    return c, j, body, journal_root(j, c), len(body)


def test_append_is_recognised_as_an_append(sealed) -> None:
    c, j, body, root, n = sealed
    j.write_text("\n".join(body + [json.dumps({"i": 99})]) + "\n", encoding="utf-8")
    # The property the old tool could not express.
    assert verify_journal(j, c, root) is False
    assert verify_append_only(j, c, root, n) is True


@pytest.mark.parametrize("pos", list(range(16)))
def test_an_edit_at_every_position_is_caught(sealed, pos: int) -> None:
    """Drawn across every sealed position, not one chosen index.

    A hand-built minimal example is where a control goes vacuous, and eleven have
    been struck in this campaign. Every position is asserted so the check cannot
    pass by happening to test a position the implementation handles.
    """
    c, j, body, root, n = sealed
    tampered = list(body)
    tampered[pos] = json.dumps({"i": pos, "v": pos * 0.5 + 1e-9})
    j.write_text("\n".join(tampered + [json.dumps({"i": 99})]) + "\n", encoding="utf-8")
    assert verify_append_only(j, c, root, n) is False


def test_deletion_inside_the_sealed_prefix_is_caught(sealed) -> None:
    """The tamper an append-only convention hides best: shorten, then re-append."""
    c, j, body, root, n = sealed
    del body[7]
    j.write_text("\n".join(body + [json.dumps({"i": 99}), json.dumps({"i": 100})])
                 + "\n", encoding="utf-8")
    assert verify_append_only(j, c, root, n) is False


def test_reordering_inside_the_sealed_prefix_is_caught(sealed) -> None:
    c, j, body, root, n = sealed
    body[3], body[4] = body[4], body[3]
    j.write_text("\n".join(body + [json.dumps({"i": 99})]) + "\n", encoding="utf-8")
    assert verify_append_only(j, c, root, n) is False


def test_truncation_below_the_sealed_count_is_caught(sealed) -> None:
    """Fewer lines than were sealed cannot be an append, whatever they hash to."""
    c, j, body, root, n = sealed
    j.write_text("\n".join(body[:10]) + "\n", encoding="utf-8")
    assert verify_append_only(j, c, root, n) is False


def test_a_changed_contract_invalidates_the_append_check_too(sealed) -> None:
    """The genesis leaf binds the append check exactly as it binds the root.

    Without this, a journal could be re-interpreted against rules it was never
    kept under merely by appending to it.
    """
    c, j, body, root, n = sealed
    j.write_text("\n".join(body + [json.dumps({"i": 99})]) + "\n", encoding="utf-8")
    assert verify_append_only(j, c, root, n) is True
    c.write_text("rules v2\n", encoding="utf-8")
    assert verify_append_only(j, c, root, n) is False


def test_unchanged_journal_is_a_valid_append_of_zero_lines(sealed) -> None:
    c, j, body, root, n = sealed
    assert verify_journal(j, c, root) is True
    assert verify_append_only(j, c, root, n) is True


def test_the_check_can_fail(sealed) -> None:
    """MUST-FIRE. Without this every assertion above could be vacuous.

    A `verify_append_only` that returned True unconditionally would satisfy the
    append and unchanged cases; a version returning False unconditionally would
    satisfy every tamper case. Both directions are required of the same function
    on the same fixture, which no constant implementation can do.
    """
    c, j, body, root, n = sealed
    j.write_text("\n".join(body + ["{}"]) + "\n", encoding="utf-8")
    assert verify_append_only(j, c, root, n) is True
    j.write_text("\n".join(["{}"] + body) + "\n", encoding="utf-8")
    assert verify_append_only(j, c, root, n) is False


def test_sealed_count_must_be_supplied_not_inferred(sealed) -> None:
    """Inferring the sealed count from the file defeats the check entirely.

    If the count came from the journal itself, any prefix that happened to root
    correctly would validate, and an attacker choosing where to cut would always
    find one. The count is an input from the seal record for that reason.
    """
    c, j, body, root, n = sealed
    with pytest.raises((TypeError, ValueError)):
        verify_append_only(j, c, root, -1)
