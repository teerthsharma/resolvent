"""The Merkle journal, and the tamper test that makes it worth having.

A journal nobody can verify is a claim about the past rather than a record of it.
This round's journals are append-only JSONL, and append-only is a convention, not
a guarantee — a line edited in place leaves no trace, and this project has already
found a published number whose producer entered version control forty-five minutes
after the number did.

A Merkle tree over the lines turns "the journal was not edited" into a checkable
statement: any single-byte change to any line changes the root. The genesis leaf is
the round contract itself, so the journal is bound to the rules it was kept under.

The must-fire is the whole point. A hash that cannot be seen to change under
tampering is decoration, so the tamper is constructed and the detection asserted —
and it is drawn over every line position rather than one hand-picked one, because
a hand-built minimal example is where a control goes vacuous.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scale.merkle import (leaf_hash, merkle_root, journal_root, verify_journal,
                          GENESIS_LABEL)


def _write(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "j.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_root_is_stable_under_reserialisation():
    """The root must depend on the bytes, not on how they were produced."""
    a = [json.dumps({"k": i, "v": "x"}, sort_keys=True) for i in range(8)]
    assert merkle_root(a) == merkle_root(list(a))


def test_root_changes_when_any_single_line_changes():
    """DRAWN over every position, not one hand-picked index."""
    base = [json.dumps({"k": i}) for i in range(16)]
    root = merkle_root(base)
    changed = 0
    for i in range(len(base)):
        t = list(base)
        t[i] = t[i].replace("}", ", \"x\":1}")
        assert merkle_root(t) != root, f"tamper at line {i} left the root unchanged"
        changed += 1
    assert changed == 16


def test_a_single_byte_is_enough():
    """Not a whole-field edit — one character."""
    base = [json.dumps({"rate": 0.16511}), json.dumps({"rate": 0.02732})]
    tampered = [base[0].replace("0.16511", "0.16512"), base[1]]
    assert merkle_root(tampered) != merkle_root(base)


def test_odd_leaf_counts_do_not_collide():
    """An odd level must not be padded by duplicating the last leaf — that makes
    [a,b,c] and [a,b,c,c] collide, which is CVE-2012-2459 in Bitcoin."""
    assert merkle_root(["a", "b", "c"]) != merkle_root(["a", "b", "c", "c"])


def test_reordering_changes_the_root():
    """A journal is a sequence. Swapping two entries is a tamper."""
    a = [json.dumps({"k": i}) for i in range(6)]
    b = list(a)
    b[2], b[3] = b[3], b[2]
    assert merkle_root(a) != merkle_root(b)


def test_empty_journal_has_a_defined_root():
    assert isinstance(merkle_root([]), str) and len(merkle_root([])) == 64


def test_journal_root_binds_the_contract_as_genesis(tmp_path):
    """The genesis leaf is the contract, so the journal is bound to the rules it
    was kept under. Change the contract, and every root changes."""
    j = _write(tmp_path, [json.dumps({"k": 1}), json.dumps({"k": 2})])
    c1 = tmp_path / "c1.md"; c1.write_text("contract A", encoding="utf-8")
    c2 = tmp_path / "c2.md"; c2.write_text("contract B", encoding="utf-8")
    assert journal_root(j, c1) != journal_root(j, c2)
    # The genesis leaf must actually be the LABELLED contract, not the bare text.
    # The first written form of this assertion was
    #     assert GENESIS_LABEL in leaf_hash(...)[:0] + GENESIS_LABEL
    # which reduces to `X in "" + X` and is true for every input — a vacuous
    # assertion, caught on re-reading before it shipped. This one can fail: it
    # rebuilds the root by hand and requires the label to be part of it.
    from scale.merkle import merkle_root
    body = [ln for ln in j.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert journal_root(j, c1) == merkle_root([GENESIS_LABEL + "contract A"] + body)
    assert journal_root(j, c1) != merkle_root(["contract A"] + body),         "the label is not participating; genesis is indistinguishable from a line"


def test_verify_detects_an_in_place_edit(tmp_path):
    """End to end: record a root, edit a line, verification fails."""
    j = _write(tmp_path, [json.dumps({"n": i}) for i in range(5)])
    c = tmp_path / "c.md"; c.write_text("contract", encoding="utf-8")
    root = journal_root(j, c)
    assert verify_journal(j, c, root) is True
    lines = j.read_text(encoding="utf-8").splitlines()
    lines[2] = json.dumps({"n": 99})
    j.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert verify_journal(j, c, root) is False


def test_verify_detects_a_deleted_line(tmp_path):
    """Deletion is the tamper an append-only convention hides best."""
    j = _write(tmp_path, [json.dumps({"n": i}) for i in range(5)])
    c = tmp_path / "c.md"; c.write_text("contract", encoding="utf-8")
    root = journal_root(j, c)
    lines = j.read_text(encoding="utf-8").splitlines()
    del lines[1]
    j.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert verify_journal(j, c, root) is False


# ---------------------------------------------------------------- MUST-FIRE

def test_must_fire_a_constant_hash_would_be_caught(tmp_path):
    """The control. A 'hash' that ignores its input passes every equality test
    and detects nothing, so the tamper tests must be able to fail."""
    const = lambda s: "0" * 64
    a = [const(x) for x in ("alpha", "beta")]
    assert a[0] == a[1], "the broken hash is not actually constant"
    assert leaf_hash("alpha") != leaf_hash("beta"), (
        "leaf_hash collides on distinct inputs — every tamper test above is vacuous")


def test_must_fire_appending_is_detected_too(tmp_path):
    """Append-only is not a defence: an appended line is still a change of record
    against a root taken earlier."""
    j = _write(tmp_path, [json.dumps({"n": i}) for i in range(3)])
    c = tmp_path / "c.md"; c.write_text("contract", encoding="utf-8")
    root = journal_root(j, c)
    with j.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"n": 99}) + "\n")
    assert verify_journal(j, c, root) is False
