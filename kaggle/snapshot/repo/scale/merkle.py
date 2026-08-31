"""A Merkle journal, so "the record was not edited" becomes checkable.

Append-only JSONL is a convention, not a guarantee. A line edited in place leaves
no trace, a deleted line leaves less, and this project has already found a
published number whose producer entered version control forty-five minutes after
the number did. A hash tree over the lines turns the claim into a statement any
later reader can test: a single byte anywhere changes the root.

THE GENESIS LEAF IS THE CONTRACT. The journal is bound to the rules it was kept
under, so a root does not merely say "these lines, in this order" — it says "these
lines, in this order, under this contract". Change the contract and every root
changes, which is the property that stops a journal being quietly re-interpreted
against rules it was never kept under.

TWO CONSTRUCTION CHOICES THAT ARE NOT COSMETIC:

**Domain separation.** Leaves are hashed with a `0x00` prefix and internal nodes
with `0x01`. Without it an internal node's digest can be presented as a leaf, and
a tree of `n` leaves can be forged as a tree of fewer. The prefix costs one byte
and removes the whole class.

**Odd levels promote, they do not duplicate.** Padding an odd level by repeating
its last node makes `[a, b, c]` and `[a, b, c, c]` produce the same root — the
Bitcoin duplicate-leaf flaw, CVE-2012-2459. An odd node is carried to the next
level unchanged instead, and a test asserts the two do not collide.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["GENESIS_LABEL", "leaf_hash", "merkle_root", "journal_root",
           "verify_append_only",
           "verify_journal"]

#: Prefixed onto the contract bytes so the genesis leaf cannot be confused with
#: an ordinary journal line that happens to hold the same text.
GENESIS_LABEL = "CEQ-GENESIS:"

_LEAF = b"\x00"
_NODE = b"\x01"


def leaf_hash(s: str) -> str:
    """SHA-256 of one journal line, domain-separated as a leaf."""
    return hashlib.sha256(_LEAF + s.encode("utf-8")).hexdigest()


def _pair(a: str, b: str) -> str:
    return hashlib.sha256(_NODE + bytes.fromhex(a) + bytes.fromhex(b)).hexdigest()


def merkle_root(lines: list[str]) -> str:
    """Root over `lines` in order. Reordering, editing, deleting or appending all
    change it.

    An empty journal has a defined root — the hash of the empty leaf — rather than
    an error, because "nothing was recorded" is itself a state worth binding.
    """
    if not lines:
        return hashlib.sha256(_LEAF).hexdigest()
    level = [leaf_hash(x) for x in lines]
    while len(level) > 1:
        nxt = [_pair(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:                 # promote, never duplicate — CVE-2012-2459
            nxt.append(level[-1])
        level = nxt
    return level[0]


def journal_root(journal: Path, contract: Path) -> str:
    """Root of `journal` with `contract` as its genesis leaf.

    Blank lines are dropped: a trailing newline is a file-format artifact, not a
    record, and treating it as a leaf would make a root depend on whether the last
    write flushed a newline.
    """
    genesis = GENESIS_LABEL + contract.read_text(encoding="utf-8", errors="replace")
    body = [ln for ln in journal.read_text(encoding="utf-8",
                                           errors="replace").splitlines() if ln.strip()]
    return merkle_root([genesis] + body)


def verify_journal(journal: Path, contract: Path, expected_root: str) -> bool:
    """True iff the journal under this contract still hashes to `expected_root`."""
    return journal_root(journal, contract) == expected_root


def verify_append_only(journal: Path, contract: Path, sealed_root: str,
                       sealed_lines: int) -> bool:
    """True iff `journal` is `sealed_lines` sealed lines, unedited, plus appends.

    WHY `verify_journal` IS NOT ENOUGH. It answers same-or-different, and that is
    not the question an append-only record raises. Every honest append changes the
    root, so a tool that only compares roots reports routine growth as a failure --
    and a check that fires on normal behaviour stops being read, which costs more
    than not having the check at all. Running the seal against this tree four
    iterations after taking it found twenty-two journals unchanged and one grown by
    three lines, and the only available answer for the grown one was False.

    WHAT DISTINGUISHES THE TWO. An append leaves the sealed prefix intact: re-rooting
    the first `sealed_lines` lines must reproduce `sealed_root` exactly. An edit to
    any line inside that prefix cannot, and neither can a deletion, a reordering, or
    a truncation. So the check is a prefix re-root, and the line count is not
    metadata -- it is the half of the seal that makes the distinction possible.

    `sealed_lines` IS AN INPUT, NEVER INFERRED. Taking it from the journal would
    defeat the check: any prefix that happened to root correctly would validate, and
    a party choosing where to cut would always find one. It comes from the seal
    record, and a negative value is an error rather than a default.
    """
    if sealed_lines < 0:
        raise ValueError(f"sealed_lines must be >= 0, got {sealed_lines}")
    genesis = GENESIS_LABEL + contract.read_text(encoding="utf-8", errors="replace")
    body = [ln for ln in journal.read_text(encoding="utf-8",
                                           errors="replace").splitlines() if ln.strip()]
    if len(body) < sealed_lines:
        return False
    return merkle_root([genesis] + body[:sealed_lines]) == sealed_root
