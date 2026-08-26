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
